"""
Counterfactual VLM bias benchmark — main engine.

For each (model, dataset, language, record):
  * build a shuffled-option MCQ from the counterfactual caption + image
  * run the VLM, parse its answer, classify image_bias / text_bias / distractor / other
  * optionally capture last-token hidden-state activations per layer (mech-interp)

Outputs (all under output_dir):
  results.jsonl                  per-record rows (config, answer, category, timing)
  aggregate_by_group.csv         counts/rates per (model, dataset, language)
  aggregate_by_model.csv         counts/rates per model
  run_config.json                exact resolved configuration
  hidden_states/<model>.npz      [n, layers, hidden] last-token activations + ids
  errors.jsonl                   any per-model / per-record failures

Usage:
  python vlm_bench.py --config config.yaml [--models qwen2.5-vl-7b ...] [overrides]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

import datasets_adapter as dsa
import models as mdl
import classify as clf


# ── Config ──────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "output_dir": "results",
    "datasets": ["pendulum", "feliren", "remote_sensing", "objects3d"],
    "languages": "all",
    "models": ["qwen2.5-vl-7b"],
    "conditions": ["inference", "perception_control"],
    "max_samples_per_group": 25,
    "shuffle_seed": 1234,
    "max_new_tokens": 24,
    "scoring": "logit",          # "logit" (1 forward pass, fast) | "generate"
    "batch_size": 8,             # logit mode: items per forward pass (OOM auto-halves)
    "dtype": "bfloat16",
    "device_map": "auto",
    "attn_impl": "sdpa",
    "save_hidden_states": True,
    "max_hidden_per_cell": 20,          # stratified: per (dataset, language) cell
    "max_hidden_state_samples": 2000,   # global safety ceiling per (model, condition)
    "force_redownload": False,
    "hf_token": None,
    "cache_dir": None,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        import yaml

        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if v is not None or k in user})
    return cfg


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Inference ───────────────────────────────────────────────────────────────────
def generate_answer(lm, inputs, max_new_tokens, want_hidden):
    import torch

    device = next(lm.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=want_hidden,
    )
    with torch.no_grad():
        out = lm.model.generate(**inputs, **gen_kwargs)

    seq = out.sequences[0]
    gen_ids = seq[input_len:]
    text = lm.processor.batch_decode(
        [gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    hidden_vec = None
    if want_hidden and getattr(out, "hidden_states", None):
        # hidden_states[0] = tuple over layers for the PROMPT forward pass,
        # each [batch, prompt_len, hidden]. Take the last prompt token per layer.
        try:
            prompt_hs = out.hidden_states[0]
            per_layer = [h[0, -1, :].float().cpu().numpy() for h in prompt_hs]
            hidden_vec = np.stack(per_layer, axis=0).astype(np.float16)
        except Exception:
            hidden_vec = None
    return text, hidden_vec


def score_answer(lm, inputs, letter_ids, want_hidden):
    """Logit-scoring fast path: a SINGLE forward pass over the prompt, then read
    the answer-letter (A/B/C/D) logits at the final position; argmax is the answer.

    No autoregressive decoding at all — this is the dominant speedup. The
    last-prompt-token hidden states fall out of the very same forward pass, so
    mech-interp capture is unchanged (and identical semantics to generate()'s
    prompt-pass hidden states).
    """
    import torch

    device = next(lm.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = lm.model(**inputs, output_hidden_states=want_hidden, use_cache=False)

    logits = out.logits[0, -1, :].float()  # next-token logits at last prompt pos
    scores = {}
    for L, ids in letter_ids.items():
        if ids:
            scores[L] = max(float(logits[i]) for i in ids)
    chosen = max(scores, key=scores.get) if scores else None

    hidden_vec = None
    if want_hidden and getattr(out, "hidden_states", None):
        try:
            # forward() returns hidden_states as a tuple over layers directly,
            # each [batch, seq, hidden]; take the last prompt token per layer.
            per_layer = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
            hidden_vec = np.stack(per_layer, axis=0).astype(np.float16)
        except Exception:
            hidden_vec = None
    # Round scores for compact, human-readable logging.
    scores = {L: round(v, 3) for L, v in scores.items()}
    return chosen, scores, hidden_vec


def score_answer_batch(lm, inputs, letter_ids, want_hidden_flags):
    """Batched logit-scoring: ONE forward pass over a padded batch of prompts.

    Inputs are RIGHT-padded so a raw forward() assigns correct position ids to
    each row's real tokens (the trailing pads get bogus positions but are never
    read, and causal attention keeps real tokens from attending to them). For
    each row we read the next-token logits at its last real position and take
    the max logit over each letter's token-id variants; argmax is the answer.
    Per-row last-prompt-token hidden states are sliced from the same pass.

    Returns a list of (chosen_letter, letter_scores, hidden_vec) per row.
    """
    import torch

    device = next(lm.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    need_hidden = any(want_hidden_flags)
    with torch.no_grad():
        out = lm.model(**inputs, output_hidden_states=need_hidden, use_cache=False)

    logits = out.logits  # [B, seq, vocab]
    B = logits.shape[0]
    attn = inputs.get("attention_mask")
    if attn is not None:
        last_idx = [int(attn[i].nonzero(as_tuple=True)[0][-1].item()) for i in range(B)]
    else:
        last_idx = [logits.shape[1] - 1] * B

    results = []
    for i in range(B):
        li = last_idx[i]
        row_logits = logits[i, li, :].float()
        scores = {}
        for L, ids in letter_ids.items():
            if ids:
                scores[L] = max(float(row_logits[t]) for t in ids)
        chosen = max(scores, key=scores.get) if scores else None

        hidden_vec = None
        if want_hidden_flags[i] and need_hidden and getattr(out, "hidden_states", None):
            try:
                per_layer = [h[i, li, :].float().cpu().numpy() for h in out.hidden_states]
                hidden_vec = np.stack(per_layer, axis=0).astype(np.float16)
            except Exception:
                hidden_vec = None
        scores = {L: round(v, 3) for L, v in scores.items()}
        results.append((chosen, scores, hidden_vec))
    return results


def run():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--max_samples_per_group", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--scoring", choices=["logit", "generate"], default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--no_hidden_states", action="store_true")
    ap.add_argument("--max_hidden_per_cell", type=int, default=None)
    ap.add_argument("--max_hidden_state_samples", type=int, default=None)
    ap.add_argument("--force_redownload", action="store_true")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--limit_smoke", type=int, default=None,
                    help="hard cap on total records per model (debug)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir: cfg["output_dir"] = args.output_dir
    if args.models: cfg["models"] = args.models
    if args.datasets: cfg["datasets"] = args.datasets
    if args.languages: cfg["languages"] = args.languages
    if args.conditions: cfg["conditions"] = args.conditions
    if args.max_samples_per_group is not None:
        cfg["max_samples_per_group"] = args.max_samples_per_group
    if args.max_new_tokens is not None: cfg["max_new_tokens"] = args.max_new_tokens
    if args.scoring: cfg["scoring"] = args.scoring
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.attn_impl: cfg["attn_impl"] = args.attn_impl
    if args.dtype: cfg["dtype"] = args.dtype
    if args.no_hidden_states: cfg["save_hidden_states"] = False
    if args.max_hidden_per_cell is not None:
        cfg["max_hidden_per_cell"] = args.max_hidden_per_cell
    if args.max_hidden_state_samples is not None:
        cfg["max_hidden_state_samples"] = args.max_hidden_state_samples
    if args.force_redownload: cfg["force_redownload"] = True
    cfg["hf_token"] = args.hf_token or cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    if cfg.get("max_samples_per_group") in (-1, 0):
        cfg["max_samples_per_group"] = None

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    hs_dir = os.path.join(out_dir, "hidden_states")
    os.makedirs(hs_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.jsonl")
    errors_path = os.path.join(out_dir, "errors.jsonl")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg_to_save = dict(cfg)
    cfg_to_save["hf_token"] = bool(cfg["hf_token"])
    cfg_to_save["run_id"] = run_id
    cfg_to_save["started_at"] = utcnow()
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    def log_err(**kw):
        kw["ts"] = utcnow()
        with open(errors_path, "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    # ── Load all records once (per dataset) ─────────────────────────────────────
    print(f"[{utcnow()}] Loading datasets: {cfg['datasets']}", flush=True)
    all_records = []
    for dkey in cfg["datasets"]:
        try:
            recs = dsa.load_records(
                dkey,
                languages=cfg["languages"],
                max_per_group=cfg["max_samples_per_group"],
                seed=cfg["shuffle_seed"],
                cache_dir=cfg["cache_dir"],
                force_redownload=cfg["force_redownload"],
            )
            print(f"  {dkey}: {len(recs)} records "
                  f"({sorted(set(r.language for r in recs))})", flush=True)
            all_records.extend(recs)
        except Exception as e:
            print(f"  ! failed loading {dkey}: {e}", flush=True)
            log_err(stage="load_dataset", dataset=dkey, error=str(e),
                    tb=traceback.format_exc())
    print(f"Total records: {len(all_records)}", flush=True)
    if not all_records:
        print("No records loaded; aborting.", flush=True)
        return

    results_f = open(results_path, "a")

    # ── Model loop ──────────────────────────────────────────────────────────────
    for mkey in cfg["models"]:
        print(f"\n[{utcnow()}] === Loading model: {mkey} ===", flush=True)
        t0 = time.time()
        try:
            lm = mdl.load_model(
                mkey,
                dtype=cfg["dtype"],
                device_map=cfg["device_map"],
                attn_impl=cfg["attn_impl"],
                cache_dir=cfg["cache_dir"],
                hf_token=cfg["hf_token"],
            )
        except Exception as e:
            print(f"  ! load failed for {mkey}: {e}", flush=True)
            log_err(stage="load_model", model=mkey, error=str(e),
                    tb=traceback.format_exc())
            continue
        print(f"  loaded in {time.time()-t0:.1f}s, layers={lm.num_layers()}", flush=True)

        scoring = cfg.get("scoring", "logit")
        letter_ids = None
        if scoring == "logit":
            letter_ids = mdl.letter_token_ids(lm)
            print(f"  scoring=logit  letter token ids: "
                  f"{ {L: v for L, v in letter_ids.items()} }", flush=True)
        else:
            print(f"  scoring=generate  max_new_tokens={cfg['max_new_tokens']}",
                  flush=True)

        records = all_records
        if args.limit_smoke:
            records = all_records[: args.limit_smoke]

        conditions = cfg["conditions"]
        # Per-condition hidden-state accumulators. Capture is STRATIFIED: up to
        # `max_hidden_per_cell` samples per (dataset, language) cell, bounded by a
        # global safety ceiling `max_hidden_state_samples`. This guarantees every
        # dataset x language cell is represented so the residual-stream activations
        # can be probed per-dataset and per-language, not just per-model.
        hs = {c: {"arrays": [], "meta": [], "cells": {}, "reserved": 0}
              for c in conditions}
        per_cell = cfg.get("max_hidden_per_cell", 20)
        global_cap = cfg.get("max_hidden_state_samples", 10**9) or 10**9
        done = 0
        total = len(records) * len(conditions)
        t_model = time.time()

        def reserve_hidden(cond, cell):
            """Decide AND reserve a stratified hidden-state slot up front, so the
            batched and per-item paths agree on the quota before the forward pass.
            A reserved slot whose capture later fails is simply left empty (rare)."""
            if not cfg["save_hidden_states"]:
                return False
            cells = hs[cond]["cells"]
            if cells.get(cell, 0) >= per_cell or hs[cond]["reserved"] >= global_cap:
                return False
            cells[cell] = cells.get(cell, 0) + 1
            hs[cond]["reserved"] += 1
            return True

        def emit_row(rec, cond, mcq, raw, res, letter_scores, hidden_vec, dt):
            nonlocal done
            if hidden_vec is not None:
                hs[cond]["arrays"].append(hidden_vec)
                hs[cond]["meta"].append(
                    {"uid": rec.uid, "category": res["category"],
                     "dataset": rec.dataset, "language": rec.language})
            row = {
                "run_id": run_id,
                "model": mkey,
                "hf_id": lm.hf_id,
                "condition": cond,
                "dataset": rec.dataset,
                "language": rec.language,
                "row_index": rec.row_index,
                "uid": rec.uid,
                "question": rec.question,
                "cf_caption": rec.cf_caption,
                "original_caption": rec.original_caption,
                "image_bias_answer": rec.image_bias_answer,
                "text_bias_answer": rec.text_bias_answer,
                "distractor": rec.distractor,
                "letter_to_cat": mcq["letter_to_cat"],
                "letter_to_text": mcq["letter_to_text"],
                "raw_output": raw,
                "chosen_letter": res["chosen_letter"],
                "chosen_text": res["chosen_text"],
                "parse_method": res["parse_method"],
                "category": res["category"],
                "scoring": scoring,
                "letter_scores": letter_scores,
                "latency_s": round(dt, 3),
                "has_hidden_state": hidden_vec is not None,
                "max_new_tokens": cfg["max_new_tokens"],
                "dtype": cfg["dtype"],
                "extra": rec.extra,
                "ts": utcnow(),
            }
            results_f.write(json.dumps(row, default=str) + "\n")
            done += 1
            if done % 40 == 0:
                results_f.flush()
                rate = done / (time.time() - t_model)
                print(f"  {mkey}: {done}/{total}  ({rate:.2f} rec/s)", flush=True)

        def run_one_item(rec, cond, mcq, want):
            try:
                inputs = mdl.build_inputs(lm, rec.image, mcq["prompt"])
                t_g = time.time()
                if scoring == "logit":
                    raw, letter_scores, hidden_vec = score_answer(
                        lm, inputs, letter_ids, want)
                    res = clf.classify_letter(raw, mcq)
                else:
                    raw, hidden_vec = generate_answer(
                        lm, inputs, cfg["max_new_tokens"], want)
                    res = clf.classify(raw, mcq)
                    letter_scores = None
                dt = time.time() - t_g
            except Exception as e:
                log_err(stage="inference", model=mkey, uid=rec.uid,
                        condition=cond, error=str(e), tb=traceback.format_exc())
                return
            emit_row(rec, cond, mcq, raw, res, letter_scores, hidden_vec, dt)

        def process_chunk(items):
            """Score a batch of (rec, cond, mcq, want) in ONE forward pass. On
            CUDA OOM, recursively halve the batch; on any other batched failure,
            fall back to per-item so no model is left unsupported."""
            imgs = [it[0].image for it in items]
            prompts = [it[2]["prompt"] for it in items]
            wants = [it[3] for it in items]
            t_g = time.time()
            try:
                inputs = mdl.build_inputs_batch(lm, imgs, prompts)
                outs = score_answer_batch(lm, inputs, letter_ids, wants)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(items) == 1:
                    rec, cond = items[0][0], items[0][1]
                    log_err(stage="inference_oom", model=mkey, uid=rec.uid,
                            condition=cond)
                    return
                mid = len(items) // 2
                process_chunk(items[:mid])
                process_chunk(items[mid:])
                return
            except Exception as e:
                log_err(stage="batch_fallback", model=mkey, error=str(e),
                        tb=traceback.format_exc())
                for rec, cond, mcq, want in items:
                    run_one_item(rec, cond, mcq, want)
                return
            dt = (time.time() - t_g) / max(len(items), 1)
            for (rec, cond, mcq, _), (chosen, scores, hv) in zip(items, outs):
                res = clf.classify_letter(chosen, mcq)
                emit_row(rec, cond, mcq, chosen, res, scores, hv, dt)

        # ── Choose path: batched logit scoring (fast) or per-item ────────────
        batch_size = int(cfg.get("batch_size", 1) or 1)
        use_batch = scoring == "logit" and batch_size > 1 and len(records) > 0
        if use_batch:
            try:  # capability probe: build + forward a 2-item batch
                p_img = [records[0].image, records[0].image]
                p_prompt = [clf.build_mcq(
                    records[0], cfg["shuffle_seed"], conditions[0])["prompt"]] * 2
                p_in = mdl.build_inputs_batch(lm, p_img, p_prompt)
                score_answer_batch(lm, p_in, letter_ids, [False, False])
                print(f"  batched logit scoring enabled (batch_size={batch_size})",
                      flush=True)
            except Exception as e:
                print(f"  batched path unavailable for {mkey} "
                      f"({type(e).__name__}: {e}); using batch=1", flush=True)
                use_batch = False

        if use_batch:
            buf = []
            for rec in records:
                for cond in conditions:
                    mcq = clf.build_mcq(rec, cfg["shuffle_seed"], cond)
                    want = reserve_hidden(cond, (rec.dataset, rec.language))
                    buf.append((rec, cond, mcq, want))
                    if len(buf) >= batch_size:
                        process_chunk(buf)
                        buf = []
            if buf:
                process_chunk(buf)
        else:
            for rec in records:
                for cond in conditions:
                    mcq = clf.build_mcq(rec, cfg["shuffle_seed"], cond)
                    want = reserve_hidden(cond, (rec.dataset, rec.language))
                    run_one_item(rec, cond, mcq, want)

        results_f.flush()
        # Save hidden states per (model, condition).
        for cond in conditions:
            arrs, meta = hs[cond]["arrays"], hs[cond]["meta"]
            if not arrs:
                continue
            try:
                arr = np.stack(arrs, axis=0)  # [n, layers, hidden]
                tag = f"{mkey}__{cond}"
                np.savez_compressed(
                    os.path.join(hs_dir, f"{tag}.npz"),
                    activations=arr,
                    uids=np.array([m["uid"] for m in meta]),
                    categories=np.array([m["category"] for m in meta]),
                    datasets=np.array([m["dataset"] for m in meta]),
                    languages=np.array([m["language"] for m in meta]),
                )
                with open(os.path.join(hs_dir, f"{tag}_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"  saved hidden states: {arr.shape} -> {tag}.npz", flush=True)
            except Exception as e:
                log_err(stage="save_hidden", model=mkey, condition=cond,
                        error=str(e), tb=traceback.format_exc())

        print(f"  {mkey}: {done} responses in {time.time()-t_model:.1f}s", flush=True)
        mdl.free_model(lm)

    results_f.close()
    aggregate(out_dir, results_path)
    print(f"\n[{utcnow()}] DONE. Results in {out_dir}", flush=True)


# ── Aggregation ─────────────────────────────────────────────────────────────────
def aggregate(out_dir: str, results_path: str):
    import pandas as pd

    if not os.path.exists(results_path):
        return
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results_flat.csv"), index=False)

    cats = clf.CATEGORIES

    def summarize(group_cols):
        g = df.groupby(group_cols)
        out = g.size().rename("n").reset_index()
        for c in cats:
            cnt = g["category"].apply(lambda s, c=c: (s == c).sum()).reset_index(
                name=f"n_{c}")
            out = out.merge(cnt, on=group_cols)
        for c in cats:
            out[f"rate_{c}"] = (out[f"n_{c}"] / out["n"]).round(4)
        return out

    has_cond = "condition" in df.columns
    cond_prefix = ["condition"] if has_cond else []

    summarize(cond_prefix + ["model", "dataset", "language"]).to_csv(
        os.path.join(out_dir, "aggregate_by_group.csv"), index=False)
    summarize(cond_prefix + ["model", "dataset"]).to_csv(
        os.path.join(out_dir, "aggregate_by_dataset.csv"), index=False)
    summarize(cond_prefix + ["model", "language"]).to_csv(
        os.path.join(out_dir, "aggregate_by_language.csv"), index=False)
    summarize(cond_prefix + ["model"]).to_csv(
        os.path.join(out_dir, "aggregate_by_model.csv"), index=False)

    # Console summary (per condition x model)
    print("\n==== Bias summary (rate per category) ====", flush=True)
    bm = summarize(cond_prefix + ["model"])
    cols = cond_prefix + ["model", "n"] + [f"rate_{c}" for c in cats]
    print(bm[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
