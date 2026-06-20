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
    "dtype": "bfloat16",
    "device_map": "auto",
    "attn_impl": None,
    "save_hidden_states": True,
    "max_hidden_state_samples": 150,
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


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--max_samples_per_group", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--no_hidden_states", action="store_true")
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
    if args.dtype: cfg["dtype"] = args.dtype
    if args.no_hidden_states: cfg["save_hidden_states"] = False
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

        records = all_records
        if args.limit_smoke:
            records = all_records[: args.limit_smoke]

        conditions = cfg["conditions"]
        # per-condition hidden-state accumulators
        hs = {c: {"arrays": [], "meta": [], "n": 0} for c in conditions}
        done = 0
        total = len(records) * len(conditions)
        t_model = time.time()
        for rec in records:
            for cond in conditions:
                mcq = clf.build_mcq(rec, cfg["shuffle_seed"], cond)
                try:
                    inputs = mdl.build_inputs(lm, rec.image, mcq["prompt"])
                    want_hidden = (
                        cfg["save_hidden_states"]
                        and hs[cond]["n"] < cfg["max_hidden_state_samples"]
                    )
                    t_g = time.time()
                    raw, hidden_vec = generate_answer(
                        lm, inputs, cfg["max_new_tokens"], want_hidden
                    )
                    dt = time.time() - t_g
                except Exception as e:
                    log_err(stage="inference", model=mkey, uid=rec.uid,
                            condition=cond, error=str(e), tb=traceback.format_exc())
                    continue

                res = clf.classify(raw, mcq)
                if hidden_vec is not None:
                    hs[cond]["arrays"].append(hidden_vec)
                    hs[cond]["meta"].append(
                        {"uid": rec.uid, "category": res["category"],
                         "dataset": rec.dataset, "language": rec.language})
                    hs[cond]["n"] += 1

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
