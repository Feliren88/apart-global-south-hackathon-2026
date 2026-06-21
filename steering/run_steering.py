"""
Cross-Lingual Contrastive Steering & Arbitration Profiling in VLMs — pipeline.

Executes the 6-phase methodology on top of the ../inference codebase (same model
loader, datasets, MCQ taxonomy). Run analysis/figures afterwards with
`analyze_steering.py`.

  Phase 1  Perception Control       image-only; keep instances the model gets
                                    right (isolate arbitration from blindness)
  Phase 2  Conflict Profiling       image + counterfactual caption (+abstain D);
                                    classify CR/VR/TR/DR/IR and cache the
                                    last-token residual stream at every layer
  Phase 3  Target-Layer Localization decoder depth band [frac_lo, frac_hi]
  Phase 4  Contrastive Vector       v_hat^(k) = norm( mu_abstain - mu_assert )
  Phase 5  Inference-Time Steering  add alpha * v_hat^(k); sweep alpha; pick the
                                    alpha maximising abstention (CR) while
                                    holding off-target perception accuracy
  Phase 6  Cross-Lingual Transfer   apply the source-language (English) vector to
                                    other languages; measure the transfer gap

Artifacts (under output_dir):
  run_config.json
  phase1_perception.jsonl          per record: perception answer + validated flag
  phase2_conflict.jsonl            per validated record: conflict answer + metric
  activations/<model>__<lang>.npz  [n,L+1,hidden] acts, lens trace, cats, split
  vectors/<model>.npz              per-language steering vectors at target layers
  phase5_alpha_sweep.jsonl         native steering CR vs off-target accuracy
  phase6_transfer.jsonl            cross-lingual source->target steering metrics

Usage:
  python run_steering.py --config config.yaml [--models qwen2.5-vl-3b ...]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch

import steer_common as sc
from steer_common import mdl, clf, dsa


DEFAULT_CONFIG = {
    "output_dir": "steering_results",
    "datasets": ["feliren", "pendulum", "remote_sensing", "objects3d"],
    "languages": "all",
    "source_language": "english",     # S in Phase 4/6 (English-fit source vector)
    "models": ["qwen2.5-vl-3b"],
    "max_samples_per_group": None,
    "shuffle_seed": 1234,
    "eval_fraction": 0.5,             # held-out split for steering evaluation
    "target_frac_lo": 0.5,            # decoder depth band for caching/steering
    "target_frac_hi": 0.8,
    "alpha_sweep": [-8.0, -4.0, 0.0, 4.0, 8.0],
    "success_categories": ["conflict_abstain"],   # abstain set for the contrast
    "fallback_success": True,         # if too few abstentions, add image_bias
    "min_class": 5,                   # min items per class to derive a vector
    "eval_cap": 80,                   # cap items per language for the steer sweep
    "batch_size": 8,                  # forward-pass batch size (all phases)
    "off_target_tol": 0.15,           # max off-target perception drop allowed when
                                      # selecting the steering alpha (Phase 5)
    "dtype": "bfloat16",
    "device_map": "auto",
    "attn_impl": "sdpa",
    "force_redownload": False,
    "hf_token": None,
    "cache_dir": None,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        import yaml
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if v is not None or k in user})
    return cfg


def split_eval(uid: str, eval_fraction: float, seed: int) -> str:
    """Deterministic fit/eval assignment from a uid hash (stable across phases)."""
    import hashlib
    h = int(hashlib.md5(f"{seed}:{uid}".encode()).hexdigest(), 16) % 1000
    return "eval" if (h / 1000.0) < eval_fraction else "fit"


class Writer:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, obj):
        self.f.write(json.dumps(obj, default=str) + "\n")

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--source_language", default=None)
    ap.add_argument("--max_samples_per_group", type=int, default=None)
    ap.add_argument("--alpha_sweep", nargs="*", type=float, default=None)
    ap.add_argument("--eval_cap", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--min_class", type=int, default=None)
    ap.add_argument("--eval_fraction", type=float, default=None)
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--force_redownload", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k in ("output_dir", "models", "datasets", "languages", "source_language",
              "alpha_sweep", "eval_cap", "attn_impl", "dtype"):
        v = getattr(args, k, None)
        if v:
            cfg[k] = v
    if args.max_samples_per_group is not None:
        cfg["max_samples_per_group"] = args.max_samples_per_group
    if args.min_class is not None:
        cfg["min_class"] = args.min_class
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.eval_fraction is not None:
        cfg["eval_fraction"] = args.eval_fraction
    if args.force_redownload:
        cfg["force_redownload"] = True
    cfg["hf_token"] = args.hf_token or cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    if cfg.get("max_samples_per_group") in (-1, 0):
        cfg["max_samples_per_group"] = None

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    act_dir = os.path.join(out_dir, "activations")
    vec_dir = os.path.join(out_dir, "vectors")
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(vec_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg_save = dict(cfg, hf_token=bool(cfg["hf_token"]), run_id=run_id,
                    started_at=utcnow())
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg_save, f, indent=2)

    errors_path = os.path.join(out_dir, "errors.jsonl")

    def log_err(**kw):
        kw["ts"] = utcnow()
        with open(errors_path, "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    seed = cfg["shuffle_seed"]
    src_lang = str(cfg["source_language"]).lower()

    # ── Load records once (shared across models) ─────────────────────────────
    print(f"[{utcnow()}] Loading datasets {cfg['datasets']}", flush=True)
    all_records = []
    for dkey in cfg["datasets"]:
        try:
            recs = dsa.load_records(
                dkey, languages=cfg["languages"],
                max_per_group=cfg["max_samples_per_group"], seed=seed,
                cache_dir=cfg["cache_dir"], force_redownload=cfg["force_redownload"])
            print(f"  {dkey}: {len(recs)} records "
                  f"{sorted(set(r.language for r in recs))}", flush=True)
            all_records.extend(recs)
        except Exception as e:
            print(f"  ! failed {dkey}: {e}", flush=True)
            log_err(stage="load_dataset", dataset=dkey, error=str(e),
                    tb=traceback.format_exc())
    if not all_records:
        print("No records; aborting.", flush=True)
        return
    print(f"Total records: {len(all_records)}", flush=True)

    w1 = Writer(os.path.join(out_dir, "phase1_perception.jsonl"))
    w2 = Writer(os.path.join(out_dir, "phase2_conflict.jsonl"))
    w5 = Writer(os.path.join(out_dir, "phase5_alpha_sweep.jsonl"))
    w6 = Writer(os.path.join(out_dir, "phase6_transfer.jsonl"))

    for mkey in cfg["models"]:
        print(f"\n[{utcnow()}] ===== MODEL {mkey} =====", flush=True)
        t0 = time.time()
        try:
            lm = mdl.load_model(mkey, dtype=cfg["dtype"], device_map=cfg["device_map"],
                                attn_impl=cfg["attn_impl"], cache_dir=cfg["cache_dir"],
                                hf_token=cfg["hf_token"])
        except Exception as e:
            print(f"  ! load failed: {e}", flush=True)
            log_err(stage="load_model", model=mkey, error=str(e),
                    tb=traceback.format_exc())
            continue
        letter_ids = mdl.letter_token_ids(lm)
        try:
            layers = sc.decoder_layers(lm)
        except Exception as e:
            print(f"  ! cannot locate decoder layers: {e}", flush=True)
            log_err(stage="decoder_layers", model=mkey, error=str(e))
            mdl.free_model(lm)
            continue
        n_layers = len(layers)
        target_idxs = sc.target_layer_indices(
            n_layers, cfg["target_frac_lo"], cfg["target_frac_hi"])
        lens = sc.LensTools(lm, letter_ids)
        device = next(lm.model.parameters()).device
        print(f"  loaded in {time.time()-t0:.1f}s | layers={n_layers} | "
              f"target band={target_idxs[0]}..{target_idxs[-1]} | "
              f"lens={'on' if lens.usable else 'off'}", flush=True)

        bs = int(cfg.get("batch_size", 8))

        def build_batches(recs, condition):
            """Right-padded batches of (inputs, [(rec, mcq), ...]) of size <= bs.
            The image preprocessor runs once per batch; for the steering phases
            the returned batches are reused across every alpha / source vector.

            If a chunk fails to batch (e.g. an image the batched processor can't
            collate), it is stored with inputs=None and scored per-item later, so
            one awkward image never blocks the rest of the run."""
            out = []
            for i in range(0, len(recs), bs):
                chunk = recs[i:i + bs]
                mcqs = [clf.build_mcq(r, seed, condition) for r in chunk]
                metas = list(zip(chunk, mcqs))
                try:
                    inputs = mdl.build_inputs_batch(
                        lm, [r.image for r in chunk], [m["prompt"] for m in mcqs])
                except Exception as e:
                    log_err(stage="build_batch", model=mkey, error=str(e),
                            n=len(metas))
                    inputs = None
                out.append((inputs, metas))
            return out

        def _score_one(rec, mcq, want_hidden, want_lens, layer_vecs, alpha):
            inputs = mdl.build_inputs(lm, rec.image, mcq["prompt"])
            lt = lens if want_lens else None
            if layer_vecs and alpha:
                with sc.steer(layers, layer_vecs, alpha):
                    ch, _, hid, ld = sc.forward_score(lm, inputs, letter_ids,
                                                      want_hidden, lt)
            else:
                ch, _, hid, ld = sc.forward_score(lm, inputs, letter_ids,
                                                  want_hidden, lt)
            return clf.classify_letter(ch, mcq), hid, ld

        def run_prebuilt(batches, stage, want_hidden=False, want_lens=False,
                         layer_vecs=None, alpha=0.0):
            """Score prebuilt batches, yielding (rec, mcq, res, hid, ld) per item.
            On a batch failure, fall back to per-item scoring so a single bad row
            never drops its whole chunk."""
            lt = lens if want_lens else None

            def per_item(metas):
                for rec, mcq in metas:
                    try:
                        res, hid, ld = _score_one(rec, mcq, want_hidden,
                                                  want_lens, layer_vecs, alpha)
                        yield rec, mcq, res, hid, ld
                    except Exception as e2:
                        log_err(stage=stage, model=mkey, uid=rec.uid, error=str(e2))

            for inputs, metas in batches:
                if inputs is None:           # chunk failed to batch-build
                    yield from per_item(metas)
                    continue
                try:
                    if layer_vecs and alpha:
                        with sc.steer(layers, layer_vecs, alpha):
                            chosen, _, hids, lds = sc.forward_score_batch(
                                lm, inputs, letter_ids, want_hidden, lt)
                    else:
                        chosen, _, hids, lds = sc.forward_score_batch(
                            lm, inputs, letter_ids, want_hidden, lt)
                    for k, (rec, mcq) in enumerate(metas):
                        yield rec, mcq, clf.classify_letter(chosen[k], mcq), \
                            hids[k], lds[k]
                except Exception as e:
                    log_err(stage=stage + "_batch", model=mkey, error=str(e),
                            n=len(metas))
                    yield from per_item(metas)

        def stream_scored(recs, condition, stage, want_hidden=False,
                          want_lens=False):
            """Build + score one batch at a time (no reuse) — for phases that see
            each item once, so batched inputs are not held in RAM all at once."""
            for i in range(0, len(recs), bs):
                batches = build_batches(recs[i:i + bs], condition)
                yield from run_prebuilt(batches, stage, want_hidden, want_lens)

        # ── Phase 1: perception control → validated set ──────────────────────
        print(f"  [P1] perception control over {len(all_records)} records "
              f"(batch={bs})", flush=True)
        validated = []
        done = 0
        for rec, mcq, res, _, _ in stream_scored(
                all_records, "perception_control", "phase1"):
            ok = res["category"] == "image_bias"
            w1.write({"model": mkey, "uid": rec.uid, "dataset": rec.dataset,
                      "language": rec.language, "chosen": res["chosen_letter"],
                      "category": res["category"], "validated": ok})
            if ok:
                validated.append(rec)
            done += 1
            if done % 100 == 0:
                w1.flush()
                print(f"      {done}/{len(all_records)}  kept={len(validated)}",
                      flush=True)
        w1.flush()
        print(f"  [P1] validated {len(validated)}/{len(all_records)}", flush=True)

        # ── Phase 2/3: conflict profiling + activation capture (per language) ─
        by_lang: dict[str, list] = {}
        for rec in validated:
            by_lang.setdefault(rec.language, []).append(rec)

        # in-memory per-language store for vector derivation + steering
        store: dict[str, dict] = {}
        for lang, recs in by_lang.items():
            acts, lensd, cats, uids, dsets, splits = [], [], [], [], [], []
            for rec, mcq, res, hid, ld in stream_scored(
                    recs, "inference", "phase2", want_hidden=True, want_lens=True):
                sp = split_eval(rec.uid, cfg["eval_fraction"], seed)
                w2.write({"model": mkey, "uid": rec.uid, "dataset": rec.dataset,
                          "language": lang, "chosen": res["chosen_letter"],
                          "category": res["category"],
                          "metric": sc.CATEGORY_TO_METRIC.get(res["category"], "IR"),
                          "split": sp})
                if hid is not None:
                    acts.append(hid)
                    lensd.append(ld if ld is not None else np.zeros(hid.shape[0],
                                                                    dtype=np.float32))
                    cats.append(res["category"])
                    uids.append(rec.uid)
                    dsets.append(rec.dataset)
                    splits.append(sp)
            w2.flush()
            if not acts:
                continue
            A = np.stack(acts, 0)                       # [n, L+1, hidden]
            store[lang] = {
                "acts": A,
                "lens": np.stack(lensd, 0),
                "cats": np.array(cats),
                "uids": np.array(uids),
                "splits": np.array(splits),
                "recs": {r.uid: r for r in recs},
            }
            np.savez_compressed(
                os.path.join(act_dir, f"{mkey}__{lang}.npz"),
                activations=A, lens_diff=store[lang]["lens"],
                categories=store[lang]["cats"], uids=store[lang]["uids"],
                datasets=np.array(dsets), splits=store[lang]["splits"],
                target_layers=np.array(target_idxs))
            metr = [sc.CATEGORY_TO_METRIC.get(c, "IR") for c in cats]
            prof = {m: round(float(np.mean(np.array(metr) == m)), 3)
                    for m in sc.METRICS}
            print(f"  [P2] {lang}: n={len(cats)} profile={prof}", flush=True)

        # ── Phase 4: contrastive vectors (fit split), per language ───────────
        succ_default = list(cfg["success_categories"])
        vectors: dict[str, dict[int, np.ndarray]] = {}
        vec_save: dict[str, np.ndarray] = {}
        meta_lang = {}
        for lang, S in store.items():
            fit = S["splits"] == "fit"
            cats = S["cats"][fit]
            acts = S["acts"][fit]
            succ_cats = succ_default
            abstain_mask = np.isin(cats, succ_cats)
            if abstain_mask.sum() < cfg["min_class"] and cfg["fallback_success"]:
                succ_cats = list(dict.fromkeys(succ_default + ["image_bias"]))
                abstain_mask = np.isin(cats, succ_cats)
            assert_mask = cats == "text_bias"
            if abstain_mask.sum() < cfg["min_class"] or assert_mask.sum() < cfg["min_class"]:
                print(f"  [P4] {lang}: SKIP (abstain={int(abstain_mask.sum())}, "
                      f"assert={int(assert_mask.sum())} < min_class)", flush=True)
                continue
            layer_vecs = {}
            stacked = []
            for j in target_idxs:
                v = sc.contrastive_vector(acts[abstain_mask], acts[assert_mask], j + 1)
                if v is None:
                    continue
                layer_vecs[j] = torch.tensor(v, device=device)
                stacked.append(v)
            if not layer_vecs:
                continue
            vectors[lang] = layer_vecs
            vec_save[lang] = np.stack(stacked, 0)       # [len(target), hidden]
            meta_lang[lang] = {"success_categories": succ_cats,
                               "n_abstain": int(abstain_mask.sum()),
                               "n_assert": int(assert_mask.sum())}
            print(f"  [P4] {lang}: vector from abstain={int(abstain_mask.sum())} "
                  f"assert={int(assert_mask.sum())} succ={succ_cats}", flush=True)
        if vec_save:
            np.savez_compressed(os.path.join(vec_dir, f"{mkey}.npz"),
                                target_layers=np.array(target_idxs),
                                source_language=src_lang, **vec_save)
            with open(os.path.join(vec_dir, f"{mkey}_meta.json"), "w") as f:
                json.dump(meta_lang, f, indent=2)

        # ── Phase 5: native alpha sweep (eval split) ─────────────────────────
        alphas = [float(a) for a in cfg["alpha_sweep"]]
        cap = cfg["eval_cap"]
        # Pre-build the eval batches ONCE per language (both conditions). They are
        # reused across every alpha here and across every source vector in Phase 6
        # — only the steering hook changes between passes, never the inputs.
        eval_inf: dict[str, list] = {}    # lang -> inference batches
        eval_perc: dict[str, list] = {}   # lang -> perception-control batches
        for lang, S in store.items():
            ev = S["splits"] == "eval"
            ev_uids = list(S["uids"][ev])[:cap]
            recs_map = S["recs"]
            ev_recs = [recs_map[u] for u in ev_uids if u in recs_map]
            eval_inf[lang] = build_batches(ev_recs, "inference")
            eval_perc[lang] = build_batches(ev_recs, "perception_control")

        best_alpha: dict[str, float] = {}      # coverage-constrained (reporting)
        best_cr_alpha: dict[str, float] = {}   # max-CR alpha incl. sign (for transfer)
        for lang, layer_vecs in vectors.items():
            results = []
            for a in alphas:
                lv = layer_vecs if a != 0.0 else None
                cr = vr = tr = n = 0
                off_ok = off_n = 0
                for rec, mcq, res, _, _ in run_prebuilt(
                        eval_inf[lang], "phase5", layer_vecs=lv, alpha=a):
                    m = sc.CATEGORY_TO_METRIC.get(res["category"], "IR")
                    cr += (m == "CR"); vr += (m == "VR"); tr += (m == "TR"); n += 1
                for rec, mcq, ro, _, _ in run_prebuilt(
                        eval_perc[lang], "phase5", layer_vecs=lv, alpha=a):
                    off_ok += (ro["category"] == "image_bias"); off_n += 1
                if n == 0:
                    continue
                row = {"model": mkey, "language": lang, "alpha": a, "n": n,
                       "CR": round(cr / n, 4), "VR": round(vr / n, 4),
                       "TR": round(tr / n, 4),
                       "off_target_acc": round(off_ok / max(off_n, 1), 4)}
                w5.write(row)
                results.append(row)
            w5.flush()
            # select alpha: maximise CR subject to off-target >= baseline - 0.1
            base = next((r for r in results if r["alpha"] == 0.0), None)
            base_off = base["off_target_acc"] if base else 0.0
            tol = cfg.get("off_target_tol", 0.15)
            ok = [r for r in results if r["off_target_acc"] >= base_off - tol]
            pick = max(ok or results, key=lambda r: r["CR"])
            best_alpha[lang] = pick["alpha"]
            # Per-source alpha for the transfer matrix: the (signed) alpha that
            # maximally activates THIS vector, so each source gets its best shot
            # (removes the polarity confound from mixed success sets).
            nonzero = [r for r in results if r["alpha"] != 0.0]
            best_cr_alpha[lang] = (max(nonzero, key=lambda r: r["CR"])["alpha"]
                                   if nonzero else 0.0)
            print(f"  [P5] {lang}: best alpha={pick['alpha']} "
                  f"CR {base['CR'] if base else 0:.2f}->{pick['CR']:.2f} "
                  f"off-target {base_off:.2f}->{pick['off_target_acc']:.2f}",
                  flush=True)

        # ── Phase 6: cross-lingual transfer — full all-pairs matrix ──────────
        # Apply every fit-language vector to every target language's eval items.
        # Each SOURCE vector is applied at its own best (signed) alpha, so the
        # matrix measures how well a language's learned abstain direction —
        # at the strength that works for it — transfers to other languages.
        # The diagonal (source==target) is the native-fit reference.
        if vectors:
            fallback_a = max(alphas, key=abs) if alphas else 0.0
            print(f"  [P6] all-pairs transfer ({len(vectors)} fit-langs), "
                  f"per-source alpha={ {l: best_cr_alpha.get(l, fallback_a) for l in vectors} }",
                  flush=True)
            for tgt in store:
                for fit_lang, fit_vecs in vectors.items():
                    a = best_cr_alpha.get(fit_lang) or fallback_a
                    cr = tr = n = 0
                    off_ok = off_n = 0
                    for rec, mcq, res, _, _ in run_prebuilt(
                            eval_inf[tgt], "phase6", layer_vecs=fit_vecs, alpha=a):
                        m = sc.CATEGORY_TO_METRIC.get(res["category"], "IR")
                        cr += (m == "CR"); tr += (m == "TR"); n += 1
                    for rec, mcq, ro, _, _ in run_prebuilt(
                            eval_perc[tgt], "phase6", layer_vecs=fit_vecs, alpha=a):
                        off_ok += (ro["category"] == "image_bias"); off_n += 1
                    if n == 0:
                        continue
                    t_alpha = a
                    w6.write({"model": mkey, "source": fit_lang, "target": tgt,
                              "fit": "native" if fit_lang == tgt else "cross",
                              "alpha": t_alpha, "n": n,
                              "CR": round(cr / n, 4), "TR": round(tr / n, 4),
                              "off_target_acc": round(off_ok / max(off_n, 1), 4)})
            w6.flush()

        print(f"  {mkey}: done in {time.time()-t0:.1f}s", flush=True)
        mdl.free_model(lm)

    for w in (w1, w2, w5, w6):
        w.close()
    print(f"\n[{utcnow()}] PIPELINE DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
