# Cross-Lingual Contrastive Steering & Arbitration Profiling

A mechanistic-interpretability pipeline that **measures and then mitigates**
cross-modal conflict in Vision-Language Models across a low-resource language
axis. Where [`../inference`](../inference) *measures* whether a VLM trusts its
eyes or defers to a misleading caption, this package goes one step further: it
extracts the model's internal **abstain-vs-assert direction** in the residual
stream and **steers** the model along it at inference time — then asks how well
that honesty direction **transfers across languages**.

Built on top of the `../inference` codebase: same model loader, same four
multilingual counterfactual datasets, same MCQ taxonomy. This package adds
activation caching, contrastive-vector derivation, forward-hook steering, and
cross-lingual transfer evaluation.

---

## The behavioural taxonomy (shared with `../inference`)

Every conflict item is a 4-option MCQ (image-bias / text-bias / distractor,
shuffled across A/B/C, plus a fixed **D = abstain**). The five outcome
categories map one-to-one onto the methodology's metric codes:

| Metric | Category (`../inference/classify.py`) | Meaning |
|--------|----------------------------------------|---------|
| **CR** | `conflict_abstain` (chose D) | Conflict Rate — honest abstention |
| **VR** | `image_bias` | Image Reliance — faithful to the image |
| **TR** | `text_bias` | Text Override — follows the misleading caption (**failure**) |
| **DR** | `distractor` | Distractor |
| **IR** | `other` | Incorrect / parse failure |

---

## The 6-phase pipeline (`run_steering.py`)

| Phase | Name | What it does |
|-------|------|--------------|
| **1** | Perception Control | Score each item image-only (no caption). **Keep only items the model gets right** (`image_bias`), so downstream failures are arbitration errors, not blindness. → `phase1_perception.jsonl` |
| **2** | Conflict Profiling | Re-score the validated set **with** the counterfactual caption (+abstain D). Classify CR/VR/TR/DR/IR and **cache the last-token residual stream at every layer** (+ a logit-lens abstain-vs-answer trace). → `phase2_conflict.jsonl`, `activations/<model>__<lang>.npz` |
| **3** | Target-Layer Localization | Select the decoder-depth band `[frac_lo, frac_hi]` (default 0.5–0.8) for caching and steering. |
| **4** | Contrastive Vector | Per language, per target layer *k*: `v̂⁽ᵏ⁾ = normalize( μ_abstain⁽ᵏ⁾ − μ_assert⁽ᵏ⁾ )` from the **fit split**. → `vectors/<model>.npz` |
| **5** | Inference-Time Steering | Add `α·v̂⁽ᵏ⁾` to the residual stream via forward hooks; sweep α; pick the α that **maximises CR while holding off-target perception accuracy**. → `phase5_alpha_sweep.jsonl` |
| **6** | Cross-Lingual Transfer | Apply every fit-language vector to every target language's **eval split** (all-pairs matrix). Transfer Gap(S→T) = TR[S→T] − TR[T→T]. → `phase6_transfer.jsonl` |

### The "success" set caveat

The contrastive vector points from *assert* (`text_bias`) toward *abstain*
(`conflict_abstain`). Most VLMs abstain rarely, so a language may have too few
abstentions to estimate `μ_abstain`. When that happens (`< min_class`) and
`fallback_success: true`, the success set is widened to `conflict_abstain ∪
image_bias` — i.e. steer toward *"do not blindly follow the misleading text"*.
Each language's actual success set is recorded in `vectors/<model>_meta.json`.

---

## How steering works mechanically (`steer_common.py`)

- **`decoder_layers(lm)`** locates the LM decoder-layer stack across VLM families
  by collecting `*DecoderLayer` modules and picking the group whose size matches
  the LM's reported layer count (skips vision-tower `*EncoderLayer` blocks).
- **`steer(layers, layer_vecs, α)`** is a context manager that registers a
  forward hook on each target layer adding `α·v̂` to that layer's output hidden
  states (all positions), and removes the hooks on exit.
- **`forward_score(...)`** does a single forward pass: argmax over A/B/C/D
  answer-token logits = the answer (the same logit-scoring path as
  `../inference`), and optionally returns per-layer last-token activations and a
  logit-lens `logit(D) − logit(best answer)` trace across depth.

No autoregressive decoding is used for scoring — steering is evaluated on the
same fast single-forward logit path as the benchmark.

---

## Outputs

```
run_config.json                       exact resolved config (hf_token as bool)
phase1_perception.jsonl               per record: perception answer + validated flag
phase2_conflict.jsonl                 per validated record: conflict answer + metric + split
activations/<model>__<lang>.npz       [n,L+1,hidden] acts, lens_diff, categories, split
vectors/<model>.npz                   per-language steering vectors at target layers
vectors/<model>_meta.json             success-set + class counts per language
phase5_alpha_sweep.jsonl              native steering: CR/VR/TR + off-target vs α
phase6_transfer.jsonl                 all-pairs source→target steered CR/TR
errors.jsonl                          per-phase failures

analysis/  (analyze_steering.py)
  table1_conflict_profile.csv         Models × Languages: CR/VR/TR/DR/IR + bootstrap std
  table2_transfer_matrix_<model>.csv  source-vector × target-eval (TR)
  transfer_gap_<model>.csv            TR[S→T], TR[T→T], transfer_gap
  summary_report.md

figures/  (analyze_steering.py)
  fig1_behavioral_drift_<model>.png   100% stacked CR/VR/TR/DR/IR vs ↓ resource
  fig2_localization_shift_<model>.png abstain-vs-answer logit-lens across depth
  fig3_honesty_tradeoff_<model>.png   dual-axis steered CR vs off-target vs α
  fig4_transfer_gap_<model>.png       native-fit vs source-fit per target
```

These cover the required deliverables: **Table 1** (conflict resolution
profile), **Table 2** (transfer matrix), **Figure 1** (behavioural drift),
**Figure 2** (localization shift), **Figure 3** (honesty tradeoff), **Figure 4**
(transfer gap).

---

## Quick Start

```bash
pip install -r ../inference/requirements.txt    # same deps as the benchmark

# Full default run (config.yaml): qwen2.5-vl-3b, all datasets/languages
./run.sh

# Quick smoke run
MODELS="qwen2.5-vl-3b" DATASETS="feliren" LANGUAGES="english hindi telugu" \
  MAX_PER_GROUP=20 EVAL_CAP=30 OUTPUT_DIR=steering_smoke ./run.sh

# Analysis only (re-draw tables/figures from existing artifacts)
python analyze_steering.py --output_dir steering_results
```

`run.sh` runs `run_steering.py` (Phases 1–6) then `analyze_steering.py`
(tables + figures). All knobs are env vars — see the header of `run.sh`.

---

## Config Reference (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | `steering_results` | Output directory |
| `datasets` | all 4 | Datasets to evaluate |
| `languages` | `all` | `"all"` or a list |
| `source_language` | `english` | S whose vector is transferred (Phase 4/6) |
| `models` | `[qwen2.5-vl-3b]` | Models to steer (keys from `../inference/models.py`) |
| `max_samples_per_group` | `null` | Cap rows per (dataset, language); `-1`/`null` = all |
| `eval_fraction` | `0.5` | Fraction of validated items used to **evaluate** steering (rest fit the vector) |
| `target_frac_lo` / `target_frac_hi` | `0.5` / `0.8` | Decoder-depth band for caching + steering |
| `alpha_sweep` | `[-8,-4,0,4,8]` | Steering coefficients (0 = baseline) |
| `success_categories` | `[conflict_abstain]` | Abstain set for the contrast |
| `fallback_success` | `true` | Widen success set with `image_bias` if too few abstentions |
| `min_class` | `5` | Min items per class to derive a vector |
| `eval_cap` | `80` | Cap eval items per language in α-sweep / transfer |
| `dtype` / `device_map` / `attn_impl` | `bfloat16` / `auto` / `sdpa` | Model load options |
| `hf_token` | `null` | **Do not hardcode** — export `HF_TOKEN` |

---

## Notes & limitations

- **Scoring is single-forward logit-argmax** (no decoding). Phases 1/2 run one
  forward per item; the α-sweep and transfer phases re-score eval items once per
  α / source — `eval_cap` bounds this.
- **Steering vectors are derived on the fit split and evaluated on the held-out
  eval split**, so reported steering effects are not fit on their test items.
- **Off-target accuracy** is measured on the *perception-control* condition under
  the same steering, guarding against a vector that buys abstention by breaking
  perception.
- Activation caching stores all `L+1` layers (for the logit-lens). For very large
  sweeps, narrow the depth band or cap rows to bound disk/RAM.
- Dataset inconsistencies are inherited from upstream and acknowledged as a
  dataset limitation, consistent with the benchmark.
