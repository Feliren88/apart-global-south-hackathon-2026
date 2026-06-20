# Benchmark Results: `multicond_4opt`

Full counterfactual VLM bias evaluation across **4 models, 4 datasets, 5 languages, 2 conditions**. Generated `2026-06-20` by `vlm_bench.py`.

---

## Run Configuration

| Setting | Value |
|---------|-------|
| Models | `qwen2.5-vl-7b`, `internvl3-8b`, `qwen3-vl-8b`, `llava-onevision-7b` |
| Datasets | `pendulum`, `feliren`, `remote_sensing`, `objects3d` |
| Languages | `english`, `hindi`, `urdu`, `telugu`, `bahasa indonesia` |
| Conditions | `inference`, `perception_control` |
| Max samples per group | 25 |
| Hidden states | captured (≤150 per condition) |
| Samples per run | 3,600 (4 models × 4 datasets × 5 languages × 2 conditions × ≤25 rows) |

---

## File Guide

### Raw Results

| File | Lines | Description |
|------|-------|-------------|
| `results.jsonl` | 3,600 | One JSON object per (model, dataset, language, condition, row) — every field including raw output, parsed answer, latency, MCQ mapping |
| `results_flat.csv` | 3,601 | Same data as flat table (header + 3,600 rows) |

### Aggregated Reports

| File | Granularity | Rows | Columns |
|------|-------------|------|---------|
| `aggregate_by_group.csv` | (model, dataset, language, condition) | per-group | `n`, `n_image_bias`, `n_text_bias`, ..., `rate_*` |
| `aggregate_by_dataset.csv` | (model, dataset) | per-dataset | same |
| `aggregate_by_language.csv` | (model, language) | per-language | same |
| `aggregate_by_model.csv` | (model) | per-model | same |

All rates are proportions `[0, 1]`. See `run_config.json` for exact configuration.

### Hidden States (`hidden_states/`)

8 compressed `.npz` files (one per model × condition):

| File | Shape | Description |
|------|-------|-------------|
| `qwen2.5-vl-7b__inference.npz` | `[150, 28, 3584]` | Last-prompt-token activations at all 28 layers, 3584-dim hidden |
| `qwen2.5-vl-7b__perception_control.npz` | same | Same model, perception-only condition |
| `internvl3-8b__*.npz` | `[150, 32, 4096]` | 32 layers |
| `llava-onevision-7b__*.npz` | `[150, 32, 4096]` | 32 layers |
| `qwen3-vl-8b__*.npz` | `[150, 32, 4096]` | 32 layers |

Each `.npz` contains:
- `activations`: `[n, layers, hidden]` float16
- `uids`, `categories`, `datasets`, `languages`: metadata arrays

Companion `_meta.json` files with the same data in readable JSON.

### Analysis (`analysis/`)

| File | Description |
|------|-------------|
| `summary_report.md` | Full report: per-model bias rates, per-language breakdowns, condition comparison (override gap), probe scores |
| `condition_comparison.csv` | Per-model: perception_ceiling, inference_image_bias, override_gap, override_share |
| `probe_scores.csv` | Per (model, layer) logistic-regression CV accuracy for decoding image_bias vs text_bias |

### Figures (`figures/`)

| File | Content |
|------|---------|
| `bias_by_model.png` | Stacked bar of 5 answer categories per model |
| `text_bias_by_language.png` | Heatmap: model × language text-following rate |
| `faithfulness_by_language.png` | Heatmap: model × language image-faithfulness rate |
| `bias_by_dataset.png` | Heatmap: model × dataset text-following rate |
| `refusal_other_by_model.png` | "other" / unparsed rate per model |
| `condition_comparison.png` | Perception ceiling vs inference image-bias (override gap) |
| `probe_layerwise_*.png` | Per model × condition: layer-wise linear probe accuracy (image vs text bias) |
| `pca_lastlayer_*.png` | Per model × condition: PCA of last-layer activations coloured by category |

---

## Headline Results (Inference Condition)

| Model | Image-Bias | Text-Bias | Conflict-Abstain | Override-Share |
|-------|-----------|-----------|-----------------|---------------|
| qwen2.5-vl-7b | 5.3% | 10.2% | 84.0% | 47.6% |
| qwen3-vl-8b | 14.9% | 20.2% | 60.4% | 26.7% |
| internvl3-8b | 24.0% | 41.6% | 29.8% | 26.7% |
| llava-onevision-7b | 33.6% | 52.9% | 3.6% | 26.7% |

Key observations:
- **Qwen2.5-VL-7B** is the most conservative — 84% abstention rate, lowest text-bias. But also the lowest image-faithfulness (5.3%), suggesting it defaults to abstention rather than resolving the conflict.
- **LLaVA-OneVision-7B** has the highest text-bias (52.9%) and near-zero abstention — it follows the caption confidently even when wrong.
- **InternVL3-8B** has the best balance of abstention, image-faithfulness, and text-bias detection.

---

## Data Dictionary

Every `results.jsonl` record:

```json
{
  "run_id":           "20260620_195253",
  "model":            "qwen2.5-vl-7b",
  "hf_id":            "Qwen/Qwen2.5-VL-7B-Instruct",
  "condition":        "inference | perception_control",
  "dataset":          "feliren | pendulum | remote_sensing | objects3d",
  "language":         "english | hindi | ...",
  "row_index":        "<int>",
  "uid":              "<dataset>__<language>__<row_index>",
  "question":         "The MCQ question",
  "cf_caption":       "The counterfactual (misleading) caption",
  "original_caption": "The faithful caption",
  "image_bias_answer":"Correct answer per image",
  "text_bias_answer": "Correct answer per caption (the misleading one)",
  "distractor":       "Plausible-but-wrong distractor",
  "letter_to_cat":    "Shuffled A/B/C/D → category mapping",
  "letter_to_text":   "Shuffled A/B/C/D → answer text mapping",
  "raw_output":       "Model's raw text output",
  "chosen_letter":    "Parsed answer letter (A/B/C/D or null)",
  "chosen_text":      "Parsed answer text",
  "parse_method":     "Which parser strategy matched",
  "category":         "image_bias | text_bias | distractor | conflict_abstain | other",
  "latency_s":        "Generation time in seconds",
  "has_hidden_state": "Whether activations were saved for this record",
  "extra":            "Dataset-specific metadata",
  "ts":               "Timestamp"
}
```

---

## Reproduction

```bash
cd inference
python vlm_bench.py --config config.yaml \
    --models qwen2.5-vl-7b internvl3-8b qwen3-vl-8b llava-onevision-7b \
    --datasets pendulum feliren remote_sensing objects3d \
    --languages all \
    --conditions inference perception_control \
    --max_samples_per_group 25 \
    --output_dir results/multicond_4opt
```

Analysis:

```bash
python analyze.py --output_dir results/multicond_4opt
```
