#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cross-lingual contrastive steering pipeline — runner.
#
# All knobs are environment variables; sensible defaults below. Examples:
#
#   ./run.sh                                          # use config.yaml as-is
#   MODELS="qwen2.5-vl-3b" ./run.sh                   # one model
#   DATASETS="feliren pendulum" ./run.sh             # subset of datasets
#   LANGUAGES="english hindi telugu" ./run.sh        # subset of languages
#   SOURCE_LANGUAGE="english" ./run.sh               # transfer source vector
#   MAX_PER_GROUP=20 EVAL_CAP=40 ./run.sh            # quick / smoke
#   ALPHAS="-8 -4 0 4 8" ./run.sh                    # custom α sweep
#   OUTPUT_DIR=my_steer_run ./run.sh
#   HF_TOKEN=hf_xxx MODELS="aya-vision-8b" ./run.sh   # gated model
#
# Runs run_steering.py (Phases 1-6) then analyze_steering.py (tables + figures).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-steering_results}"

MODELS="${MODELS:-}"
DATASETS="${DATASETS:-}"
LANGUAGES="${LANGUAGES:-}"
SOURCE_LANGUAGE="${SOURCE_LANGUAGE:-}"
MAX_PER_GROUP="${MAX_PER_GROUP:-}"
ALPHAS="${ALPHAS:-}"
EVAL_CAP="${EVAL_CAP:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
MIN_CLASS="${MIN_CLASS:-}"
EVAL_FRACTION="${EVAL_FRACTION:-}"
ATTN_IMPL="${ATTN_IMPL:-}"
DTYPE="${DTYPE:-}"
HF_TOKEN="${HF_TOKEN:-}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-}"

ARGS=(--config "$CONFIG" --output_dir "$OUTPUT_DIR")
[[ -n "$MODELS"          ]] && ARGS+=(--models $MODELS)
[[ -n "$DATASETS"        ]] && ARGS+=(--datasets $DATASETS)
[[ -n "$LANGUAGES"       ]] && ARGS+=(--languages $LANGUAGES)
[[ -n "$SOURCE_LANGUAGE" ]] && ARGS+=(--source_language "$SOURCE_LANGUAGE")
[[ -n "$MAX_PER_GROUP"   ]] && ARGS+=(--max_samples_per_group "$MAX_PER_GROUP")
[[ -n "$ALPHAS"          ]] && ARGS+=(--alpha_sweep $ALPHAS)
[[ -n "$EVAL_CAP"        ]] && ARGS+=(--eval_cap "$EVAL_CAP")
[[ -n "$BATCH_SIZE"      ]] && ARGS+=(--batch_size "$BATCH_SIZE")
[[ -n "$MIN_CLASS"       ]] && ARGS+=(--min_class "$MIN_CLASS")
[[ -n "$EVAL_FRACTION"   ]] && ARGS+=(--eval_fraction "$EVAL_FRACTION")
[[ -n "$ATTN_IMPL"       ]] && ARGS+=(--attn_impl "$ATTN_IMPL")
[[ -n "$DTYPE"           ]] && ARGS+=(--dtype "$DTYPE")
[[ -n "$HF_TOKEN"        ]] && ARGS+=(--hf_token "$HF_TOKEN")
[[ "$FORCE_REDOWNLOAD" == "1" ]] && ARGS+=(--force_redownload)

export TOKENIZERS_PARALLELISM=false
# Home filesystem has a tight quota; keep the (large) HF model cache on scratch.
export HF_HOME="${HF_HOME:-/fs04/scratch2/lf93/vfvic1/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME"

echo "Running: $PYTHON run_steering.py ${ARGS[*]}"
"$PYTHON" run_steering.py "${ARGS[@]}"

echo "Running analysis -> $OUTPUT_DIR/figures, $OUTPUT_DIR/analysis"
"$PYTHON" analyze_steering.py --output_dir "$OUTPUT_DIR"
