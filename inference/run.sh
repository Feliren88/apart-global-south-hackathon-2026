#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual VLM bias benchmark — runner.
#
# All knobs are environment variables; sensible defaults below. Examples:
#
#   ./run.sh                                   # use config.yaml as-is
#   MODELS="qwen2.5-vl-7b" ./run.sh            # one model
#   DATASETS="feliren pendulum" ./run.sh       # subset of datasets
#   LANGUAGES="english hindi" ./run.sh         # subset of languages
#   MAX_PER_GROUP=-1 ./run.sh                  # ALL rows per (dataset,language)
#   MAX_PER_GROUP=10 SMOKE=4 ./run.sh          # quick smoke test
#   SAVE_HIDDEN=0 ./run.sh                     # disable activation capture
#   HF_TOKEN=hf_xxx MODELS="aya-vision-8b" ./run.sh   # gated model
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# Optional overrides (empty => fall back to config.yaml)
MODELS="${MODELS:-}"
DATASETS="${DATASETS:-}"
LANGUAGES="${LANGUAGES:-}"
CONDITIONS="${CONDITIONS:-}"
MAX_PER_GROUP="${MAX_PER_GROUP:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
DTYPE="${DTYPE:-}"
SAVE_HIDDEN="${SAVE_HIDDEN:-}"
MAX_HIDDEN="${MAX_HIDDEN:-}"
SMOKE="${SMOKE:-}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-}"
HF_TOKEN="${HF_TOKEN:-}"

ARGS=(--config "$CONFIG" --output_dir "$OUTPUT_DIR")
[[ -n "$MODELS"        ]] && ARGS+=(--models $MODELS)
[[ -n "$DATASETS"      ]] && ARGS+=(--datasets $DATASETS)
[[ -n "$LANGUAGES"     ]] && ARGS+=(--languages $LANGUAGES)
[[ -n "$CONDITIONS"    ]] && ARGS+=(--conditions $CONDITIONS)
[[ -n "$MAX_PER_GROUP" ]] && ARGS+=(--max_samples_per_group "$MAX_PER_GROUP")
[[ -n "$MAX_NEW_TOKENS" ]] && ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
[[ -n "$DTYPE"         ]] && ARGS+=(--dtype "$DTYPE")
[[ -n "$MAX_HIDDEN"    ]] && ARGS+=(--max_hidden_state_samples "$MAX_HIDDEN")
[[ -n "$SMOKE"         ]] && ARGS+=(--limit_smoke "$SMOKE")
[[ -n "$HF_TOKEN"      ]] && ARGS+=(--hf_token "$HF_TOKEN")
[[ "$SAVE_HIDDEN" == "0" ]] && ARGS+=(--no_hidden_states)
[[ "$FORCE_REDOWNLOAD" == "1" ]] && ARGS+=(--force_redownload)

export TOKENIZERS_PARALLELISM=false
# Home filesystem has a tight quota; keep the (large) HF model cache on scratch.
export HF_HOME="${HF_HOME:-/fs04/scratch2/lf93/vfvic1/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME"

echo "Running: $PYTHON vlm_bench.py ${ARGS[*]}"
exec "$PYTHON" vlm_bench.py "${ARGS[@]}"
