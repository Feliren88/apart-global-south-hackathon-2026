#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# FULL steering run — batch_size=8 — self-contained.
#
# Runs the complete 6-phase Cross-Lingual Contrastive Steering pipeline
# (run_steering.py) then the analysis/figures (analyze_steering.py), over ALL
# four datasets and ALL languages with batched scoring (bs=8).
#
# Usage (from any terminal):
#     bash run_full.sh
#     HF_TOKEN=hf_xxx bash run_full.sh          # if using a gated model
#     OUTPUT_DIR=steering_results bash run_full.sh
#
# Everything is logged to  <OUTPUT_DIR>/run_full_<timestamp>.log  (and echoed).
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Always run from the steering/ directory this script lives in ─────────────
cd "$(dirname "$(readlink -f "$0")")"

# ── Knobs (override via env). Defaults = full-fledged run. ───────────────────
export OUTPUT_DIR="${OUTPUT_DIR:-steering_results}"
# All models that loaded OK in the inference benchmarks (result_logit_19models +
# results); load-time failures and gated aya-vision-8b excluded. Run sequentially.
export MODELS="${MODELS:-internvl3-2b internvl3-8b llava-onevision-7b glm-4.1v-9b-thinking granite-vision-3.3-2b sea-lion-v4-8b-vl qwen2.5-vl-7b qwen3-vl-8b}"
export DATASETS="${DATASETS:-feliren pendulum remote_sensing objects3d}"
export LANGUAGES="${LANGUAGES:-all}"
export BATCH_SIZE="${BATCH_SIZE:-8}"          # <-- batched scoring (the big lever)
export ALPHAS="${ALPHAS:--8 -4 0 4 8}"        # full alpha sweep
export EVAL_CAP="${EVAL_CAP:-80}"             # eval items per language in P5/P6
# MAX_PER_GROUP unset = use ALL rows per (dataset, language). Set e.g. 60 to cap.
export MAX_PER_GROUP="${MAX_PER_GROUP:-}"

# ── Environment (HF cache on scratch; home quota is tight) ───────────────────
export HF_HOME="${HF_HOME:-/fs04/scratch2/lf93/vfvic1/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

# HF auth is read from the environment — never hardcode the token in a file.
# Export HF_TOKEN before running if you need a gated model (e.g. aya-vision).
if [[ -n "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN: set (gated models OK)"; fi

LOG="$OUTPUT_DIR/run_full_$(date +%Y%m%d_%H%M%S).log"

echo "═══════════════════════════════════════════════════════════════════════"
echo " FULL steering run  (batch_size=$BATCH_SIZE)"
echo "   models     : $MODELS"
echo "   datasets   : $DATASETS"
echo "   languages  : $LANGUAGES"
echo "   alpha sweep: $ALPHAS    eval_cap: $EVAL_CAP    max_per_group: ${MAX_PER_GROUP:-all}"
echo "   output_dir : $OUTPUT_DIR"
echo "   log        : $LOG"
echo "   HF_HOME    : $HF_HOME"
echo "═══════════════════════════════════════════════════════════════════════"

# run.sh consumes all the env vars above and runs Phases 1-6 then analysis.
# tee so you watch it live AND keep a full transcript.
bash run.sh 2>&1 | tee "$LOG"

echo ""
echo "DONE. Artifacts under: $OUTPUT_DIR"
echo "  phase1_perception.jsonl  phase2_conflict.jsonl  phase5_alpha_sweep.jsonl"
echo "  phase6_transfer.jsonl    activations/  vectors/  analysis/  figures/"
echo "  full log: $LOG"
