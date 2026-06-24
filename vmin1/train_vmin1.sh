#!/bin/bash
# Train conjecture generation model on combined E + vmin1 (Vampire minimized) data
#
# Usage:
#   ./vmin1/train_vmin1.sh [GPU_ID] [HIDDEN_DIM] [EXTRA_ARGS...]
#
# Examples:
#   ./vmin1/train_vmin1.sh 0 128                          # from scratch
#   ./vmin1/train_vmin1.sh 0 128 --init_from checkpoints_af6_a_named/best_model.pt  # from E checkpoint
#   ./vmin1/train_vmin1.sh 0 256 --named_embeddings

set -euo pipefail

GPU_ID=${1:-0}
HIDDEN=${2:-128}
shift 2 2>/dev/null || shift $#

EXTRA="$*"
VARIANT="a"
NAMED=""
INIT_TAG=""
prev=""

for arg in $EXTRA; do
    case "$prev" in
        --variant) VARIANT="$arg" ;;
    esac
    if [ "$arg" = "--named_embeddings" ]; then
        NAMED="_named"
    fi
    if [ "$prev" = "--init_from" ]; then
        INIT_TAG="_initE"
    fi
    prev="$arg"
done

SAVE_DIR="checkpoints_vmin1_${VARIANT}${NAMED}_${HIDDEN}${INIT_TAG}"
LOG="train_vmin1_${VARIANT}${NAMED}_${HIDDEN}${INIT_TAG}.log"

echo "=== vmin1 Combined Training ==="
echo "GPU:        $GPU_ID"
echo "Hidden:     $HIDDEN"
echo "Variant:    $VARIANT"
echo "Save dir:   $SAVE_DIR"
echo "Log:        $LOG"
echo "Extra args: $EXTRA"
echo ""

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m conjecture_gen.train_variant \
    --variant $VARIANT \
    --problems_dir vmin1/combined_problems \
    --lemmas_file vmin1/combined_lemmas \
    --statistics_file vmin1/combined_statistics \
    --cache_dir vmin1/cache \
    --train_split vmin1/combined_train_problems.txt \
    --val_split vmin1/combined_val_problems.txt \
    --hidden_dim $HIDDEN \
    --max_nodes 3000 \
    --max_ratio 1.0 \
    --max_samples 0 \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 64 \
    --amp \
    --named_embeddings \
    --save_dir "$SAVE_DIR" \
    --seed 42 \
    $EXTRA \
    > "$LOG" 2>&1

echo "Done. Log: $LOG"
