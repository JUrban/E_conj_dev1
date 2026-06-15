#!/bin/bash
# Train conjecture generation model on Vampire dataset
#
# Usage:
#   ./v1/train_vampire.sh [GPU_ID] [HIDDEN_DIM] [EXTRA_ARGS...]
#
# Examples:
#   ./v1/train_vampire.sh 0 256
#   ./v1/train_vampire.sh 0 512 --epochs 100 --batch_size 64
#   ./v1/train_vampire.sh 0 256 --variant c --named_embeddings

set -euo pipefail

GPU_ID=${1:-0}
HIDDEN=${2:-256}
shift 2 2>/dev/null || shift $#

EXTRA="$*"
VARIANT="a"
NAMED=""
prev=""

# Extract --variant and --named_embeddings from EXTRA if present
for arg in $EXTRA; do
    case "$prev" in
        --variant) VARIANT="$arg" ;;
    esac
    if [ "$arg" = "--named_embeddings" ]; then
        NAMED="_named"
    fi
    prev="$arg"
done

SAVE_DIR="checkpoints_v1_${VARIANT}${NAMED}_${HIDDEN}"
LOG="train_v1_${VARIANT}${NAMED}_${HIDDEN}.log"

echo "=== Vampire Training ==="
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
    --problems_dir v1/cnf \
    --lemmas_file v1/lemmas_useful \
    --statistics_file v1/statistics_useful \
    --cache_dir v1/cache \
    --train_split v1/train_problems.txt \
    --val_split v1/val_problems.txt \
    --hidden_dim $HIDDEN \
    --max_nodes 5000 \
    --max_ratio 1.0 \
    --no_precompute \
    --max_batch_nodes 50000 \
    --max_samples 0 \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 16 \
    --save_dir "$SAVE_DIR" \
    --seed 42 \
    $EXTRA \
    > "$LOG" 2>&1

echo "Done. Log: $LOG"
