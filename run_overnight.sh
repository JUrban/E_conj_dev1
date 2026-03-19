#!/bin/bash
# Overnight GPU run: train + generate for multiple model variants
# Usage: nohup bash run_overnight.sh 2>&1 | tee full_run.log &

set -e
echo "=== Starting overnight run at $(date) ==="

# 1. Generate from D+dropout (40 min)
echo ""
echo "=== [1/7] Generating from D+dropout ==="
python -m conjecture_gen.bulk_generate \
  --model checkpoints_d3_dropout/best_model.pt \
  --n 30 --per_problem --batch_gen 16 \
  --output conjectures_d_arity_full/

# 2. Train A with named embeddings (2-3 hours)
echo ""
echo "=== [2/7] Training A with named embeddings ==="
rm -rf cache/precomputed cache/graph_*_named.pt
python -m conjecture_gen.train_variant --variant a --named_embeddings \
  --epochs 100 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_a_named

# 3. Generate from named A (40 min)
echo ""
echo "=== [3/7] Generating from A+named ==="
python -m conjecture_gen.bulk_generate \
  --model checkpoints_a_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 \
  --output conjectures_a_named_full/

# 4. Train C (VAE) fresh 100 epochs (2-3 hours)
echo ""
echo "=== [4/7] Training C (VAE) fresh ==="
python -m conjecture_gen.train_variant --variant c \
  --epochs 100 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_c_fresh

# 5. Generate from C (40 min)
echo ""
echo "=== [5/7] Generating from C ==="
python -m conjecture_gen.bulk_generate \
  --model checkpoints_c_fresh/best_model.pt \
  --n 30 --per_problem --batch_gen 16 \
  --output conjectures_c_arity_full/

# 6. Train A longer - 50 more epochs (1 hour)
echo ""
echo "=== [6/7] Training A for 50 more epochs ==="
python -m conjecture_gen.train_variant --variant a \
  --resume checkpoints_a2 --epochs 50 \
  --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 3e-4 --save_dir checkpoints_a3

# 7. Generate from A3 (40 min)
echo ""
echo "=== [7/7] Generating from A3 (150 epochs) ==="
python -m conjecture_gen.bulk_generate \
  --model checkpoints_a3/best_model.pt \
  --n 30 --per_problem --batch_gen 16 \
  --output conjectures_a3_arity_full/

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Conjecture sets ready for E prover evaluation:"
echo "  conjectures_d_arity_full/"
echo "  conjectures_a_named_full/"
echo "  conjectures_c_arity_full/"
echo "  conjectures_a3_arity_full/"
echo "  conjectures_a2_arity_full/  (already done)"
