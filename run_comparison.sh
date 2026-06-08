#!/bin/bash
# Compare old master vs audit-fixes branch on same data
# Run on machine with A100 GPUs
# Usage: time bash run_comparison.sh 2>&1 | tee comparison.log

set -e
export PYTHONUNBUFFERED=1

echo "=== Starting comparison at $(date) ==="
echo "GPUs available:"
nvidia-smi -L 2>/dev/null || echo "No nvidia-smi"

# ============================================================
# PART 1: Train on master (old encoder with SAGEConv)
# ============================================================
echo ""
echo "=== PART 1: master branch (SAGEConv) ==="
git stash 2>/dev/null || true
git checkout master

# Clear caches
rm -rf cache/precomputed cache/graph_*.pt cache/graph_*_named.pt

# Train A+named on master (the best config we had)
python -m conjecture_gen.train_variant --variant a --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_master_a_named

# Generate
python -m conjecture_gen.bulk_generate \
  --model checkpoints_master_a_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_master_a_named/

echo ""
echo "=== Master results ==="
echo "Val loss:"
tail -1 checkpoints_master_a_named/history.json | python3 -c "import json,sys; d=json.load(sys.stdin)[-1]; print(f'  epoch={d[\"epoch\"]} val={d[\"val\"][\"total\"]:.4f}')" 2>/dev/null || tail -3 checkpoints_master_a_named/history.json
echo "Validity:"
grep "Validity" conjectures_master_a_named/rankings.tsv 2>/dev/null || \
  python3 -c "
v=0;t=0
for line in open('conjectures_master_a_named/rankings.tsv'):
    if line.startswith('problem'): continue
    t+=1
    if '\tTrue\t' in line: v+=1
print(f'  {v}/{t} valid ({100*v/max(t,1):.1f}%)')
"

# ============================================================
# PART 2: Train on audit-fixes (GINEConv + all fixes)
# ============================================================
echo ""
echo "=== PART 2: audit-fixes branch (GINEConv + fixes) ==="
git checkout audit-fixes

# Clear caches (different encoder = different graphs)
rm -rf cache/precomputed cache/graph_*.pt cache/graph_*_named.pt

# Train A+named on audit-fixes
python -m conjecture_gen.train_variant --variant a --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_audit_a_named

# Generate
python -m conjecture_gen.bulk_generate \
  --model checkpoints_audit_a_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_audit_a_named/

echo ""
echo "=== Audit-fixes results ==="
echo "Val loss:"
tail -1 checkpoints_audit_a_named/history.json | python3 -c "import json,sys; d=json.load(sys.stdin)[-1]; print(f'  epoch={d[\"epoch\"]} val={d[\"val\"][\"total\"]:.4f}')" 2>/dev/null || tail -3 checkpoints_audit_a_named/history.json
echo "Validity:"
python3 -c "
v=0;t=0
for line in open('conjectures_audit_a_named/rankings.tsv'):
    if line.startswith('problem'): continue
    t+=1
    if '\tTrue\t' in line: v+=1
print(f'  {v}/{t} valid ({100*v/max(t,1):.1f}%)')
"

# ============================================================
# PART 3: Also train C+named on audit-fixes (fixed posterior)
# ============================================================
echo ""
echo "=== PART 3: C+named on audit-fixes (fixed VAE posterior) ==="
python -m conjecture_gen.train_variant --variant c --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_audit_c_named

python -m conjecture_gen.bulk_generate \
  --model checkpoints_audit_c_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_audit_c_named/

echo ""
echo "=== ALL DONE at $(date) ==="
echo ""
echo "Next: run E prover evaluation on both conjecture sets"
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_master_a_named/ ..."
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_audit_a_named/ ..."
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_audit_c_named/ ..."
