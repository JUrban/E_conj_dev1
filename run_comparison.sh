#!/bin/bash
# Compare master vs audit-fixes3 branch on same data
# Run on machine with A100 GPUs
# Usage: time bash run_comparison.sh 2>&1 | tee comparison.log
#
# Override branches via environment variables:
#   BASE_BRANCH=master FIX_BRANCH=audit-fixes3 bash run_comparison.sh

set -e
export PYTHONUNBUFFERED=1

BASE_BRANCH="${BASE_BRANCH:-master}"
FIX_BRANCH="${FIX_BRANCH:-audit-fixes3}"

echo "=== Starting comparison at $(date) ==="
echo "Base branch: ${BASE_BRANCH}"
echo "Fix branch:  ${FIX_BRANCH}"
echo "GPUs available:"
nvidia-smi -L 2>/dev/null || echo "No nvidia-smi"

# Record commit hashes
BASE_HASH=$(git rev-parse "${BASE_BRANCH}" 2>/dev/null || echo "unknown")
FIX_HASH=$(git rev-parse "${FIX_BRANCH}" 2>/dev/null || echo "unknown")
echo "Base commit: ${BASE_HASH}"
echo "Fix commit:  ${FIX_HASH}"

# ============================================================
# PART 1: Train on base branch (old encoder with SAGEConv)
# ============================================================
echo ""
echo "=== PART 1: ${BASE_BRANCH} branch (SAGEConv) [${BASE_HASH}] ==="
git stash 2>/dev/null || true
git checkout "${BASE_BRANCH}"

# Clear caches
rm -rf cache/precomputed cache/graph_*.pt cache/graph_*_named.pt

# Train A+named on base branch (the best config we had)
python -m conjecture_gen.train_variant --variant a --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_master_a_named

# Generate
python -m conjecture_gen.bulk_generate \
  --model checkpoints_master_a_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_master_a_named/

echo ""
echo "=== ${BASE_BRANCH} results [${BASE_HASH}] ==="
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
# PART 2: Train on fix branch (GINEConv + all fixes)
# ============================================================
echo ""
echo "=== PART 2: ${FIX_BRANCH} branch (GINEConv + fixes) [${FIX_HASH}] ==="
git checkout "${FIX_BRANCH}"

# Clear caches (different encoder = different graphs)
rm -rf cache/precomputed cache/graph_*.pt cache/graph_*_named.pt

# Train A+named on fix branch
python -m conjecture_gen.train_variant --variant a --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_audit_a_named

# Generate
python -m conjecture_gen.bulk_generate \
  --model checkpoints_audit_a_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_audit_a_named/

echo ""
echo "=== ${FIX_BRANCH} results [${FIX_HASH}] ==="
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
# PART 3: Also train C+named on fix branch (fixed posterior)
# ============================================================
echo ""
echo "=== PART 3: C+named on ${FIX_BRANCH} (fixed VAE posterior) [${FIX_HASH}] ==="
python -m conjecture_gen.train_variant --variant c --named_embeddings \
  --epochs 50 --max_samples 0 --hidden_dim 128 --max_nodes 1500 \
  --batch_size 256 --lr 1.2e-3 --save_dir checkpoints_audit_c_named

python -m conjecture_gen.bulk_generate \
  --model checkpoints_audit_c_named/best_model.pt \
  --n 30 --per_problem --batch_gen 16 --max_steps 80 \
  --output conjectures_audit_c_named/

echo ""
echo "=== ALL DONE at $(date) ==="
echo "Base: ${BASE_BRANCH} @ ${BASE_HASH}"
echo "Fix:  ${FIX_BRANCH} @ ${FIX_HASH}"
echo ""
echo "Next: run E prover evaluation on both conjecture sets"
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_master_a_named/ ..."
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_audit_a_named/ ..."
echo "  python3 -m conjecture_gen.eval_eprover --conjectures conjectures_audit_c_named/ ..."
