#!/bin/bash
# Run E prover evaluation on all conjecture sets not yet evaluated
# Already evaluated: conjectures_a2_arity_full, conjectures_d_arity_full
# Usage: time bash run_evals.sh 2>&1 | tee eval_run.log

set -e
EPROVER=~/bin/eprover
WORKERS=32
TIMEOUT=10
MAXCONJ=10

echo "=== Starting evals at $(date) ==="

# 1. A3 (150 epochs, anonymous)
echo ""
echo "=== conjectures_a3_arity_full ==="
python3 -m conjecture_gen.eval_eprover \
  --conjectures conjectures_a3_arity_full/ \
  --problems problems/ \
  --eprover $EPROVER \
  --timeout $TIMEOUT \
  --max_conjectures_per_problem $MAXCONJ \
  --workers $WORKERS

# 2. A+named (100 epochs)
echo ""
echo "=== conjectures_a_named_full ==="
python3 -m conjecture_gen.eval_eprover \
  --conjectures conjectures_a_named_full/ \
  --problems problems/ \
  --eprover $EPROVER \
  --timeout $TIMEOUT \
  --max_conjectures_per_problem $MAXCONJ \
  --workers $WORKERS

# 3. C anonymous (100 epochs)
echo ""
echo "=== conjectures_c_arity_full ==="
python3 -m conjecture_gen.eval_eprover \
  --conjectures conjectures_c_arity_full/ \
  --problems problems/ \
  --eprover $EPROVER \
  --timeout $TIMEOUT \
  --max_conjectures_per_problem $MAXCONJ \
  --workers $WORKERS

echo ""
echo "=== ALL EVALS DONE at $(date) ==="
echo ""
echo "Summary of all results:"
for d in conjectures_a2_arity_full conjectures_a3_arity_full conjectures_a_named_full conjectures_c_arity_full conjectures_d_arity_full; do
  if [ -f "$d/eprover_results.tsv" ]; then
    TOTAL=$(tail -n +2 "$d/eprover_results.tsv" | wc -l)
    BOTH=$(awk -F'\t' '$3=="proved" && $5=="proved"' "$d/eprover_results.tsv" | wc -l)
    USEFUL=$(grep -c "True" "$d/eprover_results.tsv" 2>/dev/null || echo 0)
    echo "  $d: tested=$TOTAL both_proved=$BOTH useful=$USEFUL"
  fi
done
