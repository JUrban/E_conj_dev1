#!/bin/bash
# Parallel conjecture generation: splits problems into N parts, runs on same GPU.
#
# Usage:
#   ./parallel_generate.sh <model_checkpoint> <output_dir> [N_PARTS] [GPU_ID] [extra args...]
#
# Example:
#   ./parallel_generate.sh checkpoints_af6_a_named/best_model.pt conjectures_af6_a_named 5 0 \
#       --n 50 --temperature 1.2 --top_k 15 --top_p 0.95 --batch_gen 256 --per_problem
#
# The script:
#   1. Lists problems, filters by --max_nodes (done once)
#   2. Splits into N_PARTS roughly equal chunks
#   3. Launches N_PARTS parallel python processes on GPU_ID
#   4. Each writes to output_dir with a separate rankings file
#   5. Waits for all to finish, merges rankings into rankings.tsv

set -euo pipefail

MODEL=${1:?Usage: $0 <model> <output_dir> [N_PARTS] [GPU_ID] [extra args...]}
OUTPUT=${2:?Usage: $0 <model> <output_dir> [N_PARTS] [GPU_ID] [extra args...]}
N_PARTS=${3:-5}
GPU_ID=${4:-0}
shift 4 2>/dev/null || shift $#

EXTRA_ARGS="$*"
PROBLEMS_DIR="problems"
MAX_NODES=1500
PROBLEM_LIST_FILE=""

# Extract --problems_dir, --max_nodes, --problem_list_file from extra args
prev=""
for i in "$@"; do
    case "$prev" in
        --problems_dir) PROBLEMS_DIR="$i" ;;
        --max_nodes) MAX_NODES="$i" ;;
        --problem_list_file) PROBLEM_LIST_FILE="$i" ;;
    esac
    prev="$i"
done

echo "=== Parallel Generate ==="
echo "Model:       $MODEL"
echo "Output:      $OUTPUT"
echo "Parts:       $N_PARTS"
echo "GPU:         $GPU_ID"
echo "Problems:    $PROBLEMS_DIR"
echo "Max nodes:   $MAX_NODES"
echo "Extra args:  $EXTRA_ARGS"
echo ""

# Create temp dir for problem list splits
SPLIT_DIR=$(mktemp -d /tmp/par_gen_splits.XXXXXX)
trap "rm -rf $SPLIT_DIR" EXIT

# Get problem list: from file or by scanning + filtering
if [ -n "$PROBLEM_LIST_FILE" ]; then
    echo "Using problem list from $PROBLEM_LIST_FILE..."
    cp "$PROBLEM_LIST_FILE" "$SPLIT_DIR/all_problems.txt"
else
    echo "Building problem list (max_nodes=$MAX_NODES)..."
    python3 -c "
import os, sys
sys.path.insert(0, '.')
from conjecture_gen.tptp_parser import parse_problem_file
from conjecture_gen.graph_builder import clauses_to_graph

problems_dir = '$PROBLEMS_DIR'
max_nodes = $MAX_NODES
problems = sorted(os.listdir(problems_dir))
filtered = []
for p in problems:
    try:
        clauses = parse_problem_file(os.path.join(problems_dir, p))
        graph = clauses_to_graph(clauses)
        total = sum(graph[nt].x.shape[0] for nt in graph.node_types)
        if total <= max_nodes:
            filtered.append(p)
    except Exception:
        pass

print(f'Filtered: {len(filtered)}/{len(problems)} problems', file=sys.stderr)
for p in filtered:
    print(p)
" > "$SPLIT_DIR/all_problems.txt"
fi

TOTAL=$(wc -l < "$SPLIT_DIR/all_problems.txt")
echo "Total problems after filtering: $TOTAL"

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No problems found!"
    exit 1
fi

# Split into N parts
CHUNK=$(( (TOTAL + N_PARTS - 1) / N_PARTS ))
split -l "$CHUNK" -d -a 1 "$SPLIT_DIR/all_problems.txt" "$SPLIT_DIR/part_"

# Show split sizes
echo "Split sizes:"
for f in "$SPLIT_DIR"/part_*; do
    part=$(basename "$f")
    echo "  $part: $(wc -l < "$f") problems"
done
echo ""

# Launch parallel workers
mkdir -p "$OUTPUT"
PIDS=()
PARTS=()
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Strip --problem_list_file and its value from EXTRA_ARGS (handled by this script)
CLEAN_ARGS=""
skip_next=false
for arg in $EXTRA_ARGS; do
    if $skip_next; then
        skip_next=false
        continue
    fi
    if [ "$arg" = "--problem_list_file" ]; then
        skip_next=true
        continue
    fi
    CLEAN_ARGS="$CLEAN_ARGS $arg"
done

echo "Launching $N_PARTS workers on GPU $GPU_ID..."
for f in "$SPLIT_DIR"/part_*; do
    part=$(basename "$f")
    LOG="$OUTPUT/worker_${part}.log"

    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m conjecture_gen.bulk_generate \
        --model "$MODEL" \
        --output "$OUTPUT" \
        --problems_dir "$PROBLEMS_DIR" \
        --problem_list "$f" \
        --rankings_suffix "_${part}" \
        --max_nodes 999999 \
        $CLEAN_ARGS \
        > "$LOG" 2>&1 &

    PID=$!
    PIDS+=($PID)
    PARTS+=($part)
    echo "  Worker $part: PID=$PID, $(wc -l < "$f") problems -> $LOG"
done

echo ""
echo "All $N_PARTS workers launched. Waiting..."

# Wait for all, track failures
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  WARNING: Worker ${PARTS[$i]} (PID=${PIDS[$i]}) failed!"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED/$N_PARTS workers failed. Check logs in $OUTPUT/worker_*.log"
fi

# Merge per-worker rankings into one rankings.tsv
echo "Merging rankings..."
MERGED="$OUTPUT/rankings.tsv"
HEADER_DONE=false

for part in "${PARTS[@]}"; do
    PART_FILE="$OUTPUT/rankings_${part}.tsv"
    if [ -f "$PART_FILE" ]; then
        if [ "$HEADER_DONE" = false ]; then
            cat "$PART_FILE" > "$MERGED"
            HEADER_DONE=true
        else
            # Skip header line, append rest
            tail -n +2 "$PART_FILE" >> "$MERGED"
        fi
        rm "$PART_FILE"
    else
        echo "  WARNING: $PART_FILE not found"
    fi
done

# Print summary
TOTAL_CONJ=$(( $(wc -l < "$MERGED") - 1 ))  # minus header
TOTAL_PROBS=$(tail -n +2 "$MERGED" | cut -f1 | sort -u | wc -l)
echo ""
echo "=== Done ==="
echo "Output:     $OUTPUT"
echo "Rankings:   $MERGED ($TOTAL_CONJ conjectures, $TOTAL_PROBS problems)"
echo "Logs:       $OUTPUT/worker_*.log"
