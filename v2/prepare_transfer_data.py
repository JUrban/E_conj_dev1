#!/usr/bin/env python3
"""
Create training data from the cross-prover transfer evaluation.

Reads:
  b2/00fxpN_vout1-b2  — speedup results (N=1..42 maps to gen_001..gen_042)
                         Format: problem baseline_instr conj_instr
                         fxp0 is the Avatar baseline (no conjecture), ignored.
  00allconj1           — generated conjectures
                         Format: cnf(problem__gen_NNN, axiom, (...)).

Output:
  v2/lemmas_useful       — conjecture clauses for training
  v2/statistics_useful   — statistics in E-compatible format

Usage:
  python3 v2/prepare_transfer_data.py [--max_ratio 0.9]
"""

import os
import re
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
B2_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'b2')
CONJ_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), '00allconj1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ratio', type=float, default=0.9,
                        help='Maximum ratio to include (default: 0.9 = 10%% speedup)')
    parser.add_argument('--output_dir', default=SCRIPT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Read b2 files to get useful (problem, rank) pairs
    print("Step 1: Reading b2 speedup results...")
    useful = []  # (problem, rank, baseline, with_conj, ratio)

    for fname in sorted(os.listdir(B2_DIR)):
        if not fname.endswith('-b2'):
            continue
        # Extract rank: 00fxpN_vout1-b2 -> N
        m = re.match(r'00fxp(\d+)_vout1-b2', fname)
        if not m:
            continue
        fxp_num = int(m.group(1))
        if fxp_num == 0:
            continue  # Skip Avatar baseline (no conjecture)

        # fxp1 = gen_001, fxp2 = gen_002, etc.
        gen_id = f'gen_{fxp_num:03d}'

        with open(os.path.join(B2_DIR, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                prob = parts[0]
                baseline = int(parts[1])
                with_conj = int(parts[2])
                if baseline <= 0:
                    continue
                ratio = with_conj / baseline
                if ratio <= args.max_ratio:
                    useful.append((prob, gen_id, baseline, with_conj, ratio))

    print(f"  {len(useful)} useful entries (ratio <= {args.max_ratio})")
    print(f"  {len(set(u[0] for u in useful))} unique problems")

    # Step 2: Load conjectures from 00allconj1
    print("Step 2: Loading conjectures...")
    conjectures = {}  # (problem, gen_id) -> cnf line
    with open(CONJ_FILE) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('cnf('):
                continue
            # Parse: cnf(PROBLEM__gen_NNN, axiom, (...)).
            m = re.match(r'cnf\((\w+)__(gen_\d+),', line)
            if m:
                prob, gen_id = m.group(1), m.group(2)
                conjectures[(prob, gen_id)] = line

    print(f"  {len(conjectures)} conjectures loaded")

    # Step 3: Match useful entries with conjectures and write output
    print("Step 3: Writing training data...")
    lemmas_path = os.path.join(args.output_dir, 'lemmas_useful')
    stats_path = os.path.join(args.output_dir, 'statistics_useful')

    n_written = 0
    n_missing = 0

    with open(lemmas_path, 'w') as lf, open(stats_path, 'w') as sf:
        for prob, gen_id, baseline, with_conj, ratio in sorted(useful):
            key = (prob, gen_id)
            if key not in conjectures:
                n_missing += 1
                continue

            cnf_line = conjectures[key]

            # Lemma format: ./problem/lemma_id: cnf(...)
            lf.write(f"./{prob}/{gen_id}: {cnf_line}\n")

            # Statistics E-format:
            # ratio:problem:cut_id:L1:L2:L1+L2:# label :L
            # With Avatar splitting, L1=with_conj, L2=0 (combined run)
            sf.write(f"{ratio:.6f}:{prob}:{gen_id}:"
                     f"{with_conj}:0:{with_conj}:"
                     f"# Instructions :{baseline}\n")
            n_written += 1

    print(f"\nDone:")
    print(f"  Written: {n_written}")
    print(f"  Missing conjectures: {n_missing}")
    print(f"  Wrote: {lemmas_path}")
    print(f"  Wrote: {stats_path}")

    # Summary stats
    probs = set(u[0] for u in useful if (u[0], u[1]) in conjectures)
    newly_solved = set(u[0] for u in useful if u[2] >= 100000 and (u[0], u[1]) in conjectures)
    print(f"\n  Problems with useful conjectures: {len(probs)}")
    print(f"  Of which newly solved: {len(newly_solved)}")


if __name__ == '__main__':
    main()
