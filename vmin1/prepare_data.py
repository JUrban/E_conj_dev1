#!/usr/bin/env python3
"""
Prepare training data from Vampire minimized proof claim evaluations.

Reads:
  00burned0        — baselines: ./problem:% Instructions burned: N (million)
  00burned1.gz     — claim evals: ./problem/problem__lemma:% Instructions burned: N (million)
  00allconjm11.gz  — CNF lemma clauses: cnf(problem__lemma, plain, (...)).

Output:
  lemmas_useful      — matched lemma clauses for training
  statistics_useful  — statistics in E-compatible format

Usage:
  python3 vmin1/prepare_data.py [--max_ratio 1.0] [--output_dir vmin1]
"""

import gzip
import re
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_burned(line):
    """Extract name and instruction count from a burned line."""
    # ./problem:% Instructions burned: N (million)
    m = re.match(r'(\S+):\% Instructions burned: (\d+) \(million\)', line.strip())
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ratio', type=float, default=1.0,
                        help='Maximum ratio to include (default: 1.0)')
    parser.add_argument('--output_dir', default=SCRIPT_DIR)
    parser.add_argument('--baseline', default=os.path.join(SCRIPT_DIR, '00burned0'))
    parser.add_argument('--claims', default=os.path.join(SCRIPT_DIR, '00burned1.gz'))
    parser.add_argument('--lemmas', default=os.path.join(SCRIPT_DIR, '00allconjm11.gz'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Read baselines
    print("Step 1: Reading baselines...")
    baselines = {}  # problem -> burn count
    with open(args.baseline) as f:
        for line in f:
            name, burn = parse_burned(line)
            if name:
                # ./problem -> problem
                prob = name.lstrip('./')
                baselines[prob] = burn
    print(f"  {len(baselines)} baselines loaded")

    # Step 2: Read claim evaluations and compute ratios
    print("Step 2: Reading claim evaluations...")
    useful = []  # (problem, lemma_id, baseline, claim_burn, ratio)
    n_total = 0
    n_no_baseline = 0

    with gzip.open(args.claims, 'rt') as f:
        for line in f:
            name, claim_burn = parse_burned(line)
            if not name:
                continue
            n_total += 1
            # ./problem/problem__lemma -> problem, lemma
            # strip leading ./
            name = name.lstrip('./')
            parts = name.split('/')
            if len(parts) != 2:
                continue
            prob = parts[0]
            # problem__lemma -> lemma
            full_id = parts[1]  # problem__lemma
            m = re.match(r'.+__(.+)', full_id)
            if not m:
                continue
            lemma_id = m.group(1)

            if prob not in baselines:
                n_no_baseline += 1
                continue

            baseline = baselines[prob]
            if baseline <= 0:
                continue

            ratio = claim_burn / baseline
            if ratio < args.max_ratio:
                useful.append((prob, lemma_id, full_id, baseline, claim_burn, ratio))

    print(f"  {n_total} claim evaluations read")
    print(f"  {n_no_baseline} skipped (no baseline)")
    print(f"  {len(useful)} useful (ratio < {args.max_ratio})")
    print(f"  {len(set(u[0] for u in useful))} unique problems with useful claims")

    # Step 3: Load CNF lemma clauses
    print("Step 3: Loading lemma clauses...")
    lemma_clauses = {}  # full_id (problem__lemma) -> cnf line
    with gzip.open(args.lemmas, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('cnf('):
                continue
            # cnf(problem__lemma, plain, (...)).
            m = re.match(r'cnf\((\w+),', line)
            if m:
                lemma_clauses[m.group(1)] = line
    print(f"  {len(lemma_clauses)} lemma clauses loaded")

    # Step 4: Match and write output
    print("Step 4: Writing training data...")
    lemmas_path = os.path.join(args.output_dir, 'lemmas_useful')
    stats_path = os.path.join(args.output_dir, 'statistics_useful')

    n_written = 0
    n_missing = 0

    with open(lemmas_path, 'w') as lf, open(stats_path, 'w') as sf:
        for prob, lemma_id, full_id, baseline, claim_burn, ratio in sorted(useful):
            if full_id not in lemma_clauses:
                n_missing += 1
                continue

            cnf_line = lemma_clauses[full_id]

            # Lemma format: ./problem/lemma_id: cnf(...)
            lf.write(f"./{prob}/{lemma_id}: {cnf_line}\n")

            # Statistics E-format:
            # ratio:problem:lemma_id:L1:L2:L1+L2:# label :baseline
            # With AVATAR claims, single run: L1=claim_burn, L2=0
            sf.write(f"{ratio:.6f}:{prob}:{lemma_id}:"
                     f"{claim_burn}:0:{claim_burn}:"
                     f"# Instructions :{baseline}\n")
            n_written += 1

    print(f"\nDone:")
    print(f"  Written: {n_written}")
    print(f"  Missing clauses: {n_missing}")
    print(f"  Wrote: {lemmas_path}")
    print(f"  Wrote: {stats_path}")

    # Summary stats
    if n_written > 0:
        ratios = [u[5] for u in useful if u[2] in lemma_clauses]
        probs = set(u[0] for u in useful if u[2] in lemma_clauses)
        print(f"\n  Problems with useful claims: {len(probs)}")
        print(f"  Mean ratio: {sum(ratios)/len(ratios):.3f}")
        print(f"  Ratios <= 0.5: {sum(1 for r in ratios if r <= 0.5)}")
        print(f"  Ratios <= 0.1: {sum(1 for r in ratios if r <= 0.1)}")


if __name__ == '__main__':
    main()
