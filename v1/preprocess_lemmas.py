#!/usr/bin/env python3
"""
Convert Vampire FOF lemmas to CNF format and create unified lemmas/statistics files
compatible with the existing ConjectureDataset pipeline.

Input:
  v1/cnf_vout1/          - per-problem FOF lemma files
  v1/statistics.tsv      - computed statistics with ratios

Output:
  v1/lemmas              - one CNF lemma per line: ./problem/problem__lemma_id cnf(...)
  v1/statistics_eformat  - E-prover-compatible statistics format

The FOF lemmas are all universally-quantified disjunctions (no existentials,
implications, or conjunctions), so conversion to CNF is trivial:
strip the fof() wrapper and quantifiers, emit as cnf().
"""

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fof_to_cnf(fof_line):
    """Convert a single FOF line to CNF format.

    Input:  fof(f219,plain,( ! [X0] : ( lit1 | lit2 ) )).
    Output: cnf(f219,plain,( lit1 | lit2 )).

    Returns (lemma_id, cnf_line) or None on parse failure.
    """
    # Extract: fof(NAME, ROLE, FORMULA).
    m = re.match(r'fof\((\w+)\s*,\s*(\w+)\s*,\s*\((.+)\)\s*\)\s*\.', fof_line.strip())
    if not m:
        # Try without outer parens
        m = re.match(r'fof\((\w+)\s*,\s*(\w+)\s*,\s*(.+)\s*\)\s*\.', fof_line.strip())
        if not m:
            return None

    name, role, formula = m.group(1), m.group(2), m.group(3).strip()

    # Strip universal quantifiers: ! [X0] : ! [X1] : ... body
    body = formula
    while True:
        # Match: ! [Vars] : rest
        qm = re.match(r'\s*!\s*\[([^\]]*)\]\s*:\s*(.+)', body, re.DOTALL)
        if qm:
            body = qm.group(2).strip()
        else:
            break

    # Strip outer parens if present
    if body.startswith('(') and body.endswith(')'):
        # Check balanced
        depth = 0
        balanced = True
        for i, c in enumerate(body):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(body) - 1:
                balanced = False
                break
        if balanced:
            body = body[1:-1].strip()

    cnf_line = f"cnf({name},{role},({body}))."
    return name, cnf_line


def main():
    lemma_dir = os.path.join(SCRIPT_DIR, 'cnf_vout1')
    stats_file = os.path.join(SCRIPT_DIR, 'statistics.tsv')

    # Load statistics to know which (problem, lemma) pairs exist
    print("Loading statistics...")
    stats = {}  # (problem, lemma_id) -> ratio
    stats_detail = {}  # (problem, lemma_id) -> (i_pl, i_pnl, i_base)
    with open(stats_file) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 10:
                prob, lemma = parts[0], parts[1]
                i_base, i_pl, i_pnl = int(parts[2]), int(parts[3]), int(parts[4])
                ratio = float(parts[8])
                stats[(prob, lemma)] = ratio
                stats_detail[(prob, lemma)] = (i_pl, i_pnl, i_base)
    print(f"  {len(stats)} entries")

    # Get set of problems that have stats entries
    stats_problems = set(k[0] for k in stats.keys())

    # Convert FOF lemmas to CNF and write unified lemmas file
    lemmas_path = os.path.join(SCRIPT_DIR, 'lemmas')
    statistics_path = os.path.join(SCRIPT_DIR, 'statistics_eformat')

    n_converted = 0
    n_failed = 0
    n_matched = 0

    problems = sorted(os.listdir(lemma_dir))
    print(f"Processing {len(problems)} problem lemma files...")

    with open(lemmas_path, 'w') as lf, open(statistics_path, 'w') as sf:
        for pi, prob_name in enumerate(problems):
            prob_path = os.path.join(lemma_dir, prob_name)
            if not os.path.isfile(prob_path):
                continue

            with open(prob_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('fof('):
                        continue

                    result = fof_to_cnf(line)
                    if result is None:
                        n_failed += 1
                        continue

                    lemma_id, cnf_line = result
                    n_converted += 1

                    # Write in the format expected by parse_lemma_line:
                    # ./problem/lemma_id: cnf(...)
                    lf.write(f"./{prob_name}/{lemma_id}: {cnf_line}\n")

                    # Write statistics in E-format:
                    # ratio:problem:cut_id:L1:L2:L1+L2:# Processed clauses :L
                    key = (prob_name, lemma_id)
                    if key in stats:
                        ratio = stats[key]
                        # We use instruction counts; store in the L1/L2/L fields
                        i_pl = stats_detail.get(key, (0, 0, 0))[0]
                        i_pnl = stats_detail.get(key, (0, 0, 0))[1]
                        i_base = stats_detail.get(key, (0, 0, 0))[2]
                        sf.write(f"{ratio:.6f}:{prob_name}:{lemma_id}:"
                                 f"{i_pl}:{i_pnl}:{i_pl+i_pnl}:"
                                 f"# Instructions :{i_base}\n")
                        n_matched += 1

            if (pi + 1) % 5000 == 0:
                print(f"  {pi+1}/{len(problems)}: {n_converted} converted, "
                      f"{n_matched} matched, {n_failed} failed")

    print(f"\nDone:")
    print(f"  Converted: {n_converted}")
    print(f"  Failed:    {n_failed}")
    print(f"  Matched stats: {n_matched}")
    print(f"  Wrote: {lemmas_path}")
    print(f"  Wrote: {statistics_path}")


if __name__ == '__main__':
    main()
