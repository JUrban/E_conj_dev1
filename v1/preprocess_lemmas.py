#!/usr/bin/env python3
"""
Convert Vampire FOF lemmas to CNF and create lemmas_useful + statistics_useful
files for training. Self-contained: reads raw data files directly.

Input (raw Vampire output):
  v1/00cnfburnedV50k      - baseline proof instructions (50M limit)
  v1/00vconj_out_res1     - P & L results (25M limit)
  v1/00vconj_outn_res1    - P & ~L results (25M limit)
  v1/cnf_vout1/           - per-problem FOF lemma files

Output:
  v1/lemmas_useful        - useful CNF lemmas (ratio < 1), one per line
  v1/statistics_useful    - statistics for useful pairs, E-compatible format

Usage:
  python3 v1/preprocess_lemmas.py
"""

import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(SCRIPT_DIR, '00cnfburnedV50k')
PL_FILE = os.path.join(SCRIPT_DIR, '00vconj_out_res1')
PNL_FILE = os.path.join(SCRIPT_DIR, '00vconj_outn_res1')
LEMMA_DIR = os.path.join(SCRIPT_DIR, 'cnf_vout1')
LIMIT = 25000  # 25M instruction limit for P&L and P&~L


def fof_to_cnf(fof_line):
    """Convert a single FOF line to CNF format.

    Input:  fof(f219,plain,( ! [X0] : ( lit1 | lit2 ) )).
    Output: cnf(f219,plain,( lit1 | lit2 )).

    Returns (lemma_id, cnf_line) or None on parse failure.
    """
    m = re.match(r'fof\((\w+)\s*,\s*(\w+)\s*,\s*\((.+)\)\s*\)\s*\.', fof_line.strip())
    if not m:
        m = re.match(r'fof\((\w+)\s*,\s*(\w+)\s*,\s*(.+)\s*\)\s*\.', fof_line.strip())
        if not m:
            return None

    name, role, formula = m.group(1), m.group(2), m.group(3).strip()

    # Strip universal quantifiers
    body = formula
    while True:
        qm = re.match(r'\s*!\s*\[([^\]]*)\]\s*:\s*(.+)', body, re.DOTALL)
        if qm:
            body = qm.group(2).strip()
        else:
            break

    # Strip balanced outer parens
    if body.startswith('(') and body.endswith(')'):
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
    # Step 1: Parse baseline
    print("Step 1/4: Loading baseline...")
    baseline = {}
    with open(BASELINE_FILE) as f:
        for line in f:
            m = re.match(r'^([^:]+):% Instructions burned: (\d+)', line)
            if m:
                baseline[m.group(1)] = int(m.group(2))
    print(f"  {len(baseline)} problems")

    # Step 2: Parse P&L and P&~L results
    print("Step 2/4: Loading P&L results...")
    pl = {}
    with open(PL_FILE) as f:
        for line in f:
            m = re.match(r'\./(?:.+/)?(.+?)__(.+?)\.gz:% Instructions burned: (\d+)', line)
            if m:
                pl[(m.group(1), m.group(2))] = int(m.group(3))
    print(f"  {len(pl)} entries")

    print("Step 3/4: Loading P&~L results...")
    pnl = {}
    with open(PNL_FILE) as f:
        for line in f:
            m = re.match(r'\./(?:.+/)?(.+?)__(.+?)\.gz:% Instructions burned: (\d+)', line)
            if m:
                pnl[(m.group(1), m.group(2))] = int(m.group(3))
    print(f"  {len(pnl)} entries")

    # Step 3: Compute useful pairs
    common = set(pl.keys()) & set(pnl.keys())
    print(f"  Common pairs: {len(common)}")

    useful = {}  # (problem, lemma_id) -> (ratio, i_pl, i_pnl, i_base)
    for prob, lemma in common:
        i_pl = pl[(prob, lemma)]
        i_pnl = pnl[(prob, lemma)]
        if i_pl > LIMIT or i_pnl > LIMIT:
            continue
        i_base = baseline.get(prob)
        if i_base is None or i_base <= 0:
            continue
        ratio = (i_pl + i_pnl) / i_base
        if 0 < ratio < 1.0:
            useful[(prob, lemma)] = (ratio, i_pl, i_pnl, i_base)

    print(f"  Useful pairs (0 < ratio < 1): {len(useful)}")
    useful_problems = set(k[0] for k in useful.keys())
    print(f"  Useful problems: {len(useful_problems)}")

    # Step 4: Convert FOF lemmas to CNF for useful pairs only
    print("Step 4/4: Converting FOF lemmas to CNF...")
    lemmas_path = os.path.join(SCRIPT_DIR, 'lemmas_useful')
    statistics_path = os.path.join(SCRIPT_DIR, 'statistics_useful')

    n_written = 0
    n_failed = 0

    problems = sorted(os.listdir(LEMMA_DIR))
    with open(lemmas_path, 'w') as lf, open(statistics_path, 'w') as sf:
        for pi, prob_name in enumerate(problems):
            if prob_name not in useful_problems:
                continue

            prob_path = os.path.join(LEMMA_DIR, prob_name)
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
                    key = (prob_name, lemma_id)
                    if key not in useful:
                        continue

                    ratio, i_pl, i_pnl, i_base = useful[key]

                    # Lemma format: ./problem/lemma_id: cnf(...)
                    lf.write(f"./{prob_name}/{lemma_id}: {cnf_line}\n")

                    # Statistics E-format:
                    # ratio:problem:cut_id:L1:L2:L1+L2:# label :L
                    sf.write(f"{ratio:.6f}:{prob_name}:{lemma_id}:"
                             f"{i_pl}:{i_pnl}:{i_pl+i_pnl}:"
                             f"# Instructions :{i_base}\n")
                    n_written += 1

            if (pi + 1) % 5000 == 0:
                print(f"  {pi+1}/{len(problems)}: {n_written} written, "
                      f"{n_failed} failed")

    print(f"\nDone:")
    print(f"  Written: {n_written}")
    print(f"  Failed:  {n_failed}")
    print(f"  Wrote: {lemmas_path}")
    print(f"  Wrote: {statistics_path}")


if __name__ == '__main__':
    main()
