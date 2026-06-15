#!/usr/bin/env python3
"""
Create TPTP FOF problems from Mizar60 premise selection data.

For each theorem, creates one problem file containing:
- The theorem as a conjecture (role changed from axiom to conjecture)
- Its premises as axioms

Uses the proof with the fewest dependencies (first in file order if tied).

Input:
  allsolved_and_minsub_sorted.gz  — dependency lines: theorem:prem1 prem2 ...
  statements.gz                   — FOF formulas: fof(name, axiom, ...).

Output:
  problems_fof/<theorem_name>.p   — one TPTP file per theorem

Usage:
  python3 Mizar60_premsel_data/create_problems.py [output_dir]
"""

import gzip
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPS_FILE = os.path.join(SCRIPT_DIR, 'allsolved_and_minsub_sorted.gz')
STMTS_FILE = os.path.join(SCRIPT_DIR, 'statements.gz')
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'problems_fof')


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: For each theorem, find the proof with fewest dependencies
    # (first in file order if tied)
    print("Step 1: Reading dependencies...")
    best_deps = {}  # theorem -> list of premise names
    with gzip.open(DEPS_FILE, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            theorem, premises_str = line.split(':', 1)
            premises = premises_str.strip().split()
            if theorem not in best_deps or len(premises) < len(best_deps[theorem]):
                best_deps[theorem] = premises

    print(f"  {len(best_deps)} theorems with dependencies")

    # Step 2: Load all statements
    print("Step 2: Loading statements...")
    statements = {}  # name -> full fof(...) line
    with gzip.open(STMTS_FILE, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('fof('):
                continue
            # Extract name: fof(NAME, ...)
            comma_idx = line.index(',')
            name = line[4:comma_idx].strip()
            statements[name] = line

    print(f"  {len(statements)} statements loaded")

    # Step 3: Create problem files
    print("Step 3: Creating problems...")
    created = 0
    skipped_no_theorem = 0
    skipped_no_premise = 0

    for theorem, premises in sorted(best_deps.items()):
        if theorem not in statements:
            skipped_no_theorem += 1
            continue

        # Check all premises exist
        missing = [p for p in premises if p not in statements]
        if missing:
            skipped_no_premise += 1
            continue

        # Build problem file (pure FOF, no comments/metadata)
        problem_path = os.path.join(output_dir, theorem)
        with open(problem_path, 'w') as f:
            # Write premises as axioms (they already have role=axiom)
            for prem in premises:
                f.write(statements[prem] + '\n')

            # Write theorem as conjecture (change role from axiom to conjecture)
            thm_stmt = statements[theorem]
            conj_stmt = thm_stmt.replace(', axiom,', ', conjecture,', 1)
            f.write(conj_stmt + '\n')

        created += 1
        if created % 10000 == 0:
            print(f"  {created} problems created...")

    print(f"\nDone:")
    print(f"  Created:              {created}")
    print(f"  Skipped (no theorem): {skipped_no_theorem}")
    print(f"  Skipped (no premise): {skipped_no_premise}")
    print(f"  Output:               {output_dir}")


if __name__ == '__main__':
    main()
