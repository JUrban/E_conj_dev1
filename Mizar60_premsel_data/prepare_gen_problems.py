#!/usr/bin/env python3
"""
Create problem list for conjecture generation on Mizar60 problems.

Selects:
- All unsolved problems (from 00minsub)
- All solved problems with >= 2000M instructions (from 00cnfburnedV2)

Output: 00gen_problems.txt (one problem name per line)
"""

import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load unsolved
with open(os.path.join(SCRIPT_DIR, '00minsub')) as f:
    unsolved = set(l.strip() for l in f if l.strip())
print(f"Unsolved: {len(unsolved)}")

# Load solved with instruction counts
costs = {}
with open(os.path.join(SCRIPT_DIR, '00cnfburnedV2')) as f:
    for line in f:
        m = re.match(r'(.+):% Instructions burned: (\d+)', line)
        if m:
            costs[m.group(1)] = int(m.group(2))

hard_solved = set(p for p, c in costs.items() if c >= 2000)
print(f"Hard solved (>=2000M): {len(hard_solved)}")

# Combine and verify CNF files exist
cnf_dir = os.path.join(SCRIPT_DIR, 'problems_cnf')
combined = sorted(unsolved | hard_solved)
exists = [p for p in combined if os.path.exists(os.path.join(cnf_dir, p))]
missing = len(combined) - len(exists)
print(f"Combined: {len(combined)}, exist in problems_cnf: {len(exists)}, missing: {missing}")

# Write
out_path = os.path.join(SCRIPT_DIR, '00gen_problems.txt')
with open(out_path, 'w') as f:
    for p in exists:
        f.write(p + '\n')
print(f"Wrote {len(exists)} problems to {out_path}")
