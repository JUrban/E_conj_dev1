#!/usr/bin/env python3
"""
Create a combined problems directory with symlinks to both E and v2 problems.

E problems: problems/ (original names)
V2 problems: Mizar60_premsel_data/problems_cnf/ (prefixed with v2_)

Output: v2/combined_problems/ with symlinks
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

E_DIR = os.path.join(ROOT, 'problems')
V2_DIR = os.path.join(ROOT, 'Mizar60_premsel_data', 'problems_cnf')
OUT_DIR = os.path.join(SCRIPT_DIR, 'combined_problems')

os.makedirs(OUT_DIR, exist_ok=True)

# Symlink E problems (original names)
n_e = 0
for f in os.listdir(E_DIR):
    src = os.path.join(E_DIR, f)
    dst = os.path.join(OUT_DIR, f)
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)
    n_e += 1

# Symlink V2 problems (prefixed with v2_)
n_v2 = 0
for f in os.listdir(V2_DIR):
    src = os.path.join(V2_DIR, f)
    dst = os.path.join(OUT_DIR, f'v2_{f}')
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)
    n_v2 += 1

print(f"Combined problems dir: {OUT_DIR}")
print(f"  E problems: {n_e}")
print(f"  V2 problems (v2_ prefixed): {n_v2}")
print(f"  Total: {n_e + n_v2}")
