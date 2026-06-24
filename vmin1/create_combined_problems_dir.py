#!/usr/bin/env python3
"""
Create a combined problems directory with symlinks.

E problems:    problems/          (original names)
vmin1 problems: Mizar60_premsel_data/problems_cnf/  (prefixed with 'vm_')

Output:
  vmin1/combined_problems/   — symlinks to both
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

E_PROBLEMS = os.path.join(ROOT, 'problems')
VM_PROBLEMS = os.path.join(ROOT, 'Mizar60_premsel_data', 'problems_cnf')
OUT_DIR = os.path.join(SCRIPT_DIR, 'combined_problems')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_e = 0
    n_vm = 0

    # E problems (no prefix)
    for name in os.listdir(E_PROBLEMS):
        src = os.path.join(E_PROBLEMS, name)
        dst = os.path.join(OUT_DIR, name)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        n_e += 1

    # vmin1 problems (vm_ prefix)
    for name in os.listdir(VM_PROBLEMS):
        src = os.path.join(VM_PROBLEMS, name)
        dst = os.path.join(OUT_DIR, 'vm_' + name)
        if not os.path.exists(dst):
            os.symlink(src, dst)
        n_vm += 1

    print(f"Created {n_e} E + {n_vm} vmin1 = {n_e + n_vm} symlinks in {OUT_DIR}")


if __name__ == '__main__':
    main()
