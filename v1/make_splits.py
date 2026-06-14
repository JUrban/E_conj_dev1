#!/usr/bin/env python3
"""
Create train/val/test splits for the Vampire conjecture dataset.

Splits the 34,182 solved problems (those with lemma files) into
80/10/10 train/val/test by problem. Uses a fixed random seed for
reproducibility. Stratifies to ensure problems with useful lemmas
are proportionally distributed across splits.

Also writes split statistics showing useful example counts per split.
"""

import os
import re
import random
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# test = 1 - train - val = 0.10


def load_useful_problems(stats_tsv):
    """Load set of problems that have at least one useful lemma."""
    useful = set()
    with open(stats_tsv) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 10 and parts[9] == 'True':
                useful.add(parts[0])
    return useful


def main():
    # All solved problems (those with lemma files)
    lemma_dir = os.path.join(SCRIPT_DIR, 'cnf_vout1')
    all_problems = sorted(os.listdir(lemma_dir))
    print(f"Total solved problems: {len(all_problems)}")

    # Load useful problem set from statistics
    stats_tsv = os.path.join(SCRIPT_DIR, 'statistics.tsv')
    if os.path.exists(stats_tsv):
        useful_problems = load_useful_problems(stats_tsv)
        print(f"Problems with useful lemmas: {len(useful_problems)}")
    else:
        print("WARNING: statistics.tsv not found, no stratification")
        useful_problems = set()

    # Stratified split: separate useful and non-useful problems,
    # split each proportionally, then combine
    useful_list = sorted([p for p in all_problems if p in useful_problems])
    nonuseful_list = sorted([p for p in all_problems if p not in useful_problems])
    print(f"Useful: {len(useful_list)}, Non-useful: {len(nonuseful_list)}")

    random.seed(SEED)
    random.shuffle(useful_list)
    random.shuffle(nonuseful_list)

    def split_list(lst):
        n = len(lst)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

    u_train, u_val, u_test = split_list(useful_list)
    n_train, n_val, n_test = split_list(nonuseful_list)

    train = sorted(u_train + n_train)
    val = sorted(u_val + n_val)
    test = sorted(u_test + n_test)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)} ({len(u_train)} useful + {len(n_train)} non-useful)")
    print(f"  Val:   {len(val)} ({len(u_val)} useful + {len(n_val)} non-useful)")
    print(f"  Test:  {len(test)} ({len(u_test)} useful + {len(n_test)} non-useful)")

    # Write split files
    for name, probs in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(SCRIPT_DIR, f'{name}_problems.txt')
        with open(path, 'w') as f:
            for p in probs:
                f.write(p + '\n')
        print(f"  Wrote {path}")

    # Compute per-split statistics if we have the full data
    if os.path.exists(stats_tsv):
        print(f"\nPer-split useful example counts:")
        train_set, val_set, test_set = set(train), set(val), set(test)

        split_useful = {'train': 0, 'val': 0, 'test': 0}
        split_total = {'train': 0, 'val': 0, 'test': 0}
        split_probs_useful = {'train': set(), 'val': set(), 'test': set()}

        with open(stats_tsv) as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 10:
                    continue
                prob = parts[0]
                is_useful = parts[9] == 'True'
                both = parts[7] == 'True'

                if prob in train_set:
                    s = 'train'
                elif prob in val_set:
                    s = 'val'
                elif prob in test_set:
                    s = 'test'
                else:
                    continue

                if both:
                    split_total[s] += 1
                if is_useful:
                    split_useful[s] += 1
                    split_probs_useful[s].add(prob)

        for s in ['train', 'val', 'test']:
            n_probs = len(train_set) if s == 'train' else len(val_set) if s == 'val' else len(test_set)
            n_useful_probs = len(split_probs_useful[s])
            print(f"  {s:>5}: {split_useful[s]:>8} useful / {split_total[s]:>8} total "
                  f"from {n_useful_probs:>5}/{n_probs:>5} problems")


if __name__ == '__main__':
    main()
