#!/usr/bin/env python3
"""
Combine E prover and vmin1 (Vampire minimized) datasets into a joint training set.

E data:    lemmas + statistics (original E prover, 44,895 useful from 122,356)
vmin1 data: vmin1/lemmas_useful + vmin1/statistics_useful (Vampire minimized claims)

Output:
  vmin1/combined_lemmas
  vmin1/combined_statistics
  vmin1/combined_train_problems.txt
  vmin1/combined_val_problems.txt

Since 2,481 problem names overlap between E and vmin1, vmin1 entries
are prefixed with 'vm_' to disambiguate.

Usage:
  python3 vmin1/combine_datasets.py [--val_frac 0.1]
"""

import os
import random
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# E prover data
E_LEMMAS = os.path.join(ROOT, 'lemmas')
E_STATS = os.path.join(ROOT, 'statistics')
E_TRAIN = os.path.join(ROOT, 'train_problems.txt')
E_VAL = os.path.join(ROOT, 'val_problems.txt')

# vmin1 data
VM_LEMMAS = os.path.join(SCRIPT_DIR, 'lemmas_useful')
VM_STATS = os.path.join(SCRIPT_DIR, 'statistics_useful')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of vmin1 problems for validation (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default=SCRIPT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)

    # Check overlap
    print("Checking problem name overlap...")
    e_probs = set()
    with open(E_STATS) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 3:
                e_probs.add(parts[1])

    vm_probs = set()
    with open(VM_STATS) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 3:
                vm_probs.add(parts[1])

    overlap = e_probs & vm_probs
    print(f"  E problems: {len(e_probs)}")
    print(f"  vmin1 problems: {len(vm_probs)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Prefixing vmin1 entries with 'vm_'")

    # Create vmin1 train/val split
    print("\nCreating vmin1 train/val split...")
    vm_prob_list = sorted(vm_probs)
    random.shuffle(vm_prob_list)
    n_val = max(1, int(len(vm_prob_list) * args.val_frac))
    vm_val = set(vm_prob_list[:n_val])
    vm_train = set(vm_prob_list[n_val:])
    print(f"  vmin1 train: {len(vm_train)}, val: {len(vm_val)}")

    # Combine lemmas
    out_lemmas = os.path.join(args.output_dir, 'combined_lemmas')
    n_e = 0
    n_vm = 0
    with open(out_lemmas, 'w') as out:
        with open(E_LEMMAS) as f:
            for line in f:
                out.write(line)
                n_e += 1

        with open(VM_LEMMAS) as f:
            for line in f:
                # ./prob/lid -> ./vm_prob/lid
                line = line.replace('./', './vm_', 1)
                out.write(line)
                n_vm += 1

    print(f"\nCombined lemmas: {n_e} E + {n_vm} vmin1 = {n_e + n_vm}")
    print(f"  Wrote: {out_lemmas}")

    # Combine statistics
    out_stats = os.path.join(args.output_dir, 'combined_statistics')
    n_e = 0
    n_vm = 0
    with open(out_stats, 'w') as out:
        with open(E_STATS) as f:
            for line in f:
                out.write(line)
                n_e += 1

        with open(VM_STATS) as f:
            for line in f:
                parts = line.split(':')
                parts[1] = 'vm_' + parts[1]
                line = ':'.join(parts)
                out.write(line)
                n_vm += 1

    print(f"Combined statistics: {n_e} E + {n_vm} vmin1 = {n_e + n_vm}")
    print(f"  Wrote: {out_stats}")

    # Combine train/val splits
    # E train/val
    e_train_probs = set()
    with open(E_TRAIN) as f:
        for line in f:
            p = line.strip()
            if p:
                e_train_probs.add(p)

    e_val_probs = set()
    with open(E_VAL) as f:
        for line in f:
            p = line.strip()
            if p:
                e_val_probs.add(p)

    # Combined train
    train_probs = e_train_probs | {'vm_' + p for p in vm_train}
    val_probs = e_val_probs | {'vm_' + p for p in vm_val}

    for split, probs in [('train', train_probs), ('val', val_probs)]:
        out_file = os.path.join(args.output_dir, f'combined_{split}_problems.txt')
        with open(out_file, 'w') as out:
            for p in sorted(probs):
                out.write(p + '\n')
        n_e_split = len(e_train_probs if split == 'train' else e_val_probs)
        n_vm_split = len(probs) - n_e_split
        print(f"Combined {split}: {n_e_split} E + {n_vm_split} vmin1 = {len(probs)}")
        print(f"  Wrote: {out_file}")

    # Count useful E entries (ratio < 1) for summary
    n_e_useful = 0
    with open(E_STATS) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 1:
                try:
                    if float(parts[0]) < 1.0:
                        n_e_useful += 1
                except ValueError:
                    pass
    print(f"\nSummary:")
    print(f"  E useful (ratio < 1): {n_e_useful}")
    print(f"  vmin1 useful: {n_vm}")
    print(f"  Total training examples: {n_e} E (all) + {n_vm} vmin1 (useful) = {n_e + n_vm}")


if __name__ == '__main__':
    main()
