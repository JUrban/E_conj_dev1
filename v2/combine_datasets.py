#!/usr/bin/env python3
"""
Combine E prover and v2 transfer datasets into a joint training set.

E data:  lemmas + statistics (from the original E prover experiments)
v2 data: v2/lemmas_useful + v2/statistics_useful (from E→Vampire transfer)

Output:
  v2/combined_lemmas_useful
  v2/combined_statistics_useful
  v2/combined_train_problems.txt
  v2/combined_val_problems.txt

Problems are kept disjoint across datasets (different problem dirs),
so they naturally don't overlap. The combined splits include all
problems from both datasets.
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# E prover data (original, with max_ratio filtering done at training time)
E_LEMMAS = os.path.join(ROOT, 'lemmas')
E_STATS = os.path.join(ROOT, 'statistics')
E_TRAIN = os.path.join(ROOT, 'train_problems.txt')
E_VAL = os.path.join(ROOT, 'val_problems.txt')

# v2 transfer data
V2_LEMMAS = os.path.join(SCRIPT_DIR, 'lemmas_useful')
V2_STATS = os.path.join(SCRIPT_DIR, 'statistics_useful')
V2_TRAIN = os.path.join(SCRIPT_DIR, 'train_problems.txt')
V2_VAL = os.path.join(SCRIPT_DIR, 'val_problems.txt')


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    # Prefix E data with "E:" and v2 data with "V:" in problem names
    # to keep them distinct (they use different problems_dir)
    # Actually, the problem names themselves don't overlap because
    # E problems are from problems/ and v2 from Mizar60_premsel_data/problems_cnf/
    # But the lemma/stats format uses problem names that could overlap.
    # Let's check for actual overlap first.

    print("Checking for problem name overlap...")
    e_probs = set()
    with open(E_STATS) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 3:
                e_probs.add(parts[1])

    v2_probs = set()
    with open(V2_STATS) as f:
        for line in f:
            parts = line.split(':')
            if len(parts) >= 3:
                v2_probs.add(parts[1])

    overlap = e_probs & v2_probs
    print(f"  E problems: {len(e_probs)}")
    print(f"  V2 problems: {len(v2_probs)}")
    print(f"  Overlap: {len(overlap)}")

    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping problem names!")
        print(f"  First 5: {sorted(overlap)[:5]}")
        print(f"  Will prefix V2 entries with 'v2_' to disambiguate")
        prefix_v2 = True
    else:
        prefix_v2 = False

    # Combine lemmas
    out_lemmas = os.path.join(SCRIPT_DIR, 'combined_lemmas')
    n_e = 0
    n_v2 = 0
    with open(out_lemmas, 'w') as out:
        # E lemmas (all, not just useful — ratio filtering at train time)
        with open(E_LEMMAS) as f:
            for line in f:
                out.write(line)
                n_e += 1

        # V2 lemmas (already useful-only)
        with open(V2_LEMMAS) as f:
            for line in f:
                if prefix_v2:
                    # Prefix problem name in path: ./prob/id -> ./v2_prob/id
                    line = line.replace('./', './v2_', 1)
                out.write(line)
                n_v2 += 1

    print(f"\nCombined lemmas: {n_e} E + {n_v2} v2 = {n_e + n_v2}")
    print(f"  Wrote: {out_lemmas}")

    # Combine statistics
    out_stats = os.path.join(SCRIPT_DIR, 'combined_statistics')
    n_e = 0
    n_v2 = 0
    with open(out_stats, 'w') as out:
        with open(E_STATS) as f:
            for line in f:
                out.write(line)
                n_e += 1

        with open(V2_STATS) as f:
            for line in f:
                if prefix_v2:
                    parts = line.split(':')
                    parts[1] = 'v2_' + parts[1]
                    line = ':'.join(parts)
                out.write(line)
                n_v2 += 1

    print(f"Combined statistics: {n_e} E + {n_v2} v2 = {n_e + n_v2}")
    print(f"  Wrote: {out_stats}")

    # Combine train/val splits
    for split in ['train', 'val']:
        e_file = E_TRAIN if split == 'train' else E_VAL
        v2_file = V2_TRAIN if split == 'train' else V2_VAL
        out_file = os.path.join(SCRIPT_DIR, f'combined_{split}_problems.txt')

        probs = set()
        with open(e_file) as f:
            for line in f:
                p = line.strip()
                if p:
                    probs.add(p)
        n_e_split = len(probs)

        with open(v2_file) as f:
            for line in f:
                p = line.strip()
                if p:
                    if prefix_v2:
                        p = 'v2_' + p
                    probs.add(p)
        n_v2_split = len(probs) - n_e_split

        with open(out_file, 'w') as out:
            for p in sorted(probs):
                out.write(p + '\n')

        print(f"Combined {split}: {n_e_split} E + {n_v2_split} v2 = {len(probs)}")
        print(f"  Wrote: {out_file}")


if __name__ == '__main__':
    main()
