"""
Comprehensive analysis of E prover evaluation results across all methods.

Produces:
  1. Top 20 coolest conjectures (biggest speedups with interesting math)
  2. Best conjectures to inspect manually (heuristic scoring)
  3. Overall proof-shortening metrics per method
  4. Complementarity analysis: which problems each method solves uniquely
  5. Union analysis: what's the best achievable by combining methods
  6. All results broken down by train/val/test split
"""

import os
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_split(split_file):
    """Load problem names from a split file."""
    with open(split_file) as f:
        return set(line.strip() for line in f if line.strip())


def load_baselines(path='eprover_baselines.json'):
    with open(path) as f:
        return json.load(f)


def load_results(results_dir):
    """Load eprover_results.tsv from a conjecture directory."""
    tsv_path = os.path.join(results_dir, 'eprover_results.tsv')
    if not os.path.exists(tsv_path):
        return []

    results = []
    with open(tsv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue
            try:
                r = {
                    'problem': parts[0],
                    'conjecture': parts[1],
                    'p1_status': parts[2],
                    'p1_clauses': int(parts[3]) if parts[3] != '-1' else -1,
                    'p2_status': parts[4],
                    'p2_clauses': int(parts[5]) if parts[5] != '-1' else -1,
                    'L_original': int(parts[6]) if parts[6] != '-1' else -1,
                    'ratio': float(parts[7]),
                    'speedup': parts[8] == 'True',
                    'clause': parts[9] if len(parts) > 9 else '',
                }
                results.append(r)
            except (ValueError, IndexError):
                continue
    return results


def get_useful(results):
    """Filter to useful speedups only."""
    return [r for r in results if r['speedup'] and r['ratio'] > 0]


def get_both_proved(results):
    """Filter to both P1 and P2 proved."""
    return [r for r in results if r['p1_status'] == 'proved' and r['p2_status'] == 'proved']


def main():
    # Load splits
    train_probs = load_split('train_problems.txt')
    val_probs = load_split('val_problems.txt')
    test_probs = load_split('test_problems.txt')

    def get_split(problem):
        if problem in test_probs:
            return 'test'
        elif problem in val_probs:
            return 'val'
        elif problem in train_probs:
            return 'train'
        return 'unknown'

    baselines = load_baselines()

    # Load all methods
    method_dirs = {
        'A2 (100ep)': 'conjectures_a2_arity_full',
        'A3 (150ep)': 'conjectures_a3_arity_full',
        'A+named': 'conjectures_a_named_full',
        'C (VAE)': 'conjectures_c_arity_full',
        'C+named': 'conjectures_c_named_full',
        'D (SSM)': 'conjectures_d_arity_full',
        'D+named': 'conjectures_d_named_full',
        'A2 noarity': 'conjectures_a2',
    }

    all_results = {}
    for name, dirpath in method_dirs.items():
        if os.path.isdir(dirpath):
            all_results[name] = load_results(dirpath)
            print(f"Loaded {name}: {len(all_results[name])} results")
        else:
            print(f"SKIP {name}: {dirpath} not found")

    print()

    # ================================================================
    # 1. TOP 20 COOLEST CONJECTURES (biggest speedups)
    # ================================================================
    print("=" * 80)
    print("1. TOP 20 COOLEST CONJECTURES (biggest speedups)")
    print("=" * 80)

    # Collect all useful conjectures across all methods
    all_useful = []
    for method, results in all_results.items():
        for r in get_useful(results):
            r['method'] = method
            r['split'] = get_split(r['problem'])
            r['speedup_factor'] = 1.0 / r['ratio'] if r['ratio'] > 0 else 0
            all_useful.append(r)

    # Sort by speedup factor (biggest first)
    all_useful.sort(key=lambda x: x['speedup_factor'], reverse=True)

    # Deduplicate by (problem, clause) to show unique conjectures
    seen = set()
    unique_useful = []
    for r in all_useful:
        key = (r['problem'], r['clause'][:100])
        if key not in seen:
            seen.add(key)
            unique_useful.append(r)

    print(f"\nTotal useful conjectures across all methods: {len(all_useful)}")
    print(f"Unique (by problem+clause): {len(unique_useful)}")
    print()

    for i, r in enumerate(unique_useful[:20]):
        L = r['L_original']
        L12 = r['p1_clauses'] + r['p2_clauses']
        print(f"  #{i+1}: {r['speedup_factor']:.1f}x speedup ({r['split']})")
        print(f"      Problem: {r['problem']} (L={L}, L1+L2={L12})")
        print(f"      Method: {r['method']}")
        print(f"      Clause: {r['clause'][:120]}")
        print()

    # ================================================================
    # 2. BEST CONJECTURES TO INSPECT MANUALLY (heuristic scoring)
    # ================================================================
    print("=" * 80)
    print("2. BEST CONJECTURES TO INSPECT MANUALLY")
    print("=" * 80)
    print("(Scored by: speedup factor × log(L_original) × diversity bonus)")
    print()

    # Score conjectures for manual inspection interest
    # Interesting = big speedup on hard problems with non-trivial clauses
    def inspection_score(r):
        import math
        speedup = r['speedup_factor']
        hardness = math.log(max(r['L_original'], 10))
        # Bonus for clauses with variables (more general)
        has_vars = 'X1' in r['clause'] or 'X2' in r['clause']
        var_bonus = 1.5 if has_vars else 1.0
        # Bonus for multi-literal clauses
        n_lits = r['clause'].count(' | ') + 1
        lit_bonus = min(n_lits, 4) / 2.0
        # Bonus for test set
        test_bonus = 1.3 if r['split'] == 'test' else 1.0
        return speedup * hardness * var_bonus * lit_bonus * test_bonus

    scored = [(inspection_score(r), r) for r in unique_useful]
    scored.sort(key=lambda x: x[0], reverse=True)

    for i, (score, r) in enumerate(scored[:20]):
        L = r['L_original']
        print(f"  #{i+1}: score={score:.1f} ({r['speedup_factor']:.1f}x, L={L}, {r['split']})")
        print(f"      Problem: {r['problem']}")
        print(f"      Method: {r['method']}")
        print(f"      Clause: {r['clause'][:150]}")
        print()

    # ================================================================
    # 3. OVERALL PROOF-SHORTENING METRICS PER METHOD
    # ================================================================
    print("=" * 80)
    print("3. OVERALL PROOF-SHORTENING METRICS PER METHOD")
    print("=" * 80)

    for split_name, split_set in [('ALL', train_probs | val_probs | test_probs),
                                   ('train', train_probs), ('val', val_probs), ('test', test_probs)]:
        print(f"\n--- {split_name} ---")
        print(f"{'Method':<16} {'Tested':>7} {'BothPr':>7} {'Useful':>7} {'Rate':>6} "
              f"{'AvgRatio':>8} {'BestR':>6} {'TotSaved':>10} {'ProblHelp':>9}")
        print("-" * 95)

        for method, results in sorted(all_results.items()):
            split_results = [r for r in results if r['problem'] in split_set]
            both = get_both_proved(split_results)
            useful = get_useful(split_results)

            n_tested = len(split_results)
            n_both = len(both)
            n_useful = len(useful)
            rate = n_useful / max(n_tested, 1) * 100

            avg_ratio = sum(r['ratio'] for r in useful) / max(len(useful), 1)
            best_ratio = min((r['ratio'] for r in useful), default=999)

            # Total clauses saved: sum of (L - L1 - L2) for useful conjectures
            total_saved = sum(r['L_original'] - r['p1_clauses'] - r['p2_clauses']
                            for r in useful if r['L_original'] > 0)

            # Number of unique problems helped
            problems_helped = len(set(r['problem'] for r in useful))

            print(f"{method:<16} {n_tested:>7} {n_both:>7} {n_useful:>7} {rate:>5.1f}% "
                  f"{avg_ratio:>8.3f} {best_ratio:>6.3f} {total_saved:>10,} {problems_helped:>9}")

    # ================================================================
    # 4. COMPLEMENTARITY ANALYSIS
    # ================================================================
    print()
    print("=" * 80)
    print("4. COMPLEMENTARITY ANALYSIS")
    print("=" * 80)

    # For each method, which problems get at least one useful speedup?
    method_solved = {}  # method -> set of problems with useful speedup
    method_best_ratio = {}  # method -> {problem: best_ratio}

    for method, results in all_results.items():
        useful = get_useful(results)
        solved = set()
        best = {}
        for r in useful:
            solved.add(r['problem'])
            if r['problem'] not in best or r['ratio'] < best[r['problem']]:
                best[r['problem']] = r['ratio']
        method_solved[method] = solved
        method_best_ratio[method] = best

    # Skip noarity for complementarity (it's a subset)
    comp_methods = [m for m in all_results if 'noarity' not in m]

    for split_name, split_set in [('ALL', train_probs | val_probs | test_probs),
                                   ('test', test_probs)]:
        print(f"\n--- Problems with at least one useful speedup ({split_name}) ---")

        # Per method
        for method in comp_methods:
            solved_in_split = method_solved.get(method, set()) & split_set
            print(f"  {method:<16}: {len(solved_in_split)} problems")

        # Pairwise overlaps
        print(f"\n--- Pairwise overlap ({split_name}) ---")
        print(f"{'':>16}", end='')
        for m in comp_methods:
            print(f" {m[:8]:>8}", end='')
        print()

        for m1 in comp_methods:
            s1 = method_solved.get(m1, set()) & split_set
            print(f"{m1:<16}", end='')
            for m2 in comp_methods:
                s2 = method_solved.get(m2, set()) & split_set
                overlap = len(s1 & s2)
                print(f" {overlap:>8}", end='')
            print()

        # Union analysis
        all_solved = set()
        for method in comp_methods:
            all_solved |= (method_solved.get(method, set()) & split_set)

        print(f"\n  Union (any method): {len(all_solved)} problems")

        # Incremental: which method adds the most when added to the pool?
        print(f"\n--- Greedy incremental ({split_name}) ---")
        remaining_methods = list(comp_methods)
        current_solved = set()
        order = []

        while remaining_methods:
            best_method = None
            best_new = 0
            for m in remaining_methods:
                new = len((method_solved.get(m, set()) & split_set) - current_solved)
                if new > best_new:
                    best_new = new
                    best_method = m
            if best_method is None or best_new == 0:
                break
            current_solved |= (method_solved.get(best_method, set()) & split_set)
            order.append((best_method, best_new, len(current_solved)))
            remaining_methods.remove(best_method)

        for method, new, total in order:
            print(f"  + {method:<16}: +{new} new problems = {total} total")

    # ================================================================
    # 5. BEST ACHIEVABLE BY COMBINING METHODS
    # ================================================================
    print()
    print("=" * 80)
    print("5. BEST ACHIEVABLE BY COMBINING METHODS (oracle selection)")
    print("=" * 80)

    for split_name, split_set in [('ALL', train_probs | val_probs | test_probs),
                                   ('test', test_probs)]:
        print(f"\n--- Oracle: best ratio per problem across all methods ({split_name}) ---")

        # For each problem, find the best ratio across all methods
        oracle_best = {}  # problem -> (best_ratio, method, clause)

        for method, results in all_results.items():
            if 'noarity' in method:
                continue
            for r in get_useful(results):
                if r['problem'] not in split_set:
                    continue
                if r['problem'] not in oracle_best or r['ratio'] < oracle_best[r['problem']][0]:
                    oracle_best[r['problem']] = (r['ratio'], method, r['clause'][:80])

        if not oracle_best:
            print("  No useful conjectures found")
            continue

        n_problems = len(oracle_best)
        avg_ratio = sum(r for r, _, _ in oracle_best.values()) / n_problems
        total_saved = 0
        for prob, (ratio, method, clause) in oracle_best.items():
            L = baselines.get(prob, 0)
            if L > 0:
                L12 = ratio * L
                total_saved += L - L12

        proved_in_split = sum(1 for p in split_set if baselines.get(p, -1) > 0)

        print(f"  Problems in split: {len(split_set)}")
        print(f"  Proved by E baseline: {proved_in_split}")
        print(f"  Problems with useful conjecture: {n_problems} "
              f"({100*n_problems/max(proved_in_split,1):.1f}% of proved)")
        print(f"  Average best ratio: {avg_ratio:.3f}")
        print(f"  Total clauses saved: {total_saved:,.0f}")

        # Show the 10 biggest savings
        savings = []
        for prob, (ratio, method, clause) in oracle_best.items():
            L = baselines.get(prob, 0)
            if L > 0:
                saved = L - ratio * L
                savings.append((saved, prob, ratio, method, clause, L))
        savings.sort(reverse=True)

        print(f"\n  Top 10 biggest savings:")
        for saved, prob, ratio, method, clause, L in savings[:10]:
            print(f"    {prob}: saved {saved:,.0f} clauses "
                  f"(L={L}, ratio={ratio:.3f}, {1/ratio:.1f}x, {method})")
            print(f"      {clause}")

    # ================================================================
    # 6. METHOD UNIQUENESS: which speedups does each method find exclusively?
    # ================================================================
    print()
    print("=" * 80)
    print("6. UNIQUE CONTRIBUTIONS PER METHOD")
    print("=" * 80)

    for split_name, split_set in [('test', test_probs)]:
        print(f"\n--- Unique problem speedups per method ({split_name}) ---")
        print("(Problems where ONLY this method finds a useful conjecture)")

        for method in comp_methods:
            my_solved = method_solved.get(method, set()) & split_set
            others_solved = set()
            for m2 in comp_methods:
                if m2 != method:
                    others_solved |= (method_solved.get(m2, set()) & split_set)
            unique = my_solved - others_solved

            if unique:
                print(f"\n  {method}: {len(unique)} unique problems")
                for prob in sorted(unique)[:5]:
                    ratio = method_best_ratio[method].get(prob, 999)
                    L = baselines.get(prob, 0)
                    print(f"    {prob} (L={L}, ratio={ratio:.3f}, {1/ratio:.1f}x)")

    # ================================================================
    # 7. SUMMARY STATISTICS
    # ================================================================
    print()
    print("=" * 80)
    print("7. SUMMARY")
    print("=" * 80)

    # Test set summary
    print("\n--- Test Set Summary (279 problems) ---")
    test_proved = sum(1 for p in test_probs if baselines.get(p, -1) > 0)

    all_test_solved = set()
    for method in comp_methods:
        all_test_solved |= (method_solved.get(method, set()) & test_probs)

    print(f"  Problems proved by E baseline: {test_proved}")
    print(f"  Problems helped by ANY method: {len(all_test_solved)} "
          f"({100*len(all_test_solved)/max(test_proved,1):.1f}% of proved)")

    # Per-method test summary
    print(f"\n  {'Method':<16} {'Solved':>7} {'Unique':>7} {'BestSpeedup':>12}")
    print(f"  {'-'*50}")
    for method in comp_methods:
        solved = method_solved.get(method, set()) & test_probs
        others = set()
        for m2 in comp_methods:
            if m2 != method:
                others |= (method_solved.get(m2, set()) & test_probs)
        unique = solved - others
        best = min((method_best_ratio[method].get(p, 999) for p in solved), default=999)
        best_x = 1/best if best > 0 and best < 999 else 0
        print(f"  {method:<16} {len(solved):>7} {len(unique):>7} {best_x:>11.1f}x")


if __name__ == '__main__':
    main()
