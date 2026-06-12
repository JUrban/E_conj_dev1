#!/usr/bin/env python3
"""
Comprehensive orthogonality/complementarity analysis across all conjecture methods.

Auto-discovers all conjectures_*/eprover_results.tsv files.
Reports: per-method stats, pairwise overlap, greedy set cover, weighted cover
(accounting for speedup magnitude), unique contributions — all split by train/test.

Usage:
    python3 analyze_all.py
    python3 analyze_all.py conjectures_af6_*    # explicit dirs
"""

import os
import sys
import json
import math
from collections import defaultdict


def load_split(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def load_baselines(path='eprover_baselines.json'):
    """Load baselines. Handles both formats:
    - {problem: int_clauses, ...}
    - {problem: clauses, ..., "__config__": "...", ...}
    Also tolerates nested dicts or string values.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    baselines = {}
    for k, v in raw.items():
        if k.startswith('__'):
            continue
        try:
            baselines[k] = int(v)
        except (ValueError, TypeError):
            # Skip non-integer entries (config strings, nested dicts, etc.)
            continue
    return baselines


def load_results(tsv_path):
    """Load eprover_results.tsv, return list of dicts."""
    results = []
    with open(tsv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue
            try:
                results.append({
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
                })
            except (ValueError, IndexError):
                continue
    return results


def method_stats(results, split_set):
    """Compute stats for one method on one split."""
    in_split = [r for r in results if r['problem'] in split_set]
    useful = [r for r in in_split if r['speedup'] and r['ratio'] > 0]

    # Per-problem: best ratio and total useful count
    best_per_prob = {}
    for r in useful:
        p = r['problem']
        if p not in best_per_prob or r['ratio'] < best_per_prob[p]['ratio']:
            best_per_prob[p] = r

    problems_helped = set(best_per_prob.keys())
    n_useful = len(useful)
    n_tested = len(set(r['problem'] for r in in_split))

    avg_ratio = sum(r['ratio'] for r in best_per_prob.values()) / max(len(best_per_prob), 1)

    # Total clauses saved (L - L1 - L2 for best conjecture per problem)
    total_saved = 0
    for r in best_per_prob.values():
        if r['L_original'] > 0:
            total_saved += r['L_original'] - r['p1_clauses'] - r['p2_clauses']

    # Weighted score: sum of (1 - ratio) for best per problem
    # This rewards both coverage AND magnitude of speedup
    weighted_score = sum(max(0, 1 - r['ratio']) for r in best_per_prob.values())

    # Hardness-weighted: (1 - ratio) * log(L), prioritizes harder problems
    hard_score = sum(max(0, 1 - r['ratio']) * math.log(max(r['L_original'], 10))
                     for r in best_per_prob.values())

    return {
        'n_tested': n_tested,
        'n_useful': n_useful,
        'problems_helped': problems_helped,
        'best_per_prob': best_per_prob,
        'avg_ratio': avg_ratio,
        'total_saved': total_saved,
        'weighted_score': weighted_score,
        'hard_score': hard_score,
    }


def print_table(headers, rows, fmt=None):
    """Pretty-print a table."""
    if fmt is None:
        fmt = ['<'] * len(headers)
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    # Header
    hdr = '  '.join(f'{h:>{w}}' if f == '>' else f'{h:<{w}}'
                     for h, w, f in zip(headers, widths, fmt))
    print(hdr)
    print('-' * len(hdr))
    for row in rows:
        line = '  '.join(f'{str(v):>{w}}' if f == '>' else f'{str(v):<{w}}'
                         for v, w, f in zip(row, widths, fmt))
        print(line)


def greedy_cover(method_problems, split_set, method_best_ratios=None,
                 baselines=None, mode='plain'):
    """Greedy set cover with different weighting modes.

    Modes:
        'plain':    maximize number of new problems per step
        'weighted': maximize sum of (1 - ratio) for new problems
        'hard':     maximize sum of (1 - ratio) * log(L) for new problems
        'saved':    maximize total clauses saved (L - L*ratio) for new problems
    """
    remaining = {m: s & split_set for m, s in method_problems.items()}
    covered = set()
    order = []

    def score_new(method, new_probs):
        if mode == 'plain':
            return len(new_probs)
        elif mode == 'weighted':
            return sum(max(0, 1 - method_best_ratios[method].get(p, 1.0))
                       for p in new_probs)
        elif mode == 'hard':
            s = 0
            for p in new_probs:
                r = method_best_ratios[method].get(p, 1.0)
                L = baselines.get(p, 10) if baselines else 10
                s += max(0, 1 - r) * math.log(max(L, 10))
            return s
        elif mode == 'saved':
            s = 0
            for p in new_probs:
                r = method_best_ratios[method].get(p, 1.0)
                L = baselines.get(p, 0) if baselines else 0
                s += max(0, L - L * r)
            return s
        return len(new_probs)

    while remaining:
        best_m = None
        best_val = -1

        for m, probs in remaining.items():
            new_probs = probs - covered
            if not new_probs:
                continue
            val = score_new(m, new_probs)
            if val > best_val:
                best_val = val
                best_m = m

        if best_m is None:
            break

        new = remaining[best_m] - covered
        covered |= new
        sc = score_new(best_m, new)
        if mode == 'plain':
            order.append((best_m, len(new), len(covered), ''))
        else:
            order.append((best_m, len(new), len(covered), f'{sc:.1f}'))
        del remaining[best_m]

    return order


def main():
    # Discover result directories
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        dirs = sorted(d for d in os.listdir('.')
                      if os.path.isdir(d) and d.startswith('conjectures_')
                      and os.path.exists(os.path.join(d, 'eprover_results.tsv')))

    if not dirs:
        print("No conjecture directories with eprover_results.tsv found.")
        sys.exit(1)

    # Load splits and baselines
    train_set = load_split('train_problems.txt')
    val_set = load_split('val_problems.txt')
    test_set = load_split('test_problems.txt')
    all_set = train_set | val_set | test_set
    baselines = load_baselines()

    # Load all methods
    methods = {}
    for d in dirs:
        tsv = os.path.join(d, 'eprover_results.tsv')
        if os.path.exists(tsv):
            # Short name: strip conjectures_ prefix and common suffixes
            name = d.replace('conjectures_', '').rstrip('/')
            methods[name] = load_results(tsv)

    # If baselines file was empty/missing, reconstruct from L_original in results
    if not baselines:
        print("(Reconstructing baselines from eprover_results.tsv L_original values)")
        for results in methods.values():
            for r in results:
                L = r['L_original']
                if L > 0:
                    baselines[r['problem']] = L

    # Count provable problems per split
    provable = {p for p, L in baselines.items() if L > 0}

    print(f"Loaded {len(methods)} methods: {', '.join(methods.keys())}")
    print(f"Splits: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"Provable: train={len(provable & train_set)}, "
          f"val={len(provable & val_set)}, test={len(provable & test_set)}")
    print()

    # ================================================================
    # 1. PER-METHOD STATS
    # ================================================================
    for split_name, split_set in [('test', test_set), ('train', train_set),
                                   ('val', val_set), ('ALL', all_set)]:
        n_provable = len(provable & split_set)
        print('=' * 90)
        print(f'PER-METHOD STATS — {split_name} ({len(split_set)} problems, {n_provable} provable)')
        print('=' * 90)

        rows = []
        stats_cache = {}
        for name, results in sorted(methods.items()):
            s = method_stats(results, split_set)
            stats_cache[(name, split_name)] = s
            n_helped = len(s['problems_helped'])
            pct = 100 * n_helped / max(n_provable, 1)
            rows.append((
                name,
                s['n_tested'],
                s['n_useful'],
                n_helped,
                f'{pct:.1f}%',
                f"{s['avg_ratio']:.3f}",
                f"{s['total_saved']:,}",
                f"{s['weighted_score']:.1f}",
                f"{s['hard_score']:.1f}",
            ))

        print_table(
            ['Method', 'Tested', 'Useful', 'ProblHelp', '%Provable',
             'AvgRatio', 'ClausesSaved', 'WScore', 'HardScore'],
            rows,
            ['<', '>', '>', '>', '>', '>', '>', '>', '>']
        )
        print()

    # ================================================================
    # 2. PAIRWISE OVERLAP MATRIX (test set)
    # ================================================================
    for split_name, split_set in [('test', test_set), ('train', train_set)]:
        print('=' * 90)
        print(f'PAIRWISE OVERLAP — {split_name} (problems helped by both methods)')
        print('=' * 90)

        mnames = sorted(methods.keys())
        # Build problem sets
        mprobs = {}
        for name in mnames:
            s = method_stats(methods[name], split_set)
            mprobs[name] = s['problems_helped']

        # Short names for header (max 10 chars)
        short = {n: n[:10] for n in mnames}

        # Header
        print(f"{'':>20}", end='')
        for n in mnames:
            print(f" {short[n]:>10}", end='')
        print(f" {'UNIQUE':>7}")

        # Rows
        all_helped = set()
        for n in mnames:
            all_helped |= mprobs[n]

        for n1 in mnames:
            print(f"{short[n1]:>20}", end='')
            for n2 in mnames:
                overlap = len(mprobs[n1] & mprobs[n2])
                print(f" {overlap:>10}", end='')
            # Unique: problems only this method helps
            others = set()
            for n2 in mnames:
                if n2 != n1:
                    others |= mprobs[n2]
            unique = len(mprobs[n1] - others)
            print(f" {unique:>7}")

        union = set()
        for n in mnames:
            union |= mprobs[n]
        print(f"\n  Union: {len(union)} problems")
        print()

    # ================================================================
    # 3. GREEDY SET COVER
    # ================================================================
    for split_name, split_set in [('test', test_set), ('train', train_set)]:
        print('=' * 90)
        print(f'GREEDY SET COVER — {split_name}')
        print('=' * 90)

        mprobs = {}
        mbest = {}
        for name in methods:
            s = method_stats(methods[name], split_set)
            mprobs[name] = s['problems_helped']
            mbest[name] = {p: r['ratio'] for p, r in s['best_per_prob'].items()}

        # Plain greedy (maximize problems covered)
        print("\n  Plain greedy (maximize new problems per step):")
        order = greedy_cover(mprobs, split_set, mode='plain')
        for m, new, total, _ in order:
            print(f"    + {m:<30} +{new:>4} new = {total:>4} total")

        # Weighted greedy (maximize sum of speedup magnitude)
        print("\n  Weighted greedy (maximize sum of (1-ratio) for new problems):")
        order = greedy_cover(mprobs, split_set, mbest, mode='weighted')
        for m, new, total, wscore in order:
            print(f"    + {m:<30} +{new:>4} new = {total:>4} total  (w +{wscore})")

        # Hardness-weighted greedy (prioritize harder problems)
        print("\n  Hardness-weighted greedy ((1-ratio)*log(L), harder problems count more):")
        order = greedy_cover(mprobs, split_set, mbest, baselines, mode='hard')
        for m, new, total, wscore in order:
            print(f"    + {m:<30} +{new:>4} new = {total:>4} total  (hard +{wscore})")

        # Clauses-saved greedy (maximize raw clauses saved)
        print("\n  Clauses-saved greedy (maximize L - L*ratio for new problems):")
        order = greedy_cover(mprobs, split_set, mbest, baselines, mode='saved')
        for m, new, total, wscore in order:
            print(f"    + {m:<30} +{new:>4} new = {total:>4} total  (saved +{wscore})")

        print()

    # ================================================================
    # 4. ORACLE COMBINATION
    # ================================================================
    for split_name, split_set in [('test', test_set), ('train', train_set)]:
        n_provable = len(provable & split_set)
        print('=' * 90)
        print(f'ORACLE COMBINATION — {split_name} (best ratio per problem across all methods)')
        print('=' * 90)

        oracle = {}  # problem -> (ratio, method, clause, L, L1+L2)
        for name, results in methods.items():
            for r in results:
                if not r['speedup'] or r['ratio'] <= 0:
                    continue
                if r['problem'] not in split_set:
                    continue
                p = r['problem']
                if p not in oracle or r['ratio'] < oracle[p][0]:
                    oracle[p] = (r['ratio'], name, r['clause'][:80],
                                 r['L_original'],
                                 r['p1_clauses'] + r['p2_clauses'])

        n_helped = len(oracle)
        if n_helped == 0:
            print("  No useful conjectures found")
            continue

        avg_ratio = sum(v[0] for v in oracle.values()) / n_helped
        total_saved = sum(v[3] - v[4] for v in oracle.values() if v[3] > 0)
        weighted = sum(max(0, 1 - v[0]) for v in oracle.values())

        # Which method provides the oracle winner most often?
        winner_counts = defaultdict(int)
        for _, (_, method, _, _, _) in oracle.items():
            winner_counts[method] += 1

        print(f"\n  Problems helped: {n_helped} / {n_provable} provable "
              f"({100*n_helped/max(n_provable,1):.1f}%)")
        print(f"  Avg best ratio: {avg_ratio:.3f}")
        print(f"  Total clauses saved: {total_saved:,}")
        print(f"  Weighted score: {weighted:.1f}")
        print(f"\n  Oracle winner distribution:")
        for m, cnt in sorted(winner_counts.items(), key=lambda x: -x[1]):
            print(f"    {m:<30} {cnt:>4} problems ({100*cnt/n_helped:.1f}%)")

        # Top 15 biggest speedups
        top = sorted(oracle.items(), key=lambda x: x[1][0])
        print(f"\n  Top 15 biggest speedups:")
        for i, (prob, (ratio, method, clause, L, L12)) in enumerate(top[:15]):
            speedup = 1/ratio if ratio > 0 else 0
            print(f"    {i+1:>2}. {prob:<25} {speedup:>6.1f}x  ratio={ratio:.3f}  "
                  f"L={L}  {method}")
            print(f"        {clause}")

        print()

    # ================================================================
    # 5. SUBSET ANALYSIS: best N-method combinations
    # ================================================================
    from itertools import combinations

    for split_name, split_set in [('test', test_set)]:
        print('=' * 90)
        print(f'BEST N-METHOD COMBINATIONS — {split_name}')
        print('=' * 90)

        mprobs = {}
        mbest = {}
        for name in methods:
            s = method_stats(methods[name], split_set)
            mprobs[name] = s['problems_helped']
            mbest[name] = {p: r['ratio'] for p, r in s['best_per_prob'].items()}

        mnames = list(methods.keys())

        def combo_scores(combo):
            """Compute all scores for a method combination."""
            union = set()
            for m in combo:
                union |= mprobs.get(m, set())
            w, h, sv = 0, 0, 0
            for p in union:
                best_r = min(mbest[m].get(p, 1.0) for m in combo)
                gain = max(0, 1 - best_r)
                L = baselines.get(p, 10)
                w += gain
                h += gain * math.log(max(L, 10))
                sv += max(0, L - L * best_r)
            return len(union), w, h, sv

        # Find best combo by each criterion
        for criterion, label in [('count', 'By coverage (most problems)'),
                                  ('hard', 'By hardness (harder problems matter more)'),
                                  ('saved', 'By clauses saved')]:
            print(f"\n  --- {label} ---")
            for k in range(1, min(len(mnames) + 1, 6)):
                best_combo = None
                best_key = (-1, -1)

                for combo in combinations(mnames, k):
                    cnt, w, h, sv = combo_scores(combo)
                    if criterion == 'count':
                        key = (cnt, w)
                    elif criterion == 'hard':
                        key = (h, cnt)
                    elif criterion == 'saved':
                        key = (sv, cnt)
                    if key > best_key:
                        best_key = key
                        best_combo = combo

                if best_combo:
                    cnt, w, h, sv = combo_scores(best_combo)
                    print(f"  Best {k}: {cnt:>3} problems  w={w:.1f}  "
                          f"hard={h:.1f}  saved={sv:,.0f}")
                    print(f"    {' + '.join(best_combo)}")

        print()

    # ================================================================
    # 6. PER-PROBLEM DETAIL (test, sortable)
    # ================================================================
    print('=' * 90)
    print('PER-PROBLEM DETAIL — test (which methods help each problem)')
    print('=' * 90)

    mnames = sorted(methods.keys())
    short_names = {n: n[:8] for n in mnames}

    # Build per-problem info
    prob_info = {}
    for prob in sorted(test_set):
        L = baselines.get(prob, -1)
        if L <= 0:
            continue
        info = {'L': L, 'methods': {}}
        for name in mnames:
            s = method_stats(methods[name], test_set)
            if prob in s['best_per_prob']:
                info['methods'][name] = s['best_per_prob'][prob]['ratio']
        if info['methods']:
            prob_info[prob] = info

    # Sort by number of methods that help (ascending = hardest for ensemble)
    by_coverage = sorted(prob_info.items(), key=lambda x: (len(x[1]['methods']), x[1]['L']))

    # Show problems helped by only 1 method (most fragile)
    print(f"\n  Problems helped by only 1 method ({split_name}):")
    single_method_probs = [(p, i) for p, i in by_coverage if len(i['methods']) == 1]
    for prob, info in single_method_probs[:20]:
        m, r = list(info['methods'].items())[0]
        print(f"    {prob:<28} L={info['L']:>5}  ratio={r:.3f}  {1/r:.1f}x  only: {m}")
    if len(single_method_probs) > 20:
        print(f"    ... ({len(single_method_probs)} total)")

    # Show problems helped by ALL methods (easiest)
    all_method_probs = [(p, i) for p, i in by_coverage if len(i['methods']) == len(mnames)]
    print(f"\n  Problems helped by all {len(mnames)} methods: {len(all_method_probs)}")

    print()

    # ================================================================
    # 7. RANK ANALYSIS: at what conjecture position do useful speedups appear?
    # ================================================================
    for split_name, split_set in [('test', test_set), ('train', train_set)]:
        print('=' * 90)
        print(f'RANK ANALYSIS — {split_name} (where do useful conjectures appear?)')
        print('=' * 90)

        for name in sorted(methods.keys()):
            results = methods[name]
            in_split = [r for r in results if r['problem'] in split_set]
            useful = [r for r in in_split if r['speedup'] and r['ratio'] > 0]

            if not useful:
                continue

            # Extract rank from conjecture filename (conjecture_001.p -> rank 1)
            def get_rank(r):
                cf = r['conjecture']
                try:
                    # conjecture_NNN.p -> NNN
                    return int(cf.replace('conjecture_', '').replace('.p', ''))
                except (ValueError, AttributeError):
                    return 999

            # Per problem: rank of first useful conjecture, rank of best conjecture
            prob_first_rank = {}  # problem -> rank of first useful
            prob_best_rank = {}   # problem -> rank of best ratio conjecture
            prob_best_ratio = {}  # problem -> best ratio
            all_useful_ranks = []

            for r in useful:
                p = r['problem']
                rank = get_rank(r)
                all_useful_ranks.append(rank)

                if p not in prob_first_rank or rank < prob_first_rank[p]:
                    prob_first_rank[p] = rank
                if p not in prob_best_ratio or r['ratio'] < prob_best_ratio[p]:
                    prob_best_ratio[p] = r['ratio']
                    prob_best_rank[p] = rank

            n_problems = len(prob_first_rank)
            if not all_useful_ranks:
                continue

            # Max rank tested (from all results, not just useful)
            max_rank_tested = max(get_rank(r) for r in in_split)

            # Cumulative: how many problems helped if we only evaluate top-k conjectures?
            print(f"\n  {name} ({n_problems} problems helped, "
                  f"max rank tested: {max_rank_tested}):")

            cutoffs = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]
            cutoffs = [c for c in cutoffs if c <= max_rank_tested + 1]

            print(f"    {'Top-k':>6}  {'ProblHelp':>10}  {'Useful':>7}  "
                  f"{'Marginal':>9}  {'AvgBestR':>9}")
            print(f"    {'-'*50}")

            prev_helped = 0
            for k in cutoffs:
                # Problems where first useful conjecture has rank <= k
                helped_at_k = sum(1 for p, r in prob_first_rank.items() if r <= k)
                # Useful conjectures with rank <= k
                useful_at_k = sum(1 for r in all_useful_ranks if r <= k)
                # Marginal problems from k-1 to k
                marginal = helped_at_k - prev_helped
                # Average best ratio among problems helped at this cutoff
                probs_at_k = [p for p, r in prob_first_rank.items() if r <= k]
                avg_r = sum(prob_best_ratio[p] for p in probs_at_k) / max(len(probs_at_k), 1)
                # Only show if reachable
                if helped_at_k > 0 or k <= 5:
                    print(f"    {k:>6}  {helped_at_k:>10}  {useful_at_k:>7}  "
                          f"{'+' + str(marginal):>9}  {avg_r:>9.3f}")
                prev_helped = helped_at_k

            # Rank distribution of first useful conjecture
            ranks = sorted(prob_first_rank.values())
            median_rank = ranks[len(ranks) // 2] if ranks else 0
            mean_rank = sum(ranks) / len(ranks) if ranks else 0
            print(f"    First useful rank: median={median_rank}, "
                  f"mean={mean_rank:.1f}, "
                  f"min={min(ranks)}, max={max(ranks)}")

        print()

    # ================================================================
    # 8. COST-BENEFIT: E prover calls per new problem helped
    # ================================================================
    for split_name, split_set in [('test', test_set)]:
        print('=' * 90)
        print(f'COST-BENEFIT — {split_name} (E calls per marginal problem, across methods)')
        print('=' * 90)

        # For each method: at each rank cutoff, how many E calls (= 2 * rank * n_problems)
        # vs how many problems helped
        n_provable = len(provable & split_set)

        print(f"\n  {'Method':<25} {'Top-k':>6} {'Helped':>7} {'%Prov':>6} "
              f"{'E calls':>10} {'Calls/prob':>10}")
        print(f"  {'-'*70}")

        for name in sorted(methods.keys()):
            results = methods[name]
            in_split = [r for r in results if r['problem'] in split_set]
            useful = [r for r in in_split if r['speedup'] and r['ratio'] > 0]

            if not useful:
                continue

            def get_rank(r):
                cf = r['conjecture']
                try:
                    return int(cf.replace('conjecture_', '').replace('.p', ''))
                except (ValueError, AttributeError):
                    return 999

            prob_first_rank = {}
            for r in useful:
                p = r['problem']
                rank = get_rank(r)
                if p not in prob_first_rank or rank < prob_first_rank[p]:
                    prob_first_rank[p] = rank

            n_problems_tested = len(set(r['problem'] for r in in_split))
            max_rank = max(get_rank(r) for r in in_split)

            # Show for a few useful cutoffs (deduplicated)
            seen_k = set()
            for k in [5, 10, 20, 30, 50, max_rank]:
                if k > max_rank or k in seen_k:
                    continue
                seen_k.add(k)
                helped = sum(1 for r in prob_first_rank.values() if r <= k)
                if helped == 0:
                    continue
                e_calls = 2 * k * n_problems_tested  # P1+P2 per conjecture
                cost_per_prob = e_calls / helped
                pct = 100 * helped / max(n_provable, 1)
                print(f"  {name:<25} {k:>6} {helped:>7} {pct:>5.1f}% "
                      f"{e_calls:>10,} {cost_per_prob:>10.0f}")

        print()

    # ================================================================
    # 9. BUDGET OPTIMIZER: best strategy for a given E-call budget
    # ================================================================
    for split_name, split_set in [('test', test_set)]:
        print('=' * 90)
        print(f'BUDGET OPTIMIZER — {split_name}')
        print(f'Given a total E-call budget, what is the best allocation across methods?')
        print('=' * 90)

        # Precompute per-method: at each rank cutoff, which NEW problems (and their
        # best ratios) are found. We need the incremental data.
        # For each method, build: rank -> set of problems first found at that rank
        method_rank_data = {}
        for name in sorted(methods.keys()):
            results = methods[name]
            in_split = [r for r in results if r['problem'] in split_set]
            useful = [r for r in in_split if r['speedup'] and r['ratio'] > 0]

            def get_rank(r):
                cf = r['conjecture']
                try:
                    return int(cf.replace('conjecture_', '').replace('.p', ''))
                except (ValueError, AttributeError):
                    return 999

            # Per problem: first useful rank, and best ratio
            first_rank = {}
            best_ratio = {}
            for r in useful:
                p = r['problem']
                rank = get_rank(r)
                if p not in first_rank or rank < first_rank[p]:
                    first_rank[p] = rank
                if p not in best_ratio or r['ratio'] < best_ratio[p]:
                    best_ratio[p] = r['ratio']

            n_tested = len(set(r['problem'] for r in in_split))
            max_rank = max((get_rank(r) for r in in_split), default=0)

            method_rank_data[name] = {
                'first_rank': first_rank,
                'best_ratio': best_ratio,
                'n_tested': n_tested,
                'max_rank': max_rank,
            }

        # Greedy budget allocation:
        # At each step, pick the (method, top-k increment) that gives the most
        # new problems per E-call spent.
        # Each method can be "allocated" a top-k cutoff. Increasing from k to k+1
        # costs 2*n_tested E calls (one more conjecture per problem, P1+P2).
        # The gain is however many new problems are first found at rank k+1.

        # Build incremental gains for each method at each rank
        method_increments = {}  # name -> list of (rank, new_problems, new_weighted, e_cost_increment)
        for name, data in method_rank_data.items():
            n_tested = data['n_tested']
            first_rank = data['first_rank']
            best_ratio = data['best_ratio']
            max_rank = data['max_rank']
            if max_rank == 0:
                continue

            increments = []
            for k in range(1, max_rank + 1):
                # Problems first found at exactly rank k
                new_probs = [p for p, r in first_rank.items() if r == k]
                if new_probs:
                    w = sum(max(0, 1 - best_ratio.get(p, 1.0)) for p in new_probs)
                    h = sum(max(0, 1 - best_ratio.get(p, 1.0)) * math.log(max(baselines.get(p, 10), 10))
                            for p in new_probs)
                    increments.append({
                        'rank': k,
                        'new_count': len(new_probs),
                        'new_probs': set(new_probs),
                        'weighted': w,
                        'hard': h,
                        'e_cost': 2 * n_tested,  # cost to go from k-1 to k
                    })
                else:
                    increments.append({
                        'rank': k,
                        'new_count': 0,
                        'new_probs': set(),
                        'weighted': 0,
                        'hard': 0,
                        'e_cost': 2 * n_tested,
                    })
            method_increments[name] = increments

        # Greedy: repeatedly pick the method+rank increment with best
        # (new problems not yet covered) / (E cost)
        covered = set()
        allocation = {}  # method -> current top-k
        total_e_calls = 0
        schedule = []

        # Track which increments we've consumed per method
        method_ptr = {name: 0 for name in method_increments}

        while True:
            best_choice = None
            best_efficiency = -1

            for name, increments in method_increments.items():
                ptr = method_ptr[name]
                if ptr >= len(increments):
                    continue

                inc = increments[ptr]
                # How many genuinely new problems (not covered by prior picks)?
                new_here = inc['new_probs'] - covered
                if not new_here:
                    # Skip this rank (no new problems), but still consume it
                    # to get to deeper ranks. Cost is included.
                    # Actually, for efficiency, scan ahead to find next rank
                    # with new uncovered problems
                    pass

                n_new = len(new_here)
                if n_new > 0:
                    efficiency = n_new / inc['e_cost']
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_choice = (name, ptr, new_here, inc)

            if best_choice is None:
                # Check if any method still has ranks to consume
                # (might have ranks with no new problems to skip)
                advanced = False
                for name in method_increments:
                    ptr = method_ptr[name]
                    if ptr < len(method_increments[name]):
                        method_ptr[name] += 1
                        advanced = True
                if not advanced:
                    break
                continue

            name, ptr, new_here, inc = best_choice
            covered |= new_here
            total_e_calls += inc['e_cost']
            # Consume all ranks up to and including this one for the method
            method_ptr[name] = ptr + 1
            allocation[name] = inc['rank']

            schedule.append({
                'method': name,
                'top_k': inc['rank'],
                'new': len(new_here),
                'total': len(covered),
                'e_calls': total_e_calls,
                'efficiency': len(new_here) / inc['e_cost'],
            })

            if len(schedule) > 50:  # safety limit
                break

        # Print the schedule
        n_provable = len(provable & split_set)
        print(f"\n  Greedy budget schedule (pick method+rank with best new_problems/E_cost):")
        print(f"  {'Step':>4}  {'Method':<30} {'Top-k':>5}  {'New':>4}  {'Total':>5}  "
              f"{'%Prov':>6}  {'CumEcalls':>12}  {'Eff(prob/Ecall)':>15}")
        print(f"  {'-'*95}")

        for i, s in enumerate(schedule):
            pct = 100 * s['total'] / max(n_provable, 1)
            eff_str = f"{s['efficiency']:.6f}"
            print(f"  {i+1:>4}  {s['method']:<30} {s['top_k']:>5}  {s['new']:>4}  "
                  f"{s['total']:>5}  {pct:>5.1f}%  {s['e_calls']:>12,}  {eff_str:>15}")

        # Summary at standard budgets
        print(f"\n  Summary at standard budgets:")
        print(f"  {'Budget':>12}  {'Problems':>8}  {'%Provable':>10}  {'Methods used':>50}")
        print(f"  {'-'*85}")

        for budget in [5000, 10000, 20000, 50000, 100000, 200000]:
            # Find how far we get in the schedule
            probs = 0
            methods_used = set()
            for s in schedule:
                if s['e_calls'] <= budget:
                    probs = s['total']
                    methods_used.add(f"{s['method']}@{s['top_k']}")
                else:
                    break
            pct = 100 * probs / max(n_provable, 1)
            mstr = ', '.join(sorted(methods_used)[:5])
            if len(methods_used) > 5:
                mstr += f' +{len(methods_used)-5} more'
            print(f"  {budget:>12,}  {probs:>8}  {pct:>9.1f}%  {mstr:>50}")

        print()


if __name__ == '__main__':
    main()
