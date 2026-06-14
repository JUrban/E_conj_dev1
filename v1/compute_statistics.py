#!/usr/bin/env python3
"""
Compute statistics for the Vampire conjecture dataset (v1).

Reads:
  v1/00cnfburnedV50k      - baseline proof instructions (50M limit)
  v1/00vconj_out_res1     - P & L results (25M limit)
  v1/00vconj_outn_res1    - P & ~L results (25M limit)

Outputs:
  v1/statistics.tsv       - per-(problem, lemma) statistics
  v1/statistics_summary.txt - human-readable summary
"""

import re
import os
import json
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(SCRIPT_DIR, '00cnfburnedV50k')
PL_FILE = os.path.join(SCRIPT_DIR, '00vconj_out_res1')
PNL_FILE = os.path.join(SCRIPT_DIR, '00vconj_outn_res1')
LIMIT = 25000  # 25M instruction limit for P&L and P&~L runs


def parse_burned(path):
    """Parse 'name:% Instructions burned: N (million)' format."""
    results = {}
    pattern = re.compile(r'(?:\./(?:.+/)?)?(.+?)(?:__(.+?))?\.?(?:gz)?:% Instructions burned: (\d+)')
    with open(path) as f:
        for line in f:
            # Baseline format: problem:% Instructions burned: N (million)
            m = re.match(r'^([^:./]+):% Instructions burned: (\d+)', line)
            if m:
                results[m.group(1)] = int(m.group(2))
                continue
            # P&L / P&~L format: ./problem__lemma.gz:% Instructions burned: N (million)
            m = re.match(r'\./(?:.+/)?(.+?)__(.+?)\.gz:% Instructions burned: (\d+)', line)
            if m:
                results[(m.group(1), m.group(2))] = int(m.group(3))
    return results


def main():
    print("Loading baseline...")
    baseline = parse_burned(BASELINE_FILE)
    print(f"  {len(baseline)} problems, range {min(baseline.values())}-{max(baseline.values())}M instructions")

    print("Loading P&L results...")
    pl = {}
    with open(PL_FILE) as f:
        for line in f:
            m = re.match(r'\./(?:.+/)?(.+?)__(.+?)\.gz:% Instructions burned: (\d+)', line)
            if m:
                pl[(m.group(1), m.group(2))] = int(m.group(3))
    print(f"  {len(pl)} entries")

    print("Loading P&~L results...")
    pnl = {}
    with open(PNL_FILE) as f:
        for line in f:
            m = re.match(r'\./(?:.+/)?(.+?)__(.+?)\.gz:% Instructions burned: (\d+)', line)
            if m:
                pnl[(m.group(1), m.group(2))] = int(m.group(3))
    print(f"  {len(pnl)} entries")

    common = set(pl.keys()) & set(pnl.keys())
    print(f"  Common (problem, lemma) pairs: {len(common)}")

    # Compute per-pair statistics
    rows = []
    prob_useful = Counter()
    prob_total = Counter()

    for prob, lemma in sorted(common):
        i_pl = pl[(prob, lemma)]
        i_pnl = pnl[(prob, lemma)]
        i_base = baseline.get(prob, -1)

        pl_proved = i_pl <= LIMIT
        pnl_proved = i_pnl <= LIMIT
        both = pl_proved and pnl_proved

        ratio = (i_pl + i_pnl) / i_base if both and i_base > 0 else -1
        useful = ratio > 0 and ratio < 1.0

        rows.append({
            'problem': prob,
            'lemma': lemma,
            'i_base': i_base,
            'i_pl': i_pl,
            'i_pnl': i_pnl,
            'pl_proved': pl_proved,
            'pnl_proved': pnl_proved,
            'both_proved': both,
            'ratio': ratio,
            'useful': useful,
        })

        if both:
            prob_total[prob] += 1
        if useful:
            prob_useful[prob] += 1

    # Write TSV
    tsv_path = os.path.join(SCRIPT_DIR, 'statistics.tsv')
    with open(tsv_path, 'w') as f:
        f.write("problem\tlemma\ti_base\ti_pl\ti_pnl\tpl_proved\tpnl_proved\t"
                "both_proved\tratio\tuseful\n")
        for r in rows:
            f.write(f"{r['problem']}\t{r['lemma']}\t{r['i_base']}\t{r['i_pl']}\t"
                    f"{r['i_pnl']}\t{r['pl_proved']}\t{r['pnl_proved']}\t"
                    f"{r['both_proved']}\t{r['ratio']:.6f}\t{r['useful']}\n")
    print(f"\nWrote {len(rows)} rows to {tsv_path}")

    # Summary
    n_both = sum(1 for r in rows if r['both_proved'])
    n_useful = sum(1 for r in rows if r['useful'])
    useful_ratios = [r['ratio'] for r in rows if r['useful']]

    summary_lines = []
    def p(s):
        print(s)
        summary_lines.append(s)

    p(f"\n{'='*70}")
    p(f"VAMPIRE CONJECTURE DATASET STATISTICS (v1)")
    p(f"{'='*70}")
    p(f"")
    p(f"CNF problems attempted:         {57880}")
    p(f"Problems solved by Vampire (50M): {len(baseline)}")
    p(f"Proof lemmas generated:         {len(set(k[0] for k in pl.keys()))}")
    p(f"(problem, lemma) pairs tested:  {len(common)}")
    p(f"Both P&L and P&~L proved (25M): {n_both}")
    p(f"Useful (ratio < 1.0):           {n_useful}")
    p(f"Problems with useful lemma:     {len(prob_useful)}")
    p(f"")

    p(f"{'='*70}")
    p(f"RATIO DISTRIBUTION (useful examples by speedup threshold)")
    p(f"{'='*70}")
    p(f"{'Threshold':>10} {'Examples':>10} {'Problems':>10} {'%Useful':>10} {'%Problems':>12}")
    p(f"{'-'*55}")
    for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        cnt = sum(1 for r in useful_ratios if r <= t)
        prbs = len(set(r['problem'] for r in rows if r['useful'] and r['ratio'] <= t))
        p(f"{t:>10.2f} {cnt:>10,} {prbs:>10,} {100*cnt/n_useful:>9.1f}% {100*prbs/len(prob_useful):>11.1f}%")

    p(f"")
    p(f"{'='*70}")
    p(f"DATA IMBALANCE (useful examples per problem)")
    p(f"{'='*70}")
    counts = sorted(prob_useful.values(), reverse=True)
    p(f"Mean useful/problem:   {sum(counts)/len(counts):.1f}")
    p(f"Median useful/problem: {counts[len(counts)//2]}")
    p(f"Min:                   {counts[-1]}")
    p(f"Max:                   {counts[0]}")
    p(f"")
    p(f"{'Threshold':>12} {'Problems':>10} {'Cumulative%':>12}")
    p(f"{'-'*36}")
    for threshold in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        n = sum(1 for c in counts if c <= threshold)
        p(f"  <= {threshold:>4}       {n:>10} {100*n/len(counts):>11.1f}%")

    p(f"")
    p(f"Top 20 most represented problems:")
    for prob, cnt in prob_useful.most_common(20):
        p(f"  {prob:<30} {cnt:>5} useful lemmas")

    p(f"")
    p(f"{'='*70}")
    p(f"INSTRUCTION SAVINGS DISTRIBUTION")
    p(f"{'='*70}")
    # For useful examples, how many instructions saved?
    savings = [(r['i_base'] - r['i_pl'] - r['i_pnl'], r) for r in rows if r['useful']]
    savings.sort(key=lambda x: x[0], reverse=True)
    total_saved = sum(s for s, _ in savings)
    p(f"Total instructions saved: {total_saved:,}M")
    p(f"Mean per useful example:  {total_saved/len(savings):,.0f}M")
    p(f"")
    p(f"Top 15 biggest savings:")
    for i, (saved, r) in enumerate(savings[:15]):
        p(f"  {r['problem']:<25} lemma={r['lemma']:<8} "
          f"saved={saved:>6}M  ratio={r['ratio']:.3f}  "
          f"base={r['i_base']}M")

    # Write summary
    summary_path = os.path.join(SCRIPT_DIR, 'statistics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f"\nWrote summary to {summary_path}")

    # Write JSON for easy programmatic access
    json_path = os.path.join(SCRIPT_DIR, 'statistics.json')
    stats = {
        'cnf_problems': 57880,
        'baseline_problems': len(baseline),
        'pairs_tested': len(common),
        'both_proved': n_both,
        'useful': n_useful,
        'problems_with_useful': len(prob_useful),
        'instruction_limit': LIMIT,
        'baseline_limit': 50000,
    }
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
