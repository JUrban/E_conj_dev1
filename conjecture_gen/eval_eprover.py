"""
Evaluate generated conjectures using E prover.

For each problem P and generated conjecture C, tests:
  (P1) Does C help prove P?  — add C as axiom to P, run E
  (P2) Is C provable from P? — negate C, add to P, run E

If both succeed, compare L1+L2 with original L to measure speedup.

Usage:
    python -m conjecture_gen.eval_eprover \
        --conjectures conjectures_d_fast/ \
        --problems problems/ \
        --statistics statistics \
        --eprover /path/to/eprover \
        --timeout 10 \
        --max_problems 100

Output: eval_results.tsv with per-conjecture prover results.
"""

import argparse
import os
import subprocess
import re
import time


def parse_eprover_output(output: str) -> dict:
    """Parse E prover output for processed clauses count and status."""
    result = {
        'status': 'unknown',
        'processed_clauses': -1,
    }

    for line in output.split('\n'):
        if '# SZS status' in line:
            if 'Unsatisfiable' in line or 'Theorem' in line:
                result['status'] = 'proved'
            elif 'Satisfiable' in line or 'CounterSatisfiable' in line:
                result['status'] = 'disproved'
            elif 'ResourceOut' in line or 'Timeout' in line:
                result['status'] = 'timeout'
            else:
                result['status'] = line.strip()

        m = re.search(r'# Processed clauses\s*:\s*(\d+)', line)
        if m:
            result['processed_clauses'] = int(m.group(1))

    return result


def run_eprover(problem_file: str, extra_axioms: str = None,
                eprover: str = 'eprover', timeout: int = 10) -> dict:
    """Run E prover on a problem, optionally with extra axioms.

    Uses a temp file to combine problem + extra axioms, since E
    works best with file input (not stdin). Strips # comments
    which are not standard TPTP.
    """
    import tempfile

    try:
        with open(problem_file) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {'status': 'file_not_found', 'processed_clauses': -1}

    # Keep only cnf() lines (strip # comments and blank lines)
    content = ''.join(line for line in lines if line.strip().startswith('cnf('))

    if extra_axioms:
        content = content + '\n' + extra_axioms + '\n'

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        proc = subprocess.run(
            [eprover, '--auto', '--cpu-limit=' + str(timeout),
             '-s', '--print-statistics', tmp_path],
            capture_output=True, text=True,
            timeout=timeout + 5,
        )
        os.unlink(tmp_path)
        return parse_eprover_output(proc.stdout + proc.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return {'status': f'error:{type(e).__name__}', 'processed_clauses': -1}


def negate_clause(clause_text: str) -> str:
    """Negate a CNF clause for the P2 test.

    A CNF clause ∀X. (L1(X) | L2(X) | ... | Ln(X)) negates to:
    ∃X. (~L1(X) ∧ ~L2(X) ∧ ... ∧ ~Ln(X))

    The existential is Skolemized: replace each variable Xi with a
    fresh Skolem constant negsk_i. Then each negated literal becomes
    a separate ground unit clause (implicitly conjoined in CNF).
    """
    # Collect all variables used in the clause
    var_pattern = re.compile(r'\b(X\d+)\b')
    all_vars = set(var_pattern.findall(clause_text))

    # Create Skolem substitution: X1 -> negsk_1, X2 -> negsk_2, etc.
    skolem_map = {v: f'negsk_{v[1:]}' for v in sorted(all_vars)}

    # Apply substitution to the whole clause
    skolemized = clause_text
    for var, sk in sorted(skolem_map.items(), key=lambda x: -len(x[0])):
        # Replace whole-word only (longer vars first to avoid X1 matching in X10)
        skolemized = re.sub(r'\b' + var + r'\b', sk, skolemized)

    # Split into literals and negate each
    literals = skolemized.split(' | ')
    negated_clauses = []
    for i, lit in enumerate(literals):
        lit = lit.strip()
        if not lit:
            continue
        if lit.startswith('~'):
            neg_lit = lit[1:]
        else:
            neg_lit = '~' + lit
        negated_clauses.append(
            f"cnf(neg_{i}, negated_conjecture, ({neg_lit}))."
        )
    return '\n'.join(negated_clauses)


def load_original_stats(statistics_file: str) -> dict:
    """Load original proof search lengths from statistics file."""
    stats = {}  # problem -> L_original (min processed clauses)
    with open(statistics_file) as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 8:
                try:
                    problem = parts[1]
                    L = int(parts[7].strip())
                    if problem not in stats or L < stats[problem]:
                        stats[problem] = L
                except (ValueError, IndexError):
                    continue
    return stats


# Module-level worker functions (must be picklable for ProcessPoolExecutor)
# Config is passed through the task tuple, not globals.


def _run_baseline_worker(args_tuple):
    problem_name, problems_dir, eprover, timeout = args_tuple
    path = os.path.join(problems_dir, problem_name)
    result = run_eprover(path, eprover=eprover, timeout=timeout)
    return problem_name, result


def _eval_one_worker(task):
    problem_name, conj_file, conj_line, conj_text, L_orig, problems_dir, eprover, timeout = task
    problem_path = os.path.join(problems_dir, problem_name)

    p1 = run_eprover(problem_path, conj_line,
                      eprover=eprover, timeout=timeout)
    neg_clauses = negate_clause(conj_text)
    p2 = run_eprover(problem_path, neg_clauses,
                      eprover=eprover, timeout=timeout)

    both_proved = (p1['status'] == 'proved' and p2['status'] == 'proved')
    ratio = -1.0
    speedup = False
    if both_proved and L_orig > 0:
        L1 = p1['processed_clauses']
        L2 = p2['processed_clauses']
        if L1 >= 0 and L2 >= 0:
            ratio = (L1 + L2) / L_orig
            speedup = ratio < 1.0

    return {
        'problem': problem_name, 'conj_file': conj_file,
        'conj_text': conj_text,
        'p1': p1, 'p2': p2, 'L_orig': L_orig,
        'ratio': ratio, 'speedup': speedup,
    }


def compute_baselines(problems_dir: str, problem_names: list[str],
                      eprover: str, timeout: int, workers: int,
                      cache_path: str = None) -> dict:
    """Compute baseline proof search lengths using our E prover setup.

    Runs E on each problem without any conjectures to get the true
    baseline L for comparison. Results are cached to disk.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if cache_path and os.path.exists(cache_path):
        import json
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} cached baselines from {cache_path}")
        # Only compute missing ones
        missing = [p for p in problem_names if p not in cached]
        if not missing:
            return cached
        print(f"  Computing {len(missing)} missing baselines...")
        problem_names = missing
    else:
        cached = {}

    stats = dict(cached)
    done = 0

    baseline_tasks = [(p, problems_dir, eprover, timeout) for p in problem_names]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_baseline_worker, t): t for t in baseline_tasks}
        for future in as_completed(futures):
            try:
                pname, result = future.result()
                if result['status'] == 'proved' and result['processed_clauses'] >= 0:
                    stats[pname] = result['processed_clauses']
                else:
                    stats[pname] = -1
                done += 1
                if done % 50 == 0 or done <= 3:
                    print(f"    baselines: {done}/{len(problem_names)} "
                          f"({pname}: {result['status']}, L={result['processed_clauses']})")
            except Exception as e:
                done += 1
                print(f"    baseline ERROR: {e}")

    proved = sum(1 for v in stats.values() if v > 0)
    print(f"  Baselines: {proved}/{len(stats)} proved within {timeout}s")

    if cache_path:
        import json
        with open(cache_path, 'w') as f:
            json.dump(stats, f)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate conjectures with E prover')
    parser.add_argument('--conjectures', required=True,
                        help='Directory with generated conjectures')
    parser.add_argument('--problems', default='problems',
                        help='Directory with original CNF problems')
    parser.add_argument('--eprover', default='eprover',
                        help='Path to E prover binary')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Per-proof time limit in seconds')
    parser.add_argument('--max_problems', type=int, default=0, help='0=all')
    parser.add_argument('--max_conjectures_per_problem', type=int, default=5,
                        help='Max conjectures to test per problem')
    parser.add_argument('--output', default=None)
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel E prover processes')
    parser.add_argument('--baseline_cache', default='eprover_baselines.json',
                        help='Cache file for baseline proof search lengths')

    args = parser.parse_args()

    # Check E prover
    try:
        proc = subprocess.run([args.eprover, '--version'],
                              capture_output=True, text=True, timeout=5)
        print(f"E prover: {proc.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"WARNING: E prover not found at '{args.eprover}'")

    # Find problems with generated conjectures
    conj_dir = args.conjectures
    problems = [d for d in sorted(os.listdir(conj_dir))
                if os.path.isdir(os.path.join(conj_dir, d))]
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
    print(f"{len(problems)} problems with conjectures, {args.workers} workers")

    # Compute baselines with OUR E prover (not relying on old stats)
    print("Computing baselines (same E, same timeout)...")
    orig_stats = compute_baselines(
        args.problems, problems, args.eprover, args.timeout,
        args.workers, cache_path=args.baseline_cache,
    )

    # Output file
    output_path = args.output or os.path.join(conj_dir, 'eprover_results.tsv')

    # Build all tasks first
    tasks = []  # (problem_name, conj_file, conj_line, conj_text, L_orig)
    for problem_name in problems:
        problem_path = os.path.join(args.problems, problem_name)
        if not os.path.exists(problem_path):
            continue

        L_orig = orig_stats.get(problem_name, -1)
        conj_path = os.path.join(conj_dir, problem_name)
        try:
            conj_files = sorted([f for f in os.listdir(conj_path) if f.endswith('.p')])
        except FileNotFoundError:
            continue
        conj_files = conj_files[:args.max_conjectures_per_problem]

        for conj_file in conj_files:
            with open(os.path.join(conj_path, conj_file)) as f:
                lines = f.readlines()
            conj_line = None
            conj_text = None
            for line in lines:
                line = line.strip()
                if line.startswith('cnf('):
                    conj_line = line
                    m = re.match(r'cnf\([^,]+,\s*[^,]+,\s*\((.+)\)\)\.\s*$', line)
                    if m:
                        conj_text = m.group(1)
                    break
            if conj_line and conj_text:
                tasks.append((problem_name, conj_file, conj_line, conj_text, L_orig,
                              args.problems, args.eprover, args.timeout))

    print(f"  Total tasks: {len(tasks)} (P1+P2 = {len(tasks)*2} prover calls)")

    # Run in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total_tested = 0
    total_p1_proved = 0
    total_p2_proved = 0
    total_both = 0
    total_useful = 0
    speedups = []
    results = []

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_eval_one_worker, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as e:
                continue

            results.append(r)
            total_tested += 1

            if r['p1']['status'] == 'proved':
                total_p1_proved += 1
            if r['p2']['status'] == 'proved':
                total_p2_proved += 1
            if r['p1']['status'] == 'proved' and r['p2']['status'] == 'proved':
                total_both += 1
            if r['speedup']:
                total_useful += 1
                speedups.append(r['ratio'])

            if total_tested % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {total_tested}/{len(tasks)}: "
                      f"p1={total_p1_proved} p2={total_p2_proved} "
                      f"both={total_both} useful={total_useful} "
                      f"({elapsed:.0f}s)")

    # Sort results by problem + conjecture for consistent output
    results.sort(key=lambda r: (r['problem'], r['conj_file']))

    with open(output_path, 'w') as out_f:
        out_f.write("problem\tconjecture\tp1_status\tp1_clauses\t"
                    "p2_status\tp2_clauses\tL_original\tratio\tspeedup\tclause\n")
        for r in results:
            out_f.write(
                f"{r['problem']}\t{r['conj_file']}\t"
                f"{r['p1']['status']}\t{r['p1']['processed_clauses']}\t"
                f"{r['p2']['status']}\t{r['p2']['processed_clauses']}\t"
                f"{r['L_orig']}\t{r['ratio']:.4f}\t{r['speedup']}\t"
                f"{r['conj_text']}\n"
            )

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"E Prover Evaluation Results")
    print(f"{'='*60}")
    print(f"Problems:           {len(problems)}")
    print(f"Conjectures tested: {total_tested}")
    print(f"P1 proved (C helps): {total_p1_proved} ({100*total_p1_proved/max(total_tested,1):.1f}%)")
    print(f"P2 proved (C valid): {total_p2_proved} ({100*total_p2_proved/max(total_tested,1):.1f}%)")
    print(f"Both proved:         {total_both} ({100*total_both/max(total_tested,1):.1f}%)")
    print(f"Useful (speedup):    {total_useful} ({100*total_useful/max(total_tested,1):.1f}%)")
    if speedups:
        avg_ratio = sum(speedups) / len(speedups)
        best_ratio = min(speedups)
        print(f"Avg speedup ratio:   {avg_ratio:.3f}")
        print(f"Best speedup ratio:  {best_ratio:.3f}")
    print(f"Time: {elapsed:.0f}s")
    print(f"Results: {output_path}")


if __name__ == '__main__':
    main()
