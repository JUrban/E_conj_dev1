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

    Args:
        problem_file: path to the CNF problem file
        extra_axioms: string of additional cnf() clauses to append
        eprover: path to E prover binary
        timeout: time limit in seconds

    Returns: dict with 'status', 'processed_clauses'
    """
    # Build input: problem + extra axioms
    try:
        with open(problem_file) as f:
            content = f.read()
    except FileNotFoundError:
        return {'status': 'file_not_found', 'processed_clauses': -1}

    if extra_axioms:
        content = content + '\n' + extra_axioms + '\n'

    try:
        proc = subprocess.run(
            [eprover, '--auto', '--cpu-limit=' + str(timeout),
             '--tstp-format', '-s'],
            input=content, capture_output=True, text=True,
            timeout=timeout + 5,
        )
        return parse_eprover_output(proc.stdout + proc.stderr)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate conjectures with E prover')
    parser.add_argument('--conjectures', required=True,
                        help='Directory with generated conjectures')
    parser.add_argument('--problems', default='problems',
                        help='Directory with original CNF problems')
    parser.add_argument('--statistics', default='statistics',
                        help='Original statistics file for baseline comparison')
    parser.add_argument('--eprover', default='eprover',
                        help='Path to E prover binary')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Per-proof time limit in seconds')
    parser.add_argument('--max_problems', type=int, default=0, help='0=all')
    parser.add_argument('--max_conjectures_per_problem', type=int, default=5,
                        help='Max conjectures to test per problem')
    parser.add_argument('--output', default=None)

    args = parser.parse_args()

    # Load baseline stats
    print("Loading original statistics...")
    orig_stats = load_original_stats(args.statistics)
    print(f"  {len(orig_stats)} problems with baseline stats")

    # Check E prover
    try:
        proc = subprocess.run([args.eprover, '--version'],
                              capture_output=True, text=True, timeout=5)
        print(f"  E prover: {proc.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"  WARNING: E prover not found at '{args.eprover}'")

    # Find problems with generated conjectures
    conj_dir = args.conjectures
    problems = [d for d in sorted(os.listdir(conj_dir))
                if os.path.isdir(os.path.join(conj_dir, d))]
    if args.max_problems > 0:
        problems = problems[:args.max_problems]
    print(f"  {len(problems)} problems with conjectures")

    # Output file
    output_path = args.output or os.path.join(conj_dir, 'eprover_results.tsv')

    total_tested = 0
    total_p1_proved = 0
    total_p2_proved = 0
    total_both = 0
    total_useful = 0  # both proved AND speedup
    speedups = []

    t0 = time.time()

    with open(output_path, 'w') as out_f:
        out_f.write("problem\tconjecture\tp1_status\tp1_clauses\t"
                    "p2_status\tp2_clauses\tL_original\tratio\tspeedup\n")

        for pi, problem_name in enumerate(problems):
            problem_path = os.path.join(args.problems, problem_name)
            if not os.path.exists(problem_path):
                continue

            L_orig = orig_stats.get(problem_name, -1)

            # Get conjecture files (sorted by rank)
            conj_path = os.path.join(conj_dir, problem_name)
            conj_files = sorted([f for f in os.listdir(conj_path) if f.endswith('.p')])
            conj_files = conj_files[:args.max_conjectures_per_problem]

            for conj_file in conj_files:
                # Read conjecture
                with open(os.path.join(conj_path, conj_file)) as f:
                    lines = f.readlines()
                # Find the cnf line
                conj_line = None
                conj_text = None
                for line in lines:
                    line = line.strip()
                    if line.startswith('cnf('):
                        conj_line = line
                        # Extract the clause body
                        m = re.match(r'cnf\([^,]+,\s*[^,]+,\s*\((.+)\)\)\.\s*$', line)
                        if m:
                            conj_text = m.group(1)
                        break

                if not conj_line or not conj_text:
                    continue

                total_tested += 1

                # P1: prove P with C added as axiom
                p1 = run_eprover(problem_path, conj_line,
                                 eprover=args.eprover, timeout=args.timeout)

                # P2: prove C from P (negate C, add to P)
                neg_clauses = negate_clause(conj_text)
                p2 = run_eprover(problem_path, neg_clauses,
                                 eprover=args.eprover, timeout=args.timeout)

                if p1['status'] == 'proved':
                    total_p1_proved += 1
                if p2['status'] == 'proved':
                    total_p2_proved += 1

                both_proved = (p1['status'] == 'proved' and p2['status'] == 'proved')
                if both_proved:
                    total_both += 1

                # Compute speedup ratio
                ratio = -1.0
                speedup = False
                if both_proved and L_orig > 0:
                    L1 = p1['processed_clauses']
                    L2 = p2['processed_clauses']
                    if L1 >= 0 and L2 >= 0:
                        ratio = (L1 + L2) / L_orig
                        if ratio < 1.0:
                            speedup = True
                            total_useful += 1
                            speedups.append(ratio)

                out_f.write(
                    f"{problem_name}\t{conj_file}\t"
                    f"{p1['status']}\t{p1['processed_clauses']}\t"
                    f"{p2['status']}\t{p2['processed_clauses']}\t"
                    f"{L_orig}\t{ratio:.4f}\t{speedup}\n"
                )

            if (pi + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {pi+1}/{len(problems)}: tested={total_tested} "
                      f"p1_proved={total_p1_proved} p2_proved={total_p2_proved} "
                      f"both={total_both} useful={total_useful} "
                      f"({elapsed:.0f}s)")

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
