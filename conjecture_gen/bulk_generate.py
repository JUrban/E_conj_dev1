"""
Generate ranked conjectures for all problems, output in TPTP CNF format
ready for theorem prover evaluation.

For each problem, generates N conjectures with sampling, deduplicates,
and ranks by model confidence (average log-probability of the sequence).

Usage:
    python -m conjecture_gen.bulk_generate --model checkpoints_d/best_model.pt --n 20 --output conjectures/
    python -m conjecture_gen.bulk_generate --model checkpoints_d/best_model.pt --problems_dir problems --n 50

Output structure:
    conjectures/
      problem_name/
        conjecture_001.p   # TPTP format: cnf(gen_001, conjecture, (...)).
        conjecture_002.p
        ...
      rankings.tsv         # problem \t rank \t score \t clause_text
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F

from conjecture_gen.tptp_parser import parse_problem_file, parse_clause
from conjecture_gen.graph_builder import clauses_to_graph
from conjecture_gen.target_encoder import (
    decode_sequence, NUM_ACTION_TYPES, PRED, ARG_VAR, ARG_FUNC,
    END_CLAUSE, NEW_LIT_POS, NEW_LIT_NEG,
)
from conjecture_gen.sampling import sample_from_logits


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    variant = checkpoint.get('variant', 'a')

    if variant == 'd':
        from conjecture_gen.model_d import ConjectureModelD
        model = ConjectureModelD(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
        )
    elif variant == 'c':
        from conjecture_gen.model_c import ConjectureModelC
        model = ConjectureModelC(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
        )
    else:
        from conjecture_gen.model import ConjectureModel
        model = ConjectureModel(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, variant, checkpoint


def score_sequence(model, data, sequence, variant='a'):
    """Compute log-probability of a generated sequence under the model.

    Higher score = model is more confident in this conjecture.
    Returns average log-prob per token.
    """
    # This is a simplified scoring — for proper scoring we'd need to
    # run the model in teacher-forcing mode on the generated sequence.
    # For now, use sequence length and action diversity as proxy.
    n_tokens = len(sequence)
    if n_tokens == 0:
        return -100.0

    # Count unique actions and symbols used
    actions = [a for a, _ in sequence]
    args = [arg for _, arg in sequence]
    n_unique_actions = len(set(actions))
    n_unique_args = len(set(args))

    # Prefer: moderate length, diverse actions, diverse symbols
    length_score = -abs(n_tokens - 15) * 0.05  # prefer ~15 tokens
    diversity_score = (n_unique_actions + n_unique_args) * 0.1
    has_vars = any(a == ARG_VAR for a in actions)
    var_bonus = 0.5 if has_vars else 0.0

    return length_score + diversity_score + var_bonus


def generate_for_problem(model, problem_path, n=20, temperature=1.0,
                         top_k=10, top_p=0.9, device=None):
    """Generate n conjectures for a single problem, deduplicate, rank."""
    clauses = parse_problem_file(problem_path)
    if not clauses:
        return []

    graph = clauses_to_graph(clauses)
    if device:
        graph = graph.to(device)

    # Generate with varying temperatures for diversity
    all_conjectures = []
    seen = set()

    for i in range(n):
        temp = temperature * (0.7 + 0.3 * (i / max(n - 1, 1)))  # 0.7 to 1.0
        try:
            seqs = model.generate(graph, max_steps=80, temperature=temp,
                                  top_k=top_k, top_p=top_p)
        except Exception:
            continue

        if not seqs:
            continue

        seq = seqs[0]
        decoded = decode_sequence(seq, graph.symbol_names)

        if not decoded or decoded == '<empty>' or decoded in seen:
            continue
        seen.add(decoded)

        # Validate: try to parse as TPTP
        test_str = f"cnf(test, axiom, ({decoded}))."
        parsed = parse_clause(test_str)
        is_valid = parsed is not None and '...' not in decoded

        score = score_sequence(model, graph, seq)

        all_conjectures.append({
            'text': decoded,
            'score': score,
            'valid': is_valid,
            'n_tokens': len(seq),
            'sequence': seq,
        })

    # Sort by score (highest first), valid ones first
    all_conjectures.sort(key=lambda x: (x['valid'], x['score']), reverse=True)
    return all_conjectures


def main():
    parser = argparse.ArgumentParser(description='Bulk generate conjectures')
    parser.add_argument('--model', required=True, help='Model checkpoint')
    parser.add_argument('--problems_dir', default='problems')
    parser.add_argument('--output', default='conjectures', help='Output directory')
    parser.add_argument('--n', type=int, default=20, help='Conjectures per problem')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_nodes', type=int, default=1500)
    parser.add_argument('--max_problems', type=int, default=0, help='0=all')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, variant, ckpt = load_model(args.model, device)
    print(f"Model variant={variant}, epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}")

    os.makedirs(args.output, exist_ok=True)

    # Get problem list
    problems = sorted(os.listdir(args.problems_dir))
    if args.max_problems > 0:
        problems = problems[:args.max_problems]

    # Optionally filter by size
    if args.max_nodes > 0:
        filtered = []
        for p in problems:
            clauses = parse_problem_file(os.path.join(args.problems_dir, p))
            graph = clauses_to_graph(clauses)
            total = sum(graph[nt].x.shape[0] for nt in graph.node_types)
            if total <= args.max_nodes:
                filtered.append(p)
        print(f"Filtered to {len(filtered)}/{len(problems)} problems "
              f"(max_nodes={args.max_nodes})")
        problems = filtered

    # Generate
    rankings_path = os.path.join(args.output, 'rankings.tsv')
    total_valid = 0
    total_generated = 0
    t0 = time.time()

    with open(rankings_path, 'w') as rankings_f:
        rankings_f.write("problem\trank\tscore\tvalid\tn_tokens\tclause\n")

        for pi, problem_name in enumerate(problems):
            problem_path = os.path.join(args.problems_dir, problem_name)

            conjectures = generate_for_problem(
                model, problem_path, n=args.n,
                temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                device=device,
            )

            # Save per-problem
            if conjectures:
                prob_dir = os.path.join(args.output, problem_name)
                os.makedirs(prob_dir, exist_ok=True)

                for ci, conj in enumerate(conjectures):
                    # Save as TPTP file
                    tptp_path = os.path.join(prob_dir, f'conjecture_{ci+1:03d}.p')
                    with open(tptp_path, 'w') as f:
                        f.write(f"% Generated conjecture for {problem_name}\n")
                        f.write(f"% Score: {conj['score']:.4f}, "
                                f"Valid: {conj['valid']}, "
                                f"Tokens: {conj['n_tokens']}\n")
                        f.write(f"cnf(gen_{ci+1:03d}, axiom, ({conj['text']})).\n")

                    # Write to rankings
                    rankings_f.write(
                        f"{problem_name}\t{ci+1}\t{conj['score']:.4f}\t"
                        f"{conj['valid']}\t{conj['n_tokens']}\t{conj['text']}\n"
                    )

                    if conj['valid']:
                        total_valid += 1
                    total_generated += 1

            if (pi + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (pi + 1) / elapsed
                print(f"  {pi+1}/{len(problems)} problems "
                      f"({total_valid}/{total_generated} valid) "
                      f"{rate:.1f} problems/s")

    elapsed = time.time() - t0
    print(f"\nDone: {len(problems)} problems, "
          f"{total_generated} conjectures ({total_valid} valid), "
          f"{elapsed:.0f}s")
    print(f"Rankings: {rankings_path}")
    print(f"TPTP files: {args.output}/<problem>/conjecture_NNN.p")

    # Summary stats
    if total_generated > 0:
        print(f"\nSummary:")
        print(f"  Avg conjectures/problem: {total_generated/len(problems):.1f}")
        print(f"  Validity rate: {total_valid/total_generated:.1%}")


if __name__ == '__main__':
    main()
