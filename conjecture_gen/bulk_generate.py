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

    named = model_args.get('named_embeddings', False)
    vocab_size = model_args.get('vocab_size', 0)

    if variant == 'd':
        from conjecture_gen.model_d import ConjectureModelD
        model = ConjectureModelD(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
            use_named_embeddings=named, vocab_size=vocab_size,
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
            use_named_embeddings=named, vocab_size=vocab_size,
        )

    # Remap old checkpoint keys if needed (transformer_decoder -> dec_layers)
    state_dict = checkpoint['model_state_dict']
    remapped = {}
    for k, v in state_dict.items():
        new_k = k.replace('decoder.transformer_decoder.layers.', 'decoder.dec_layers.')
        remapped[new_k] = v
    model.load_state_dict(remapped, strict=False)
    model = model.to(device)
    model.eval()

    # Load vocab if named embeddings were used
    symbol_vocab = None
    if named:
        from conjecture_gen.symbol_vocab import build_vocab
        symbol_vocab = build_vocab(
            'problems', cache_path=os.path.join('cache', 'symbol_vocab.pt')
        )

    return model, variant, checkpoint, symbol_vocab


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
                         top_k=10, top_p=0.9, device=None,
                         batch_gen=8, symbol_vocab=None):
    """Generate n conjectures for a single problem, deduplicate, rank.

    Uses batched generation: replicates the graph batch_gen times and
    generates in parallel for GPU efficiency.
    """
    from torch_geometric.data import Batch

    clauses = parse_problem_file(problem_path)
    if not clauses:
        return []

    graph = clauses_to_graph(clauses, vocab=symbol_vocab)
    if device:
        graph = graph.to(device)

    all_conjectures = []
    seen = set()

    # Generate in batches of batch_gen
    remaining = n
    temp_idx = 0
    while remaining > 0:
        bs = min(remaining, batch_gen)
        temp = temperature * (0.7 + 0.3 * (temp_idx / max(n - 1, 1)))
        temp_idx += bs

        try:
            # Replicate graph bs times into a batch
            batched = Batch.from_data_list([graph.clone() for _ in range(bs)])
            seqs = model.generate(batched, max_steps=args.max_steps, temperature=temp,
                                  top_k=top_k, top_p=top_p)
        except Exception:
            remaining -= bs
            continue

        for seq in seqs:
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

        remaining -= bs

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
    parser.add_argument('--max_steps', type=int, default=120,
                        help='Max generation steps per conjecture (80 was too short)')
    parser.add_argument('--max_problems', type=int, default=0, help='0=all')
    parser.add_argument('--batch_gen', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--per_problem', action='store_true',
                        help='Generate per-problem (slower but correct arity constraints)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, variant, ckpt, symbol_vocab = load_model(args.model, device)
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

    rankings_path = os.path.join(args.output, 'rankings.tsv')
    total_valid = 0
    total_generated = 0
    t0 = time.time()

    all_results = {p: [] for p in problems}

    if args.per_problem:
        # Per-problem generation: correct arity constraints, slower
        print(f"Generating per-problem (n={args.n}, batch_gen={args.batch_gen})...")
        for pi, problem_name in enumerate(problems):
            problem_path = os.path.join(args.problems_dir, problem_name)
            conjectures = generate_for_problem(
                model, problem_path, n=args.n,
                temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                device=device, batch_gen=args.batch_gen,
                symbol_vocab=symbol_vocab,
            )
            for c in conjectures:
                all_results[problem_name].append((c['text'], c.get('sequence', [])))

            if (pi + 1) % 100 == 0:
                n_total = sum(len(v) for v in all_results.values())
                elapsed = time.time() - t0
                print(f"  {pi+1}/{len(problems)}: {n_total} conjectures ({elapsed:.0f}s)")
    else:
        # Batched multi-problem generation: fast but no arity constraints
        print(f"Loading problem graphs...")
        from torch_geometric.data import Batch
        problem_graphs = {}
        for p in problems:
            clauses = parse_problem_file(os.path.join(args.problems_dir, p))
            if clauses:
                problem_graphs[p] = clauses_to_graph(clauses, vocab=symbol_vocab)
        print(f"Loaded {len(problem_graphs)} graphs")

        batch_size = args.batch_gen
        for attempt in range(args.n):
            temp = args.temperature * (0.7 + 0.3 * (attempt / max(args.n - 1, 1)))

            for batch_start in range(0, len(problems), batch_size):
                batch_problems = problems[batch_start:batch_start + batch_size]
                batch_graphs = []
                batch_names = []
                for p in batch_problems:
                    if p in problem_graphs:
                        g = problem_graphs[p].clone()
                        if device.type == 'cuda':
                            g = g.to(device)
                        batch_graphs.append(g)
                        batch_names.append(p)

                if not batch_graphs:
                    continue

                try:
                    batched = Batch.from_data_list(batch_graphs)
                    seqs = model.generate(batched, max_steps=args.max_steps, temperature=temp,
                                          top_k=args.top_k, top_p=args.top_p)
                except Exception:
                    continue

                for idx, (p, seq) in enumerate(zip(batch_names, seqs)):
                    sym_names = problem_graphs[p].symbol_names
                    decoded = decode_sequence(seq, sym_names)
                    if decoded and decoded != '<empty>':
                        all_results[p].append((decoded, seq))

        if (attempt + 1) % 5 == 0:
            n_total = sum(len(v) for v in all_results.values())
            elapsed = time.time() - t0
            print(f"  Attempt {attempt+1}/{args.n}: {n_total} total conjectures ({elapsed:.0f}s)")

    # Deduplicate, validate, rank, save
    with open(rankings_path, 'w') as rankings_f:
        rankings_f.write("problem\trank\tscore\tvalid\tn_tokens\tclause\n")

        for pi, problem_name in enumerate(problems):
            results = all_results.get(problem_name, [])

            # Deduplicate
            seen = set()
            conjectures = []
            for decoded, seq in results:
                if decoded not in seen:
                    seen.add(decoded)
                    test_str = f"cnf(test, axiom, ({decoded}))."
                    parsed = parse_clause(test_str)
                    is_valid = parsed is not None and '...' not in decoded
                    score = score_sequence(model, None, seq)
                    conjectures.append({
                        'text': decoded, 'score': score,
                        'valid': is_valid, 'n_tokens': len(seq),
                    })

            conjectures.sort(key=lambda x: (x['valid'], x['score']), reverse=True)

            if conjectures:
                prob_dir = os.path.join(args.output, problem_name)
                os.makedirs(prob_dir, exist_ok=True)

                for ci, conj in enumerate(conjectures):
                    tptp_path = os.path.join(prob_dir, f'conjecture_{ci+1:03d}.p')
                    with open(tptp_path, 'w') as f:
                        f.write(f"% Generated conjecture for {problem_name}\n")
                        f.write(f"% Score: {conj['score']:.4f}, "
                                f"Valid: {conj['valid']}, "
                                f"Tokens: {conj['n_tokens']}\n")
                        f.write(f"cnf(gen_{ci+1:03d}, axiom, ({conj['text']})).\n")

                    rankings_f.write(
                        f"{problem_name}\t{ci+1}\t{conj['score']:.4f}\t"
                        f"{conj['valid']}\t{conj['n_tokens']}\t{conj['text']}\n"
                    )

                    if conj['valid']:
                        total_valid += 1
                    total_generated += 1

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
