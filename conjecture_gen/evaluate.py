"""
Evaluate a trained conjecture generation model on the test set.

Metrics:
  - Syntactic validity: % of generated clauses that parse back to valid CNF
  - Diversity: avg unique conjectures per problem (out of N attempts)
  - Symbol precision: % of symbols in conjecture that exist in the problem
  - Variable usage: % of conjectures that use at least one variable
  - Length stats: avg number of literals per generated conjecture
  - Action accuracy: how well the model predicts action types on held-out data
  - Pointer accuracy: how well the model predicts symbol pointers

Usage:
    python -m conjecture_gen.evaluate --model checkpoints/best_model.pt
    python -m conjecture_gen.evaluate --model checkpoints/best_model.pt --n 10 --split test
"""

import argparse
import json
import os
import time
import torch
from collections import Counter

from conjecture_gen.dataset import ConjectureDataset
from conjecture_gen.model import ConjectureModel
from conjecture_gen.tptp_parser import parse_clause
from conjecture_gen.target_encoder import (
    decode_sequence, NUM_ACTION_TYPES, PRED, ARG_VAR, ARG_FUNC,
    END_CLAUSE, NEW_LIT_POS, NEW_LIT_NEG, END_ARGS,
)
from conjecture_gen.train import collate_fn, compute_loss

import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    model = ConjectureModel(
        hidden_dim=args['hidden_dim'],
        num_gnn_layers=args['num_gnn_layers'],
        max_vars=args.get('max_vars', 20),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args, checkpoint


def check_syntactic_validity(clause_str: str, symbol_names: list[str]) -> dict:
    """Check if a generated clause string is syntactically valid."""
    if not clause_str or clause_str == '<empty>':
        return {'valid': False, 'reason': 'empty'}
    if '...' in clause_str:
        return {'valid': False, 'reason': 'truncated'}

    # Try to parse it as a TPTP clause
    tptp_str = f"cnf(gen, axiom, ({clause_str}))."
    parsed = parse_clause(tptp_str)
    if parsed is None:
        return {'valid': False, 'reason': 'parse_error'}

    # Check symbol coverage: all symbols in conjecture should be from the problem
    used_symbols = set()
    for lit in parsed.literals:
        used_symbols.add(lit.predicate)
        for arg in lit.args:
            _collect_term_symbols(arg, used_symbols)

    known = set(symbol_names)
    unknown = used_symbols - known - {'$eq'}
    precision = 1.0 - len(unknown) / max(len(used_symbols), 1)

    # Check variable usage
    has_vars = any(
        _term_has_vars(arg)
        for lit in parsed.literals
        for arg in lit.args
    )

    return {
        'valid': True,
        'n_literals': len(parsed.literals),
        'n_symbols': len(used_symbols),
        'symbol_precision': precision,
        'has_variables': has_vars,
        'unknown_symbols': list(unknown),
    }


def _collect_term_symbols(term, symbols):
    if not term.is_variable:
        symbols.add(term.name)
        for arg in term.args:
            _collect_term_symbols(arg, symbols)


def _term_has_vars(term):
    if term.is_variable:
        return True
    return any(_term_has_vars(a) for a in term.args)


def evaluate_generation(model, dataset, device, n_per_problem=5, max_problems=200,
                        top_k=10, top_p=0.9):
    """Generate conjectures and compute quality metrics."""
    model.eval()

    # Group samples by problem
    problem_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        p = sample['problem']
        if p not in problem_to_indices:
            problem_to_indices[p] = []
        problem_to_indices[p].append(idx)

    problems = list(problem_to_indices.keys())[:max_problems]

    results = {
        'n_problems': len(problems),
        'n_per_problem': n_per_problem,
        'per_problem': [],
    }

    all_valid = 0
    all_total = 0
    all_unique_counts = []
    all_n_literals = []
    all_sym_precision = []
    all_has_vars = 0
    gen_times = []

    for pi, problem in enumerate(problems):
        idx = problem_to_indices[problem][0]
        data = dataset[idx].to(device)

        # Generate n conjectures with varying temperature
        conjectures = []
        t0 = time.time()
        for i in range(n_per_problem):
            temp = 0.8 + 0.1 * i  # 0.8, 0.9, 1.0, 1.1, ...
            seqs = model.generate(data, max_steps=80, temperature=temp,
                                  top_k=top_k, top_p=top_p)
            if seqs:
                decoded = decode_sequence(seqs[0], data.symbol_names)
                conjectures.append(decoded)
        gen_time = time.time() - t0
        gen_times.append(gen_time)

        # Analyze
        unique = list(set(c for c in conjectures if c != '<empty>'))
        all_unique_counts.append(len(unique))

        problem_results = {
            'problem': problem,
            'n_generated': len(conjectures),
            'n_unique': len(unique),
            'conjectures': [],
        }

        for conj in unique:
            check = check_syntactic_validity(conj, data.symbol_names)
            problem_results['conjectures'].append({
                'text': conj[:200],
                **check,
            })
            all_total += 1
            if check['valid']:
                all_valid += 1
                all_n_literals.append(check['n_literals'])
                all_sym_precision.append(check['symbol_precision'])
                if check['has_variables']:
                    all_has_vars += 1

        results['per_problem'].append(problem_results)

        if (pi + 1) % 50 == 0:
            print(f"  Evaluated {pi+1}/{len(problems)} problems...")

    # Aggregate metrics
    results['metrics'] = {
        'syntactic_validity': all_valid / max(all_total, 1),
        'avg_unique_per_problem': sum(all_unique_counts) / max(len(all_unique_counts), 1),
        'avg_literals': sum(all_n_literals) / max(len(all_n_literals), 1),
        'avg_symbol_precision': sum(all_sym_precision) / max(len(all_sym_precision), 1),
        'variable_usage_rate': all_has_vars / max(all_valid, 1),
        'avg_gen_time_per_problem': sum(gen_times) / max(len(gen_times), 1),
        'total_generated': all_total,
        'total_valid': all_valid,
    }

    return results


def evaluate_loss(model, dataset, device, max_batches=100):
    """Compute loss metrics on a dataset split."""
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    model.eval()
    total = {'total': 0, 'action': 0, 'pointer': 0, 'variable': 0}
    n = 0
    correct_actions = 0
    total_actions = 0
    correct_ptrs = 0
    total_ptrs = 0

    with torch.no_grad():
        for batch in loader:
            if n >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            losses = compute_loss(output, batch)
            for k in total:
                total[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()

            # Action accuracy
            pred_actions = output['action_logits'].argmax(dim=-1)
            mask = torch.arange(batch.target_actions.shape[1], device=device).unsqueeze(0) < batch.target_length.unsqueeze(1)
            correct_actions += ((pred_actions == batch.target_actions) & mask).sum().item()
            total_actions += mask.sum().item()

            # Pointer accuracy (on PRED/ARG_FUNC positions)
            ptr_positions = ((batch.target_actions == PRED) | (batch.target_actions == ARG_FUNC)) & mask
            if ptr_positions.any():
                pred_ptrs = output['pointer_logits'].argmax(dim=-1)
                max_sym = output['pointer_logits'].shape[-1]
                target_ptrs = batch.target_arguments.clamp(0, max_sym - 1)
                correct_ptrs += ((pred_ptrs == target_ptrs) & ptr_positions).sum().item()
                total_ptrs += ptr_positions.sum().item()

            n += 1

    for k in total:
        total[k] /= max(n, 1)

    total['action_accuracy'] = correct_actions / max(total_actions, 1)
    total['pointer_accuracy'] = correct_ptrs / max(total_ptrs, 1)
    return total


def main():
    parser = argparse.ArgumentParser(description='Evaluate conjecture generator')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--problems_dir', default='problems')
    parser.add_argument('--lemmas_file', default='lemmas')
    parser.add_argument('--statistics_file', default='statistics')
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    parser.add_argument('--max_ratio', type=float, default=1.0)
    parser.add_argument('--max_nodes', type=int, default=1500)
    parser.add_argument('--n', type=int, default=5, help='Conjectures per problem')
    parser.add_argument('--max_problems', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=10, help='Top-k sampling (0=greedy)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling (0=disabled)')
    parser.add_argument('--output', default=None, help='Save results to JSON file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, model_args, checkpoint = load_model(args.model, device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    # Load dataset
    dataset = ConjectureDataset(
        problems_dir=args.problems_dir,
        lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file,
        cache_dir=args.cache_dir,
        max_ratio=args.max_ratio,
        max_nodes=args.max_nodes,
        split=args.split,
    )

    # 1. Loss-based metrics
    print(f"\n--- Loss metrics on {args.split} set ---")
    loss_metrics = evaluate_loss(model, dataset, device)
    for k, v in loss_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 2. Generation metrics
    print(f"\n--- Generation metrics ({args.n} per problem, up to {args.max_problems} problems) ---")
    gen_results = evaluate_generation(
        model, dataset, device,
        n_per_problem=args.n,
        max_problems=args.max_problems,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    metrics = gen_results['metrics']
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 3. Show some examples
    print(f"\n--- Sample conjectures ---")
    shown = 0
    for pr in gen_results['per_problem']:
        if pr['conjectures'] and shown < 10:
            print(f"  Problem: {pr['problem']} ({pr['n_unique']} unique / {pr['n_generated']} generated)")
            for c in pr['conjectures'][:2]:
                valid_str = "VALID" if c['valid'] else f"INVALID({c.get('reason', '?')})"
                print(f"    [{valid_str}] {c['text']}")
            shown += 1

    # Save results
    all_results = {
        'model_path': args.model,
        'model_args': model_args,
        'epoch': checkpoint['epoch'],
        'split': args.split,
        'loss_metrics': loss_metrics,
        'generation_metrics': metrics,
        'samples': gen_results['per_problem'][:50],  # save first 50
    }

    output_path = args.output
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        output_path = f"eval_{model_name}_{args.split}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
