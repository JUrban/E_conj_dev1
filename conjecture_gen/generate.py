"""
Generate conjectures for a given CNF problem using a trained model.

Usage:
    python -m conjecture_gen.generate --model checkpoints/best_model.pt --problem problems/t8_prob_1
    python -m conjecture_gen.generate --model checkpoints/best_model.pt --problem problems/t8_prob_1 --n 10
"""

import argparse
import torch

from conjecture_gen.tptp_parser import parse_problem_file
from conjecture_gen.graph_builder import clauses_to_graph
from conjecture_gen.model import ConjectureModel
from conjecture_gen.target_encoder import decode_sequence


def load_model(checkpoint_path: str, device: torch.device = None):
    """Load a trained model from a checkpoint."""
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']

    model = ConjectureModel(
        hidden_dim=model_args['hidden_dim'],
        num_gnn_layers=model_args['num_gnn_layers'],
        max_vars=model_args.get('max_vars', 20),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")
    return model


def generate_conjectures(model, problem_path: str, n: int = 5,
                         temperature: float = 1.0,
                         max_steps: int = 80) -> list[str]:
    """Generate n conjectures for a problem."""
    clauses = parse_problem_file(problem_path)
    graph = clauses_to_graph(clauses)

    results = []
    for i in range(n):
        # Use increasing temperature for diversity
        temp = temperature * (1.0 + 0.1 * i)
        seqs = model.generate(graph, max_steps=max_steps, temperature=temp)
        if seqs:
            decoded = decode_sequence(seqs[0], graph.symbol_names)
            results.append(decoded)

    # Deduplicate
    seen = set()
    unique = []
    for r in results:
        if r not in seen and r != '<empty>':
            seen.add(r)
            unique.append(r)

    return unique


def main():
    parser = argparse.ArgumentParser(description='Generate conjectures')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--problem', required=True, help='Path to CNF problem file')
    parser.add_argument('--n', type=int, default=5, help='Number of conjectures to generate')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=80)

    args = parser.parse_args()

    model = load_model(args.model)
    conjectures = generate_conjectures(
        model, args.problem,
        n=args.n, temperature=args.temperature,
        max_steps=args.max_steps,
    )

    print(f"\nGenerated {len(conjectures)} unique conjectures for {args.problem}:")
    for i, c in enumerate(conjectures, 1):
        print(f"  {i}. {c}")


if __name__ == '__main__':
    main()
