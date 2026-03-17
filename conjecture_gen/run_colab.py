"""
Colab/AWS training script for conjecture generation.

Usage on Google Colab:
  1. Upload this repo (or clone from git)
  2. Run this cell:
     !pip install torch torch-geometric
  3. Run training:
     !python -m conjecture_gen.run_colab

Or from a terminal:
  python -m conjecture_gen.run_colab [--full] [--hidden_dim 128] ...

Presets:
  --dev    : Small quick run for testing (default)
  --medium : Medium run, good for initial results
  --full   : Full training on all good cuts
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='Conjecture generation training')

    # Presets
    parser.add_argument('--dev', action='store_true', help='Quick dev run')
    parser.add_argument('--medium', action='store_true', help='Medium run')
    parser.add_argument('--full', action='store_true', help='Full training')

    # Override any setting
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_gnn_layers', type=int, default=None)
    parser.add_argument('--max_vars', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_ratio', type=float, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_nodes', type=int, default=None,
                        help='Max graph nodes per problem (1500 for T4, 3000 for A100)')
    parser.add_argument('--problems_dir', default='problems')
    parser.add_argument('--lemmas_file', default='lemmas')
    parser.add_argument('--statistics_file', default='statistics')

    args = parser.parse_args()

    # Determine preset
    if args.full:
        preset = 'full'
    elif args.medium:
        preset = 'medium'
    else:
        preset = 'dev'

    configs = {
        'dev': {
            'hidden_dim': 64,
            'num_gnn_layers': 4,
            'max_vars': 20,
            'batch_size': 32,
            'lr': 3e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'epochs': 5,
            'max_ratio': 0.5,
            'max_samples': 500,
            'max_nodes': 0,
            'log_every': 10,
            'sample_every': 1,
            'num_workers': 0,
            'save_dir': 'checkpoints_dev',
        },
        'medium': {
            'hidden_dim': 128,
            'num_gnn_layers': 6,
            'max_vars': 20,
            'batch_size': 32,
            'lr': 3e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'epochs': 30,
            'max_ratio': 1.0,
            'max_samples': 5000,
            'max_nodes': 1500,  # safe for T4 16GB
            'log_every': 50,
            'sample_every': 5,
            'num_workers': 4,
            'save_dir': 'checkpoints_medium',
        },
        'full': {
            'hidden_dim': 128,
            'num_gnn_layers': 6,
            'max_vars': 20,
            'batch_size': 64,
            'lr': 3e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'epochs': 50,
            'max_ratio': 1.0,
            'max_samples': 0,  # all data
            'max_nodes': 1500,  # safe for T4 16GB
            'log_every': 100,
            'sample_every': 5,
            'num_workers': 4,
            'save_dir': 'checkpoints_full',
        },
    }

    config = configs[preset]

    # Apply overrides
    for key in ['hidden_dim', 'num_gnn_layers', 'max_vars', 'batch_size',
                'lr', 'epochs', 'max_ratio', 'max_samples', 'max_nodes']:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    config['problems_dir'] = args.problems_dir
    config['lemmas_file'] = args.lemmas_file
    config['statistics_file'] = args.statistics_file
    config['cache_dir'] = 'cache'

    print(f"=== Conjecture Generation Training ({preset} preset) ===")
    print(f"Config: {config}")
    print()

    # Build args namespace
    from argparse import Namespace
    train_args = Namespace(**config)

    from conjecture_gen.train import train
    train(train_args)

    # Run evaluation on test set
    print("\n=== Evaluation on test set ===")
    import torch
    from conjecture_gen.evaluate import load_model, evaluate_loss, evaluate_generation
    from conjecture_gen.dataset import ConjectureDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(config['save_dir'], 'best_model.pt')
    model, model_args, checkpoint = load_model(model_path, device)

    test_ds = ConjectureDataset(
        problems_dir=config['problems_dir'],
        lemmas_file=config['lemmas_file'],
        statistics_file=config['statistics_file'],
        cache_dir=config['cache_dir'],
        max_ratio=config['max_ratio'],
        max_nodes=config.get('max_nodes', 0),
        split='test',
    )

    loss_metrics = evaluate_loss(model, test_ds, device)
    print("Loss metrics:")
    for k, v in loss_metrics.items():
        print(f"  {k}: {v:.4f}")

    gen_results = evaluate_generation(model, test_ds, device, n_per_problem=5, max_problems=100)
    print("Generation metrics:")
    for k, v in gen_results['metrics'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nSample conjectures:")
    shown = 0
    for pr in gen_results['per_problem']:
        if pr['conjectures'] and shown < 5:
            print(f"  {pr['problem']} ({pr['n_unique']} unique):")
            for c in pr['conjectures'][:2]:
                tag = "OK" if c['valid'] else "BAD"
                print(f"    [{tag}] {c['text']}")
            shown += 1

    # Save
    import json
    eval_path = os.path.join(config['save_dir'], f'eval_test.json')
    with open(eval_path, 'w') as f:
        json.dump({
            'loss_metrics': loss_metrics,
            'generation_metrics': gen_results['metrics'],
            'samples': gen_results['per_problem'][:50],
        }, f, indent=2)
    print(f"\nEvaluation saved to {eval_path}")


if __name__ == '__main__':
    main()
