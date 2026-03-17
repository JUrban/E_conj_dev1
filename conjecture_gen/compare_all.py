"""
Train and compare all model variants (A, B, C, D, E) on the same data.

Usage:
    python -m conjecture_gen.compare_all [--epochs 5] [--max_samples 500]

Produces a comparison table at the end.
"""

import argparse
import json
import os
import time
import torch
from conjecture_gen.train_variant import get_model_and_loss, collate_fn
from conjecture_gen.dataset import ConjectureDataset
from conjecture_gen.target_encoder import decode_sequence
from torch.utils.data import DataLoader
import torch.nn as nn


def train_one_variant(variant, train_ds, val_ds, args, device):
    """Train a single variant and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Training variant {variant.upper()}")
    print(f"{'='*60}")

    model, loss_fn = get_model_and_loss(variant, args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    use_cuda = device.type == 'cuda'
    nw = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=nw,
        pin_memory=False, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=nw,
        pin_memory=False, persistent_workers=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    save_dir = f'checkpoints_{variant}'
    os.makedirs(save_dir, exist_ok=True)
    best_val = float('inf')
    train_losses = []

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            losses = loss_fn(output, batch)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses['total'].item()
            n += 1
        scheduler.step()
        train_loss = epoch_loss / max(n, 1)

        # Val
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if n_val >= 20:
                    break
                batch = batch.to(device)
                output = model(batch)
                losses = loss_fn(output, batch)
                val_loss += losses['total'].item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val, 'args': vars(args), 'variant': variant,
            }, os.path.join(save_dir, 'best_model.pt'))

        train_losses.append(train_loss)
        print(f"  Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

    total_time = time.time() - t_start

    # Generate samples
    model.eval()
    samples = []
    for idx in range(min(3, len(val_ds))):
        data = val_ds[idx].to(device)
        seqs = model.generate(data)
        if seqs:
            decoded = decode_sequence(seqs[0], data.symbol_names)
            samples.append(decoded[:150])

    return {
        'variant': variant,
        'params': n_params,
        'best_val_loss': best_val,
        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
        'train_time': total_time,
        'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--max_vars', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_ratio', type=float, default=0.5)
    parser.add_argument('--max_nodes', type=int, default=0)
    parser.add_argument('--variants', default='a,b,c,d,e', help='Comma-separated variant list')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets once
    train_ds = ConjectureDataset(
        problems_dir='problems', lemmas_file='lemmas', statistics_file='statistics',
        max_ratio=args.max_ratio, split='train', max_samples=args.max_samples,
        max_nodes=args.max_nodes,
    )
    val_ds = ConjectureDataset(
        problems_dir='problems', lemmas_file='lemmas', statistics_file='statistics',
        max_ratio=args.max_ratio, split='val',
        max_samples=args.max_samples // 4 if args.max_samples > 0 else 0,
        max_nodes=args.max_nodes,
    )

    # Precompute into RAM on GPU for zero-overhead data loading
    if device.type == 'cuda':
        train_ds.precompute()
        val_ds.precompute()

    variants = [v.strip() for v in args.variants.split(',')]
    results = []

    for variant in variants:
        try:
            r = train_one_variant(variant, train_ds, val_ds, args, device)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in variant {variant}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'variant': variant, 'error': str(e)})

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON ({args.epochs} epochs, {args.max_samples} samples, hidden={args.hidden_dim})")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Params':>10} {'Val Loss':>10} {'Train Loss':>12} {'Time (s)':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for r in results:
        if 'error' in r:
            print(f"{r['variant'].upper():<12} {'ERROR':>10} {r['error'][:30]}")
        else:
            print(f"{r['variant'].upper():<12} {r['params']:>10,} {r['best_val_loss']:>10.4f} "
                  f"{r['final_train_loss']:>12.4f} {r['train_time']:>10.0f}")

    print(f"\nSample outputs:")
    for r in results:
        if 'samples' in r and r['samples']:
            print(f"\n  [{r['variant'].upper()}]")
            for s in r['samples'][:2]:
                print(f"    {s}")

    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to comparison_results.json")


if __name__ == '__main__':
    main()
