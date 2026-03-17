"""
Unified training script that supports all model variants (A/E/D/C/B).

Usage:
    python -m conjecture_gen.train_variant --variant e --epochs 2 --max_samples 200 ...
"""

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from conjecture_gen.dataset import ConjectureDataset
from conjecture_gen.train import collate_fn, compute_loss, generate_samples
from conjecture_gen.target_encoder import decode_sequence, END_CLAUSE


def get_model_and_loss(variant, args):
    """Return (model, loss_fn) for the given variant."""
    if variant == 'a':
        from conjecture_gen.model import ConjectureModel
        model = ConjectureModel(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
        )
        return model, compute_loss

    elif variant == 'e':
        from conjecture_gen.model_e import ConjectureModelE, compute_slot_loss
        model = ConjectureModelE(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
        )
        return model, compute_slot_loss

    elif variant == 'd':
        from conjecture_gen.model_d import ConjectureModelD
        model = ConjectureModelD(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
        )
        return model, compute_loss

    elif variant == 'c':
        from conjecture_gen.model_c import ConjectureModelC, compute_vae_loss
        model = ConjectureModelC(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
        )
        return model, compute_vae_loss

    elif variant == 'b':
        from conjecture_gen.model_b import ConjectureModelB, compute_graph_grow_loss
        model = ConjectureModelB(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
        )
        return model, compute_graph_grow_loss

    else:
        raise ValueError(f"Unknown variant: {variant}")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    # Dataset
    train_ds = ConjectureDataset(
        problems_dir=args.problems_dir, lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file, cache_dir=args.cache_dir,
        max_ratio=args.max_ratio, split='train',
        max_samples=args.max_samples, max_nodes=args.max_nodes,
    )
    val_ds = ConjectureDataset(
        problems_dir=args.problems_dir, lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file, cache_dir=args.cache_dir,
        max_ratio=args.max_ratio, split='val',
        max_samples=args.max_samples // 4 if args.max_samples > 0 else 0,
        max_nodes=args.max_nodes,
    )

    if device.type == 'cuda':
        train_ds.precompute()
        val_ds.precompute()

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

    model, loss_fn = get_model_and_loss(args.variant, args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model variant={args.variant}: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {'total': 0, 'action': 0, 'pointer': 0, 'variable': 0}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            losses = loss_fn(output, batch)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % args.log_every == 0:
                print(f"  [{epoch}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={epoch_losses['total']/n_batches:.4f} ({time.time()-t0:.0f}s)", flush=True)

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation
        model.eval()
        val_losses = {'total': 0, 'action': 0, 'pointer': 0, 'variable': 0}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                if n_val >= 50:
                    break
                batch = batch.to(device)
                output = model(batch)
                losses = loss_fn(output, batch)
                for k in val_losses:
                    val_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
                n_val += 1
        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
              f"train={epoch_losses['total']:.4f} val={val_losses['total']:.4f}")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args), 'variant': args.variant,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> Saved best (val={best_val_loss:.4f})")

        # Sample generation
        if epoch % args.sample_every == 0:
            model.eval()
            for idx in range(min(3, len(val_ds))):
                data = val_ds[idx].to(device)
                seqs = model.generate(data, top_k=10, top_p=0.9)
                if seqs:
                    decoded = decode_sequence(seqs[0], data.symbol_names)
                    print(f"  [{val_ds.samples[idx]['problem']}] {decoded[:150]}")

        history.append({'epoch': epoch, 'train': epoch_losses, 'val': val_losses, 'time': elapsed})
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=['a', 'b', 'c', 'd', 'e'])
    p.add_argument('--problems_dir', default='problems')
    p.add_argument('--lemmas_file', default='lemmas')
    p.add_argument('--statistics_file', default='statistics')
    p.add_argument('--cache_dir', default='cache')
    p.add_argument('--save_dir', default=None)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_gnn_layers', type=int, default=4)
    p.add_argument('--max_vars', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max_ratio', type=float, default=0.5)
    p.add_argument('--max_samples', type=int, default=200)
    p.add_argument('--max_nodes', type=int, default=0)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--sample_every', type=int, default=1)
    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = f'checkpoints_{args.variant}'

    train(args)


if __name__ == '__main__':
    main()
