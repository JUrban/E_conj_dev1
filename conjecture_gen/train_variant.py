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
    named = getattr(args, 'named_embeddings', False)
    vocab_size = getattr(args, 'vocab_size', 0)

    if variant == 'a':
        from conjecture_gen.model import ConjectureModel
        model = ConjectureModel(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
            use_named_embeddings=named, vocab_size=vocab_size,
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
            use_named_embeddings=named, vocab_size=vocab_size,
        )
        return model, compute_loss

    elif variant == 'c':
        from conjecture_gen.model_c import ConjectureModelC, compute_vae_loss
        model = ConjectureModelC(
            hidden_dim=args.hidden_dim, num_gnn_layers=args.num_gnn_layers,
            max_vars=args.max_vars,
            use_named_embeddings=named, vocab_size=vocab_size,
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

    # Seed for reproducibility
    seed = getattr(args, 'seed', 42)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build symbol vocab if using named embeddings
    os.makedirs(args.cache_dir, exist_ok=True)
    symbol_vocab = None
    if getattr(args, 'named_embeddings', False):
        from conjecture_gen.symbol_vocab import build_vocab
        vocab_cache = os.path.join(args.cache_dir, 'symbol_vocab.pt')
        # Only scan problems that have training data (much faster than all files)
        prob_list = None
        if args.train_split:
            from conjecture_gen.tptp_parser import parse_statistics_line
            # Get problems that appear in the statistics file
            stats_probs = set()
            with open(args.statistics_file) as f:
                for line in f:
                    s = parse_statistics_line(line)
                    if s:
                        stats_probs.add(s['problem'])
            prob_list = [p for p in sorted(stats_probs)
                         if os.path.exists(os.path.join(args.problems_dir, p))]
            print(f"Scanning {len(prob_list)} problems for symbol vocab "
                  f"(not all {len(os.listdir(args.problems_dir))})")
        symbol_vocab = build_vocab(args.problems_dir, cache_path=vocab_cache,
                                   problem_list=prob_list)
        args.vocab_size = len(symbol_vocab)
        print(f"Named embeddings: vocab_size={args.vocab_size}")

    # Dataset
    train_ds = ConjectureDataset(
        problems_dir=args.problems_dir, lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file, cache_dir=args.cache_dir,
        max_ratio=args.max_ratio,
        split='train' if not args.train_split else 'all',
        split_file=args.train_split,
        max_samples=args.max_samples, max_nodes=args.max_nodes,
        symbol_vocab=symbol_vocab,
    )
    val_ds = ConjectureDataset(
        problems_dir=args.problems_dir, lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file, cache_dir=args.cache_dir,
        max_ratio=args.max_ratio,
        split='val' if not args.val_split else 'all',
        split_file=args.val_split,
        max_samples=args.max_samples // 4 if args.max_samples > 0 else 0,
        max_nodes=args.max_nodes,
        symbol_vocab=symbol_vocab,
    )

    if device.type == 'cuda':
        if args.no_precompute:
            # Warm graph cache + pre-encode targets for fast __getitem__
            print("Warming graph cache...")
            for pi, pname in enumerate(sorted(set(s['problem'] for s in train_ds.samples))):
                train_ds._get_problem_graph(pname)
                if (pi + 1) % 500 == 0:
                    print(f"  {pi+1} graphs loaded...")
            for pname in sorted(set(s['problem'] for s in val_ds.samples)):
                val_ds._get_problem_graph(pname)
            print(f"  {len(train_ds._graph_cache)} graphs in RAM")
            # Pre-encode targets (~70s, ~14MB)
            train_ds.precompute_targets()
            val_ds.precompute_targets()
        else:
            train_ds.precompute(load_into_ram=True)
            val_ds.precompute(load_into_ram=True)

    use_cuda = device.type == 'cuda'
    nw = args.num_workers
    if args.no_precompute and hasattr(train_ds, '_problem_sizes'):
        # Size-aware batching: cap total nodes per batch to avoid VRAM spikes
        from conjecture_gen.batch_by_size import SizeAwareBatchSampler
        max_total = args.max_batch_nodes
        print(f"Size-aware batching: max_total_nodes={max_total}")
        train_sampler = SizeAwareBatchSampler(train_ds, max_total_nodes=max_total,
                                              shuffle=True)
        val_sampler = SizeAwareBatchSampler(val_ds, max_total_nodes=max_total,
                                            shuffle=False)
        # Use num_workers for parallel graph.clone() — fork-based workers
        # share the parent's _graph_cache and _targets via copy-on-write.
        loader_nw = min(nw, 2) if nw > 0 else 2
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            collate_fn=collate_fn, num_workers=loader_nw,
            pin_memory=False, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler,
            collate_fn=collate_fn, num_workers=loader_nw,
            pin_memory=False, persistent_workers=True,
        )
    else:
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

    if args.variant in ('b', 'e'):
        import warnings
        warnings.warn(
            f"Variant '{args.variant}' is EXPERIMENTAL / UNMAINTAINED and showed "
            f"poor results in comparison experiments. "
            f"Consider using variants a, c, or d instead.",
            UserWarning,
            stacklevel=1,
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
    start_epoch = 1

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_path = os.path.join(args.resume, 'best_model.pt')
        if os.path.exists(ckpt_path):
            from conjecture_gen.checkpoints import load_checkpoint
            _resumed_model, ckpt, _sv = load_checkpoint(
                ckpt_path, device, allow_partial=True,
            )
            # Copy the loaded weights into our model (which may have
            # been constructed with different args for fine-tuning)
            sd = ckpt['model_state_dict']
            sd = {k.replace('decoder.transformer_decoder.layers.', 'decoder.dec_layers.'): v
                  for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            best_val_loss = ckpt.get('val_loss', float('inf'))
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resumed from {ckpt_path}: epoch {start_epoch-1}, "
                  f"val_loss={best_val_loss:.4f}")
            # Load history if available
            hist_path = os.path.join(args.resume, 'history.json')
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
        else:
            print(f"WARNING: --resume {args.resume} but no best_model.pt found, starting fresh")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = {'total': 0, 'action': 0, 'pointer': 0, 'variable': 0}
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            try:
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
            except torch.cuda.OutOfMemoryError:
                # Skip oversized batches, free memory, continue training
                torch.cuda.empty_cache()
                print(f"  [{epoch}] batch {batch_idx+1}: CUDA OOM, skipped", flush=True)
                continue

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
    p.add_argument('--train_split', default=None,
                   help='File listing train problem names (one per line)')
    p.add_argument('--val_split', default=None,
                   help='File listing val problem names (one per line)')
    p.add_argument('--save_dir', default=None)
    p.add_argument('--resume', default=None,
                   help='Resume from checkpoint dir (e.g., checkpoints_d)')
    p.add_argument('--named_embeddings', action='store_true',
                   help='Use learnable name embeddings for Mizar symbols')
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
    p.add_argument('--no_precompute', action='store_true',
                   help='Disable precomputing all samples into RAM (needed for large datasets)')
    p.add_argument('--max_batch_nodes', type=int, default=50000,
                   help='Max total graph nodes per batch (size-aware batching)')
    p.add_argument('--max_samples', type=int, default=200)
    p.add_argument('--max_nodes', type=int, default=0)
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--sample_every', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = f'checkpoints_{args.variant}'

    train(args)


if __name__ == '__main__':
    main()
