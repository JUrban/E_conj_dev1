"""
Training loop for the conjecture generation model.

Usage:
    python -m conjecture_gen.train [--hidden_dim 128] [--epochs 50] [--batch_size 32] ...
"""

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from conjecture_gen.dataset import ConjectureDataset
from conjecture_gen.model import ConjectureModel
from conjecture_gen.target_encoder import (
    NUM_ACTION_TYPES, PRED, ARG_VAR, ARG_FUNC, END_CLAUSE,
    decode_sequence,
)


def compute_loss(model_output, batch, pad_value=-1):
    """Compute the combined loss for action prediction, pointer, and variable selection.

    Uses quality_weight to weight each sample.
    """
    action_logits = model_output['action_logits']   # (B, T, 7)
    pointer_logits = model_output['pointer_logits']  # (B, T, max_sym)
    var_logits = model_output['var_logits']           # (B, T, max_vars)

    target_actions = batch.target_actions    # (B, T)
    target_args = batch.target_arguments     # (B, T)
    target_lengths = batch.target_length     # (B,)
    weights = batch.quality_weight           # (B,)

    B, T = target_actions.shape
    device = action_logits.device

    # Create mask for valid timesteps
    mask = torch.arange(T, device=device).unsqueeze(0) < target_lengths.unsqueeze(1)
    mask = mask.float()  # (B, T)

    # --- Action type loss ---
    action_loss = F.cross_entropy(
        action_logits.reshape(-1, NUM_ACTION_TYPES),
        target_actions.reshape(-1),
        reduction='none',
    ).reshape(B, T)
    action_loss = (action_loss * mask * weights.unsqueeze(1)).sum() / mask.sum().clamp(min=1)

    # --- Pointer loss (for PRED and ARG_FUNC actions) ---
    pointer_mask = ((target_actions == PRED) | (target_actions == ARG_FUNC)) & mask.bool()
    if pointer_mask.any():
        # Clamp target args to valid range
        max_sym = pointer_logits.shape[2]
        ptr_targets = target_args.clamp(0, max_sym - 1)
        ptr_loss = F.cross_entropy(
            pointer_logits.reshape(-1, max_sym),
            ptr_targets.reshape(-1),
            reduction='none',
        ).reshape(B, T)
        ptr_loss = (ptr_loss * pointer_mask.float() * weights.unsqueeze(1)).sum() / pointer_mask.sum().clamp(min=1)
    else:
        ptr_loss = torch.tensor(0.0, device=device)

    # --- Variable slot loss (for ARG_VAR actions) ---
    var_mask = (target_actions == ARG_VAR) & mask.bool()
    if var_mask.any():
        max_vars = var_logits.shape[2]
        var_targets = target_args.clamp(0, max_vars - 1)
        var_loss = F.cross_entropy(
            var_logits.reshape(-1, max_vars),
            var_targets.reshape(-1),
            reduction='none',
        ).reshape(B, T)
        var_loss = (var_loss * var_mask.float() * weights.unsqueeze(1)).sum() / var_mask.sum().clamp(min=1)
    else:
        var_loss = torch.tensor(0.0, device=device)

    total_loss = action_loss + ptr_loss + var_loss
    return {
        'total': total_loss,
        'action': action_loss.item(),
        'pointer': ptr_loss.item(),
        'variable': var_loss.item(),
    }


def collate_fn(batch_list):
    """Custom collate: pad target sequences and batch graphs.

    We handle target tensors manually (stack into 2D) because PyG's
    Batch.from_data_list concatenates 1D tensors instead of stacking.
    Clones items to avoid mutating the in-memory dataset cache.
    """
    from torch_geometric.data import Batch

    # Find max target length in this batch
    max_len = max(item.target_length.item() for item in batch_list)
    batch_size = len(batch_list)

    # Extract and pad targets before batching
    all_actions = []
    all_arguments = []
    all_lengths = []
    all_weights = []
    all_ratios = []
    all_num_symbols = []

    # Clone items so we don't mutate the dataset's in-memory cache
    cloned = [item.clone() for item in batch_list]

    for item in cloned:
        cur_len = item.target_actions.shape[0]
        pad_len = max_len - cur_len
        all_actions.append(F.pad(item.target_actions, (0, pad_len), value=END_CLAUSE))
        all_arguments.append(F.pad(item.target_arguments, (0, pad_len), value=0))
        all_lengths.append(item.target_length)
        all_weights.append(item.quality_weight)
        all_ratios.append(item.ratio)
        all_num_symbols.append(item.num_symbols)

        # Remove these from the clone so PyG doesn't try to batch them
        del item.target_actions
        del item.target_arguments
        del item.target_length
        del item.quality_weight
        del item.ratio
        del item.num_symbols

    # Batch the graphs (using clones, not originals)
    batch = Batch.from_data_list(cloned)

    # Re-attach properly stacked targets
    batch.target_actions = torch.stack(all_actions)         # (B, T)
    batch.target_arguments = torch.stack(all_arguments)     # (B, T)
    batch.target_length = torch.stack(all_lengths)          # (B,)
    batch.quality_weight = torch.stack(all_weights)         # (B,)
    batch.ratio = torch.stack(all_ratios)                   # (B,)
    batch.num_symbols = torch.stack(all_num_symbols)        # (B,)

    return batch


def evaluate(model, dataloader, device, max_batches=50):
    """Evaluate on validation set."""
    model.eval()
    total_losses = {'total': 0, 'action': 0, 'pointer': 0, 'variable': 0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if n_batches >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            losses = compute_loss(output, batch)
            for k in total_losses:
                total_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n_batches += 1

    for k in total_losses:
        total_losses[k] /= max(n_batches, 1)
    return total_losses


def generate_samples(model, dataset, device, n_samples=5):
    """Generate a few sample conjectures for inspection."""
    model.eval()
    results = []

    indices = list(range(min(n_samples, len(dataset))))
    for idx in indices:
        data = dataset[idx].to(device)
        seqs = model.generate(data, max_steps=80)
        if seqs:
            decoded = decode_sequence(seqs[0], data.symbol_names)
            results.append({
                'problem': dataset.samples[idx]['problem'],
                'ratio': dataset.samples[idx]['ratio'],
                'generated': decoded,
            })
    return results


def train(args):
    device = torch.device('cpu')  # CPU training; GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    # Dataset
    print("\nLoading dataset...")
    train_ds = ConjectureDataset(
        problems_dir=args.problems_dir,
        lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file,
        cache_dir=args.cache_dir,
        max_ratio=args.max_ratio,
        split='train',
        max_samples=args.max_samples,
        max_nodes=args.max_nodes,
    )
    val_ds = ConjectureDataset(
        problems_dir=args.problems_dir,
        lemmas_file=args.lemmas_file,
        statistics_file=args.statistics_file,
        cache_dir=args.cache_dir,
        max_ratio=args.max_ratio,
        split='val',
        max_samples=args.max_samples // 4 if args.max_samples > 0 else 0,
    )

    # Precompute samples for faster data loading on GPU
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

    # Model
    model = ConjectureModel(
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        max_vars=args.max_vars,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
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
            losses = compute_loss(output, batch)
            losses['total'].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % args.log_every == 0:
                avg_loss = epoch_losses['total'] / n_batches
                elapsed = time.time() - t0
                print(f"  [{epoch}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg_loss:.4f} "
                      f"({elapsed:.1f}s)", flush=True)

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validation
        val_losses = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        record = {
            'epoch': epoch,
            'train': epoch_losses,
            'val': val_losses,
            'lr': scheduler.get_last_lr()[0],
            'time': elapsed,
        }
        history.append(record)

        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
              f"train_loss={epoch_losses['total']:.4f} "
              f"val_loss={val_losses['total']:.4f} "
              f"(act={val_losses['action']:.3f} "
              f"ptr={val_losses['pointer']:.3f} "
              f"var={val_losses['variable']:.3f})")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'args': vars(args),
            }, save_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Generate samples every few epochs
        if epoch % args.sample_every == 0 or epoch == 1:
            samples = generate_samples(model, val_ds, device, n_samples=3)
            print("  Sample conjectures:")
            for s in samples:
                print(f"    [{s['problem']} r={s['ratio']:.3f}] {s['generated']}")

        # Save history
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining done. Best val_loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train conjecture generator')
    parser.add_argument('--problems_dir', default='problems')
    parser.add_argument('--lemmas_file', default='lemmas')
    parser.add_argument('--statistics_file', default='statistics')
    parser.add_argument('--cache_dir', default='cache')
    parser.add_argument('--save_dir', default='checkpoints')

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_gnn_layers', type=int, default=6)
    parser.add_argument('--max_vars', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_ratio', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--max_nodes', type=int, default=0,
                        help='Max total graph nodes per problem (0=no limit, 1500 recommended for T4)')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--sample_every', type=int, default=5)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
