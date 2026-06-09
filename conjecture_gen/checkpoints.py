"""
Shared checkpoint loading for all model variants.

Centralizes the logic for:
  - Reading variant from checkpoint args
  - Building the correct model class (A/B/C/D/E) with correct args
  - Loading state dict with old key remapping
  - Loading symbol_vocab if named_embeddings=True

Usage:
    from conjecture_gen.checkpoints import load_checkpoint
    model, checkpoint, symbol_vocab = load_checkpoint('checkpoints_d/best_model.pt', device)
"""

import os
import torch


def load_checkpoint(path, device, allow_partial=False):
    """Load a model checkpoint, returning (model, checkpoint, symbol_vocab).

    Args:
        path: Path to the checkpoint .pt file.
        device: torch.device to map the checkpoint onto.
        allow_partial: If True, use strict=False when loading state dict.
                       If False (default), raise on any key mismatch.

    Returns:
        model: The model with loaded weights, on the given device, in eval mode.
        checkpoint: The raw checkpoint dict (contains 'args', 'epoch', etc.).
        symbol_vocab: The symbol vocabulary dict if named_embeddings was used,
                      else None.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_args = checkpoint['args']
    variant = checkpoint.get('variant', 'a')

    named = model_args.get('named_embeddings', False)
    vocab_size = model_args.get('vocab_size', 0)

    # Build the correct model class
    if variant == 'a':
        from conjecture_gen.model import ConjectureModel
        model = ConjectureModel(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
            use_named_embeddings=named, vocab_size=vocab_size,
        )
    elif variant == 'b':
        from conjecture_gen.model_b import ConjectureModelB
        model = ConjectureModelB(
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
            use_named_embeddings=named, vocab_size=vocab_size,
        )
    elif variant == 'd':
        from conjecture_gen.model_d import ConjectureModelD
        model = ConjectureModelD(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
            use_named_embeddings=named, vocab_size=vocab_size,
        )
    elif variant == 'e':
        from conjecture_gen.model_e import ConjectureModelE
        model = ConjectureModelE(
            hidden_dim=model_args['hidden_dim'],
            num_gnn_layers=model_args['num_gnn_layers'],
            max_vars=model_args.get('max_vars', 20),
        )
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    # Remap old checkpoint key names (transformer_decoder -> dec_layers)
    state_dict = checkpoint['model_state_dict']
    remapped = {}
    for k, v in state_dict.items():
        new_k = k.replace('decoder.transformer_decoder.layers.', 'decoder.dec_layers.')
        remapped[new_k] = v

    # Load state dict
    strict = not allow_partial
    model.load_state_dict(remapped, strict=strict)
    model = model.to(device)
    model.eval()

    # Load symbol vocab if named embeddings were used
    symbol_vocab = None
    if named:
        from conjecture_gen.symbol_vocab import build_vocab
        # Derive problems_dir and cache_dir from checkpoint args
        problems_dir = model_args.get('problems_dir', 'problems')
        cache_dir = model_args.get('cache_dir', 'cache')
        vocab_cache = os.path.join(cache_dir, 'symbol_vocab.pt')
        symbol_vocab = build_vocab(problems_dir, cache_path=vocab_cache)

    return model, checkpoint, symbol_vocab
