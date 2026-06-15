#!/usr/bin/env python3
"""
Warm up the PyG JIT compilation by loading the model once before
parallel workers start. Run this before parallel_generate.sh to
avoid race conditions on PyG's propagation code compilation.

Usage: python3 Mizar60_premsel_data/warmup_model.py <checkpoint>
"""

import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python3 warmup_model.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from conjecture_gen.checkpoints import load_checkpoint
model, ckpt, symbol_vocab = load_checkpoint(checkpoint_path, device, allow_partial=False)
print(f"Model loaded: variant={ckpt.get('variant','?')}, "
      f"epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?'):.4f}")
print("PyG JIT compilation complete. Safe to launch parallel workers.")
