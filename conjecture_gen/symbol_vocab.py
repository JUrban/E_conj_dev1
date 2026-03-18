"""
Symbol vocabulary for named embeddings.

Builds a mapping from Mizar symbol names to vocabulary indices.
Skolem symbols (esk*_N) are mapped to a shared UNK index.
"""

import os
import re
import torch


SKOLEM_PATTERN = re.compile(r'^esk\d+_\d+$')


def is_skolem(name: str) -> bool:
    return bool(SKOLEM_PATTERN.match(name))


def build_vocab(problems_dir: str, cache_path: str = None,
                min_count: int = 2) -> dict[str, int]:
    """Build vocabulary from all problem files.

    Returns dict mapping symbol_name -> vocab_index.
    Index 0 is reserved for UNK (Skolem symbols and rare Mizar symbols).
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    from conjecture_gen.tptp_parser import parse_problem_file
    from conjecture_gen.graph_builder import clauses_to_graph

    counts = {}
    for fname in sorted(os.listdir(problems_dir)):
        g = clauses_to_graph(parse_problem_file(os.path.join(problems_dir, fname)))
        for name in g.symbol_names:
            if not is_skolem(name):
                counts[name] = counts.get(name, 0) + 1

    # Build vocab: index 0 = UNK, then sorted by frequency
    vocab = {'<UNK>': 0}
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= min_count:
            vocab[name] = len(vocab)

    print(f"Symbol vocab: {len(vocab)} entries "
          f"({len(vocab)-1} Mizar + UNK, min_count={min_count})")

    if cache_path:
        torch.save(vocab, cache_path)

    return vocab


def names_to_indices(symbol_names: list[str], vocab: dict[str, int]) -> list[int]:
    """Convert a list of symbol names to vocab indices."""
    unk = vocab.get('<UNK>', 0)
    return [vocab.get(name, unk) if not is_skolem(name) else unk
            for name in symbol_names]


if __name__ == '__main__':
    vocab = build_vocab('problems')
    print(f"Vocab size: {len(vocab)}")
    print(f"Sample: {list(vocab.items())[:10]}")
