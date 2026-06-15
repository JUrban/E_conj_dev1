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
                min_count: int = 2,
                problem_list: list[str] = None) -> dict[str, int]:
    """Build vocabulary from problem files.

    Args:
        problems_dir: directory containing CNF problem files
        cache_path: optional path to cache the vocab
        min_count: minimum frequency to include a symbol
        problem_list: if given, only scan these problems (not the full dir)

    Returns dict mapping symbol_name -> vocab_index.
    Index 0 is reserved for UNK (Skolem symbols and rare Mizar symbols).
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    from conjecture_gen.tptp_parser import parse_problem_file
    from conjecture_gen.graph_builder import clauses_to_graph

    if problem_list is None:
        all_files = sorted(os.listdir(problems_dir))
    else:
        all_files = sorted(problem_list)

    counts = {}
    print(f"Building symbol vocab from {len(all_files)} problem files...")
    for fi, fname in enumerate(all_files):
        try:
            g = clauses_to_graph(parse_problem_file(os.path.join(problems_dir, fname)))
            for name in g.symbol_names:
                if not is_skolem(name):
                    counts[name] = counts.get(name, 0) + 1
        except Exception:
            pass
        if (fi + 1) % 1000 == 0:
            print(f"  {fi+1}/{len(all_files)} files, {len(counts)} symbols...")

    # Build vocab: index 0 = UNK, then sorted by frequency
    vocab = {'<UNK>': 0}
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count >= min_count:
            vocab[name] = len(vocab)

    print(f"Symbol vocab: {len(vocab)} entries "
          f"({len(vocab)-1} Mizar + UNK, min_count={min_count})")

    if cache_path:
        tmp = cache_path + f".tmp.{os.getpid()}"
        torch.save(vocab, tmp)
        os.replace(tmp, cache_path)

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
