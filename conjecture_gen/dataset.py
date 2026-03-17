"""
Dataset: pairs each problem graph with its good conjecture targets.

Preprocesses all problems and lemmas once, saves to disk as .pt files.
Supports lazy loading for large datasets.
"""

import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from conjecture_gen.tptp_parser import (
    parse_problem_file, parse_lemma_line, parse_statistics_line,
)
from conjecture_gen.graph_builder import clauses_to_graph
from conjecture_gen.target_encoder import encode_conjecture


class ConjectureDataset(Dataset):
    """Dataset of (problem_graph, target_sequence, quality_weight) triples.

    Each item corresponds to one good cut for one problem.
    """

    def __init__(
        self,
        problems_dir: str,
        lemmas_file: str,
        statistics_file: str,
        cache_dir: str = None,
        max_ratio: float = 1.0,
        min_ratio: float = 0.0,
        split: str = 'all',  # 'train', 'val', 'test', or 'all'
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        max_samples: int = 0,  # 0 = no limit
        max_nodes: int = 0,  # 0 = no limit; max total graph nodes per problem
    ):
        self.problems_dir = problems_dir
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(problems_dir), 'cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Load or build the full index (all ratios)
        index_path = os.path.join(cache_dir, 'index.pt')
        if os.path.exists(index_path):
            index = torch.load(index_path, weights_only=False)
            self.samples = index['samples']
            self.problem_names = index['problem_names']
        else:
            print("Building dataset index (first time)...")
            self._build_index(problems_dir, lemmas_file, statistics_file)
            torch.save({
                'samples': self.samples,
                'problem_names': self.problem_names,
            }, index_path)

        # Apply ratio filter
        self.samples = [
            s for s in self.samples
            if min_ratio <= s['ratio'] <= max_ratio
        ]
        self.problem_names = sorted(set(s['problem'] for s in self.samples))

        # Filter out problems with too-large graphs (prevents GPU OOM)
        if max_nodes > 0:
            size_cache_path = os.path.join(cache_dir, 'problem_sizes.pt')
            if os.path.exists(size_cache_path):
                problem_sizes = torch.load(size_cache_path, weights_only=False)
            else:
                print(f"Computing problem graph sizes (first time)...")
                problem_sizes = self._compute_problem_sizes(problems_dir)
                torch.save(problem_sizes, size_cache_path)

            big_problems = {
                p for p, sz in problem_sizes.items() if sz > max_nodes
            }
            if big_problems:
                before = len(self.samples)
                self.samples = [
                    s for s in self.samples
                    if s['problem'] not in big_problems
                ]
                self.problem_names = sorted(
                    set(s['problem'] for s in self.samples)
                )
                print(f"max_nodes={max_nodes}: dropped {len(big_problems)} "
                      f"large problems ({before - len(self.samples)} samples)")

        # Split by problem (not by sample!) for proper evaluation
        all_problems = sorted(self.problem_names)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(all_problems), generator=rng).tolist()
        n_val = int(len(all_problems) * val_frac)
        n_test = int(len(all_problems) * test_frac)

        test_problems = set(all_problems[perm[i]] for i in range(n_test))
        val_problems = set(all_problems[perm[i]] for i in range(n_test, n_test + n_val))
        train_problems = set(all_problems) - test_problems - val_problems

        if split == 'train':
            keep = train_problems
        elif split == 'val':
            keep = val_problems
        elif split == 'test':
            keep = test_problems
        else:
            keep = set(all_problems)

        self.samples = [s for s in self.samples if s['problem'] in keep]
        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Split '{split}': {len(keep)} problems, {len(self.samples)} samples")

    def _compute_problem_sizes(self, problems_dir: str) -> dict[str, int]:
        """Count total graph nodes per problem (for filtering large ones)."""
        from conjecture_gen.graph_builder import clauses_to_graph
        sizes = {}
        unique_problems = sorted(set(s['problem'] for s in self.samples))
        for i, pname in enumerate(unique_problems):
            path = os.path.join(problems_dir, pname)
            try:
                clauses = parse_problem_file(path)
                graph = clauses_to_graph(clauses)
                total = sum(
                    graph[nt].x.shape[0] for nt in graph.node_types
                )
                sizes[pname] = total
            except Exception:
                sizes[pname] = 999999  # mark as large on error
            if (i + 1) % 500 == 0:
                print(f"  sized {i+1}/{len(unique_problems)} problems...")
        return sizes

    def _build_index(self, problems_dir, lemmas_file, statistics_file):
        """Parse all files and build the sample index."""
        # 1. Parse statistics to get quality info
        print("  Parsing statistics...")
        stats = {}  # (problem, cut_id) -> ratio
        with open(statistics_file) as f:
            for line in f:
                s = parse_statistics_line(line)
                if s is not None:
                    stats[(s['problem'], s['cut_id'])] = s['ratio']

        # 2. Parse lemmas
        print("  Parsing lemmas...")
        lemmas = {}  # (problem, cut_id) -> Clause
        with open(lemmas_file) as f:
            for line in f:
                result = parse_lemma_line(line)
                if result is not None:
                    problem, lemma_id, clause = result
                    lemmas[(problem, lemma_id)] = clause

        # 3. Build samples
        print("  Building samples...")
        self.samples = []
        self.problem_names = set()

        for (problem, cut_id), ratio in stats.items():
            if ratio < self.min_ratio or ratio > self.max_ratio:
                continue
            if (problem, cut_id) not in lemmas:
                continue

            self.problem_names.add(problem)
            self.samples.append({
                'problem': problem,
                'cut_id': cut_id,
                'ratio': ratio,
            })

        self.problem_names = sorted(self.problem_names)
        print(f"  Found {len(self.samples)} samples across "
              f"{len(self.problem_names)} problems")

    def _get_problem_graph(self, problem_name: str) -> HeteroData:
        """Load or build the problem graph."""
        cache_path = os.path.join(self.cache_dir, f'graph_{problem_name}.pt')
        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        # Parse and build
        problem_path = os.path.join(self.problems_dir, problem_name)
        clauses = parse_problem_file(problem_path)
        graph = clauses_to_graph(clauses)

        torch.save(graph, cache_path)
        return graph

    def _get_lemma_clause(self, problem_name: str, cut_id: str):
        """Re-parse a specific lemma from the lemmas file.

        For efficiency, we cache per-problem lemma dicts.
        """
        cache_path = os.path.join(
            self.cache_dir, f'lemmas_{problem_name}.pt'
        )
        if os.path.exists(cache_path):
            lemma_dict = torch.load(cache_path, weights_only=False)
            return lemma_dict.get(cut_id)

        # Parse all lemmas for this problem and cache
        lemma_dict = {}
        lemma_file = os.path.join(
            os.path.dirname(self.problems_dir), 'lemmas'
        )
        prefix = f'./{problem_name}/'
        with open(lemma_file) as f:
            for line in f:
                if not line.startswith(prefix):
                    continue
                result = parse_lemma_line(line)
                if result is not None:
                    _, lid, clause = result
                    lemma_dict[lid] = clause

        torch.save(lemma_dict, cache_path)
        return lemma_dict.get(cut_id)

    def precompute(self):
        """Precompute all samples and load into RAM for zero-overhead __getitem__.

        First call builds disk cache, then loads everything into memory.
        Subsequent calls just load from disk cache into memory.
        """
        precomp_dir = os.path.join(self.cache_dir, 'precomputed')
        os.makedirs(precomp_dir, exist_ok=True)

        # Build disk cache if needed
        marker = os.path.join(precomp_dir, f'done_{len(self.samples)}.marker')
        if not os.path.exists(marker):
            print(f"Precomputing {len(self.samples)} samples to disk...")
            for idx in range(len(self.samples)):
                out_path = os.path.join(precomp_dir, f'sample_{idx}.pt')
                if not os.path.exists(out_path):
                    item = self._build_item(idx)
                    torch.save(item, out_path)
                if (idx + 1) % 2000 == 0:
                    print(f"  precomputed {idx+1}/{len(self.samples)}...")
            with open(marker, 'w') as f:
                f.write('done')

        # Load everything into RAM
        print(f"Loading {len(self.samples)} precomputed samples into RAM...")
        self._inmemory = []
        for idx in range(len(self.samples)):
            path = os.path.join(precomp_dir, f'sample_{idx}.pt')
            self._inmemory.append(torch.load(path, weights_only=False))
            if (idx + 1) % 2000 == 0:
                print(f"  loaded {idx+1}/{len(self.samples)}...")
        print(f"  All {len(self._inmemory)} samples in RAM.")

    def __len__(self):
        return len(self.samples)

    def _build_item(self, idx):
        """Build a single sample (used by both __getitem__ and precompute)."""
        sample = self.samples[idx]
        problem_name = sample['problem']
        cut_id = sample['cut_id']
        ratio = sample['ratio']

        graph = self._get_problem_graph(problem_name)

        clause = self._get_lemma_clause(problem_name, cut_id)
        if clause is None:
            target_seq = [(6, 0)]
        else:
            target_seq = encode_conjecture(clause, graph.symbol_names)

        weight = 1.0 / (1.0 + ratio)

        actions = torch.tensor([a for a, _ in target_seq], dtype=torch.long)
        arguments = torch.tensor([arg for _, arg in target_seq], dtype=torch.long)

        graph = graph.clone()
        graph.target_actions = actions
        graph.target_arguments = arguments
        graph.target_length = torch.tensor(len(target_seq), dtype=torch.long)
        graph.quality_weight = torch.tensor(weight, dtype=torch.float)
        graph.ratio = torch.tensor(ratio, dtype=torch.float)
        graph.num_symbols = torch.tensor(len(graph.symbol_names), dtype=torch.long)
        return graph

    def __getitem__(self, idx):
        # Use in-memory cache if available (fastest)
        # Must clone to prevent .to(device) from mutating the cache
        if hasattr(self, '_inmemory') and self._inmemory:
            return self._inmemory[idx].clone()
        return self._build_item(idx)


if __name__ == '__main__':
    import time

    t0 = time.time()
    ds = ConjectureDataset(
        problems_dir='problems',
        lemmas_file='lemmas',
        statistics_file='statistics',
        max_ratio=1.0,  # only good cuts
        split='train',
    )
    t1 = time.time()
    print(f"\nBuilt in {t1-t0:.1f}s")
    print(f"Training samples: {len(ds)}")

    # Inspect one sample
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  Problem: {ds.samples[0]['problem']}")
    print(f"  Ratio: {sample.ratio.item():.4f}")
    print(f"  Weight: {sample.quality_weight.item():.4f}")
    print(f"  Target length: {sample.target_length.item()}")
    print(f"  Target actions: {sample.target_actions.tolist()}")
    print(f"  Target args:    {sample.target_arguments.tolist()}")
    print(f"  Symbols: {sample.num_symbols.item()}")
    print(f"  Graph: {sample}")
