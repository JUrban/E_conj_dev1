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


CACHE_SCHEMA_VERSION = 5


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
        split_file: str = None,  # path to file listing problem names for this split
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
        max_samples: int = 0,  # 0 = no limit
        max_nodes: int = 0,  # 0 = no limit; max total graph nodes per problem
        symbol_vocab: dict = None,  # if set, adds named embeddings to graphs
    ):
        self.problems_dir = problems_dir
        self.lemmas_file = lemmas_file
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        self.symbol_vocab = symbol_vocab

        # Pre-index lemma file: group raw lines by problem name.
        # Class-level so second dataset (val) reuses the index.
        if not ConjectureDataset._lemma_str_index:
            print(f"Indexing lemma file...")
            with open(lemmas_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    colon_idx = line.find(': cnf(')
                    if colon_idx == -1:
                        continue
                    parts = line[:colon_idx].split('/')
                    if len(parts) >= 3:
                        prob, lid = parts[-2], parts[-1]
                        if prob not in ConjectureDataset._lemma_str_index:
                            ConjectureDataset._lemma_str_index[prob] = {}
                        ConjectureDataset._lemma_str_index[prob][lid] = line
            print(f"  Indexed {sum(len(v) for v in ConjectureDataset._lemma_str_index.values())} "
                  f"lemmas for {len(ConjectureDataset._lemma_str_index)} problems")
        else:
            print(f"Reusing lemma index ({len(ConjectureDataset._lemma_str_index)} problems)")

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(problems_dir), 'cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Load or build the full index (all ratios)
        index_path = os.path.join(cache_dir, 'index.pt')
        rebuild = True
        if os.path.exists(index_path):
            index = torch.load(index_path, weights_only=False)
            cached_schema = index.get('schema', None)
            if cached_schema == CACHE_SCHEMA_VERSION:
                self.samples = index['samples']
                self.problem_names = index['problem_names']
                rebuild = False
            else:
                print(f"Cache schema mismatch (cached={cached_schema}, "
                      f"current={CACHE_SCHEMA_VERSION}), rebuilding...")
        if rebuild:
            print("Building dataset index (first time)...")
            self._build_index(problems_dir, lemmas_file, statistics_file)
            tmp = index_path + f".tmp.{os.getpid()}"
            torch.save({
                'schema': CACHE_SCHEMA_VERSION,
                'samples': self.samples,
                'problem_names': self.problem_names,
            }, tmp)
            os.replace(tmp, index_path)

        # Apply ratio filter
        before_ratio = len(self.samples)
        self.samples = [
            s for s in self.samples
            if min_ratio <= s['ratio'] <= max_ratio
        ]
        self.problem_names = sorted(set(s['problem'] for s in self.samples))
        print(f"Ratio filter [{min_ratio},{max_ratio}]: {before_ratio} -> "
              f"{len(self.samples)} samples, {len(self.problem_names)} problems")

        # Filter out problems with too-large graphs (prevents GPU OOM)
        if max_nodes > 0:
            size_cache_path = os.path.join(cache_dir, 'problem_sizes.pt')
            if os.path.exists(size_cache_path):
                problem_sizes = torch.load(size_cache_path, weights_only=False)
                print(f"Loaded cached problem sizes: {len(problem_sizes)} entries")
            else:
                print(f"Computing problem graph sizes (first time)...")
                problem_sizes = self._compute_problem_sizes(problems_dir)
                tmp = size_cache_path + f".tmp.{os.getpid()}"
                torch.save(problem_sizes, tmp)
                os.replace(tmp, size_cache_path)

            big_problems = {
                p for p, sz in problem_sizes.items() if sz > max_nodes
            }
            # Store sizes for batch sampler
            self._problem_sizes = problem_sizes

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
        if split_file is not None:
            # Use external split file (one problem name per line)
            with open(split_file) as f:
                keep = set(line.strip() for line in f if line.strip())
            # Intersect with available problems
            keep = keep & set(self.problem_names)
        elif split != 'all':
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
        else:
            keep = set(self.problem_names)

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

        # 2. Scan lemma file for available (problem, lemma_id) keys.
        # Only check existence here — full clause parsing is deferred to
        # _get_lemma_clause (called lazily per sample). This avoids holding
        # 196K+ parsed clause objects in memory during index construction.
        print("  Scanning lemma keys...")
        lemma_keys = set()
        with open(lemmas_file) as f:
            for line in f:
                # Fast key extraction without full clause parsing:
                # format is "./problem/lemma_id: cnf(...)"
                line = line.strip()
                if not line:
                    continue
                colon_idx = line.find(': cnf(')
                if colon_idx == -1:
                    continue
                path_part = line[:colon_idx]
                parts = path_part.split('/')
                if len(parts) >= 3:
                    lemma_keys.add((parts[-2], parts[-1]))

        # 3. Build samples
        print(f"  Building samples ({len(stats)} stats, {len(lemma_keys)} lemma keys)...")
        self.samples = []
        self.problem_names = set()

        for (problem, cut_id), ratio in stats.items():
            # Store ALL samples regardless of ratio — ratio filtering is
            # applied after loading the index so the cached index.pt is not
            # permanently limited by the first run's ratio range.
            if (problem, cut_id) not in lemma_keys:
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

    # Class-level caches shared across train/val instances:
    _graph_cache = {}       # problem_name -> HeteroData (~2-3GB)
    _lemma_cache = {}       # problem_name -> {lemma_id: Clause} (~128MB)
    _lemma_str_index = {}   # problem_name -> {lemma_id: raw_line} (~94MB)

    def _get_problem_graph(self, problem_name: str) -> HeteroData:
        """Load or build the problem graph. Cached in RAM."""
        if problem_name in self._graph_cache:
            return self._graph_cache[problem_name]

        suffix = '_named' if self.symbol_vocab else ''
        cache_path = os.path.join(self.cache_dir, f'graph_{problem_name}{suffix}.pt')
        if os.path.exists(cache_path):
            graph = torch.load(cache_path, weights_only=False)
        else:
            problem_path = os.path.join(self.problems_dir, problem_name)
            clauses = parse_problem_file(problem_path)
            graph = clauses_to_graph(clauses, vocab=self.symbol_vocab)
            tmp = cache_path + f".tmp.{os.getpid()}"
            torch.save(graph, tmp)
            os.replace(tmp, cache_path)

        self._graph_cache[problem_name] = graph
        return graph

    def _get_lemma_clause(self, problem_name: str, cut_id: str):
        """Get a parsed lemma clause. Cached in RAM per problem."""
        if problem_name in self._lemma_cache:
            return self._lemma_cache[problem_name].get(cut_id)

        # Build lemma dict for this problem
        lemma_dict = {}

        # Try disk cache first
        cache_path = os.path.join(self.cache_dir, f'lemmas_{problem_name}.pt')
        if os.path.exists(cache_path):
            lemma_dict = torch.load(cache_path, weights_only=False)
        elif problem_name in self._lemma_str_index:
            for lid, line in self._lemma_str_index[problem_name].items():
                result = parse_lemma_line(line)
                if result is not None:
                    _, _, clause = result
                    lemma_dict[lid] = clause
            tmp = cache_path + f".tmp.{os.getpid()}"
            torch.save(lemma_dict, tmp)
            os.replace(tmp, cache_path)
        else:
            # Fallback: scan the file
            prefix = f'./{problem_name}/'
            with open(self.lemmas_file) as f:
                for line in f:
                    if not line.startswith(prefix):
                        continue
                    result = parse_lemma_line(line)
                    if result is not None:
                        _, lid, clause = result
                        lemma_dict[lid] = clause
            tmp = cache_path + f".tmp.{os.getpid()}"
            torch.save(lemma_dict, tmp)
            os.replace(tmp, cache_path)

        self._lemma_cache[problem_name] = lemma_dict
        return lemma_dict.get(cut_id)

    def precompute(self, load_into_ram=True):
        """Precompute all samples into RAM for zero-overhead __getitem__.

        First run builds all samples and saves as a single .pt file (~8GB).
        Subsequent runs load the file directly (~1-2 min vs ~30 min build).
        """
        cache_path = os.path.join(self.cache_dir,
                                   f'precomputed_{len(self.samples)}.pt')
        if os.path.exists(cache_path):
            print(f"Loading {len(self.samples)} precomputed samples from {cache_path}...")
            self._inmemory = torch.load(cache_path, weights_only=False)
            print(f"  Loaded {len(self._inmemory)} samples into RAM.")
            return

        # Parallel precompute using threads (graph.clone releases GIL)
        import concurrent.futures
        n_workers = min(os.cpu_count() or 1, 8)
        n = len(self.samples)
        print(f"Precomputing {n} samples ({n_workers} threads)...")

        self._inmemory = [None] * n

        def build_one(idx):
            return idx, self._build_item(idx)

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            for idx, item in pool.map(build_one, range(n)):
                self._inmemory[idx] = item
                done += 1
                if done % 5000 == 0:
                    print(f"  precomputed {done}/{n}...")
        print(f"  Saving to {cache_path}...")
        tmp = cache_path + f".tmp.{os.getpid()}"
        torch.save(self._inmemory, tmp)
        os.replace(tmp, cache_path)
        print(f"  All {len(self._inmemory)} samples in RAM.")

    def __len__(self):
        return len(self.samples)

    # Encoding stats accumulator for periodic logging
    _encoding_stats_accum = {'exact_hits': 0, 'role_fallback_hits': 0,
                             'name_fallback_hits': 0, 'unk_hits': 0, 'count': 0}

    def precompute_targets(self):
        """Pre-encode all target sequences. ~14MB for 138K samples, ~70s.

        After this, __getitem__ skips encode_conjecture() and lemma parsing.
        Must be called after graph cache is warmed.
        """
        print(f"Pre-encoding {len(self.samples)} target sequences...")
        self._targets = []
        n_fail = 0
        for idx, sample in enumerate(self.samples):
            problem_name = sample['problem']
            cut_id = sample['cut_id']
            ratio = sample['ratio']

            graph = self._get_problem_graph(problem_name)
            clause = self._get_lemma_clause(problem_name, cut_id)
            if clause is None:
                n_fail += 1
                self._targets.append(None)
                continue

            target_seq = encode_conjecture(
                clause, graph.symbol_names,
                symbol_is_pred=getattr(graph, 'symbol_is_pred', None),
                symbol_arities=getattr(graph, 'symbol_arities', None),
                strict=False,
            )
            weight = 1.0 / (1.0 + ratio)
            self._targets.append({
                'actions': torch.tensor([a for a, _ in target_seq], dtype=torch.long),
                'arguments': torch.tensor([arg for _, arg in target_seq], dtype=torch.long),
                'length': len(target_seq),
                'weight': weight,
                'num_symbols': len(graph.symbol_names),
            })
            if (idx + 1) % 10000 == 0:
                print(f"  encoded {idx+1}/{len(self.samples)}...")
        print(f"  Done ({n_fail} failures)")

    def _build_item(self, idx):
        """Build a single sample (used by both __getitem__ and precompute)."""
        sample = self.samples[idx]
        problem_name = sample['problem']
        cut_id = sample['cut_id']
        ratio = sample['ratio']

        graph = self._get_problem_graph(problem_name)

        # Use pre-encoded targets if available
        if hasattr(self, '_targets') and self._targets and self._targets[idx] is not None:
            t = self._targets[idx]
            graph = graph.clone()
            graph.target_actions = t['actions']
            graph.target_arguments = t['arguments']
            graph.target_length = torch.tensor(t['length'], dtype=torch.long)
            graph.quality_weight = torch.tensor(t['weight'], dtype=torch.float)
            graph.ratio = torch.tensor(ratio, dtype=torch.float)
            graph.num_symbols = torch.tensor(t['num_symbols'], dtype=torch.long)
            return graph

        clause = self._get_lemma_clause(problem_name, cut_id)
        if clause is None:
            raise KeyError(
                f"Lemma not found: problem={problem_name!r}, cut_id={cut_id!r}. "
                f"Check that the lemmas file '{self.lemmas_file}' contains this entry."
            )
        target_seq = encode_conjecture(
            clause, graph.symbol_names,
            symbol_is_pred=getattr(graph, 'symbol_is_pred', None),
            symbol_arities=getattr(graph, 'symbol_arities', None),
            strict=False,
        )

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
        # Fallback: build on the fly
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
