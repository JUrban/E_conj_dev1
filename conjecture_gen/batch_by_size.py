"""
Size-aware batching: group samples by graph size so batches have
similar total node counts, avoiding VRAM spikes from mixing
small and large graphs.
"""

from torch.utils.data import Sampler
import torch


class SizeAwareBatchSampler(Sampler):
    """Yields batches where total node count stays below max_total_nodes.

    Sorts samples by graph size, then greedily fills batches up to the
    node budget. This ensures no batch exceeds the VRAM limit regardless
    of graph size variance.

    Args:
        dataset: ConjectureDataset with .samples list
        max_total_nodes: maximum total nodes across all graphs in a batch
        shuffle: whether to shuffle the order of batches (not within)
        seed: random seed for shuffling
    """

    def __init__(self, dataset, max_total_nodes: int = 50000,
                 shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.max_total_nodes = max_total_nodes
        self.shuffle = shuffle
        self.seed = seed

        # Get graph sizes for each sample's problem
        # Uses the cached problem_sizes if available
        self._build_batches()

    def _get_sample_size(self, idx):
        """Get node count for sample idx's problem graph."""
        problem = self.dataset.samples[idx]['problem']
        # Try cached sizes from dataset
        if hasattr(self.dataset, '_problem_sizes'):
            return self.dataset._problem_sizes.get(problem, 1000)
        # Fallback: estimate from graph cache
        if problem in self.dataset._graph_cache:
            g = self.dataset._graph_cache[problem]
            return sum(g[nt].x.shape[0] for nt in g.node_types)
        return 1000  # default estimate

    def _build_batches(self):
        """Group samples into size-bounded batches."""
        # Sort indices by problem graph size
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: self._get_sample_size(i))

        # Greedily fill batches
        self.batches = []
        current_batch = []
        current_nodes = 0

        for idx in indices:
            size = self._get_sample_size(idx)
            if current_batch and current_nodes + size > self.max_total_nodes:
                self.batches.append(current_batch)
                current_batch = []
                current_nodes = 0
            current_batch.append(idx)
            current_nodes += size

        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + getattr(self, '_epoch', 0))
            perm = torch.randperm(len(self.batches), generator=g).tolist()
            for i in perm:
                yield self.batches[i]
        else:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        """Update epoch for deterministic shuffling."""
        self._epoch = epoch
