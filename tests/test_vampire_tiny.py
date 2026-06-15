"""End-to-end test of the Vampire-scale training pipeline on tiny data."""

import os
import shutil
import tempfile

import pytest
import torch

from conjecture_gen.dataset import ConjectureDataset
from conjecture_gen.batch_by_size import SizeAwareBatchSampler
from conjecture_gen.symbol_vocab import build_vocab


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'vampire_tiny')


@pytest.fixture(autouse=True)
def _clear_class_caches():
    """Reset ConjectureDataset class-level caches between tests."""
    ConjectureDataset._graph_cache.clear()
    ConjectureDataset._lemma_cache.clear()
    ConjectureDataset._lemma_str_index.clear()
    yield
    ConjectureDataset._graph_cache.clear()
    ConjectureDataset._lemma_cache.clear()
    ConjectureDataset._lemma_str_index.clear()


@pytest.fixture
def cache_dir():
    """Provide a fresh temporary cache directory per test."""
    d = tempfile.mkdtemp(prefix='vampire_tiny_test_')
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_train_ds(cache_dir):
    return ConjectureDataset(
        problems_dir=os.path.join(FIXTURE_DIR, 'problems'),
        lemmas_file=os.path.join(FIXTURE_DIR, 'lemmas'),
        statistics_file=os.path.join(FIXTURE_DIR, 'statistics'),
        cache_dir=cache_dir,
        max_ratio=1.0,
        split_file=os.path.join(FIXTURE_DIR, 'train_problems.txt'),
    )


def _make_val_ds(cache_dir):
    return ConjectureDataset(
        problems_dir=os.path.join(FIXTURE_DIR, 'problems'),
        lemmas_file=os.path.join(FIXTURE_DIR, 'lemmas'),
        statistics_file=os.path.join(FIXTURE_DIR, 'statistics'),
        cache_dir=cache_dir,
        max_ratio=1.0,
        split_file=os.path.join(FIXTURE_DIR, 'val_problems.txt'),
    )


# -----------------------------------------------------------------------
# 1. Dataset loading with split_file
# -----------------------------------------------------------------------

def test_dataset_loading(cache_dir):
    """ConjectureDataset loads with split_file, correct sample counts."""
    train_ds = _make_train_ds(cache_dir)

    # train_problems.txt has p1 and p2.
    # With max_ratio=1.0:
    #   p1/f1 ratio=0.5 (kept), p1/f2 ratio=0.8 (kept)
    #   p2/f3 ratio=0.3 (kept), p2/f4 ratio=1.5 (filtered by max_ratio)
    # So 3 samples survive for the train split.
    assert len(train_ds) == 3, f"Expected 3 train samples, got {len(train_ds)}"

    # Check all surviving samples belong to p1 or p2
    for s in train_ds.samples:
        assert s['problem'] in ('p1', 'p2'), (
            f"Unexpected problem {s['problem']} in train split"
        )
        assert s['ratio'] <= 1.0

    # Val split: p3 has f5 (ratio=0.6, kept) and f6 (ratio=1.2, filtered)
    val_ds = _make_val_ds(cache_dir)
    assert len(val_ds) == 1, f"Expected 1 val sample, got {len(val_ds)}"
    assert val_ds.samples[0]['problem'] == 'p3'


# -----------------------------------------------------------------------
# 2. Precompute and __getitem__
# -----------------------------------------------------------------------

def test_precompute_and_getitem(cache_dir):
    """precompute() works, __getitem__ returns valid HeteroData with target_actions."""
    ds = _make_train_ds(cache_dir)
    ds.precompute()

    for idx in range(len(ds)):
        item = ds[idx]

        # Must be a HeteroData (or at least have these attrs)
        assert hasattr(item, 'target_actions'), (
            f"Sample {idx} missing target_actions"
        )
        assert hasattr(item, 'target_arguments'), (
            f"Sample {idx} missing target_arguments"
        )
        assert hasattr(item, 'target_length'), (
            f"Sample {idx} missing target_length"
        )
        assert hasattr(item, 'quality_weight'), (
            f"Sample {idx} missing quality_weight"
        )
        assert hasattr(item, 'num_symbols'), (
            f"Sample {idx} missing num_symbols"
        )

        # target_actions is a 1-D long tensor with length matching target_length
        assert item.target_actions.dtype == torch.long
        assert item.target_actions.dim() == 1
        assert item.target_actions.shape[0] == item.target_length.item()

        # target_arguments same shape
        assert item.target_arguments.shape == item.target_actions.shape

        # Last action should be END_CLAUSE (6)
        from conjecture_gen.target_encoder import END_CLAUSE
        assert item.target_actions[-1].item() == END_CLAUSE

        # Graph should have clause nodes
        assert 'clause' in item.node_types
        assert item['clause'].x.shape[0] > 0


# -----------------------------------------------------------------------
# 3. SizeAwareBatchSampler
# -----------------------------------------------------------------------

def test_size_aware_batching(cache_dir):
    """SizeAwareBatchSampler produces valid batches covering all indices."""
    ds = _make_train_ds(cache_dir)

    # Warm graph cache so the sampler can estimate sizes
    for s in ds.samples:
        ds._get_problem_graph(s['problem'])

    sampler = SizeAwareBatchSampler(
        ds, max_total_nodes=50000, shuffle=False,
    )
    batches = list(sampler)

    # At least one batch
    assert len(batches) >= 1

    # All batches are non-empty lists of ints
    for batch in batches:
        assert isinstance(batch, list)
        assert len(batch) > 0
        for idx in batch:
            assert isinstance(idx, int)
            assert 0 <= idx < len(ds)

    # Every sample index appears exactly once
    all_indices = []
    for batch in batches:
        all_indices.extend(batch)
    assert sorted(all_indices) == list(range(len(ds))), (
        f"Batch indices {sorted(all_indices)} != expected {list(range(len(ds)))}"
    )

    # __len__ matches actual batch count
    assert len(sampler) == len(batches)


# -----------------------------------------------------------------------
# 4. build_vocab with problem_list
# -----------------------------------------------------------------------

def test_symbol_vocab_with_problem_list():
    """build_vocab with problem_list works and produces a valid vocabulary."""
    vocab = build_vocab(
        os.path.join(FIXTURE_DIR, 'problems'),
        problem_list=['p1', 'p2', 'p3'],
        min_count=1,
    )

    # Must have <UNK> at index 0
    assert '<UNK>' in vocab
    assert vocab['<UNK>'] == 0

    # Should contain well-known Mizar symbols from the fixtures
    for sym in ['r2_hidden', 'v1_xboole_0', 'r1_tarski', 'k1_xboole_0']:
        assert sym in vocab, f"Expected symbol '{sym}' in vocab"

    # Skolem symbols should NOT appear (they get mapped to UNK)
    for sym in ['esk1_0', 'esk2_2', 'esk3_0', 'esk4_0', 'esk5_0', 'esk6_0']:
        assert sym not in vocab, f"Skolem symbol '{sym}' should not be in vocab"

    # All indices should be unique non-negative integers
    indices = list(vocab.values())
    assert len(set(indices)) == len(indices), "Duplicate vocab indices"
    assert all(i >= 0 for i in indices)


# -----------------------------------------------------------------------
# 5. Shared _lemma_str_index across datasets
# -----------------------------------------------------------------------

def test_lemma_index_shared(cache_dir):
    """Two datasets share _lemma_str_index (class-level cache)."""
    train_ds = _make_train_ds(cache_dir)
    val_ds = _make_val_ds(cache_dir)

    # Both should reference the exact same dict object
    assert train_ds._lemma_str_index is val_ds._lemma_str_index, (
        "_lemma_str_index is not shared between train and val datasets"
    )

    # The shared index should contain all 3 problems
    assert 'p1' in train_ds._lemma_str_index
    assert 'p2' in train_ds._lemma_str_index
    assert 'p3' in train_ds._lemma_str_index

    # Total lemma count should be 6 (2 per problem)
    total = sum(len(v) for v in train_ds._lemma_str_index.values())
    assert total == 6, f"Expected 6 lemmas in index, got {total}"
