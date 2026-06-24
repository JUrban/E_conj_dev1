# Vampire Minimized Proof Lemma Dataset (vmin1)

**Addendum to:** *Neural Conjecturing for Saturation Theorem Provers* (LPAR short paper)

## Motivation

The main paper describes two sources of cut training data:

1. **E prover data** — 3,161 post-proof minimized CNF problems, 122,356 evaluated proof-lemma pairs, 44,895 useful (ratio < 1).
2. **Vampire full-axiom data (v1)** — 57,880 Mizar40 full-axiom CNF problems, 2.47M evaluated pairs, 196,199 useful.

The E data has small, clean graphs (post-proof minimized) but limited scale.
The Vampire v1 data is large but uses full-axiom problems with thousands of
clauses, causing memory and batching difficulties during training.

The new **vmin1** dataset bridges this gap: Vampire proof lemmas extracted
from **Mizar60 premise-selected** problems, which have moderate graph sizes
(median 34 clauses) and large scale.  This combines the training-friendly
graph sizes of the E data with the large lemma volume of the Vampire data.

## Data Extraction

### Source problems

The 55,832 premise-selected Mizar60 CNF problems from the paper's
evaluation corpus.  These problems use premise selection to include only
relevant axioms, giving much smaller clause sets than full-axiom problems
but larger than post-proof minimized ones.

### Proof lemma extraction

The Deepire 2.0 non-AVATAR model (described in the main paper, Section 2)
solves a subset of these problems.  For each solved problem, intermediate
proof clauses are extracted and introduced as AVATAR claims in a separate
Vampire evaluation run.  This yields 2,557,482 claim evaluations from
37,427 problems.

The claims are originally in FOF format with universal quantifiers
(e.g., `fof(name, claim, (! [X] : (body)))`).  For training compatibility
with the existing CNF-based pipeline, we strip the quantifiers and convert
to CNF format: `cnf(name, plain, (body)).`  Of the 2,557,482 claims,
1,306,475 (51%) had quantifiers that were removed.

### Evaluation

Each claim is evaluated as an AVATAR assertion: Vampire runs the
premise-selected problem with the claim added.  The baseline run (without
claims) provides the reference instruction count.  A claim is useful when:

- Vampire solves the problem with the claim present, and
- the instruction count with the claim is lower than the baseline (ratio < 1).

| Metric | Count |
|--------|-------|
| Baseline problems | 37,427 |
| Claim evaluations | 2,552,691 |
| Useful claims (ratio < 1) | 745,285 |
| Unique problems with useful claims | 29,887 |
| Useful rate | 29.2% |
| Claims with ≥ 2× speedup (ratio ≤ 0.5) | 93,650 |
| Claims with ≥ 10× speedup (ratio ≤ 0.1) | 13,213 |
| Mean ratio (useful claims) | 0.840 |

The 29.2% useful rate is much higher than the 7.9% rate on the full-axiom
Vampire data (196,199 / 2,470,292), likely because premise-selected problems
are smaller and more amenable to case-splitting speedups.

### Comparison of cut datasets

| Dataset | Problems | Evaluated | Useful | Useful rate | Graph size |
|---------|----------|-----------|--------|-------------|------------|
| E minimized | 3,161 | 122,356 | 44,895 | 36.7% | very small |
| Vampire full-axiom (v1) | 57,880 | 2,470,292 | 196,199 | 7.9% | very large |
| **Vampire premsel (vmin1)** | **37,427** | **2,552,691** | **745,285** | **29.2%** | **moderate** |

## Combined Training Data

We combine the E data (all 122,356 pairs, ratio filtering at training time)
with the vmin1 useful pairs (745,285) for a total of **867,641 training examples**.

Since 2,481 problem names overlap between the E and vmin1 corpora (the E
problems are a subset of Mizar40 which overlaps with Mizar60), vmin1
entries are prefixed with `vm_` to disambiguate.  A combined problems
directory provides symlinks to both `problems/` (E, 3,161 files) and
`Mizar60_premsel_data/problems_cnf/` (vmin1, 55,832 files).

### Split statistics

| Split | E problems | vmin1 problems | Total | Samples (after ratio+size filter) |
|-------|-----------|----------------|-------|-----------------------------------|
| Train | 2,239 | 26,899 | 29,138 | 696,130 |
| Val | 279 | 2,988 | 3,267 | 77,038 |

After applying `max_ratio=1.0` (keep all useful) and `max_nodes=3000`
(drop 188 large problems, 11,448 samples), the training set has 696,130
samples and the validation set has 77,038 samples.

## Training Setup

### Model

The same heterogeneous GNN encoder + Transformer pointer decoder
architecture described in the main paper (Section 3).  Default
configuration: hidden dimension 128, 4 GNN layers, named Mizar symbol
embeddings (vocab size 9,026), 3-layer Transformer decoder with 4
attention heads.

### Training parameters

- Batch size: 64
- Learning rate: 1e-4, Adam with weight decay 1e-5
- Mixed precision (AMP)
- Size-aware batching (max 3,000 nodes per problem)
- 8 DataLoader workers
- Checkpoints saved every epoch for post-hoc model selection

### Planned experiments

1. **From scratch (d=128):** Train on the combined E + vmin1 data.
2. **Init from E checkpoint (d=128):** Initialize from the best
   E-trained model (`checkpoints_af6_a_named/best_model.pt`) and fine-tune
   on combined data.  This worked well for the v2 iteration in the main
   paper (Section 4.2, "Closing the loop").

### Estimated training time

Based on previous experiments, d=128 on 59K samples ran at ~130s/epoch.
With ~696K samples (11.8×), estimated ~25-30 min/epoch.  100 epochs ≈
2 days on a single A100.  Per-epoch checkpoints allow early stopping or
selection of the best generation model independently of validation loss.

## Pipeline Scripts

All scripts are in `vmin1/`:

| Script | Purpose |
|--------|---------|
| `convert_fof_to_cnf.py` | Strip FOF quantifiers → CNF format |
| `prepare_data.py` | Parse burned baseline/claim files → `statistics_useful` + `lemmas_useful` |
| `combine_datasets.py` | Merge E + vmin1 with `vm_` prefix → combined training files |
| `create_combined_problems_dir.py` | Symlink E + M60 premsel problems |
| `train_vmin1.sh` | Training launcher with default parameters |

## Next Steps

1. Complete training (both from-scratch and init-from-E).
2. Generate conjectures for the 11,881 hard/unsolved Mizar60 problems
   (same evaluation set as the main paper).
3. Evaluate with Vampire AVATAR + Deepire 2.0, measuring newly solved
   problems and speedups.
4. Compare with the first two iterations from the main paper to assess
   whether the larger, premise-selected training data improves generation
   quality and transfer.
