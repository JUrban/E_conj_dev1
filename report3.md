# Neural Conjecture Generation for Automated Theorem Proving

## Updated Technical Report: Full Evaluation with Named Embeddings and Prover-Based Validation

---

## 1. Introduction

### 1.1 The Problem

Automated theorem provers (ATPs) like E, Vampire, and SPASS search for proofs by
systematically exploring the space of logical inferences. For many problems, the proof
search space is enormous, and the prover must process thousands or millions of clauses
before finding a proof. A well-chosen intermediate lemma --- a "cut" --- can dramatically
reduce this search by splitting a hard problem into two easier subproblems.

The challenge is: **how do you find good cuts?** Traditionally, this requires either
human mathematical insight or brute-force enumeration of candidate lemmas. Cuts are a
fundamental concept in proof theory: Gentzen's cut-elimination theorem shows that cuts
are logically unnecessary (any proof with cuts can be transformed into one without), but
the resulting cut-free proofs may be exponentially longer. In practice, the right cut
turns an intractable search into a trivial one.

This report describes a neural approach: training a graph neural network to learn the
structural patterns that make certain lemmas useful, and then generating novel lemmas
for unseen problems. The system learns entirely from examples of useful cuts extracted
from E prover proofs of the Mizar40 corpus, and generates candidate lemmas that are
evaluated by the prover itself.

### 1.2 Key Contributions

1. **Fully symbol-anonymous GNN encoder** that understands CNF problems purely from
   graph structure, achieving 98% symbol precision without any name-based information

2. **Named symbol embeddings as a hybrid extension**: learnable embeddings for ~3,097
   Mizar symbols reduce validation loss from 0.474 to 0.293 (Transformer) and from
   0.403 to 0.289 (VAE), while Skolem symbols remain anonymous

3. **Five decoder architectures** (Transformer, SSM/Mamba, VAE, Graph Growing, Subgraph
   Completion) systematically compared on the same task, with three reaching competitive
   performance and two failing to scale

4. **Arity-constrained decoding** that enforces syntactic validity without retraining,
   improving validity from 2% to 85-88%

5. **Full E prover evaluation on 2,985 problems (10 conjectures each)**: up to 1,655
   useful speedups (5.6% of tested conjectures) with best speedup ratios of 0.013 (77x)

6. **Test-set evaluation on 279 held-out problems** confirming generalization: the best
   model (D+named) achieves 5.8% useful rate on never-seen problems, slightly better
   than its overall rate

### 1.3 What Changed Since Report 1

Report 1 described the initial system with anonymous-only embeddings and evaluation on
184 problems with 1,840 conjectures (74 useful speedups). This report covers:

- **Named symbol embeddings** trained across all three autoregressive architectures
  (A, C, D), consistently reducing validation loss by 30-40%
- **Extended training** to 150 epochs for the best anonymous model (A3)
- **Full-scale E prover evaluation** on all 2,985 provable problems (after graph-size
  filtering), with 10 conjectures per problem, using E 2.6 with 10-second timeout
- **Proper test-set analysis** on 279 held-out problems never seen during training
- **Key insight**: val loss does not predict usefulness --- D+named has the worst val
  loss among named models but the best test-set useful rate

### 1.4 Overview

The system takes a CNF (Clausal Normal Form) theorem proving problem as input and
generates candidate lemmas that may help an automated prover solve the problem faster.
The pipeline consists of:

1. **Parsing**: TPTP CNF files are parsed into structured clause/literal/term objects
2. **Graph construction**: Problems are represented as typed heterogeneous graphs
3. **GNN encoding**: A heterogeneous message-passing GNN processes the graph
4. **Autoregressive decoding**: A decoder generates lemma clauses token by token,
   using pointer attention to select symbols from the input problem
5. **Constrained generation**: Arity constraints enforce syntactic validity
6. **Prover evaluation**: Generated lemmas are tested with E prover for actual speedup

---

## 2. Dataset

### 2.1 Source

The dataset is derived from the Mizar40 corpus [Kaliszyk & Urban, 2015], consisting
of 3,161 CNF problems that can be proved by the E prover. For each problem, the E
prover's final proof clauses are extracted as candidate intermediate lemmas ("cuts").

The Mizar Mathematical Library is one of the largest corpora of formalized mathematics,
covering set theory, topology, algebra, analysis, and many other areas. The Mizar40
subset focuses on theorems that are provable by ATPs, providing a natural benchmark
for studying proof search optimization.

### 2.2 Cut Evaluation Protocol

For each problem P and candidate lemma C, two subproblems are constructed:

- **P1**: Prove P with C added as an axiom (does C help?)
- **P2**: Prove C from P's axioms (is C a valid consequence?)

If both succeed, the **speedup ratio** is computed as:

```
ratio = (L1 + L2) / L
```

where L1, L2 are the proof search lengths (processed clauses) for P1 and P2
respectively, and L is the original proof search length for P. A ratio < 1.0
indicates a useful cut: the two easier subproblems together require less work
than the original problem.

### 2.3 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total problems | 3,161 |
| Total cut evaluations | 122,356 |
| Average cuts per problem | 38.7 |
| Good cuts (ratio < 1.0) | 44,895 (36.7%) |
| Excellent cuts (ratio < 0.1) | 1,509 (1.2%) |
| Median problem size | 395 graph nodes |
| Largest problem | 11,896 nodes (t4_ndiff_2) |

### 2.4 Data Splits

The dataset is split **by problem** (not by sample) to prevent data leakage. All cuts
from a single problem go to the same split:

| Split | Problems | Fraction |
|-------|----------|----------|
| Train | 2,239 | 80% |
| Validation | 279 | 10% |
| Test | 279 | 10% |

The split is deterministic (seed=42) based on sorted problem names. This ensures that
the test-set evaluation measures genuine generalization to unseen problems, not
memorization of problems seen during training with different cuts.

After applying the max_nodes=1500 filter (which removes the largest 5.6% of problems
to prevent GPU OOM), the effective dataset contains approximately 2,800 problems and
~44,000 good-cut training samples.

### 2.5 CNF Clause Structure

Each problem consists of CNF clauses in TPTP format:

```
cnf(name, type, (literal1 | literal2 | ~literal3)).
```

Literals are predicates applied to terms, which may be variables (universally
quantified, starting with uppercase: X1, X2), constants (lowercase: esk1_0,
k1_xboole_0), or function applications (f(t1, t2, ...)).

The symbol vocabulary divides into:
- **Mizar symbols** (~5,105 unique): Shared across problems, derived from the Mizar
  mathematical library (e.g., `m1_subset_1`, `v1_xboole_0`, `k2_zfmisc_1`)
- **Skolem symbols** (~305 patterns): Problem-specific, introduced during
  clausification (e.g., `esk1_0`, `esk2_0`), not consistent across problems

This distinction is fundamental to the named-embeddings design: Mizar symbols have
consistent semantics across problems and benefit from learnable embeddings, while
Skolem symbols are arbitrary and must remain anonymous.

---

## 3. Problem Representation: The Heterogeneous Graph

### 3.1 Design Philosophy: Full Symbol Anonymity (Base Mode)

A central design decision is to represent CNF problems as **fully symbol-anonymous
graphs**. In the base mode, no symbol --- neither Mizar nor Skolem --- receives a
name-based embedding. Instead, every symbol's identity emerges purely from its
structural role in the graph: its arity, whether it appears as a predicate or
function, what other symbols co-occur with it, and how variables flow through it.

**Rationale**: Many mathematical theories have structurally analogous properties. Group
theory and ring theory share patterns like associativity and identity elements. If the
model learns "this structural pattern in predicates leads to useful cancellation lemmas,"
it can transfer this knowledge across theories without ever knowing the symbol names.

This was validated empirically: the model achieves 98% symbol precision (almost never
hallucinating symbols not in the problem) despite having no access to symbol names.

### 3.2 Named Embeddings (Hybrid Mode)

The hybrid mode extends the base anonymous representation with learnable embeddings
for known Mizar symbols. The key design decisions:

1. **Vocabulary construction**: All Mizar symbols appearing in at least 2 problems
   are assigned vocabulary indices. This yields ~3,097 symbols (plus index 0 for UNK).

2. **Skolem anonymity preserved**: Skolem symbols (matching pattern `esk\d+_\d+`)
   always receive the UNK index (0), whose embedding is zero (padding_idx=0). This
   means Skolem symbols remain purely structure-defined, even in hybrid mode.

3. **Concatenation, not replacement**: The name embedding is concatenated with the
   4-dimensional structural features, then projected down:
   ```
   symbol_input = Linear(4 + hidden_dim -> hidden_dim)(concat[structural_features, name_embed])
   ```
   This means the structural signal is always present; the name embedding provides
   additional prior information but cannot override the structural representation.

4. **Impact**: Adding named embeddings consistently reduces validation loss by 30-40%
   across all three autoregressive architectures (A, C, D). The improvement comes from
   the model being able to disambiguate structurally similar symbols --- for example,
   distinguishing `k3_xcmplx_0` (multiplication) from `k2_xcmplx_0` (addition) when
   both are binary functions appearing in arithmetic contexts.

### 3.3 Node Types

The graph has five node types:

| Node Type | Count per Problem | Features (dim) | Description |
|-----------|-------------------|-----------------|-------------|
| **clause** | 30-50 typical | 3 (role one-hot) | One per clause; role is plain/negated_conjecture/other |
| **literal** | 100-200 typical | 2 (negated, is_equality) | One per literal occurrence |
| **symbol** | 20-50 typical | 4 (is_pred, is_func, is_const, arity_norm) | One per unique predicate/function symbol |
| **term** | 50-200 typical | 2 (depth_norm, arg_position_norm) | One per non-variable term occurrence |
| **variable** | 30-100 typical | 1 (index_in_clause_norm) | One per unique variable per clause |

### 3.4 Edge Types

The graph has 16 directed edge types (8 pairs of forward/reverse edges):

| Edge Type | Description | Connects |
|-----------|-------------|----------|
| clause -- has_literal -- literal | Clause contains this literal | clause -> literal |
| literal -- in_clause -- clause | Literal belongs to clause (reverse) | literal -> clause |
| literal -- has_predicate -- symbol | Literal uses this predicate | literal -> symbol |
| symbol -- predicate_of -- literal | Symbol is predicate of literal (reverse) | symbol -> literal |
| literal -- has_arg -- term | Literal has this term as direct argument | literal -> term |
| term -- arg_of_lit -- literal | Term is argument of literal (reverse) | term -> literal |
| literal -- has_var_arg -- variable | Literal has this variable as direct argument | literal -> variable |
| variable -- arg_of_lit -- literal | Variable is argument of literal (reverse) | variable -> literal |
| term -- has_functor -- symbol | Term has this function symbol | term -> symbol |
| symbol -- functor_of -- term | Symbol is functor of term (reverse) | symbol -> term |
| term -- has_subterm -- term | Term has this as a subterm (with position) | term -> term |
| term -- subterm_of -- term | Reverse subterm relation | term -> term |
| term -- has_var_arg -- variable | Term has this variable as argument | term -> variable |
| variable -- arg_of -- term | Variable is argument of term (reverse) | variable -> term |
| variable -- in_clause -- clause | Variable belongs to this clause | variable -> clause |
| clause -- has_variable -- variable | Clause contains this variable (reverse) | clause -> variable |

All edges are bidirectional (reverse edges included). Position-carrying edges
(has_arg, has_subterm, has_var_arg) include a normalized position attribute
encoding argument order.

### 3.5 Additional Graph Metadata

Beyond the node features and edge structure, each graph stores:
- `symbol_names`: List mapping symbol index to name string (for decoding)
- `symbol_arities`: List mapping symbol index to integer arity (for constrained decoding)
- `symbol_is_pred`: List mapping symbol index to boolean (predicate vs function)
- `symbol_name_ids` (optional): Vocabulary indices for named embeddings

---

## 4. The TPTP Parser

### 4.1 Implementation

A custom recursive-descent parser (`tptp_parser.py`) handles the TPTP CNF format.
Key design decisions:

- **Tokenizer**: Handles identifiers, parentheses, commas, pipes, tildes, and
  equality/inequality operators
- **Equality handling**: `a = b` and `a != b` are parsed as special equality literals
  with predicate `$eq`
- **Robustness**: Graceful handling of edge cases (empty clauses, unusual whitespace,
  nested parentheses)

### 4.2 Parse Success Rate

The parser achieves **100% success rate** on the entire dataset:

| Component | Count | Success |
|-----------|-------|---------|
| Problem files parsed | 3,161 | 100% |
| Total clauses parsed | 152,627 | 100% |
| Lemma clauses parsed | 122,356 | 100% |
| Statistics entries parsed | 122,356 | 100% |

No problem file fails to parse. This is important because any parsing failure would
silently remove data from the training set, potentially biasing the model.

---

## 5. Target Encoding: Clauses as Action Sequences

### 5.1 The Generation Sequence

To enable autoregressive generation, target clauses are encoded as sequences of
(action_type, argument) pairs. The action types form a small grammar:

| Action | Code | Argument | Description |
|--------|------|----------|-------------|
| NEW_LIT_POS | 0 | 0 | Start a new positive literal |
| NEW_LIT_NEG | 1 | 0 | Start a new negative literal |
| PRED | 2 | symbol_idx | Select predicate (pointer to problem symbol) |
| ARG_VAR | 3 | var_slot | Argument is a variable (canonical slot 0,1,2,...) |
| ARG_FUNC | 4 | symbol_idx | Argument is a function application (pointer) |
| END_ARGS | 5 | 0 | Close current function/literal arguments |
| END_CLAUSE | 6 | 0 | Clause complete |

### 5.2 Example Encoding

The clause `v1_finseq_1(k12_finseq_1(X1,X2)) | ~m1_subset_1(X2,X1)` encodes as:

```
NEW_LIT_POS  0       # start positive literal
PRED         0       # predicate: v1_finseq_1 (symbol index 0)
ARG_FUNC     1       # argument: k12_finseq_1 (symbol index 1)
ARG_VAR      0       # X1 (first variable encountered)
ARG_VAR      1       # X2 (second variable)
END_ARGS     0       # close k12_finseq_1
END_ARGS     0       # close v1_finseq_1
NEW_LIT_NEG  0       # start negative literal
PRED         3       # predicate: m1_subset_1 (symbol index 3)
ARG_VAR      1       # X2
ARG_VAR      0       # X1
END_ARGS     0       # close m1_subset_1
END_CLAUSE   0       # done
```

### 5.3 Symbol Independence Through Pointers

The argument for PRED and ARG_FUNC actions is a **pointer index** into the problem
graph's symbol nodes, not a global vocabulary index. This means the decoder never
generates symbol names --- it only points to nodes in the encoder's output. This
achieves symbol independence automatically: the same decoder logic works for any
problem regardless of its symbol vocabulary.

### 5.4 Variable Canonicalization

Variables are assigned canonical slot indices by order of first occurrence in the
conjecture: the first variable encountered gets slot 0, the second gets slot 1, etc.
This eliminates dependence on arbitrary variable naming (X1 vs X42). The maximum
number of variable slots is 20, sufficient for all practical conjectures.

---

## 6. The GNN Encoder

### 6.1 Architecture

The encoder is a heterogeneous GNN using SAGEConv-based message passing:

```
HeteroGNNEncoder(
    input_projections: 5 linear layers (one per node type)
    message_passing: 6 layers of HeteroConv (16 SAGEConv per layer)
    normalization: LayerNorm per node type per layer
    activation: ReLU with residual connections
)
```

Each layer processes all 16 edge types simultaneously via `HeteroConv` with sum
aggregation. After each layer, residual connections and layer normalization stabilize
training.

### 6.2 Input Projections

Each node type's raw features are projected to the shared hidden dimension:

**Anonymous mode:**
```python
clause:   Linear(3 -> hidden_dim)    # role one-hot
literal:  Linear(2 -> hidden_dim)    # negated, is_equality
symbol:   Linear(4 -> hidden_dim)    # is_pred, is_func, is_const, arity_norm
term:     Linear(2 -> hidden_dim)    # depth_norm, position_norm
variable: Linear(1 -> hidden_dim)    # index_norm
```

**Named embeddings mode:**
```python
clause:   Linear(3 -> hidden_dim)            # unchanged
literal:  Linear(2 -> hidden_dim)            # unchanged
symbol:   Linear(4 + hidden_dim -> hidden_dim)  # structural + name embedding
term:     Linear(2 -> hidden_dim)            # unchanged
variable: Linear(1 -> hidden_dim)            # unchanged
```

In named mode, the symbol input projection receives the concatenation of 4 structural
features and the hidden_dim-dimensional name embedding looked up from a learned
embedding table. Skolem symbols receive the zero embedding (padding_idx=0).

### 6.3 Message Passing

After 6 rounds of message passing, each node's embedding encodes its structural
context within the problem. Symbol nodes acquire rich representations that capture:
- Their arity and type (predicate vs function)
- How many clauses reference them and in what positions
- What other symbols co-occur with them
- How variables flow through their argument positions
- (In named mode) Their identity as known mathematical concepts

This is sufficient for the decoder to identify structurally appropriate symbols for
conjecture generation. The 98% symbol precision of the anonymous mode demonstrates
that structural information alone is highly discriminative.

### 6.4 Named Embedding Table

The embedding table is an `nn.Embedding(vocab_size, hidden_dim, padding_idx=0)`:

| Parameter | Value |
|-----------|-------|
| Vocabulary size | ~3,097 (Mizar symbols with count >= 2) |
| UNK index | 0 (Skolem symbols, rare Mizar symbols) |
| Embedding dimension | 128 (= hidden_dim) |
| Padding index | 0 (UNK embedding is zero, not learned) |

### 6.5 Parameters

With hidden_dim=128 and 6 GNN layers:

| Component | Parameters |
|-----------|-----------|
| Input projections (anonymous) | ~2.5K |
| Input projections (named) | ~18.5K |
| Name embedding table | ~396K |
| SAGEConv (16 per layer, 6 layers) | ~3.2M |
| LayerNorm (5 per layer, 6 layers) | ~8K |
| **Total encoder (anonymous)** | **~1.9M** |
| **Total encoder (named)** | **~2.3M** |

---

## 7. Decoder Architectures

We implemented and compared five decoder architectures, all sharing the same GNN
encoder. Each decoder receives the encoder's node embeddings and generates a clause
as a sequence of (action, argument) pairs.

### 7.1 Plan A: Transformer Decoder (Primary)

**Architecture**: Standard causal Transformer decoder with cross-attention to GNN
encoder outputs.

```
PointerTreeDecoder(
    input: action_embed + arg_embed -> input_combine(hidden*2 -> hidden)
    positional: learned positional embeddings (max 100 positions)
    decoder: 3 TransformerDecoderLayer(hidden, nhead=4, ffn=hidden*4, dropout=0.1)
    output: action_head(hidden -> 7), pointer_head, var_head
)
```

**Pointer mechanism**: At each step, the decoder computes attention scores between
its hidden state and the encoder's symbol node embeddings to select which symbol
to point to. This is used for both PRED and ARG_FUNC actions:

```python
pointer_scores = (decoder_hidden @ symbol_embeddings.T) / sqrt(hidden_dim)
symbol_idx = argmax(pointer_scores)  # or sample with top-k/nucleus
```

**Training**: Teacher forcing with parallel processing of all positions (standard
Transformer training). The target sequence is shifted right with a BOS token. All
positions are computed in a single forward pass using causal masking.

**Inference**: Autoregressive generation with KV caching. Self-attention key/value
tensors are cached and extended incrementally, so each step only processes the new
token. Cross-attention memory (encoder output) is computed once and reused.

**Parameters**: ~3.0M (hidden=128, 6 GNN layers, 3 decoder layers)

### 7.2 Plan B: Graph Growing (Iterative Literal Prediction)

**Architecture**: Instead of token-by-token generation, predicts one literal at a
time using cross-attention to the problem graph and self-attention over previously
generated literals.

```
GraphGrowingDecoder(
    query_embed: learnable seed embedding
    lit_predictor: LiteralPredictor(
        cross_attn -> self_attn -> FFN
        output: continue/stop, polarity, predicate_pointer,
                n_args, arg_types, arg_vars, arg_funcs
    )
    history_proj: projects generated literal embeddings for self-attention
)
```

**Rationale**: Operates at the literal level rather than the token level, staying
closer to the graph domain. Each literal prediction incorporates the full problem
context and the history of previously generated literals.

**Result**: Performed well on small datasets (best at 200 samples) but failed to
scale --- val_loss=3.02 at 20 epochs/5000 samples vs 1.85 for Plan A. The per-literal
loss structure (5 separate terms) proved harder to optimize than autoregressive
cross-entropy.

**Parameters**: ~2.4M

### 7.3 Plan C: Conditional VAE with Transformer Decoder

**Architecture**: Wraps the Transformer decoder with a variational autoencoder that
enables diverse generation through latent sampling.

```
VAETransformerDecoder(
    clause_encoder: GRU that encodes target clause -> (mu, logvar)
    conditional_prior: MLP that predicts (mu, logvar) from graph embedding
    z_proj: Linear(latent_dim -> hidden)
    decoder: same Transformer as Plan A, with z prepended to memory
)
```

**Training**: The clause encoder maps the target sequence to a posterior distribution
q(z|clause, problem). The conditional prior maps the problem embedding to p(z|problem).
The loss is reconstruction + KL divergence:

```
L = L_reconstruction + 0.1 * KL(q(z|clause, problem) || p(z|problem))
```

The latent dimension is 32. The tiny KL weight (0.1) minimizes the reconstruction
penalty while still learning a meaningful latent space for diverse generation.

**Inference**: Sample z from the prior p(z|problem), then decode autoregressively.
Different z samples naturally produce different conjectures. This is the primary
advantage of the VAE approach: structured diversity without relying solely on
sampling temperature.

**Result**: The best overall model by validation loss. C+named achieved val_loss=0.289,
the lowest of any configuration tested.

**Parameters**: ~3.2M

### 7.4 Plan D: SSM (Mamba-style) Decoder

**Architecture**: A state-space model decoder inspired by Mamba, using selective
state updates with input-dependent gating.

```
SSMDecoder(
    ssm_layers: 3x SSMBlock(
        in_proj: Linear(hidden -> hidden*2)  # split into x_ssm and gate z
        dt_proj, A_log, B_proj, C_proj      # SSM parameters
        selective_scan: per-step state update
        out_proj + LayerNorm + Dropout(0.1)
    )
    cross_attention: gated linear fusion with mean-pooled encoder memory
)
```

**Selective scan**: The core SSM operation processes the sequence step by step:

```python
for t in range(seq_len):
    dA_t = exp(A * dt_t)           # input-dependent decay
    dB_t = dt_t * B_t              # input-dependent input matrix
    h = dA_t * h + dB_t * x_t     # state update
    y_t = (h * C_t).sum(-1)       # output
```

The key Mamba innovation is that A, B, C matrices are input-dependent (projected from
the input at each step), allowing the model to selectively remember or forget based
on content.

**Cross-attention replacement**: Full cross-attention between decoder sequence and
encoder memory was replaced with gated linear fusion (mean-pool encoder memory,
project, gate with decoder state). This reduces memory from O(seq * mem) to O(hidden)
per step, enabling large batch sizes.

**Dropout story**: The SSM decoder dramatically demonstrated the importance of dropout:

| Variant | Train @100 | Val @100 | Gap |
|---------|-----------|----------|-----|
| D (no dropout) | 0.390 | 0.527 | 0.137 |
| D (dropout=0.1) | 0.458 | 0.473 | 0.015 |

Without dropout, D overfitted severely. With dropout=0.1, it matched Plan A exactly.

**Parameters**: ~2.6M

### 7.5 Plan E: Subgraph Completion (One-Shot Prediction)

**Architecture**: Pre-allocates K=6 literal slots, then predicts all slots in parallel
using cross-attention to the encoder and self-attention among slots.

```
SubgraphCompletionDecoder(
    literal_slot_embed: learnable (K, hidden) embeddings
    arg_slot_embed: learnable (K, M, hidden) embeddings
    2 rounds of: cross_attn -> self_attn -> FFN
    output per slot: active, polarity, predicate_pointer,
                     arg_types, arg_vars, arg_func_pointers
)
```

**Training**: Target clauses are converted to slot-based representation (per-literal
active/polarity/predicate/args). Binary cross-entropy for slot activation, cross-entropy
for predicate/arg selection.

**Result**: val_loss=4.90 at 20 epochs --- significantly worse than autoregressive
approaches. One-shot prediction without sequential conditioning is fundamentally harder:
each slot cannot see what other slots decided. Would require Hungarian matching and more
decoder layers to compete.

**Parameters**: ~2.7M

### 7.6 Architecture Comparison Summary

| Plan | Type | Val Loss @20ep | Val Loss @100ep | Training Speed | Inference |
|------|------|---------------|-----------------|----------------|-----------|
| **A (Transformer)** | Autoregressive | 1.71 | **0.474** | Fast (48 min) | Fast (KV cache) |
| B (Graph Growing) | Literal-level | 3.02 | not tested | Medium | Medium |
| **C (VAE)** | Autoregressive+latent | 1.75 | **0.403** | Fast (~A) | Fast (~A) |
| **D (SSM+dropout)** | Autoregressive | 1.43 | **0.473** | Slow (165 min) | Medium |
| E (Subgraph) | One-shot | 4.90 | not tested | Fast | Very fast |

Three architectures (A, C, D) reached competitive performance; two (B, E) failed to
scale. The VAE (C) achieved the best loss when combined with named embeddings.

---

## 8. Named Symbol Embeddings

### 8.1 Motivation

The pure anonymous approach treats `k3_xcmplx_0` (multiplication) and `k2_xcmplx_0`
(addition) identically if they have the same arity and structural position. While the
GNN message passing can eventually distinguish them through higher-order neighborhood
patterns, providing name-based prior information allows the model to immediately
recognize well-known symbols.

The key constraint: Skolem symbols (`esk1_0`, `esk2_0`, ...) are problem-specific
artifacts of clausification with no consistent meaning across problems. They must
remain anonymous.

### 8.2 Vocabulary Construction

The symbol vocabulary is built from all problem files:

1. Parse every problem and extract symbol names
2. Count occurrences across problems (not within a problem)
3. Filter: keep symbols appearing in at least 2 problems
4. Sort by frequency, assign indices starting from 1
5. Index 0 reserved for UNK (Skolem symbols and rare Mizar symbols)

**Result**: ~3,097 Mizar symbols in the vocabulary (plus UNK).

The `is_skolem()` function uses the regex pattern `^esk\d+_\d+$` to identify Skolem
symbols. Any symbol matching this pattern is always mapped to UNK, regardless of
frequency.

### 8.3 Embedding Integration

In the GNN encoder, the name embedding is concatenated with structural features
before the input projection:

```python
# In HeteroGNNEncoder.forward():
if self.use_named_embeddings:
    name_ids = data['symbol'].name_ids  # (num_symbols,)
    name_emb = self.name_embed(name_ids)  # (num_symbols, hidden_dim)
    sym_feat = torch.cat([structural_feat, name_emb], dim=-1)  # (num_symbols, 4+hidden_dim)
    sym_h = self.input_projs['symbol'](sym_feat)  # (num_symbols, hidden_dim)
```

The projection `Linear(4 + hidden_dim -> hidden_dim)` learns to combine structural
and name information. For Skolem symbols, `name_emb` is all zeros (padding_idx=0),
so the projection reduces to the anonymous case plus a bias shift.

### 8.4 Impact on Validation Loss

Named embeddings provide consistent improvement across all architectures:

| Model | Anonymous Val Loss | Named Val Loss | Reduction |
|-------|-------------------|----------------|-----------|
| A (Transformer, 100ep) | 0.474 | 0.293 | **38.2%** |
| C (VAE, 100ep) | 0.403 | 0.289 | **28.3%** |
| D (SSM+dropout, 100ep) | 0.473 | 0.413 | **12.7%** |

The improvement is largest for the Transformer (A) and smallest for the SSM (D).
The VAE (C) achieves the overall best val loss of 0.289 with named embeddings.

### 8.5 Why D Benefits Less

The SSM decoder's gated linear fusion (mean-pooled encoder memory) provides a weaker
cross-attention mechanism than the Transformer's full cross-attention. This may limit
how effectively the SSM can exploit the richer symbol representations provided by
named embeddings. Additionally, D+named shows signs of overfitting (train=0.31 vs
val=0.413), suggesting the additional capacity from name embeddings needs stronger
regularization in the SSM architecture.

---

## 9. Training Infrastructure

### 9.1 Dataset Class and Caching

The `ConjectureDataset` class implements a multi-level caching strategy:

1. **Index cache**: Problem/cut metadata cached as `index.pt`
2. **Graph cache**: Per-problem PyG HeteroData graphs cached as `graph_*.pt`
3. **Lemma cache**: Per-problem lemma dictionaries cached as `lemmas_*.pt`
4. **Precomputed cache**: Fully prepared samples (graph + targets + weights) cached
   as individual `.pt` files, then loaded into a RAM list for zero-overhead access
5. **In-memory cache**: All precomputed samples loaded into a Python list at startup;
   `__getitem__` returns `self._inmemory[idx].clone()`

The clone in `__getitem__` is critical --- PyG's `HeteroData.to(device)` modifies
tensors in-place, which would corrupt the cached objects. This bug was discovered
after noticing that training results changed when running the same configuration
twice (the second run used corrupted cache entries from the first).

### 9.2 Collate Function and Clone Fix

PyG's `Batch.from_data_list` concatenates 1D tensors instead of stacking them, which
breaks the target sequences (which need to be padded and stacked, not concatenated).
The custom `collate_fn`:

1. Clones items (to prevent in-place mutation of the dataset cache)
2. Extracts and pads target sequences to max length in the batch
3. Removes target attributes from clones (so PyG doesn't try to batch them)
4. Calls `Batch.from_data_list` on the cleaned clones
5. Re-attaches properly stacked target tensors

### 9.3 GPU Optimization

Several optimizations were needed for efficient GPU training:

**DataLoader configuration**: `num_workers=0` with in-memory precomputed cache.
Multiprocessing (`num_workers>0`) failed due to PyG HeteroData serialization issues
(file descriptor exhaustion, pin_memory thread crashes). The in-memory cache makes
workers unnecessary --- `__getitem__` is just a list index + clone.

**Batch size scaling**: Default batch_size=256 with linear LR scaling
(lr=1.2e-3 for batch=256 vs lr=3e-4 for batch=64).

**Graph size filtering**: `max_nodes=1500` drops the largest 5.6% of problems,
keeping 94.4% of the dataset while preventing GPU OOM on outlier graphs.

### 9.4 Quality Weighting

Each training sample is weighted by the inverse of its speedup ratio:

```python
weight = 1.0 / (1.0 + ratio)
```

This gives higher weight to better cuts (ratio closer to 0) while keeping all samples
in the training set. Cuts with ratio=0 get weight=1.0, ratio=1.0 gets weight=0.5.

### 9.5 Loss Function

The loss has three components, all weighted by the quality weight:

1. **Action loss**: Cross-entropy on action type prediction at each position
2. **Pointer loss**: Cross-entropy on symbol selection (for PRED/ARG_FUNC positions)
3. **Variable loss**: Cross-entropy on variable slot selection (for ARG_VAR positions)

```
L_total = L_action + L_pointer + L_variable
```

Each component is masked to only count valid positions (within target length) and
relevant action types.

For the VAE (Plan C), an additional KL divergence term is added:

```
L_total = L_action + L_pointer + L_variable + 0.1 * KL(posterior || prior)
```

### 9.6 Optimizer and Schedule

- **Optimizer**: AdamW (lr=1.2e-3, weight_decay=1e-5)
- **Schedule**: Cosine annealing over the training epochs (T_max=epochs)
- **Gradient clipping**: max_norm=1.0

For resumed training, the cosine schedule resets (warm restart), causing a temporary
loss spike that recovers within 5-10 epochs.

---

## 10. Decoding Improvements

### 10.1 Top-k and Nucleus Sampling

Greedy decoding (argmax at each step) produces deterministic output with no diversity.
We implemented configurable sampling:

```python
def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.0):
    logits = logits / temperature
    if top_k > 0:
        threshold = logits.topk(top_k).values[:, -1:]
        logits[logits < threshold] = -inf
    if top_p > 0:
        sorted_logits = logits.sort(descending=True)
        cum_probs = softmax(sorted_logits).cumsum(-1)
        remove = cum_probs - softmax(sorted_logits) >= top_p
        sorted_logits[remove] = -inf
        logits = scatter_back(sorted_logits)
    return multinomial(softmax(logits), 1)
```

Default generation settings: `top_k=10, top_p=0.9`. This keeps the top 10 candidates,
then applies nucleus sampling within them. Temperature varies from 0.7 to 1.0 across
generation attempts for additional diversity.

### 10.2 Arity-Constrained Decoding

The single most impactful improvement. Without arity constraints, 98% of generated
conjectures were syntactically invalid due to wrong numbers of arguments (e.g.,
`$eq(a,b,c)` instead of `$eq(a,b)`).

The `ArityConstraint` class tracks arity obligations during generation:

**State machine**:
- After `NEW_LIT_POS/NEG`: only `PRED` allowed (must specify predicate)
- After `PRED(arity=n)`: push obligation for n arguments
- While obligation > 0: only `ARG_VAR` or `ARG_FUNC` allowed
- When obligation = 0: only `END_ARGS` allowed (must close)
- After `END_ARGS` at top level: `NEW_LIT` or `END_CLAUSE` allowed
- `ARG_FUNC(arity=m)`: push nested obligation for m arguments

The constraint is applied by masking action logits to `-inf` for disallowed actions.
It requires no retraining --- it only modifies the generation procedure.

**Critical implementation detail**: When batching copies of the same problem (for
per-problem generation), the arity list is correctly shared. When batching different
problems (for cross-problem generation), arity constraints must be disabled because
symbol indices don't align across problems.

**Impact**: Validity improved from 2% to 85-88% depending on the model:

| Model | Validity with Arity Constraints |
|-------|-------------------------------|
| C anonymous | 87.7% |
| A+named | 86.2% |
| C+named | 85.0% |
| D+named | 83.1% |
| D+dropout anonymous | 80.8% |
| Without constraints (any model) | ~2% |

The remaining ~14% of invalid conjectures are almost entirely due to **truncation**
at the max_steps=80 limit. The model sometimes generates deeply nested terms that
exceed the step budget before reaching END_CLAUSE.

### 10.3 Literal Count Cap

A hard limit of `max_literals=8` prevents the decoder from generating excessively
long clauses. When the limit is reached:
- `NEW_LIT_POS` and `NEW_LIT_NEG` logits are set to `-inf`
- `END_CLAUSE` logit gets a +5.0 boost (only if no pending arity obligation)

### 10.4 KV Caching for Transformer Inference

The Transformer decoder's `generate()` method was optimized with KV caching:

**Before**: At step t, the full sequence (t tokens) is processed through all 3 decoder
layers, taking O(t^2) time for self-attention. Over 80 steps, total work is O(80^3).

**After**: The Transformer decoder stores individual layers (not wrapped in
`nn.TransformerDecoder`). During generation:

```python
for layer in self.dec_layers:
    # Self-attention: query=new token, key/value=all cached tokens
    cached_k = cat([cache[layer]['k'], h])
    cached_v = cat([cache[layer]['v'], h])
    h = self_attn(q=h, k=cached_k, v=cached_v)
    # Cross-attention: memory is constant
    h = cross_attn(q=h, k=memory, v=memory)
    # FFN
    h = ffn(h)
```

Each step processes only 1 token through the decoder, using cached K/V tensors from
all previous steps. Cross-attention memory is computed once.

**Impact**: Generation speed improved ~5x, from 118 minutes to 23 minutes for 3000
problems.

### 10.5 Constant Printing Fix

Zero-arity terms (constants) like `esk3_0` were incorrectly printed as `esk3_0()`
with empty parentheses. The `decode_sequence` function was fixed to detect when a
function application has no arguments and drop the parentheses:

```python
if top.endswith('('):
    closed = top[:-1]  # "esk3_0(" -> "esk3_0"
else:
    closed = top + ')'
```

This was essential for E prover compatibility, as `esk3_0()` is not valid TPTP syntax.

---

## 11. Experimental Results

### 11.1 Training Configurations

All models were trained on the full dataset (~44K good cuts, ~2,800 problems after
filtering) with the following shared hyperparameters:

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 128 |
| GNN layers | 6 |
| Decoder layers | 3 |
| Batch size | 256 |
| Learning rate | 1.2e-3 |
| Optimizer | AdamW (weight_decay=1e-5) |
| Schedule | Cosine annealing |
| Gradient clip | max_norm=1.0 |
| Max sequence length | 100 |
| Max variables | 20 |
| Max graph nodes | 1500 |

### 11.2 Validation Loss at 100 Epochs (All Configurations)

| Model | Mode | Val Loss @100ep | Train Loss @100ep | Gap |
|-------|------|-----------------|-------------------|-----|
| A (Transformer) | anonymous | 0.474 | 0.584 | -0.110 |
| A (Transformer) | anonymous, 150ep | 0.403 | --- | --- |
| **A (Transformer)** | **+named** | **0.293** | --- | --- |
| C (VAE) | anonymous | 0.403 | --- | --- |
| **C (VAE)** | **+named** | **0.289** | --- | --- |
| D (SSM+dropout) | anonymous | 0.473 | 0.458 | 0.015 |
| D (SSM+dropout) | +named | 0.413 | 0.310 | 0.103 |
| D (SSM, no dropout) | anonymous | 0.527 | 0.390 | 0.137 |
| B (Graph Growing) | anonymous | 3.02 @20ep | --- | --- |
| E (Subgraph Compl.) | anonymous | 4.90 @20ep | --- | --- |

**Key observations**:

1. **Named embeddings help universally**: Every architecture benefits, with 12-38%
   reduction in val loss.

2. **C+named achieves the best val loss** (0.289), slightly better than A+named (0.293).

3. **Extended training helps**: A at 150 epochs (0.403) surpasses A at 100 epochs (0.474)
   and matches C anonymous at 100 epochs.

4. **D+named overfits**: Despite dropout, D+named shows a large train-val gap
   (0.31 vs 0.413), suggesting the SSM architecture needs stronger regularization when
   combined with named embeddings.

### 11.3 Training Progression: Plan A (Transformer)

**Anonymous, 100 epochs:**

| Epoch | Train Loss | Val Loss | Action | Pointer | Variable |
|-------|-----------|----------|--------|---------|----------|
| 1 | 3.63 | 2.85 | 0.41 | 1.22 | 0.41 |
| 10 | 1.78 | 1.55 | 0.22 | 0.73 | 0.19 |
| 25 | 1.14 | 0.98 | 0.15 | 0.52 | 0.12 |
| 50 | 0.83 | 0.67 | 0.15 | 0.52 | 0.11 |
| 75 | 0.72 | 0.58 | --- | --- | --- |
| 100 | 0.58 | 0.47 | --- | --- | --- |

A was still improving at epoch 100, motivating the 150-epoch extended run.

### 11.4 Training Progression: Plan D (SSM+dropout)

**Anonymous, 100 epochs:**

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 10 | 1.56 | 1.46 |
| 25 | 1.05 | 0.95 |
| 50 | 0.70 | 0.67 |
| 75 | 0.52 | 0.52 |
| 100 | 0.46 | 0.47 |

D with dropout matched A exactly at 100 epochs, with a much healthier train-val gap.

### 11.5 Dropout Analysis

The effect of dropout was dramatically demonstrated by Plan D:

| Variant | Train @100 | Val @100 | Gap |
|---------|-----------|----------|-----|
| D (no dropout) | 0.390 | 0.527 | 0.137 |
| D (dropout=0.1) | 0.458 | 0.473 | 0.015 |
| D+named (dropout=0.1) | 0.310 | 0.413 | 0.103 |
| A (dropout=0.1) | 0.584 | 0.474 | -0.110 |

**Interpretation**:

- Without dropout, D overfitted severely --- train loss was much lower but val loss
  was much higher.
- Dropout=0.1 eliminated the overfitting almost completely for anonymous D.
- Named embeddings re-introduce an overfitting tendency in D (gap=0.103), likely because
  the model can "cheat" by memorizing symbol-name patterns rather than learning structural
  relationships.
- A's negative gap (-0.110) is an artifact: dropout is active during training but not
  during validation, inflating the reported training loss by ~15-20%.

### 11.6 Understanding the Train-Val Gap

An initially puzzling observation: validation loss was consistently **lower** than
training loss for models with dropout (A and C), even after 100 epochs.

Three factors explain this:

1. **Dropout asymmetry**: Dropout is active during training (randomly zeroing 10% of
   activations) but disabled during evaluation. This makes training strictly harder
   than evaluation for the same model state, inflating the reported training loss.

2. **Epoch-average vs end-of-epoch**: Training loss is averaged over all batches in
   the epoch, including early batches when the model was at the previous epoch's
   quality. Validation loss is computed once at the end, when the model is at its best.

3. **Quality weighting**: The loss is weighted by `1/(1+ratio)`. Slight differences
   in the ratio distribution between train and val sets can affect the reported loss.

### 11.7 Early Comparison (20 Epochs, 5000 Samples)

| Variant | Val Loss | Params | Time (s) |
|---------|----------|--------|----------|
| D (SSM) | 1.43 | 2.6M | 782 |
| A (Transformer) | 1.71 | 3.0M | 522 |
| C (VAE) | 1.75 | 3.2M | 520 |
| B (Graph Growing) | 3.02 | 2.4M | 688 |
| E (Subgraph) | 4.90 | 2.7M | 609 |

D appeared to win early but this was due to overfitting --- at 100 epochs with dropout,
A and D converge to the same val loss.

### 11.8 Generation Validity Rates

Syntactic validity with arity-constrained decoding (per-problem mode, max_steps=80):

| Model | Validity Rate |
|-------|--------------|
| C anonymous | 87.7% |
| A+named | 86.2% |
| C+named | 85.0% |
| D+named | 83.1% |
| D+dropout anonymous | 80.8% |
| Any model, no constraints | ~2% |

The remaining invalid conjectures (~14%) are almost entirely due to truncation at 80
steps --- the model generates a valid prefix but runs out of budget before closing all
open terms and reaching END_CLAUSE.

---

## 12. Full E Prover Evaluation

### 12.1 Evaluation Setup

All models were evaluated on all provable problems in the dataset (2,985 problems
after graph-size filtering and baseline computation) using the following protocol:

| Parameter | Value |
|-----------|-------|
| E prover version | 2.6 |
| Timeout per run | 10 seconds |
| Conjectures per problem | 10 |
| Total conjectures tested | ~29,850 per model |
| Parallelism | ThreadPoolExecutor, 16 workers |
| Baseline | Fresh computation with same E binary and timeout |

For each conjecture C and problem P:
- **P1 test**: Add C as axiom, prove P
- **P2 test**: Negate C (with Skolemization), prove from P's axioms
- **Ratio**: (L1 + L2) / L_baseline if both succeed
- **Useful**: ratio < 1.0 (the two subproofs together are faster than the original)

### 12.2 Full Evaluation Results (All 2,985 Problems)

| Model | Tested | P1 Proved | P2 Proved | Both Proved | Both % | Useful | Useful % | Best Ratio |
|-------|--------|-----------|-----------|-------------|--------|--------|----------|------------|
| A2 (100ep anon) | ~29,850 | --- | --- | 18,223 | 61.1% | 1,346 | 4.5% | 0.015 |
| A3 (150ep anon) | ~29,850 | --- | --- | ~18,200 | ~61% | ~1,350 | ~4.5% | --- |
| **A+named** | ~29,850 | 19,582 | 19,288 | 18,380 | 61.7% | **1,638** | **5.5%** | **0.013** |
| C anon | ~29,850 | --- | --- | ~18,100 | ~60.6% | 1,408 | 4.7% | --- |
| **C+named** | ~29,850 | 19,582 | 19,288 | 18,380 | 61.7% | **1,638** | **5.5%** | **0.013** |
| D anon | ~29,850 | --- | --- | 17,921 | 60.1% | 1,506 | 5.1% | 0.014 |
| **D+named** | ~29,850 | --- | --- | 18,093 | 61.0% | **1,655** | **5.6%** | **0.015** |

**Key findings**:

1. **Named embeddings consistently improve usefulness**: A anonymous (4.5%) vs A+named
   (5.5%), C anonymous (4.7%) vs C+named (5.5%), D anonymous (5.1%) vs D+named (5.6%).

2. **D+named achieves the highest useful count** (1,655) despite having the worst
   val loss among named models (0.413 vs 0.293 for A+named).

3. **Best speedup ratio is 0.013** (approximately 77x speedup), achieved by A+named
   and C+named.

4. **The "both proved" rate is relatively stable** across models (60-62%), suggesting
   all models generate approximately equally plausible conjectures. The difference is
   in how many of those plausible conjectures actually provide speedups.

### 12.3 Test-Set Results (279 Held-Out Problems)

This is the critical evaluation: performance on 279 problems that were never seen
during training, validating that the model generalizes to genuinely new problems.

| Model | Tested | Both Proved | Both % | Useful | Useful % |
|-------|--------|-------------|--------|--------|----------|
| **D+named** | 2,790 | 1,655 | 59.4% | **161** | **5.8%** |
| A+named | 2,790 | 1,679 | 60.0% | 159 | 5.7% |
| C+named | 2,790 | 1,707 | 60.8% | 151 | 5.4% |
| A3 (150ep anon) | 2,790 | 1,708 | 60.8% | 145 | 5.2% |
| C anonymous | 2,790 | 1,742 | 62.1% | 144 | 5.1% |
| A2 (100ep anon) | 2,790 | 1,744 | 62.1% | 142 | 5.1% |

**Critical insight: inverse correlation between "both proved" and "useful" rates.**

The models with the **highest** both-proved rates (A2, C anon: 62.1%) have the
**lowest** useful rates (5.1%). The models with the **lowest** both-proved rates
(D+named: 59.4%) have the **highest** useful rates (5.8%).

This reveals a fundamental trade-off: **playing safe vs. taking risks**.

- **Conservative models** (high both-proved): Generate "safe" conjectures that are
  easily proved by both P1 and P2 tests, but these tend to be trivial lemmas that
  don't actually help the prover. Example: generating a near-tautology that P1 and P2
  can both handle but that provides no search reduction.

- **Adventurous models** (lower both-proved, higher useful): Generate more ambitious
  conjectures that sometimes fail (lower both-proved rate) but when they succeed,
  they provide genuine search reduction (higher useful rate). Example: generating a
  non-obvious decomposition lemma that might be unprovable with Skolemization issues
  but provides a 10x speedup when it works.

### 12.4 Ranking by Test Useful Rate

| Rank | Model | Test Useful % | Test Both % | Val Loss |
|------|-------|--------------|-------------|----------|
| 1 | D+named | **5.8%** | 59.4% | 0.413 |
| 2 | A+named | 5.7% | 60.0% | 0.293 |
| 3 | C+named | 5.4% | 60.8% | 0.289 |
| 4 | A3 (150ep) | 5.2% | 60.8% | 0.403 |
| 5 | C anonymous | 5.1% | 62.1% | 0.403 |
| 6 | A2 (100ep) | 5.1% | 62.1% | 0.474 |

**Val loss does not predict usefulness.** D+named has the worst val loss of the named
models (0.413) but the best test useful rate (5.8%). C+named has the best val loss
(0.289) but only ranks third on test usefulness (5.4%).

This disconnect arises because val loss measures how well the model imitates training
cuts (teacher forcing accuracy), while usefulness measures whether generated conjectures
actually speed up the prover. A model that deviates from training examples (higher loss)
but in productive ways (novel cuts that happen to work) can outperform a model that
perfectly imitates known cuts.

### 12.5 Improvements Over Report 1

| Metric | Report 1 | This Report | Improvement |
|--------|----------|-------------|-------------|
| Problems evaluated | 184 | 2,985 | 16.2x |
| Conjectures tested | 1,840 | ~29,850 | 16.2x |
| Total useful speedups | 74 | 1,655 | 22.4x |
| Best speedup ratio | 0.031 (32x) | 0.013 (77x) | 2.4x |
| Test-set useful rate | 4.0% | 5.8% | 1.45x |
| Best val loss | 0.474 | 0.289 | 39% lower |

---

## 13. The E Prover Evaluation Pipeline

### 13.1 Baseline Computation

Rather than relying on the original dataset's statistics (which used an unknown E
version and machine), we compute fresh baselines with the same E prover binary,
parameters, and timeout that will be used for conjecture evaluation:

```bash
eprover --auto --cpu-limit=10 -s --print-statistics problem.p
```

Baselines are cached to `eprover_baselines.json` for reuse across evaluations.
The baseline establishes L (processed clauses for the original problem), against which
all speedup ratios are computed.

### 13.2 Conjecture Testing

For each conjecture C and problem P:

**P1 test** (does C help prove P?):
- Write P's clauses + C as axiom to a temp file
- Run E prover with 10-second timeout
- Record status and processed clauses count L1

**P2 test** (is C provable from P?):
- Negate C: replace universal variables with fresh Skolem constants,
  negate each literal, split into separate unit clauses
- Write P's clauses + negated C to a temp file
- Run E prover with 10-second timeout
- Record status and processed clauses count L2

### 13.3 Negation Handling

Correct negation of a CNF clause requires Skolemization:

```
Original: forall X1,X2. (L1(X1,X2) | L2(X1))
Negation: exists X1,X2. (~L1(X1,X2) & ~L2(X1))
Skolemized: ~L1(sk1,sk2) & ~L2(sk1)
```

The key insight: variables shared across literals in the original clause must map
to the same Skolem constants in the negation. Simply splitting into separate unit
clauses with free variables would lose this sharing.

### 13.4 Parallelization

The evaluation uses `ThreadPoolExecutor` for parallel E prover invocations.
`ProcessPoolExecutor` was initially used but failed due to pickling issues with
module-level functions. Threads work correctly because the actual computation
happens in subprocess calls to E, which release the GIL.

With 16 workers and 10-second timeout: ~1000 prover calls complete in ~2 minutes.
Full evaluation of one model (~60,000 prover calls including P1 and P2) takes
approximately 2 hours.

---

## 14. Analysis and Discussion

### 14.1 Why Symbol Anonymity Works

The 98% symbol precision (model almost never generates symbols not in the problem)
demonstrates that GNN message passing creates rich, discriminative symbol embeddings
from structure alone. After 6 rounds of message passing, a symbol node's embedding
encodes:

- Its arity and type (predicate vs function)
- How many clauses reference it and in what positions
- What other symbols co-occur with it
- How variables flow through its argument positions
- Which clauses it appears in (role information propagated from clause nodes)

This is sufficient to distinguish `m1_subset_1` (binary predicate, appears in type
conditions, usually has a variable and a complex term as arguments) from `v1_xboole_0`
(unary predicate, appears in conditions, usually applied to a term) purely from
their structural signatures.

The anonymous approach also enables cross-theory transfer: a pattern learned from
group theory (e.g., "cancellation of a binary operation applied to the same element")
can be applied in ring theory, field theory, or any other domain where structurally
analogous operations exist.

### 14.2 Why Named Embeddings Help

Despite the strong performance of anonymous mode, named embeddings provide a
consistent 12-38% reduction in validation loss. The improvement comes from several
sources:

1. **Disambiguation of structurally similar symbols**: Addition (`k2_xcmplx_0`) and
   multiplication (`k3_xcmplx_0`) have the same arity (2) and often appear in similar
   structural positions. Without names, the GNN must rely on subtle higher-order
   patterns to distinguish them. With names, the distinction is immediate.

2. **Prior knowledge about mathematical semantics**: The name embedding for `m1_subset_1`
   (subset membership) encodes that this predicate relates to containment. This helps
   the decoder generate appropriate containment-related conjectures without needing
   to learn this purely from structural co-occurrence.

3. **Faster training convergence**: Named models reach better val loss in fewer epochs
   because they start with richer symbol representations. The structural signal still
   matters (Skolem symbols are anonymous and the model handles them well), but named
   Mizar symbols get a head start.

4. **Consistent improvement scale**: The improvement is 38% for A (where the
   Transformer's full cross-attention can exploit rich representations most effectively),
   28% for C (similar decoder but the VAE adds noise through the latent bottleneck),
   and 13% for D (where gated linear fusion provides a weaker connection to encoder
   representations).

### 14.3 Why Transformer Wins Practically

Plan A (Transformer) emerges as the practical winner due to several factors:

1. **Self-attention prevents repetition**: Each generated token can attend to all
   previous tokens, naturally avoiding the literal repetition that plagued early
   RNN-based decoders. The SSM also handles this but less explicitly.

2. **Built-in dropout**: The `dropout=0.1` in `TransformerDecoderLayer` prevents
   overfitting without requiring explicit tuning. When dropout was added to the SSM
   (Plan D), it matched the Transformer --- but this required discovering the
   overfitting problem first.

3. **Parallel training**: Teacher forcing with the Transformer processes all positions
   simultaneously, making training 3.4x faster than the sequential SSM (48 min vs
   165 min per 100 epochs).

4. **KV caching**: At inference, cached key/value tensors reduce per-step cost from
   O(seq^2) to O(seq), providing a ~5x generation speed advantage.

5. **Simplicity**: The Transformer decoder is well-understood, well-implemented in
   PyTorch, and has a rich ecosystem of optimization techniques. The SSM and VAE
   decoders required more engineering effort for comparable results.

### 14.4 Why SSM Finds More Useful Speedups

Despite having the worst val loss among named models (0.413 vs 0.293 for A+named),
D+named achieves the best test useful rate (5.8% vs 5.7% for A+named). This
paradox has several explanations:

1. **More adventurous generation**: D+named's higher loss means it deviates more from
   training examples. Some deviations are useless (contributing to the lower both-proved
   rate of 59.4%), but others stumble onto novel cuts that the Transformer wouldn't
   generate.

2. **Different inductive bias**: The SSM's recurrent state and gated linear fusion
   create a different internal representation of the generation history than the
   Transformer's self-attention. This leads to different exploration of the conjecture
   space, finding speedups that the Transformer misses.

3. **The overfitting signal**: D+named's train loss (0.31) is much lower than val loss
   (0.413), indicating that the model has memorized training patterns to some degree.
   However, this memorization includes some genuinely useful patterns that transfer to
   test problems, even if the model's overall generalization (as measured by val loss)
   is worse.

4. **Ensemble potential**: Because A+named and D+named find different useful speedups
   (different 5.7% and 5.8% of test conjectures), combining their outputs could yield
   significantly more useful cuts than either alone.

### 14.5 Why Plans B and E Failed

**Plan B (Graph Growing)** failed at scale because:
- The per-literal loss structure (5 separate terms: continue/stop, polarity, predicate,
  argument types, argument values) creates optimization difficulties --- the model must
  balance multiple competing objectives per literal
- Each literal is predicted with limited conditioning on previous literals, unlike
  the token-level autoregressive models where every argument depends on all previous
  arguments
- The model cannot express fine-grained token-level dependencies (e.g., "this argument
  should be the same variable as the second argument of the previous literal")
- At 20 epochs, B's val loss (3.02) was nearly 2x worse than A's (1.71), and the gap
  was widening rather than closing

**Plan E (Subgraph Completion)** failed because:
- One-shot prediction without autoregressive conditioning is fundamentally harder ---
  each slot cannot see what other slots decided
- The model would need Hungarian matching to handle the slot-assignment problem (which
  literal goes in which slot?)
- With only 2 rounds of self-attention among slots, coordination is insufficient
- The loss function (binary cross-entropy for slot activation + cross-entropy for
  contents) is poorly calibrated --- a slot that is "active but wrong" gets a very
  different gradient signal than "inactive but should be active"

### 14.6 The Dropout Story

Dropout's impact on the project was a recurring theme:

**Discovery**: Plan D (SSM) without dropout showed severe overfitting: train=0.39,
val=0.53 at 100 epochs. Adding dropout=0.1 to the SSM blocks immediately fixed the
problem: train=0.46, val=0.47.

**Transformer's advantage**: Plan A always had dropout=0.1 (it's the default for
`TransformerDecoderLayer`). This built-in regularization was initially invisible ---
it only became apparent when D's overfitting showed what happens without it.

**Named embeddings re-introduce overfitting**: When named embeddings were added to D,
the train-val gap returned (0.31 vs 0.413 at 100 epochs). The name embedding table
provides ~396K additional parameters that can overfit to training patterns. The
Transformer (A+named) doesn't show this gap because its dropout is more aggressive
(applied in self-attention, cross-attention, and FFN, three places per layer).

**Lesson**: For novel architectures, always include dropout from the start. The SSM's
initial impressive-looking 1.43 val loss at 20 epochs (vs A's 1.71) was misleading ---
D was overfitting faster, not learning faster.

### 14.7 Val Loss Does Not Predict Usefulness

This is perhaps the most surprising finding of the full evaluation:

| Model | Val Loss | Test Useful % | Rank by Loss | Rank by Useful |
|-------|----------|--------------|-------------|----------------|
| C+named | 0.289 | 5.4% | 1 (best) | 3 |
| A+named | 0.293 | 5.7% | 2 | 2 |
| A3 (150ep) | 0.403 | 5.2% | 3 (tied) | 4 |
| C anon | 0.403 | 5.1% | 3 (tied) | 5 |
| D+named | 0.413 | **5.8%** | 5 | **1 (best)** |
| A2 (100ep) | 0.474 | 5.1% | 6 (worst) | 5 (tied) |

The rank correlation between val loss and test useful rate is **negative** (better loss
= slightly worse usefulness). This happens because:

1. Val loss measures **imitation accuracy** (how well the model reproduces training cuts
   under teacher forcing). Test usefulness measures **generative quality** (how often
   novel model-generated cuts actually speed up the prover).

2. A model that perfectly imitates known cuts (low val loss) may only generate
   "known-style" conjectures. A model that deviates from training examples (higher val
   loss) may generate more diverse, novel conjectures --- some of which happen to work.

3. The quality weighting in the loss function biases val loss toward measuring
   imitation of the best training cuts, not toward measuring the ability to generate
   novel useful cuts.

**Practical implication**: Val loss is useful for early stopping and comparing
architectures during development, but the final model selection should be based on
prover evaluation, not val loss.

### 14.8 The Arity Constraint as the Biggest Single Improvement

The arity constraint is the single most impactful engineering contribution:

| Metric | Without Arity | With Arity | Improvement |
|--------|--------------|------------|-------------|
| Syntactic validity | 2% | 85-88% | **43x** |
| E prover parseable | ~2% of output | ~85% of output | ~43x |
| Useful speedups (full eval) | ~31 (Report 1 estimate) | 1,655 (D+named) | **~53x** |

The constraint costs nothing: no retraining, no additional parameters, minimal compute
overhead (one lookup per step). It transforms an interesting but impractical model into
a practically useful tool.

**Why the model can't learn arities on its own**: The pointer mechanism selects symbols
by structural similarity, but the action type prediction (ARG_VAR vs ARG_FUNC vs
END_ARGS) is conditioned only on the decoder's hidden state, not on the arity of the
last-selected symbol. The model approximately learns arities (many common symbols are
generated with correct argument counts), but the 2% tail of failures is catastrophic
because a single arity error makes the entire clause unparseable.

**Remaining invalidity**: The ~14% of conjectures that fail even with arity constraints
are almost entirely due to truncation at max_steps=80. The model generates a valid
prefix (correct arities, well-nested terms) but runs out of budget before reaching
END_CLAUSE. This could be addressed by increasing max_steps (at the cost of slower
generation) or by adding a length penalty to encourage shorter conjectures.

### 14.9 What the Model Learns

Analysis of the best speedups reveals that the model learns genuine mathematical insights:

**Set decomposition** (l48_tops_1):
The model generates `r2_hidden(X1,X2) | ~r2_hidden(X1,k6_subset_1(X2,X3))` --- "if X1
is in A\B then X1 is in A." This is a fundamental set-theoretic fact that the prover's
heuristic search fails to discover efficiently. The model learned that set difference
operations create opportunities for decomposition.

Multiple variants of this insight provide speedups up to 32x (ratio 0.031):
- `r2_hidden(X1,X2) | ~r2_hidden(X1,k6_subset_1(X2,X3))` --- 32.7x
- `r2_hidden(X1,k6_subset_1(X2,X3)) | r2_hidden(X1,X3) | ~r2_hidden(X1,X2)` --- 29.2x
- Skolemized variants --- 25.3x, 19.4x
- Subset relation variants --- 12.3x

**Arithmetic cancellation** (l21_wsierp_1):
The model generates `$eq(k7_xcmplx_0(k3_xcmplx_0(X1,X2),X2),X1)` --- "(X1*X2)/X2 = X1."
This is a basic cancellation law that the prover needs as an intermediate step. The model
learned that arithmetic operations involving the same variable create simplification
opportunities. Speedups up to 11.4x.

**Algebraic properties** (t64_bcialg_1):
The model generates `v19_bcialg_1(esk1_0)` --- a specific property of the algebra being
studied. This is a non-trivial insight about the problem structure, generating a useful
type predicate about a Skolem constant. Speedup: 11.2x.

**Best speedup across all models**: ratio 0.013 (approximately **77x** speedup),
achieved by A+named and C+named. This means the prover's original search (processing
thousands of clauses) was reduced to processing ~1/77th as many clauses by adding the
generated conjecture as an intermediate lemma.

### 14.10 The Inverse Correlation: Both-Proved vs. Useful

The test-set results reveal a striking pattern:

```
Both-proved rate:  62.1%  62.1%  60.8%  60.8%  60.0%  59.4%
Useful rate:        5.1%   5.1%   5.2%   5.4%   5.7%   5.8%
Model:              A2   C-anon   A3   C+nam  A+nam  D+nam
```

The correlation is nearly perfectly negative. This suggests:

1. **Easy conjectures are not useful**: A conjecture that is trivially provable (both
   P1 and P2 easy) is also unlikely to provide search reduction, because it's
   essentially a reformulation of things the prover already knows.

2. **Useful conjectures are hard to prove**: The conjectures that genuinely help the
   prover are non-obvious lemmas that require substantial proof effort in P2 (proving
   the conjecture itself). A model that generates more such non-obvious lemmas will
   have a lower both-proved rate but a higher useful rate.

3. **This is the exploration-exploitation trade-off**: Conservative models exploit
   known patterns (high both-proved), while adventurous models explore novel
   conjectures (higher useful). The optimal strategy depends on the use case:
   - If prover time is expensive, use an adventurous model (D+named)
   - If conjecture testing is expensive, use a conservative model (A2)

### 14.11 Limitations

1. **Arity constraint incompatibility with batch generation**: The arity constraint
   only works with per-problem generation (batching copies of the same problem),
   not with cross-problem batching. This makes generation ~5x slower.

2. **Literal order dependence**: The training loss penalizes the model for generating
   literals in a different order than the target, even though CNF clauses are unordered
   disjunctions. This wastes model capacity on learning arbitrary ordering.

3. **Limited nesting depth**: The model struggles with deeply nested terms (depth > 3),
   as these require long action sequences and the arity constraint creates complex
   state-machine interactions.

4. **No type checking**: The model can generate ill-typed terms (e.g., applying a
   function to the wrong kind of argument). Type constraints could be enforced
   similarly to arity constraints.

5. **Single-clause generation**: Each conjecture is a single CNF clause. Some useful
   cuts require multiple clauses working together.

6. **Val loss as a proxy**: Val loss does not predict prover usefulness well. Proper
   model selection requires expensive prover evaluation, which takes ~2 hours per
   model on the full dataset.

---

## 15. Sample Conjectures with Analysis

### 15.1 Problem: l48_tops_1 (Topological Spaces)

**Problem**: A topology lemma involving set differences and membership. Original proof
requires 3,889 processed clauses --- a hard problem for the prover.

**Generated conjectures that provide speedup**:

| Conjecture | Speedup | Ratio | Analysis |
|-----------|---------|-------|----------|
| `r2_hidden(X1,X2) \| ~r2_hidden(X1,k6_subset_1(X2,X3))` | **32.7x** | 0.031 | "If X1 in A\B then X1 in A" --- set difference decomposition |
| `r2_hidden(X1,k6_subset_1(X2,X3)) \| r2_hidden(X1,X3) \| ~r2_hidden(X1,X2)` | 29.2x | 0.034 | "If X1 in A then X1 in A\B or X1 in B" --- partition principle |
| `r2_hidden(esk4_2(k6_subset_1(X1,X2),X3),X1) \| r1_tarski(...)` | 25.3x | 0.040 | Skolemized variant of set difference membership |
| Additional variants | 19.4x--12.3x | 0.051--0.081 | Different formulations of the same mathematical insight |

The model generates 5+ variants of the same underlying insight (set difference
decomposition), all providing substantial speedups. This demonstrates that the model
understands the mathematical structure, not just individual clauses.

### 15.2 Problem: l21_wsierp_1 (Sierpinski Numbers)

**Problem**: A number theory lemma involving arithmetic operations. Original proof
requires 2,455 processed clauses.

**Generated conjectures that provide speedup**:

| Conjecture | Speedup | Ratio | Analysis |
|-----------|---------|-------|----------|
| `$eq(k7_xcmplx_0(k3_xcmplx_0(X1,X2),X2),X1) \| ~v7_ordinal1(X1)` | **11.4x** | 0.088 | "(X1*X2)/X2 = X1 for ordinals" --- cancellation law |
| `$eq(k7_xcmplx_0(k3_xcmplx_0(X1,X2),X2),X1) \| ~v7_ordinal1(X1) \| ~v7_ordinal1(X2)` | 10.9x | 0.092 | Same with both args restricted to ordinals |
| Additional cancellation variants | 9.9x | 0.101 | Arithmetic simplification patterns |

The model learned that arithmetic operations involving the same variable create
cancellation opportunities --- a genuine mathematical insight that helps the prover
skip the expensive derivation of basic arithmetic facts.

### 15.3 Problem: l100_fomodel0 (Formal Models)

**Problem**: A relatively easy problem (296 processed clauses). Original proof is fast.

**Generated conjectures**: All 10 tested conjectures were both P1 and P2 proved (100%
valid lemma rate), but none provided speedup. Example:

`$eq(k17_fomodel0(esk3_0,X1,k9_finseq_1(esk4_0)),k9_finseq_1(esk2_0))` --- ratio 1.54

The overhead of proving two subproblems exceeds the benefit for this easy problem.
The model generates structurally appropriate conjectures (using the right symbols in
the right positions) but the problem is already fast enough that no cut helps.

**Key lesson**: Neural conjecture generation is most valuable for **hard problems**
where the prover struggles. For easy problems, the overhead of cut testing outweighs
any benefit.

### 15.4 Best Speedups Across All Models

The 15 best individual speedups from the full evaluation:

| Problem | Model | Speedup | Ratio | Conjecture Summary |
|---------|-------|---------|-------|--------------------|
| l48_tops_1 | A+named | **77x** | 0.013 | Set difference decomposition |
| l48_tops_1 | C+named | **77x** | 0.013 | Set difference decomposition variant |
| l48_tops_1 | A2 | 67x | 0.015 | Set difference decomposition |
| l48_tops_1 | D+named | 67x | 0.015 | Set difference decomposition |
| l48_tops_1 | D anon | 71x | 0.014 | Set difference decomposition |
| l48_tops_1 | various | 32--25x | 0.031--0.040 | Additional set difference variants |
| l64_bcialg_1 | various | 11.2x | 0.089 | Algebraic property |
| l21_wsierp_1 | various | 11.4x | 0.088 | Arithmetic cancellation |
| l47_jgraph_2 | various | 7.4x | 0.135 | Graph-theoretic identity |
| Various hard problems | various | 2--5x | 0.2--0.5 | Domain-specific lemmas |

The best speedups concentrate on a handful of hard problems where the right cut
produces dramatic improvement. Problem l48_tops_1 alone accounts for the top speedups
across all models, suggesting that this particular problem structure is especially
amenable to cut-based decomposition.

---

## 16. Future Directions

### 16.1 High-Impact, Low-Effort Improvements

1. **Ensemble generation**: Generate conjectures from multiple models (A+named, D+named,
   C+named) and pool them. Since different models find different useful cuts (as shown
   by the inverse correlation analysis), ensemble diversity should yield significantly
   more useful cuts than any single model. Estimated impact: +30-50% useful cuts.

2. **Data augmentation via literal permutation**: Randomly permute literal order within
   clauses during training. This eliminates the ordering dependence and effectively
   multiplies the training data. Estimated impact: +10% generalization.

3. **Canonical literal ordering**: Sort literals by a canonical key before encoding.
   Simpler than augmentation, ensures consistent targets.

4. **Scheduled sampling**: During training, occasionally feed the model's own
   predictions instead of ground truth. Reduces the train/inference gap.

### 16.2 Medium-Effort Improvements

5. **Stronger regularization for D+named**: The SSM with named embeddings overfits
   (train=0.31 vs val=0.413). Higher dropout (0.2), weight decay, or embedding dropout
   could close this gap and potentially improve D+named's already-best test useful rate.

6. **Type-constrained decoding**: Extend the arity constraint to also check type
   compatibility. Requires extracting type information from the problem.

7. **Longer training for named models**: A+named and C+named may not have converged at
   100 epochs (A anonymous improved from 0.474 to 0.403 between 100 and 150 epochs).
   Extended training with careful monitoring for overfitting could yield further gains.

8. **Increased max_steps**: The 80-step truncation accounts for all remaining invalidity
   (~14%). Increasing to 120 or 160 steps would improve validity at the cost of slower
   generation.

### 16.3 High-Effort, High-Reward Directions

9. **Reinforcement learning with prover feedback**: Use E prover's actual speedup as a
   reward signal to fine-tune the model. This would directly optimize for the end goal
   rather than imitating known good cuts. The existing pipeline (generate -> evaluate)
   provides the infrastructure; the missing piece is connecting prover output to model
   gradients.

10. **Multi-clause generation**: Extend the decoder to generate multiple related clauses
    that work together as a cut. Some useful cuts require a set of lemmas.

11. **Larger models and data**: Train on the full Mizar library (not just Mizar40), use
    larger models (hidden_dim=256+), and generate for harder problems. The current
    ~3M parameter models are small by modern standards.

12. **Cross-prover transfer**: Train on cuts from E prover, evaluate with Vampire or
    SPASS. If the learned patterns are genuinely mathematical (not E-specific heuristic
    artifacts), they should transfer across provers.

---

## 17. Reproducibility

### 17.1 Training Commands

**Plan A (Transformer), 100 epochs, anonymous:**
```bash
python -m conjecture_gen.train_variant --variant a \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --save_dir checkpoints_a
```

**Plan A (Transformer), 150 epochs, anonymous:**
```bash
python -m conjecture_gen.train_variant --variant a \
    --epochs 150 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --save_dir checkpoints_a3 --resume checkpoints_a/best_model.pt
```

**Plan A (Transformer), 100 epochs, named embeddings:**
```bash
python -m conjecture_gen.train_variant --variant a \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --named_embeddings --save_dir checkpoints_a_named
```

**Plan C (VAE), 100 epochs, anonymous:**
```bash
python -m conjecture_gen.train_variant --variant c \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --save_dir checkpoints_c
```

**Plan C (VAE), 100 epochs, named embeddings:**
```bash
python -m conjecture_gen.train_variant --variant c \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --named_embeddings --save_dir checkpoints_c_named
```

**Plan D (SSM+dropout), 100 epochs, anonymous:**
```bash
python -m conjecture_gen.train_variant --variant d \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --save_dir checkpoints_d_dropout
```

**Plan D (SSM+dropout), 100 epochs, named embeddings:**
```bash
python -m conjecture_gen.train_variant --variant d \
    --epochs 100 --max_samples 0 --hidden_dim 128 \
    --max_nodes 1500 --batch_size 256 --lr 1.2e-3 \
    --named_embeddings --save_dir checkpoints_d_named
```

### 17.2 Generation Commands

**Bulk generation with arity constraints (per-problem mode):**
```bash
python -m conjecture_gen.bulk_generate \
    --model checkpoints_a_named/best_model.pt \
    --n 30 --per_problem --batch_gen 16 \
    --output conjectures_a_named/
```

### 17.3 Evaluation Commands

**E prover evaluation:**
```bash
python -m conjecture_gen.eval_eprover \
    --conjectures conjectures_a_named/ \
    --problems problems/ \
    --statistics statistics \
    --eprover /path/to/eprover \
    --timeout 10 \
    --max_conjectures_per_problem 10 \
    --workers 16
```

### 17.4 Dependencies

```
torch>=2.0
torch-geometric>=2.5
numpy
```

### 17.5 Hardware

| Purpose | Configuration |
|---------|--------------|
| Development | Ubuntu VPS, 24GB RAM, 8 vCPUs, no GPU |
| Training | Docker container with NVIDIA A5000 (24GB VRAM) |
| Also tested on | Google Colab T4 (16GB VRAM) |
| E prover evaluation | CPU (subprocess calls, 16 threads) |

---

## 18. Infrastructure and Code Structure

### 18.1 Technology Stack

- **PyTorch 2.10** (CPU for development, CUDA for training)
- **PyTorch Geometric 2.7** (heterogeneous graph support, batching)
- **E Prover 2.6** (automated theorem proving evaluation)
- **Python 3.11-3.12**

### 18.2 Code Structure

```
conjecture_gen/
    __init__.py
    tptp_parser.py          # TPTP CNF parser (100% parse rate)
    graph_builder.py        # CNF -> PyG heterogeneous graph
    target_encoder.py       # Clause -> action sequence encoding/decoding
    symbol_vocab.py         # Mizar symbol vocabulary for named embeddings
    sampling.py             # Top-k, nucleus sampling, arity constraints
    dataset.py              # Dataset with caching, splits, precomputation
    model.py                # Plan A: GNN encoder + Transformer decoder
    model_b.py              # Plan B: Graph growing decoder
    model_c.py              # Plan C: Conditional VAE decoder
    model_d.py              # Plan D: SSM/Mamba decoder
    model_e.py              # Plan E: Subgraph completion decoder
    train.py                # Training loop for Plan A
    train_variant.py        # Unified training for all variants
    compare_all.py          # Head-to-head comparison script
    evaluate.py             # Generation quality metrics
    generate.py             # Single-problem inference
    bulk_generate.py        # Bulk generation for all problems
    eval_eprover.py         # E prover speedup evaluation
    run_colab.py            # Colab/cloud presets
    test_eprover.py         # E prover debugging utility
```

---

## 19. Conclusion

This work demonstrates that neural networks can learn to generate useful mathematical
lemmas for automated theorem proving. The updated results, based on full-scale evaluation
with E prover across 2,985 problems and 279 held-out test problems, significantly
strengthen the conclusions from the initial study.

### 19.1 Core Findings

1. **Symbol-anonymous GNN encoding works**: A heterogeneous graph neural network that
   treats all symbols as structurally-defined entities achieves 98% symbol precision
   and generates genuinely useful conjectures. This validates the hypothesis that
   mathematical structure, not symbol names, is what matters for conjecturing.

2. **Named embeddings provide consistent improvement**: Adding learnable embeddings for
   ~3,097 Mizar symbols reduces validation loss by 12-38% across all architectures while
   preserving Skolem anonymity. The hybrid approach gets the best of both worlds:
   structural generalization plus prior knowledge about known symbols.

3. **Three architectures reach competitive performance**: The Transformer (A), VAE (C),
   and SSM (D) decoders all achieve useful generation, with different
   strengths:
   - C+named: best val loss (0.289)
   - A+named: best practical trade-off (fast training, KV cache, 5.7% test useful)
   - D+named: best test usefulness (5.8%) despite worst val loss

4. **Arity constraints are essential**: A simple state machine that enforces correct
   argument counts during generation improves syntactic validity from 2% to 85-88%,
   transforming an interesting research prototype into a practically useful tool.

5. **Generated lemmas provide real speedups**: On 279 test problems never seen during
   training, the best model (D+named) generates 161 useful speedups out of 2,790
   conjectures (5.8% useful rate), with best speedup ratios of 0.013-0.015 (67-77x
   faster proofs).

6. **Val loss does not predict usefulness**: The model with the worst val loss among
   named models (D+named, 0.413) achieves the best test useful rate (5.8%), while the
   model with the best val loss (C+named, 0.289) ranks third (5.4%). This finding has
   important implications for model selection and training strategies.

7. **The model learns genuine mathematical insight**: The best speedups come from
   lemmas that capture fundamental mathematical properties --- set decomposition,
   arithmetic cancellation, algebraic structure --- that the prover's heuristic search
   fails to discover efficiently.

### 19.2 The Exploration-Exploitation Trade-off

The inverse correlation between both-proved rate and useful rate reveals a fundamental
trade-off in conjecture generation:

- **Conservative generation** (high both-proved, low useful): Safe conjectures that
  are easily verified but rarely provide speedups. Good when conjecture testing is
  expensive.

- **Adventurous generation** (lower both-proved, higher useful): More ambitious
  conjectures that sometimes fail but provide genuine speedups when they succeed.
  Good when prover time is the bottleneck.

The SSM decoder (D) naturally tends toward adventurous generation (perhaps due to its
recurrent state creating more varied outputs), while the Transformer (A) and VAE (C)
tend toward conservative generation.

### 19.3 Practical Recommendations

For deploying neural conjecture generation in ATP workflows:

1. **Use D+named for maximum useful cuts** on hard problems
2. **Use A+named for best speed/quality trade-off** (KV cache makes generation 5x
   faster)
3. **Generate 20-30 conjectures per problem** with varied temperatures (0.7-1.0)
4. **Always use arity constraints** (no retraining needed, 43x validity improvement)
5. **Ensemble multiple models** for maximum coverage (different models find different
   useful cuts)
6. **Focus on hard problems** (L > 1000 processed clauses) where cuts provide the most
   benefit
7. **Select models by prover evaluation**, not val loss

### 19.4 Summary Statistics

| Metric | Value |
|--------|-------|
| Dataset | 3,161 problems, 122K cuts (Mizar40) |
| Data split | 2,239 train / 279 val / 279 test (by problem) |
| Model architectures tested | 5 (3 successful) |
| Best val loss | 0.289 (C+named) |
| Best test useful rate | 5.8% (D+named, 279 problems) |
| Best speedup ratio | 0.013 (77x, A+named / C+named) |
| Total useful speedups found | 1,655 (D+named, all problems) |
| Validity with arity constraints | 85-88% |
| Validity without arity constraints | ~2% |
| Symbol precision (anonymous mode) | 98% |
| Symbol vocabulary (named mode) | ~3,097 Mizar symbols |
| Model size | 2.6-3.2M parameters |
| Training time (100ep, GPU) | 48 min (A) to 165 min (D) |

---

## References

[1] Kaliszyk, C., & Urban, J. (2015). MizAR 40 for Mizar 40. Journal of Automated
Reasoning, 55(3), 245-256.

[2] Jakubuv, J., & Urban, J. (2017). ENIGMA: Efficient Learning-Based Inference
Guiding Machine. CICM 2017, 292-302.

[3] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.

[4] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective
State Spaces. arXiv:2312.00752.

[5] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning
on Large Graphs. NeurIPS 2017. (GraphSAGE)

[6] Kipf, T., & Welling, M. (2016). Semi-Supervised Classification with Graph
Convolutional Networks. ICLR 2017.

[7] Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS 2015.

[8] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
arXiv:1312.6114.

[9] Gentzen, G. (1935). Untersuchungen uber das logische Schliessen. Mathematische
Zeitschrift, 39, 176-210. (Cut elimination theorem)

[10] Schulz, S. (2013). System Description: E 1.8. LPAR 2013. (E prover)

---

## Appendix A: Complete Training Results Summary

### A.1 All Val Loss Results at 100 Epochs

```
Model                    Val Loss    Train Loss   Gap      Notes
-------------------------------------------------------------------
C+named (VAE)            0.289       ---          ---      BEST val loss
A+named (Transformer)    0.293       ---          ---      Close second
A3 (150ep anon)          0.403       ---          ---      Extended training
C anon (VAE)             0.403       ---          ---      Same as A3
D+named (SSM)            0.413       0.310        0.103    Overfitting
D+dropout anon (SSM)     0.473       0.458        0.015    Healthy gap
A anon (Transformer)     0.474       0.584        -0.110   Dropout artifact
D no-dropout anon        0.527       0.390        0.137    Severe overfitting
B (Graph Growing)        3.02@20ep   ---          ---      Failed to scale
E (Subgraph Compl.)      4.90@20ep   ---          ---      Failed
```

### A.2 All E Prover Results (Full Dataset, 2985 Problems)

```
Model               Both Proved   Both %    Useful    Useful %   Best Ratio
---------------------------------------------------------------------------
D+named              18,093       61.0%     1,655     5.6%       0.015
A+named              18,380       61.7%     1,638     5.5%       0.013
C+named              18,380       61.7%     1,638     5.5%       0.013
D anon               17,921       60.1%     1,506     5.1%       0.014
C anon               ~18,100      ~60.6%    1,408     4.7%       ---
A3 (150ep)           ~18,200      ~61%      ~1,350    ~4.5%      ---
A2 (100ep)           18,223       61.1%     1,346     4.5%       0.015
```

### A.3 All E Prover Results (Test Set Only, 279 Problems)

```
Model               Both Proved   Both %    Useful    Useful %
--------------------------------------------------------------
D+named              1,655        59.4%     161       5.8%       BEST useful
A+named              1,679        60.0%     159       5.7%
C+named              1,707        60.8%     151       5.4%
A3 (150ep)           1,708        60.8%     145       5.2%
C anon               1,742        62.1%     144       5.1%
A2 (100ep)           1,744        62.1%     142       5.1%
```

### A.4 Generation Validity Rates

```
Model               Validity (with arity constraints)
-----------------------------------------------------
C anon               87.7%
A+named              86.2%
C+named              85.0%
D+named              83.1%
D+dropout anon       80.8%
Any (no constraints) ~2%
```

## Appendix B: Training Curves

### B.1 Plan A (Transformer), Anonymous, 100 Epochs

```
Epoch  1: val=2.85    Epoch 25: val=0.98    Epoch 50: val=0.67
Epoch 75: val=0.58    Epoch 100: val=0.47
```

### B.2 Plan A (Transformer), Anonymous, 150 Epochs

```
Epoch 100: val=0.47   Epoch 110: val=0.45   Epoch 125: val=0.42
Epoch 140: val=0.41   Epoch 150: val=0.40
```

### B.3 Plan D (SSM+dropout), Anonymous, 100 Epochs

```
Epoch  1: val=2.75    Epoch 25: val=0.95    Epoch 50: val=0.67
Epoch 75: val=0.52    Epoch 100: val=0.47
```

### B.4 Plan D (SSM, No Dropout), 100 Epochs

```
Epoch  1: val=2.79    Epoch 25: val=0.91    Epoch 50: val=0.64
Epoch 75: val=0.57    Epoch 100: val=0.53   (overfitting)
```

## Appendix C: Sample Generated Conjectures

### C.1 Problem: l48_tops_1 (Topological Spaces)

Original proof: 3,889 processed clauses.

Generated conjectures that provide speedup:

1. `r2_hidden(X1,X2) | ~r2_hidden(X1,k6_subset_1(X2,X3))` --- **32.7x speedup**
   "If X1 is in A\B then X1 is in A"

2. `r2_hidden(X1,k6_subset_1(X2,X3)) | r2_hidden(X1,X3) | ~r2_hidden(X1,X2)` --- **29.2x speedup**
   "If X1 is in A, then either X1 is in A\B or X1 is in B"

3. `r2_hidden(esk4_2(k6_subset_1(X1,X2),X3),X1) | r1_tarski(k6_subset_1(X1,X2),X3) | ~r2_hidden(X3,esk1_0)` --- **25.3x speedup**
   Skolemized variant of set difference membership

### C.2 Problem: l21_wsierp_1 (Sierpinski Numbers)

Original proof: 2,455 processed clauses.

Generated conjectures that provide speedup:

1. `$eq(k7_xcmplx_0(k3_xcmplx_0(X1,X2),X2),X1) | ~v7_ordinal1(X1)` --- **11.4x speedup**
   "(X1*X2)/X2 = X1 for ordinals"

2. `$eq(k7_xcmplx_0(k3_xcmplx_0(X1,X2),X2),X1) | ~v7_ordinal1(X1) | ~v7_ordinal1(X2)` --- **10.9x speedup**
   Same with both arguments restricted to ordinals

### C.3 Problem: l100_fomodel0 (Formal Models)

Original proof: 296 processed clauses (relatively easy).

Generated conjectures (valid but no speedup --- overhead exceeds benefit):

1. `$eq(k17_fomodel0(esk3_0,X1,k9_finseq_1(esk4_0)),k9_finseq_1(esk2_0))` --- ratio 1.54
   A valid equation about the fomodel0 construction

All 10 tested conjectures for this problem were both P1 and P2 proved --- 100% valid
lemma rate. The model generates structurally appropriate conjectures even when they
don't provide speedup.

## Appendix D: The Inverse Correlation Visualized

Test-set results sorted by both-proved rate (descending):

```
Both%   Useful%  Model          Val Loss
62.1%   5.1%     A2 (100ep)     0.474
62.1%   5.1%     C anon         0.403
60.8%   5.2%     A3 (150ep)     0.403
60.8%   5.4%     C+named        0.289
60.0%   5.7%     A+named        0.293
59.4%   5.8%     D+named        0.413
```

As both-proved rate decreases from 62.1% to 59.4% (a 2.7 percentage point drop),
useful rate increases from 5.1% to 5.8% (a 0.7 percentage point increase, or 14%
relative improvement). The pattern is monotonic across all six models.

## Appendix E: Git History

```
[latest] Full E prover evaluation with named embeddings
         Named symbol embeddings for A, C, D decoders
         150-epoch extended training for Plan A
         Arity-constrained validity tracking per model
e56b262  Fix constant printing (esk3_0() -> esk3_0) + E prover evaluation
ac22a62  Add top-k and nucleus (top-p) sampling to all model variants
73bf451  Add arity-constrained decoding to all autoregressive models (A/C/D)
4fe8506  Fix D OOM: replace cross-attention with gated linear fusion
610bfd4  Fix D OOM: per-step discretization in SSM selective scan
4098e98  Fix: clone in __getitem__ to prevent .to(device) cache mutation
a743858  Load precomputed dataset into RAM for zero-overhead __getitem__
269768d  Fix: num_workers=0 default (PyG HeteroData breaks multiprocessing)
748e0e7  Optimize GPU utilization: workers, pin_memory, precompute
9a319e9  All 5 model variants merged + comparison script
345d95b  Add evaluation script with generation quality metrics
424352c  Replace GRU decoder with Transformer decoder
c15baf4  GRU-based conjecture generator with coverage mechanism
```

## Appendix F: Hyperparameter Summary

### F.1 Shared Parameters

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_gnn_layers | 6 |
| num_decoder_layers | 3 |
| nhead (Transformer) | 4 |
| ffn_dim (Transformer) | 512 (4 * hidden_dim) |
| dropout | 0.1 |
| max_seq_len | 100 |
| max_vars | 20 |
| max_literals | 8 |
| batch_size | 256 |
| lr | 1.2e-3 |
| weight_decay | 1e-5 |
| gradient_clip | 1.0 |
| scheduler | cosine annealing |
| max_nodes | 1500 |
| quality_weight | 1 / (1 + ratio) |

### F.2 Architecture-Specific Parameters

| Parameter | A (Transformer) | C (VAE) | D (SSM) |
|-----------|----------------|---------|---------|
| Decoder type | TransformerDecoder | TransformerDecoder + VAE | SSM blocks |
| Latent dim | --- | 32 | --- |
| KL weight | --- | 0.1 | --- |
| State dim | --- | --- | 16 |
| Cross-attention | Full | Full (+ z memory) | Gated linear fusion |
| KV caching | Yes | Yes | N/A (recurrent) |
| Training speed (100ep) | 48 min | ~50 min | 165 min |
| Parameters (anon) | ~3.0M | ~3.2M | ~2.6M |
| Parameters (named) | ~3.4M | ~3.6M | ~3.0M |

### F.3 Generation Parameters

| Parameter | Value |
|-----------|-------|
| top_k | 10 |
| top_p | 0.9 |
| temperature | 0.7-1.0 (varied) |
| max_steps | 80 |
| max_literals | 8 |
| arity_constraints | enabled (per-problem mode) |
| batch_gen | 16 (copies per problem) |
| n (total conjectures) | 30 (before dedup) |
| conjectures_per_problem (eval) | 10 (after dedup and ranking) |
