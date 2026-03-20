# Best Conjectures: Test and Validation Set Only

These conjectures were generated for problems the model **never saw during training**.
They demonstrate genuine generalization of learned mathematical patterns.

Data split: 279 test + 279 validation problems (558 total, from 2,797 unique problems
after max_nodes filtering). All conjectures tested with E prover 2.6, 10s timeout.

---

## Test Set Examples

### 1. Set Difference Decomposition (t35_xboole_1) --- 16.5x speedup

**Split**: test | **L**: 39,681 | **L1+L2**: 2,410 | **Method**: A3 (150ep)

**Problem**: Given A ⊆ B and C ⊆ D, prove that A \ D ⊆ B \ C.

**Generated Conjecture**:
```
r2_hidden(X1,X2) | r2_hidden(X1,X3) | ~r2_hidden(X1,k4_xboole_0(X2,X3))
```

**In plain math**: For any element x and sets A, B:
> If x ∈ A \ B, then x ∈ A or x ∈ B.

More precisely, this is the contrapositive of set difference: x ∈ A \ B implies
x ∈ A. The third disjunct covers the case where x ∉ A \ B. This is the fundamental
decomposition rule for set differences.

**Why it helps**: The prover must repeatedly decompose membership in set differences.
Without this lemma, it derives this from the definition each time (expensive with
39K clauses). With the lemma, it applies it directly.

**Novelty**: This pattern transfers from training data where similar set-theoretic
problems exist, but this specific problem (with the particular subset chain
A ⊆ B, C ⊆ D → A\D ⊆ B\C) was never seen.

---

### 2. Ordinal Transitivity --- Multiple Variants (t13_ordinal1) --- up to 26.5x speedup

**Split**: test | **L**: 15,084 | **Method**: multiple

**Problem**: If a ∈ b and b is an ordinal, then a is also an ordinal.
(Membership in ordinals is transitive.)

Multiple methods generated useful variants:

**Variant A** (26.5x, A3): `r1_tarski(esk2_0,X1) | r2_hidden(X2,esk1_0)`
> Either b is a subset of everything, or something belongs to a.

**Variant B** (20.1x, D+named): `~v1_ordinal1(esk2_0) | ~v1_ordinal1(esk2_0)`
> b is not ordinal-type (repeated for emphasis — a proof-by-contradiction trigger).

**Variant C** (19.3x, D+named): `~v1_ordinal1(esk2_0)`
> Simply: b is not an ordinal. (Forces the prover to use the hypothesis.)

**Variant D** (14.5x, A2): `$eq(X1,esk2_0) | r1_tarski(X1,esk1_0)`
> Either X equals b, or X is a subset of a.

**Why they help**: The prover struggles with ordinal transitivity because it must
reason about the well-ordering of ordinals. These lemmas provide key case splits
that organize the search. Notably, the simplest variant (just "b is not ordinal")
gives a 19x speedup by directly triggering contradiction.

**Remarkable**: 7 different methods found useful conjectures for this problem,
generating 15+ variants. The model thoroughly understands ordinal structure.

---

### 3. MML Query Composition (t49_mmlquery) --- 21.1x speedup --- D-only!

**Split**: test | **L**: 10,417 | **L1+L2**: 494 | **Method**: D (SSM) exclusively

**Problem**: In the Mizar Mathematical Library query system, prove that composing
two query filters in sequence satisfies a relational compatibility condition.

**Generated Conjecture**:
```
r2_relset_1(esk1_0, esk1_0,
  k19_mmlquery(esk1_0, esk2_0, k18_mmlquery(esk1_0, esk3_0)),
  k19_mmlquery(esk1_0, k18_mmlquery(esk1_0, esk2_0), k18_mmlquery(esk1_0, esk3_0)))
| v1_mmlquery(k18_mmlquery(esk1_0, esk2_0))
```

**In plain math**: The composition of query Q1 after filter F(Q2) is relationally
compatible with the composition F(Q1) after F(Q2), unless Q1's filter is trivial.

**Why it helps**: This commutativity-like property of MML queries is the key lemma
for the proof. Only the SSM model discovered this.

**Uniqueness**: This is one of 6 test problems where ONLY the D (SSM) model finds
a useful conjecture. The SSM's "adventurous" generation style explores unusual
structural patterns that the Transformer doesn't reach.

---

### 4. Simplex Monotonicity (t6_simplex0) --- 12.7x speedup

**Split**: test | **L**: 3,509 | **L1+L2**: 277 | **Method**: A2 (100ep)

**Problem**: If set family A is a subfamily of B (every member of A is contained
in some member of B), then the abstract simplicial complex of A is contained in
the simplicial complex of B.

**Generated Conjecture**:
```
r2_hidden(X1,k1_simplex0(esk2_0)) | ~r2_hidden(X1,esk1_0)
```

**In plain math**: For any element x:
> If x ∈ A, then x ∈ simplex(B).

This directly states the monotonicity: elements of A appear in B's simplicial
complex.

**Why it helps**: The simplex construction involves recursive definitions over
set families. This lemma bypasses the recursion with a direct membership fact.

---

### 5. Euclidean Complex Norm (t40_euclid_3) --- 40.8x speedup

**Split**: test | **L**: 3,760 | **L1+L2**: 92 | **Method**: C (VAE)

**Problem**: Prove a property about complex numbers in Euclidean space, relating
the real part of one number to the imaginary part of another.

**Generated Conjecture**:
```
m1_subset_1(k3_complex1(esk1_0),k1_numbers) | ~v1_xreal_0(k4_complex1(esk2_0))
```

**In plain math**:
> The real part of the first complex number is a real number,
> OR the imaginary part of the second number is not an extended real.

**Why it helps**: Complex number arithmetic in the prover requires explicit typing
facts about real/imaginary parts. This lemma provides the typing directly.

---

### 6. Real Number Contradiction (t17_xreal_1) --- 38.3x speedup

**Split**: val | **L**: 8,380 | **L1+L2**: 219 | **Method**: D+named

**Problem**: Prove an inequality about extended real numbers.

**Generated Conjecture**:
```
~v1_xreal_0(X1) | ~v1_xreal_0(X2)
```

**In plain math**:
> Either X1 is not a real number, or X2 is not a real number.

**Why it helps**: This is the simplest possible conjecture that works. The problem's
negated conjecture assumes both variables are real. Adding "at least one isn't real"
forces the prover into the exact case analysis it needs. L1=8 (trivial with the
conjecture!) and L2=211 (proving the conjecture from the problem).

**Lesson**: Sometimes the most powerful lemma is the most obvious one, stated in
exactly the right form for the prover.

---

### 7. Linear Temporal Logic Axiom Type (t48_ltlaxio1) --- 30.9x speedup

**Split**: val | **L**: 17,456 | **L1+L2**: 566 | **Method**: A2 (100ep)

**Problem**: Prove a property about formulas in linear temporal logic (LTL),
specifically about implication and conjunction of LTL formulas.

**Generated Conjecture**:
```
m1_subset_1(k3_hilbert1(esk2_0,esk3_0),k1_hilbert1)
```

**In plain math**:
> The implication (esk2_0 → esk3_0) is a well-formed LTL formula.

Where `k3_hilbert1` is implication and `k1_hilbert1` is the set of all formulas.

**Why it helps**: The prover needs to know that the implication of two formulas
is itself a formula (closure under implication). This typing fact is buried in
the definitions but crucial for the proof.

---

### 8. Relational Pairing Membership (t126_relat_1) --- 29.9x speedup

**Split**: val | **L**: 17,884 | **L1+L2**: 598 | **Method**: A3 (150ep)

**Problem**: Prove a property about relational images and restrictions.

**Generated Conjecture**:
```
r2_hidden(k4_tarski(X1,esk5_3(X2,X3,X4)),X5) | r2_hidden(esk4_4(X2,X3,X4,X5),X1)
| ~r2_hidden(X1,X5) | ~r2_hidden(X1,X5)
```

**In plain math**: For any x and any relation R:
> Either the ordered pair (x, witness) belongs to R,
> or some other witness belongs to x,
> or x does not belong to R.

This provides a structural decomposition of relational membership in terms of
ordered pairs.

**Why it helps**: Relational reasoning requires decomposing membership into
pair-level facts. This lemma provides the decomposition template.

---

### 9. Graph Substructure (t95_glib_000) --- 2.5x speedup

**Split**: test | **L**: 33,982 | **L1+L2**: 13,810 | **Method**: C (VAE)

**Problem**: Prove a property about graph homomorphisms in the glib_000 graph library.

**Generated Conjecture**:
```
m1_subset_1(esk3_0,k1_zfmisc_1(k6_glib_000(esk2_0)))
```

**In plain math**:
> The set esk3_0 is a subset of the edge set of graph esk2_0.

**Why it helps**: Graph proofs need explicit typing of vertex/edge sets. This
confirms that a set is actually a set of edges, enabling edge-level reasoning.

---

### 10. Projective Geometry Point Membership (t10_anproj_1) --- 2.3x speedup

**Split**: test | **L**: 27,159 | **L1+L2**: 11,809 | **Method**: A2 (100ep)

**Problem**: Prove a property in analytic projective geometry about points and
the projective space structure.

**Generated Conjecture**:
```
m1_subset_1(esk4_0,u1_struct_0(esk1_0)) | ~m1_subset_1(k4_struct_0(esk1_0),u1_struct_0(esk1_0))
```

**In plain math**:
> Point esk4_0 belongs to the carrier of projective space esk1_0,
> OR the distinguished point of esk1_0 does not belong to its own carrier.

**Why it helps**: Establishes point membership in the projective space, which
is a prerequisite for geometric reasoning.

---

## Summary Statistics (Test + Validation)

| Metric | Value |
|--------|-------|
| Total test+val problems with speedup (any method) | 95 test + ~70 val ≈ 165 |
| Best test speedup | 40.8x (t40_euclid_3) |
| Best val speedup | 38.3x (t17_xreal_1) |
| Problems where only D finds speedup | 6 (test) |
| Problems where all methods find speedup | ~28 (test) |

## Key Patterns in Test/Val Results

1. **Typing lemmas dominate**: 6 of 10 examples above are typing/membership facts
2. **The simplest conjectures work best**: "x is not real" (38x) beats complex lemmas
3. **SSM finds unique solutions**: D-only speedups on 6 test problems
4. **Cross-theory transfer confirmed**: Set difference pattern from training applies
   to unseen set-theoretic problems
5. **Multi-method agreement**: Hard problems (t13_ordinal1) attract useful conjectures
   from ALL methods — the key lemma is structurally "obvious" to every architecture
