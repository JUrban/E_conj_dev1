"""
Core test suite for the conjecture generation system.

Covers: parser, validation, sampler, target encoding, and model integration.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# (a) Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    """Tests for the TPTP CNF parser."""

    def test_valid_clause_parses(self):
        from conjecture_gen.tptp_parser import parse_clause
        c = parse_clause(
            'cnf(t1, axiom, (p(X1,X2) | ~q(a))).'
        )
        assert c is not None
        assert c.name == 't1'
        assert c.role == 'axiom'
        assert len(c.literals) == 2
        assert c.literals[0].predicate == 'p'
        assert not c.literals[0].negated
        assert c.literals[1].predicate == 'q'
        assert c.literals[1].negated

    def test_equality_clause(self):
        from conjecture_gen.tptp_parser import parse_clause
        c = parse_clause('cnf(t2, axiom, (X1=f(X2))).')
        assert c is not None
        assert c.literals[0].is_equality
        assert c.literals[0].predicate == '$eq'

    def test_inequality_clause(self):
        from conjecture_gen.tptp_parser import parse_clause
        c = parse_clause('cnf(t3, axiom, (X1!=X2)).')
        assert c is not None
        assert c.literals[0].is_equality
        assert c.literals[0].negated

    def test_nested_function(self):
        from conjecture_gen.tptp_parser import parse_clause
        c = parse_clause('cnf(t4, axiom, (p(f(g(X1,a),X2)))).')
        assert c is not None
        assert len(c.literals) == 1
        assert c.literals[0].predicate == 'p'
        assert len(c.literals[0].args) == 1
        # f(g(X1,a),X2)
        f_term = c.literals[0].args[0]
        assert f_term.name == 'f'
        assert len(f_term.args) == 2
        assert f_term.args[0].name == 'g'
        assert len(f_term.args[0].args) == 2

    def test_empty_parens_rejected_strict(self):
        from conjecture_gen.tptp_parser import parse_clause, TPTPParseError
        with pytest.raises(TPTPParseError):
            parse_clause('cnf(t, axiom, ()).', strict=True)

    def test_empty_args_rejected_strict(self):
        from conjecture_gen.tptp_parser import parse_clause, TPTPParseError
        with pytest.raises(TPTPParseError):
            parse_clause('cnf(t, axiom, (p())).', strict=True)

    def test_trailing_comma_rejected_strict(self):
        from conjecture_gen.tptp_parser import parse_clause, TPTPParseError
        with pytest.raises(TPTPParseError):
            parse_clause('cnf(t, axiom, (p(a,))).', strict=True)

    def test_constant_predicate(self):
        """A predicate with no args is valid in TPTP: cnf(t, axiom, (p))."""
        from conjecture_gen.tptp_parser import parse_clause
        c = parse_clause('cnf(t, axiom, (p)).')
        assert c is not None
        assert c.literals[0].predicate == 'p'
        assert len(c.literals[0].args) == 0

    def test_problem_files_parse(self):
        """If problem files are available, they should all parse."""
        import os
        from conjecture_gen.tptp_parser import parse_problem_file
        problems_dir = '/project/problems'
        if not os.path.isdir(problems_dir):
            pytest.skip("No problems directory found")
        files = os.listdir(problems_dir)[:5]  # test first 5
        for fname in files:
            path = os.path.join(problems_dir, fname)
            clauses = parse_problem_file(path)
            assert len(clauses) > 0, f"No clauses parsed from {fname}"


# ---------------------------------------------------------------------------
# (b) Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for the clause validation helper."""

    def test_empty_rejected(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('')
        assert not r['valid']
        assert r['reason'] == 'empty'

    def test_empty_tag_rejected(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('<empty>')
        assert not r['valid']
        assert r['reason'] == 'empty'

    def test_truncated_rejected(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('p(a)...')
        assert not r['valid']
        assert r['reason'] == 'truncated'

    def test_unk_rejected(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('p(?UNK3)')
        assert not r['valid']
        assert r['reason'] == 'unknown_symbol'

    def test_valid_body_passes(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('p(a,X1) | ~q(X2)')
        assert r['valid']
        assert r['n_literals'] == 2

    def test_equality_body_passes(self):
        from conjecture_gen.validation import validate_clause_text
        r = validate_clause_text('X1=f(X2)')
        assert r['valid']


# ---------------------------------------------------------------------------
# (c) Sampler tests
# ---------------------------------------------------------------------------

class TestSampler:
    """Tests for sample_from_logits and related utilities."""

    def test_all_inf_action_with_fallback_returns_end_clause(self):
        from conjecture_gen.sampling import sample_from_logits
        from conjecture_gen.target_encoder import END_CLAUSE
        logits = torch.full((1, 7), float('-inf'))
        result = sample_from_logits(logits, fallback_idx=END_CLAUSE)
        assert result.item() == END_CLAUSE

    def test_all_inf_pointer_without_fallback_raises(self):
        from conjecture_gen.sampling import sample_from_logits, SamplingError
        logits = torch.full((1, 10), float('-inf'))
        with pytest.raises(SamplingError):
            sample_from_logits(logits)

    def test_deterministic_all_inf_with_fallback(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.full((2, 10), float('-inf'))
        result = sample_from_logits(logits, temperature=0.0, fallback_idx=5)
        assert (result == 5).all()

    def test_deterministic_all_inf_without_fallback_raises(self):
        from conjecture_gen.sampling import sample_from_logits, SamplingError
        logits = torch.full((1, 5), float('-inf'))
        with pytest.raises(SamplingError):
            sample_from_logits(logits, temperature=0.0)

    def test_top_p_validation(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError):
            sample_from_logits(logits, top_p=1.5)
        with pytest.raises(ValueError):
            sample_from_logits(logits, top_p=-0.1)

    def test_top_k_validation(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError):
            sample_from_logits(logits, top_k=-1)

    def test_ndim_validation(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(10)  # 1D, should fail
        with pytest.raises(ValueError):
            sample_from_logits(logits)

    def test_sample_action_logits_uses_end_clause(self):
        from conjecture_gen.sampling import sample_action_logits
        from conjecture_gen.target_encoder import END_CLAUSE
        logits = torch.full((1, 7), float('-inf'))
        result = sample_action_logits(logits)
        assert result.item() == END_CLAUSE

    def test_normal_sampling_works(self):
        from conjecture_gen.sampling import sample_from_logits
        torch.manual_seed(42)
        logits = torch.zeros(4, 10)
        logits[:, 3] = 100.0  # make index 3 dominant
        result = sample_from_logits(logits, temperature=0.1)
        assert (result == 3).all()

    def test_top_p_zero_is_disabled(self):
        """top_p=0 should behave like no nucleus filtering."""
        from conjecture_gen.sampling import sample_from_logits
        torch.manual_seed(42)
        logits = torch.zeros(1, 5)
        logits[0, 2] = 10.0
        result = sample_from_logits(logits, top_p=0.0, temperature=0.01)
        assert result.item() == 2

    def test_valid_top_p_boundaries(self):
        """top_p=0.0 and top_p=1.0 should both be valid."""
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        sample_from_logits(logits, top_p=0.0)
        sample_from_logits(logits, top_p=1.0)


# ---------------------------------------------------------------------------
# (d) Target encoding tests
# ---------------------------------------------------------------------------

class TestTargetEncoding:
    """Tests for encode_conjecture and decode_sequence."""

    def _make_clause(self, text):
        from conjecture_gen.tptp_parser import parse_clause
        return parse_clause(f'cnf(t, axiom, ({text})).')

    def test_role_aware_distinguishes_pred_from_func(self):
        """predicate f/1 and function f/1 should map to different indices."""
        from conjecture_gen.target_encoder import encode_conjecture, PRED, ARG_FUNC
        # symbol_names has f as pred (idx 0) and f as func (idx 1)
        symbol_names = ['f', 'f']
        symbol_is_pred = [True, False]
        symbol_arities = [1, 1]

        clause = self._make_clause('f(f(X1))')
        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)

        # Find PRED and ARG_FUNC actions
        pred_indices = [arg for act, arg in seq if act == PRED]
        func_indices = [arg for act, arg in seq if act == ARG_FUNC]

        assert len(pred_indices) == 1
        assert len(func_indices) == 1
        assert pred_indices[0] == 0  # pred f -> idx 0
        assert func_indices[0] == 1  # func f -> idx 1

    def test_arity_aware_distinguishes_func_arities(self):
        """function f/1 and function f/2 should map to different indices when arities differ."""
        from conjecture_gen.target_encoder import encode_conjecture, ARG_FUNC
        # f/1 at index 1, f/2 at index 2
        symbol_names = ['p', 'f', 'f']
        symbol_is_pred = [True, False, False]
        symbol_arities = [1, 1, 2]

        # p(f(f(X1, X2))) -- outer f has arity 1, inner f has arity 2
        clause = self._make_clause('p(f(f(X1,X2)))')
        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)

        func_indices = [arg for act, arg in seq if act == ARG_FUNC]
        assert len(func_indices) == 2
        # First ARG_FUNC is outer f/1 -> idx 1
        assert func_indices[0] == 1
        # Second ARG_FUNC is inner f/2 -> idx 2
        assert func_indices[1] == 2

    def test_equality_round_trips(self):
        """Equality clauses should encode and decode correctly."""
        from conjecture_gen.target_encoder import encode_conjecture, decode_sequence
        clause = self._make_clause('X1=f(X2)')
        symbol_names = ['$eq', 'f']
        symbol_is_pred = [True, False]
        symbol_arities = [2, 1]

        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)
        decoded = decode_sequence(seq, symbol_names)
        assert 'X1' in decoded
        assert 'X2' in decoded
        assert '=' in decoded

    def test_nested_equality_round_trips(self):
        """Nested equality like f(X1)=g(X2) should round-trip."""
        from conjecture_gen.target_encoder import encode_conjecture, decode_sequence
        clause = self._make_clause('f(X1)=g(X2)')
        symbol_names = ['$eq', 'f', 'g']
        symbol_is_pred = [True, False, False]
        symbol_arities = [2, 1, 1]

        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)
        decoded = decode_sequence(seq, symbol_names)
        assert '=' in decoded
        # Check that both f and g appear
        assert 'f(' in decoded
        assert 'g(' in decoded

    def test_fallback_without_arities(self):
        """Without symbol_arities, encoding should still work (backward compat)."""
        from conjecture_gen.target_encoder import encode_conjecture, END_CLAUSE
        clause = self._make_clause('p(a)')
        symbol_names = ['p', 'a']
        seq = encode_conjecture(clause, symbol_names)
        assert seq[-1][0] == END_CLAUSE

    def test_unk_for_missing_symbol(self):
        """Symbols not in the list get UNK index."""
        from conjecture_gen.target_encoder import encode_conjecture, PRED
        clause = self._make_clause('unknown_pred(X1)')
        symbol_names = ['p', 'q']
        seq = encode_conjecture(clause, symbol_names)
        pred_args = [arg for act, arg in seq if act == PRED]
        assert pred_args[0] == len(symbol_names)  # UNK index


# ---------------------------------------------------------------------------
# (e) Integration test: small model forward + backward
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests with a small model."""

    @pytest.fixture
    def small_graph(self):
        """Build a small graph from a minimal problem."""
        from conjecture_gen.tptp_parser import parse_clause
        from conjecture_gen.graph_builder import clauses_to_graph
        clauses = [
            parse_clause('cnf(c1, axiom, (p(X1) | ~q(f(X1,a)))).')
        ]
        return clauses_to_graph(clauses)

    def test_model_a_forward_backward(self, small_graph):
        """Model A: encode -> decode (teacher-forced) -> loss -> backward."""
        from conjecture_gen.model import ConjectureModel
        from conjecture_gen.target_encoder import encode_conjecture
        from conjecture_gen.tptp_parser import parse_clause
        from conjecture_gen.train import compute_loss

        model = ConjectureModel(hidden_dim=32, num_gnn_layers=2, dec_layers=1, dec_nhead=2)

        # Encode a target conjecture
        target_clause = parse_clause('cnf(t, axiom, (p(a))).')
        seq = encode_conjecture(
            target_clause, small_graph.symbol_names,
            symbol_is_pred=small_graph.symbol_is_pred,
            symbol_arities=small_graph.symbol_arities,
        )

        # Prepare batch-like data
        graph = small_graph.clone()
        actions = torch.tensor([a for a, _ in seq], dtype=torch.long)
        arguments = torch.tensor([arg for _, arg in seq], dtype=torch.long)
        graph.target_actions = actions.unsqueeze(0)
        graph.target_arguments = arguments.unsqueeze(0)
        graph.target_length = torch.tensor([len(seq)], dtype=torch.long)
        graph.quality_weight = torch.tensor([1.0], dtype=torch.float)
        graph.num_symbols = torch.tensor([len(small_graph.symbol_names)], dtype=torch.long)

        # Forward
        x_dict = model.encoder(graph)
        output = model.decoder(
            x_dict,
            graph.target_actions, graph.target_arguments,
            graph.target_length, graph.num_symbols,
            batch_data=graph,
        )
        assert 'action_logits' in output
        assert 'pointer_logits' in output
        assert 'var_logits' in output

        # Loss
        loss_dict = compute_loss(output, graph)
        assert 'total' in loss_dict
        assert loss_dict['total'].requires_grad

        # Backward
        loss_dict['total'].backward()
        # Check gradients exist
        grads_found = 0
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_found += 1
        assert grads_found > 0, "No nonzero gradients after backward"

    def test_model_a_generate(self, small_graph):
        """Model A: generation should produce valid sequences."""
        from conjecture_gen.model import ConjectureModel
        from conjecture_gen.target_encoder import END_CLAUSE

        model = ConjectureModel(hidden_dim=32, num_gnn_layers=2, dec_layers=1, dec_nhead=2)
        model.eval()

        seqs = model.generate(small_graph, max_steps=30, temperature=1.0)
        assert len(seqs) == 1
        seq = seqs[0]
        assert len(seq) > 0
        # Last action should be END_CLAUSE (or generation was truncated)
        if len(seq) < 30:
            assert seq[-1][0] == END_CLAUSE

    def test_model_c_forward(self, small_graph):
        """Model C (VAE): basic forward pass works."""
        from conjecture_gen.model_c import ConjectureModelC
        from conjecture_gen.target_encoder import encode_conjecture
        from conjecture_gen.tptp_parser import parse_clause

        model = ConjectureModelC(hidden_dim=32, num_gnn_layers=2)

        target_clause = parse_clause('cnf(t, axiom, (p(a))).')
        seq = encode_conjecture(
            target_clause, small_graph.symbol_names,
            symbol_is_pred=small_graph.symbol_is_pred,
            symbol_arities=small_graph.symbol_arities,
        )
        graph = small_graph.clone()
        actions = torch.tensor([a for a, _ in seq], dtype=torch.long)
        arguments = torch.tensor([arg for _, arg in seq], dtype=torch.long)
        graph.target_actions = actions.unsqueeze(0)
        graph.target_arguments = arguments.unsqueeze(0)
        graph.target_length = torch.tensor([len(seq)], dtype=torch.long)
        graph.quality_weight = torch.tensor([1.0], dtype=torch.float)
        graph.num_symbols = torch.tensor([len(small_graph.symbol_names)], dtype=torch.long)

        output = model(graph)
        assert 'action_logits' in output
        assert 'post_mu' in output
        assert 'prior_mu' in output

    def test_model_d_forward(self, small_graph):
        """Model D (SSM): basic forward pass works."""
        from conjecture_gen.model_d import ConjectureModelD
        from conjecture_gen.target_encoder import encode_conjecture
        from conjecture_gen.tptp_parser import parse_clause

        model = ConjectureModelD(hidden_dim=32, num_gnn_layers=2)

        target_clause = parse_clause('cnf(t, axiom, (p(a))).')
        seq = encode_conjecture(
            target_clause, small_graph.symbol_names,
            symbol_is_pred=small_graph.symbol_is_pred,
            symbol_arities=small_graph.symbol_arities,
        )
        graph = small_graph.clone()
        actions = torch.tensor([a for a, _ in seq], dtype=torch.long)
        arguments = torch.tensor([arg for _, arg in seq], dtype=torch.long)
        graph.target_actions = actions.unsqueeze(0)
        graph.target_arguments = arguments.unsqueeze(0)
        graph.target_length = torch.tensor([len(seq)], dtype=torch.long)
        graph.quality_weight = torch.tensor([1.0], dtype=torch.float)
        graph.num_symbols = torch.tensor([len(small_graph.symbol_names)], dtype=torch.long)

        output = model(graph)
        assert 'action_logits' in output
        assert 'pointer_logits' in output

    def test_sampling_error_class_importable(self):
        """SamplingError should be importable from sampling module."""
        from conjecture_gen.sampling import SamplingError
        assert issubclass(SamplingError, Exception)

    def test_build_arg_embeddings_shared(self):
        """The shared build_arg_embeddings function works correctly."""
        from conjecture_gen.model import build_arg_embeddings
        from conjecture_gen.target_encoder import PRED, ARG_VAR, ARG_FUNC, END_CLAUSE

        hidden = 16
        max_vars = 5
        B, T, S = 2, 4, 3

        arg_sym_proj = nn.Linear(hidden, hidden)
        var_slot_embed = nn.Embedding(max_vars, hidden)
        unk_sym = torch.randn(hidden)
        unk_var = torch.randn(hidden)
        symbol_embeds = torch.randn(B, S, hidden)

        actions = torch.tensor([
            [PRED, ARG_VAR, ARG_FUNC, END_CLAUSE],
            [PRED, ARG_FUNC, ARG_VAR, END_CLAUSE],
        ])
        arguments = torch.tensor([
            [0, 1, 2, 0],
            [1, 0, 0, 0],
        ])

        result = build_arg_embeddings(
            actions, arguments, symbol_embeds,
            arg_sym_proj, var_slot_embed, max_vars,
            unk_sym, unk_var,
        )
        assert result.shape == (B, T, hidden)
        # END_CLAUSE positions should be zero
        assert result[0, 3].abs().sum() == 0
        assert result[1, 3].abs().sum() == 0
        # PRED/ARG_FUNC/ARG_VAR positions should be nonzero (in general)
        # (could be zero by chance but very unlikely with random init)


# ---------------------------------------------------------------------------
# (f) Action masking tests (R02)
# ---------------------------------------------------------------------------

class TestActionMasking:
    """Tests for role-availability action masking in generation."""

    @pytest.fixture
    def funcs_only_graph(self):
        """Build a graph that has only function symbols (no predicates except
        implicitly via equality)."""
        from conjecture_gen.tptp_parser import parse_clause
        from conjecture_gen.graph_builder import clauses_to_graph
        # X1=f(a) uses $eq as predicate and f, a as functions
        clauses = [parse_clause('cnf(c1, axiom, (X1=f(a))).')]
        graph = clauses_to_graph(clauses)
        return graph

    @pytest.fixture
    def preds_only_graph(self):
        """Build a graph that has only predicate symbols (no function symbols)."""
        from conjecture_gen.tptp_parser import parse_clause
        from conjecture_gen.graph_builder import clauses_to_graph
        # p(X1) | ~q(X2) -- p and q are predicates, no functions
        clauses = [parse_clause('cnf(c1, axiom, (p(X1) | ~q(X2))).')]
        graph = clauses_to_graph(clauses)
        return graph

    def test_no_functions_masks_arg_func(self, preds_only_graph):
        """When graph has no function symbols, ARG_FUNC should be pre-masked."""
        from conjecture_gen.model import ConjectureModel
        from conjecture_gen.target_encoder import ARG_FUNC, END_CLAUSE

        model = ConjectureModel(hidden_dim=32, num_gnn_layers=2, dec_layers=1, dec_nhead=2)
        model.eval()
        torch.manual_seed(123)

        seqs = model.generate(preds_only_graph, max_steps=30, temperature=1.0)
        assert len(seqs) == 1
        seq = seqs[0]
        # No ARG_FUNC should appear since there are no function symbols
        func_actions = [act for act, _ in seq if act == ARG_FUNC]
        assert len(func_actions) == 0, (
            f"ARG_FUNC appeared {len(func_actions)} times despite no function symbols"
        )

    def test_no_predicates_masks_pred(self):
        """When graph has no predicate symbols, PRED should be pre-masked,
        leading to END_CLAUSE immediately (no way to start a literal)."""
        from conjecture_gen.model import ConjectureModel
        from conjecture_gen.target_encoder import PRED, END_CLAUSE
        from conjecture_gen.tptp_parser import parse_clause
        from conjecture_gen.graph_builder import clauses_to_graph

        # Build graph where all symbols are functions (artificial case)
        # We manually modify symbol_is_pred to all False
        clauses = [parse_clause('cnf(c1, axiom, (p(f(a)))).')]
        graph = clauses_to_graph(clauses)
        # Override: pretend all symbols are functions
        graph.symbol_is_pred = [False] * len(graph.symbol_names)

        model = ConjectureModel(hidden_dim=32, num_gnn_layers=2, dec_layers=1, dec_nhead=2)
        model.eval()
        torch.manual_seed(42)

        seqs = model.generate(graph, max_steps=30, temperature=1.0)
        seq = seqs[0]
        # PRED should never appear because there are no predicate symbols
        pred_actions = [act for act, _ in seq if act == PRED]
        assert len(pred_actions) == 0, (
            f"PRED appeared {len(pred_actions)} times despite no predicate symbols"
        )


# ---------------------------------------------------------------------------
# (g) Strict target encoding tests (R03)
# ---------------------------------------------------------------------------

class TestStrictTargetEncoding:
    """Tests for strict mode in encode_conjecture."""

    def _make_clause(self, text):
        from conjecture_gen.tptp_parser import parse_clause
        return parse_clause(f'cnf(t, axiom, ({text})).')

    def test_mismatched_lengths_raises(self):
        """Mismatched metadata lengths should raise ValueError in strict mode."""
        from conjecture_gen.target_encoder import encode_conjecture
        clause = self._make_clause('p(a)')
        with pytest.raises(ValueError, match="Metadata length mismatch"):
            encode_conjecture(
                clause,
                symbol_names=['p', 'a'],
                symbol_is_pred=[True],  # length 1, should be 2
                symbol_arities=[1, 0],
                strict=True,
            )

    def test_mismatched_arities_length_raises(self):
        """Mismatched symbol_arities length should raise ValueError in strict mode."""
        from conjecture_gen.target_encoder import encode_conjecture
        clause = self._make_clause('p(a)')
        with pytest.raises(ValueError, match="Metadata length mismatch"):
            encode_conjecture(
                clause,
                symbol_names=['p', 'a'],
                symbol_is_pred=[True, False],
                symbol_arities=[1],  # length 1, should be 2
                strict=True,
            )

    def test_duplicate_keys_raises(self):
        """Duplicate (name, is_pred, arity) keys should raise ValueError in strict mode."""
        from conjecture_gen.target_encoder import encode_conjecture
        clause = self._make_clause('p(a)')
        with pytest.raises(ValueError, match="Duplicate symbol key"):
            encode_conjecture(
                clause,
                symbol_names=['p', 'p'],
                symbol_is_pred=[True, True],
                symbol_arities=[1, 1],  # same key: ('p', True, 1)
                strict=True,
            )

    def test_strict_stats_correct(self):
        """Stats dict should have correct counts in strict mode."""
        from conjecture_gen.target_encoder import encode_conjecture
        clause = self._make_clause('p(a)')
        symbol_names = ['p', 'a']
        symbol_is_pred = [True, False]
        symbol_arities = [1, 0]

        seq, stats = encode_conjecture(
            clause, symbol_names, symbol_is_pred, symbol_arities, strict=True,
        )
        assert isinstance(stats, dict)
        assert stats['exact_hits'] == 2  # p as pred, a as func
        assert stats['unk_hits'] == 0
        assert stats['role_fallback_hits'] == 0
        assert stats['name_fallback_hits'] == 0

    def test_strict_no_fallback_uses_unk(self):
        """In strict mode, missing symbols should get UNK, not cascading fallback."""
        from conjecture_gen.target_encoder import encode_conjecture, PRED
        # 'p' is registered as (p, True, 2) but clause uses p with arity 1
        clause = self._make_clause('p(a)')
        symbol_names = ['p', 'a']
        symbol_is_pred = [True, False]
        symbol_arities = [2, 0]  # p has arity 2 in metadata, but used with arity 1

        seq, stats = encode_conjecture(
            clause, symbol_names, symbol_is_pred, symbol_arities, strict=True,
        )
        # In strict mode, p(arity=1) won't match (p, True, 2), so UNK
        pred_args = [arg for act, arg in seq if act == PRED]
        assert pred_args[0] == len(symbol_names)  # UNK index
        assert stats['unk_hits'] >= 1

    def test_nonstrict_returns_list_not_tuple(self):
        """Non-strict mode (default) should return just the sequence list."""
        from conjecture_gen.target_encoder import encode_conjecture
        clause = self._make_clause('p(a)')
        result = encode_conjecture(clause, ['p', 'a'])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# (h) Variant C posterior UNK tests (R06)
# ---------------------------------------------------------------------------

class TestClauseEncoderUNK:
    """Tests for ClauseEncoder UNK handling for out-of-range arguments."""

    def test_out_of_range_uses_unk_not_clamp(self):
        """Out-of-range argument IDs should use the UNK embedding, not clamp."""
        from conjecture_gen.model_c import ClauseEncoder
        from conjecture_gen.target_encoder import PRED, ARG_FUNC, END_CLAUSE, END_ARGS

        enc = ClauseEncoder(hidden_dim=32, latent_dim=8, max_arg_vocab=1024)

        # Arguments well within range
        actions = torch.tensor([[PRED, ARG_FUNC, END_ARGS, END_CLAUSE]])
        args_in_range = torch.tensor([[0, 1, 0, 0]])
        lengths = torch.tensor([4])

        mu1, _ = enc(actions, args_in_range, lengths)

        # Arguments out of range (2000 > max_arg_vocab=1024)
        args_out_range = torch.tensor([[2000, 5000, 0, 0]])
        mu2, _ = enc(actions, args_out_range, lengths)

        # Both should succeed without errors
        assert mu1.shape == (1, 8)
        assert mu2.shape == (1, 8)

        # The out-of-range args should produce different embeddings
        # than the in-range args (since UNK != embed[0])
        # (technically could be equal by chance, but extremely unlikely)
        assert not torch.allclose(mu1, mu2, atol=1e-6), (
            "Out-of-range and in-range args produced identical latents"
        )

    def test_negative_args_use_unk(self):
        """Negative argument values should use UNK embedding."""
        from conjecture_gen.model_c import ClauseEncoder
        from conjecture_gen.target_encoder import PRED, END_CLAUSE

        enc = ClauseEncoder(hidden_dim=32, latent_dim=8)
        actions = torch.tensor([[PRED, END_CLAUSE]])
        args_neg = torch.tensor([[-1, 0]])
        lengths = torch.tensor([2])

        # Should not raise
        mu, logvar = enc(actions, args_neg, lengths)
        assert mu.shape == (1, 8)


# ---------------------------------------------------------------------------
# (i) Sampler boundary validation tests (R04)
# ---------------------------------------------------------------------------

class TestSamplerBoundaryValidation:
    """Tests for new boundary validations in sample_from_logits."""

    def test_fallback_idx_negative_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError, match="fallback_idx must be in"):
            sample_from_logits(logits, fallback_idx=-1)

    def test_fallback_idx_equal_to_vocab_size_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)  # vocab_size=10
        with pytest.raises(ValueError, match="fallback_idx must be in"):
            sample_from_logits(logits, fallback_idx=10)

    def test_fallback_idx_beyond_vocab_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 5)
        with pytest.raises(ValueError, match="fallback_idx must be in"):
            sample_from_logits(logits, fallback_idx=100)

    def test_temperature_nan_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError, match="temperature must be finite"):
            sample_from_logits(logits, temperature=float('nan'))

    def test_temperature_inf_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError, match="temperature must be finite"):
            sample_from_logits(logits, temperature=float('inf'))

    def test_temperature_neg_inf_raises(self):
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError, match="temperature must be finite"):
            sample_from_logits(logits, temperature=float('-inf'))

    def test_valid_fallback_idx_works(self):
        """Valid fallback_idx should not raise."""
        from conjecture_gen.sampling import sample_from_logits
        logits = torch.full((1, 10), float('-inf'))
        result = sample_from_logits(logits, fallback_idx=5)
        assert result.item() == 5


# ---------------------------------------------------------------------------
# (j) Exact equality serialization test
# ---------------------------------------------------------------------------

class TestEqualitySerialization:
    """Tests for exact round-trip of equality clauses."""

    def _make_clause(self, text):
        from conjecture_gen.tptp_parser import parse_clause
        return parse_clause(f'cnf(t, axiom, ({text})).')

    def test_eq_roundtrip_exact(self):
        """$eq(f(a,b),g(c,d)) should decode to exactly f(a,b)=g(c,d)."""
        from conjecture_gen.target_encoder import encode_conjecture, decode_sequence
        clause = self._make_clause('f(a,b)=g(c,d)')
        symbol_names = ['$eq', 'f', 'g', 'a', 'b', 'c', 'd']
        symbol_is_pred = [True, False, False, False, False, False, False]
        symbol_arities = [2, 2, 2, 0, 0, 0, 0]

        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)
        decoded = decode_sequence(seq, symbol_names)
        assert decoded == 'f(a,b)=g(c,d)', f"Expected 'f(a,b)=g(c,d)', got '{decoded}'"

    def test_neq_roundtrip_exact(self):
        """Negated equality should round-trip to a!=b format."""
        from conjecture_gen.target_encoder import encode_conjecture, decode_sequence
        clause = self._make_clause('a!=b')
        symbol_names = ['$eq', 'a', 'b']
        symbol_is_pred = [True, False, False]
        symbol_arities = [2, 0, 0]

        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)
        decoded = decode_sequence(seq, symbol_names)
        assert decoded == 'a!=b', f"Expected 'a!=b', got '{decoded}'"

    def test_nested_eq_roundtrip(self):
        """f(a,b)=g(c,d) should round-trip exactly, not be a substring match."""
        from conjecture_gen.target_encoder import encode_conjecture, decode_sequence
        clause = self._make_clause('f(a,b)=g(c,d)')
        symbol_names = ['$eq', 'f', 'g', 'a', 'b', 'c', 'd']
        symbol_is_pred = [True, False, False, False, False, False, False]
        symbol_arities = [2, 2, 2, 0, 0, 0, 0]

        seq = encode_conjecture(clause, symbol_names, symbol_is_pred, symbol_arities)
        decoded = decode_sequence(seq, symbol_names)
        # Must be exact, not just containing '='
        assert decoded == 'f(a,b)=g(c,d)'
        # Verify it doesn't contain $eq
        assert '$eq' not in decoded
