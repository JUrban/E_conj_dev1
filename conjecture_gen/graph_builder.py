"""
Converts parsed TPTP clauses into PyTorch Geometric heterogeneous graphs.

Fully symbol-anonymous: all symbols (Mizar and Skolem) are identified only
by structural features (type, arity, occurrence pattern). Identity emerges
from GNN message passing.

Node types:
  - clause:   one per clause in the problem
  - literal:  one per literal occurrence
  - symbol:   one per unique predicate/function symbol (shared across uses)
  - term:     one per term occurrence (non-variable)
  - variable: one per unique variable per clause

Edge types (bidirectional where noted):
  - (clause, has_literal, literal) + reverse
  - (literal, has_predicate, symbol) + reverse
  - (literal, has_arg, term) + reverse          [with position encoding]
  - (term, has_functor, symbol) + reverse
  - (term, has_subterm, term) + reverse          [with position encoding]
  - (term, is_var, variable) + reverse
  - (variable, in_clause, clause) + reverse
"""

import torch
from torch_geometric.data import HeteroData
from conjecture_gen.tptp_parser import Clause, Literal, Term


# Feature dimensions
CLAUSE_FEAT_DIM = 3    # role one-hot: plain, negated_conjecture, other
LITERAL_FEAT_DIM = 2   # [negated, is_equality]
SYMBOL_FEAT_DIM = 4    # [is_predicate, is_function, is_constant(arity0), arity_normalized]
TERM_FEAT_DIM = 2      # [depth_normalized, arg_position_normalized]
VARIABLE_FEAT_DIM = 1  # [index_in_clause normalized]

ROLE_MAP = {'plain': 0, 'negated_conjecture': 1}


def _collect_symbols(clauses: list[Clause]) -> dict[str, dict]:
    """Collect all unique predicate and function symbols with their arities."""
    symbols = {}

    def _visit_term(t: Term):
        if t.is_variable:
            return
        arity = len(t.args)
        key = (t.name, 'func', arity)
        if t.name not in symbols:
            symbols[t.name] = {'is_pred': False, 'is_func': True, 'arity': arity}
        else:
            # Could appear as both pred and func in weird cases; keep both flags
            symbols[t.name]['is_func'] = True
            symbols[t.name]['arity'] = max(symbols[t.name]['arity'], arity)
        for arg in t.args:
            _visit_term(arg)

    for clause in clauses:
        for lit in clause.literals:
            if lit.predicate not in symbols:
                symbols[lit.predicate] = {
                    'is_pred': True, 'is_func': False,
                    'arity': len(lit.args),
                }
            else:
                symbols[lit.predicate]['is_pred'] = True
                symbols[lit.predicate]['arity'] = max(
                    symbols[lit.predicate]['arity'], len(lit.args)
                )
            for arg in lit.args:
                _visit_term(arg)

    return symbols


class GraphBuilder:
    """Builds a HeteroData graph from a list of clauses."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Node lists
        self.clause_feats = []
        self.literal_feats = []
        self.symbol_feats = []
        self.term_feats = []
        self.variable_feats = []

        # Edge index lists: dict of (src_type, edge_type, dst_type) -> [src_list, dst_list]
        self.edges = {}
        # Edge attr lists (for position encoding)
        self.edge_attrs = {}

        # Symbol name -> node index
        self.symbol_map = {}
        # (clause_idx, var_name) -> node index
        self.variable_map = {}

        # Counters
        self.n_clauses = 0
        self.n_literals = 0
        self.n_symbols = 0
        self.n_terms = 0
        self.n_variables = 0

        # For reconstructing clause text from graph
        self.symbol_names = []  # index -> name
        self.symbol_arities = []  # index -> arity (int)
        self.symbol_is_pred = []  # index -> bool

    def _add_edge(self, edge_type: tuple, src: int, dst: int, attr: float = None):
        if edge_type not in self.edges:
            self.edges[edge_type] = [[], []]
            if attr is not None:
                self.edge_attrs[edge_type] = []
        self.edges[edge_type][0].append(src)
        self.edges[edge_type][1].append(dst)
        if attr is not None:
            self.edge_attrs[edge_type].append(attr)

    def _get_symbol_idx(self, name: str, is_pred: bool, arity: int) -> int:
        if name in self.symbol_map:
            return self.symbol_map[name]
        idx = self.n_symbols
        self.symbol_map[name] = idx
        self.symbol_names.append(name)
        self.symbol_arities.append(arity)
        self.symbol_is_pred.append(is_pred)
        is_const = (not is_pred) and (arity == 0)
        arity_norm = min(arity, 10) / 10.0
        self.symbol_feats.append([
            float(is_pred),
            float(not is_pred),  # is_function
            float(is_const),
            arity_norm,
        ])
        self.n_symbols += 1
        return idx

    def _get_variable_idx(self, clause_idx: int, var_name: str) -> int:
        key = (clause_idx, var_name)
        if key in self.variable_map:
            return self.variable_map[key]
        idx = self.n_variables
        self.variable_map[key] = idx
        # Count how many variables already in this clause
        clause_var_count = sum(
            1 for (ci, _) in self.variable_map if ci == clause_idx
        )
        self.variable_feats.append([min(clause_var_count, 10) / 10.0])
        self.n_variables += 1

        # variable <-> clause edge
        self._add_edge(('variable', 'in_clause', 'clause'), idx, clause_idx)
        self._add_edge(('clause', 'has_variable', 'variable'), clause_idx, idx)

        return idx

    def _add_term(self, term: Term, clause_idx: int, depth: int) -> int:
        """Add a term node and return its index. Variables are handled separately."""
        if term.is_variable:
            # Don't create a term node for variables; return -1 as sentinel
            return -1

        idx = self.n_terms
        self.n_terms += 1
        depth_norm = min(depth, 5) / 5.0
        self.term_feats.append([depth_norm, 0.0])  # position filled by caller

        # term -> functor symbol
        sym_idx = self._get_symbol_idx(term.name, is_pred=False, arity=len(term.args))
        self._add_edge(('term', 'has_functor', 'symbol'), idx, sym_idx)
        self._add_edge(('symbol', 'functor_of', 'term'), sym_idx, idx)

        # Recurse into arguments
        for pos, arg in enumerate(term.args):
            pos_norm = pos / max(len(term.args), 1)
            if arg.is_variable:
                var_idx = self._get_variable_idx(clause_idx, arg.name)
                self._add_edge(('term', 'has_var_arg', 'variable'), idx, var_idx, attr=pos_norm)
                self._add_edge(('variable', 'arg_of', 'term'), var_idx, idx, attr=pos_norm)
            else:
                child_idx = self._add_term(arg, clause_idx, depth + 1)
                # Update child's position feature
                self.term_feats[child_idx][1] = pos_norm
                self._add_edge(('term', 'has_subterm', 'term'), idx, child_idx, attr=pos_norm)
                self._add_edge(('term', 'subterm_of', 'term'), child_idx, idx, attr=pos_norm)

        return idx

    def build(self, clauses: list[Clause]) -> HeteroData:
        """Build a heterogeneous graph from a list of clauses."""
        self.reset()

        for ci, clause in enumerate(clauses):
            # Add clause node
            role_idx = ROLE_MAP.get(clause.role, 2)
            feat = [0.0, 0.0, 0.0]
            feat[role_idx] = 1.0
            self.clause_feats.append(feat)
            self.n_clauses += 1

            for li, lit in enumerate(clause.literals):
                # Add literal node
                lit_idx = self.n_literals
                self.n_literals += 1
                self.literal_feats.append([
                    float(lit.negated),
                    float(lit.is_equality),
                ])

                # clause <-> literal
                self._add_edge(('clause', 'has_literal', 'literal'), ci, lit_idx)
                self._add_edge(('literal', 'in_clause', 'clause'), lit_idx, ci)

                # literal -> predicate symbol
                sym_idx = self._get_symbol_idx(
                    lit.predicate, is_pred=True, arity=len(lit.args)
                )
                self._add_edge(('literal', 'has_predicate', 'symbol'), lit_idx, sym_idx)
                self._add_edge(('symbol', 'predicate_of', 'literal'), sym_idx, lit_idx)

                # literal -> argument terms
                for pos, arg in enumerate(lit.args):
                    pos_norm = pos / max(len(lit.args), 1)
                    if arg.is_variable:
                        var_idx = self._get_variable_idx(ci, arg.name)
                        self._add_edge(
                            ('literal', 'has_var_arg', 'variable'),
                            lit_idx, var_idx, attr=pos_norm,
                        )
                        self._add_edge(
                            ('variable', 'arg_of_lit', 'literal'),
                            var_idx, lit_idx, attr=pos_norm,
                        )
                    else:
                        term_idx = self._add_term(arg, ci, depth=0)
                        self.term_feats[term_idx][1] = pos_norm
                        self._add_edge(
                            ('literal', 'has_arg', 'term'),
                            lit_idx, term_idx, attr=pos_norm,
                        )
                        self._add_edge(
                            ('term', 'arg_of_lit', 'literal'),
                            term_idx, lit_idx, attr=pos_norm,
                        )

        # Build HeteroData
        data = HeteroData()

        # Node features
        if self.n_clauses > 0:
            data['clause'].x = torch.tensor(self.clause_feats, dtype=torch.float)
        if self.n_literals > 0:
            data['literal'].x = torch.tensor(self.literal_feats, dtype=torch.float)
        if self.n_symbols > 0:
            data['symbol'].x = torch.tensor(self.symbol_feats, dtype=torch.float)
        if self.n_terms > 0:
            data['term'].x = torch.tensor(self.term_feats, dtype=torch.float)
        if self.n_variables > 0:
            data['variable'].x = torch.tensor(self.variable_feats, dtype=torch.float)

        # Edge indices
        for edge_type, (src, dst) in self.edges.items():
            data[edge_type].edge_index = torch.tensor(
                [src, dst], dtype=torch.long
            )
            if edge_type in self.edge_attrs:
                data[edge_type].edge_attr = torch.tensor(
                    self.edge_attrs[edge_type], dtype=torch.float
                ).unsqueeze(1)

        # Store symbol names for decoding
        data.symbol_names = self.symbol_names
        data.symbol_arities = self.symbol_arities
        data.symbol_is_pred = self.symbol_is_pred

        return data


def clauses_to_graph(clauses: list[Clause], vocab: dict = None) -> HeteroData:
    """Convenience function: parse clauses into a graph.

    If vocab is provided, adds symbol_name_ids for named embeddings.
    """
    builder = GraphBuilder()
    graph = builder.build(clauses)

    if vocab is not None:
        from conjecture_gen.symbol_vocab import names_to_indices
        graph.symbol_name_ids = names_to_indices(graph.symbol_names, vocab)

    return graph


if __name__ == '__main__':
    from conjecture_gen.tptp_parser import parse_problem_file
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else 'problems/l100_fomodel0'
    clauses = parse_problem_file(path)
    graph = clauses_to_graph(clauses)

    print(f"Problem: {path}")
    print(f"Clauses: {len(clauses)}")
    print(f"\nGraph structure:")
    print(f"  clause nodes:   {graph['clause'].x.shape[0]}")
    print(f"  literal nodes:  {graph['literal'].x.shape[0]}")
    print(f"  symbol nodes:   {graph['symbol'].x.shape[0]}")
    print(f"  term nodes:     {graph['term'].x.shape[0]}")
    print(f"  variable nodes: {graph['variable'].x.shape[0]}")
    print(f"\nEdge types:")
    for edge_type in graph.edge_types:
        ei = graph[edge_type].edge_index
        print(f"  {edge_type}: {ei.shape[1]} edges")
    print(f"\nSymbols: {graph.symbol_names[:10]}...")
