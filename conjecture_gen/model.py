"""
Conjecture generation model: GNN encoder + pointer-based tree decoder.

Encoder: Heterogeneous GNN (SAGEConv-based) that processes the problem graph.
         All symbols are anonymous — identity comes only from structure.

Decoder: GRU-based autoregressive decoder that generates the conjecture
         clause as a sequence of (action, argument) pairs.
         Uses pointer attention over GNN symbol embeddings to select
         predicates and function symbols.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData

from conjecture_gen.target_encoder import (
    NUM_ACTION_TYPES, NEW_LIT_POS, NEW_LIT_NEG, PRED,
    ARG_VAR, ARG_FUNC, END_ARGS, END_CLAUSE,
)


class HeteroGNNEncoder(nn.Module):
    """Heterogeneous GNN encoder for CNF problem graphs.

    Projects each node type's features to a shared hidden dimension,
    then runs multiple rounds of heterogeneous message passing.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections for each node type
        # Features: clause=3, literal=2, symbol=4, term=2, variable=1
        self.input_projs = nn.ModuleDict({
            'clause': nn.Linear(3, hidden_dim),
            'literal': nn.Linear(2, hidden_dim),
            'symbol': nn.Linear(4, hidden_dim),
            'term': nn.Linear(2, hidden_dim),
            'variable': nn.Linear(1, hidden_dim),
        })

        # Heterogeneous message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            # Define convolutions for each edge type
            edge_types = [
                ('clause', 'has_literal', 'literal'),
                ('literal', 'in_clause', 'clause'),
                ('literal', 'has_predicate', 'symbol'),
                ('symbol', 'predicate_of', 'literal'),
                ('term', 'has_functor', 'symbol'),
                ('symbol', 'functor_of', 'term'),
                ('variable', 'in_clause', 'clause'),
                ('clause', 'has_variable', 'variable'),
                ('term', 'has_var_arg', 'variable'),
                ('variable', 'arg_of', 'term'),
                ('literal', 'has_arg', 'term'),
                ('term', 'arg_of_lit', 'literal'),
                ('literal', 'has_var_arg', 'variable'),
                ('variable', 'arg_of_lit', 'literal'),
                ('term', 'has_subterm', 'term'),
                ('term', 'subterm_of', 'term'),
            ]
            for et in edge_types:
                conv_dict[et] = SAGEConv(hidden_dim, hidden_dim)

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            # Layer norms per node type
            self.norms.append(nn.ModuleDict({
                'clause': nn.LayerNorm(hidden_dim),
                'literal': nn.LayerNorm(hidden_dim),
                'symbol': nn.LayerNorm(hidden_dim),
                'term': nn.LayerNorm(hidden_dim),
                'variable': nn.LayerNorm(hidden_dim),
            }))

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode the problem graph.

        Returns dict mapping node type -> embeddings tensor.
        """
        # Project inputs
        x_dict = {}
        dev = next(self.parameters()).device
        for ntype in ['clause', 'literal', 'symbol', 'term', 'variable']:
            store = data.get(ntype, None) if hasattr(data, 'get') else None
            # Also try direct attribute access for HeteroData
            if store is None:
                try:
                    store = data[ntype]
                except (KeyError, AttributeError):
                    store = None
            if store is not None and hasattr(store, 'x') and store.x is not None and store.x.shape[0] > 0:
                x_dict[ntype] = self.input_projs[ntype](store.x)
            else:
                x_dict[ntype] = torch.zeros(0, self.hidden_dim, device=dev)

        # Collect edge indices
        edge_index_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_index'):
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # Message passing with residual connections
        for conv, norm_dict in zip(self.convs, self.norms):
            x_out = conv(x_dict, edge_index_dict)
            for ntype in x_dict:
                if ntype in x_out and x_out[ntype].shape[0] > 0:
                    # Residual + norm + activation
                    x_dict[ntype] = F.relu(
                        norm_dict[ntype](x_out[ntype] + x_dict[ntype])
                    )

        return x_dict


class PointerTreeDecoder(nn.Module):
    """Autoregressive tree decoder with pointer mechanism.

    At each step, predicts:
    1. Action type (NEW_LIT_POS/NEG, PRED, ARG_VAR, ARG_FUNC, END_ARGS, END_CLAUSE)
    2. Argument (pointer to symbol, or variable slot index)

    Uses GRU hidden state + attention over encoder outputs.
    """

    def __init__(self, hidden_dim: int = 128, max_vars: int = 20,
                 max_literals: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals

        # Embedding for action tokens (fed back as input at each step)
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)

        # GRU cell
        # Input: action_embedding + context_vector + argument_embedding
        self.gru = nn.GRUCell(hidden_dim * 3, hidden_dim)

        # Attention over all encoder nodes (for context vector)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)

        # Action type prediction — takes hidden state + literal count feature
        self.action_head = nn.Linear(hidden_dim + 1, NUM_ACTION_TYPES)

        # Pointer head: attention over symbol embeddings
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)

        # Coverage: learnable penalty weight for repeated pointer attention
        self.coverage_weight = nn.Parameter(torch.tensor(-1.0))

        # Variable slot prediction
        self.var_head = nn.Linear(hidden_dim, max_vars)

        # Argument embedding: project symbol embedding or variable slot
        self.arg_sym_proj = nn.Linear(hidden_dim, hidden_dim)
        self.var_slot_embed = nn.Embedding(max_vars, hidden_dim)

        # Initial hidden state projection from global graph embedding
        self.init_hidden = nn.Linear(hidden_dim, hidden_dim)

    def _compute_context(self, h: torch.Tensor,
                         all_node_embeds: torch.Tensor) -> torch.Tensor:
        """Attention over all encoder node embeddings."""
        # h: (batch, hidden)  all_node_embeds: (batch, N, hidden)
        query = self.attn_query(h).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.attn_key(all_node_embeds)     # (batch, N, hidden)
        scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, 1, N)
        scores = scores / (self.hidden_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, all_node_embeds).squeeze(1)  # (batch, hidden)
        return context

    def _pointer_scores(self, h: torch.Tensor,
                        symbol_embeds: torch.Tensor,
                        symbol_mask: torch.Tensor = None,
                        coverage: torch.Tensor = None) -> torch.Tensor:
        """Compute pointer attention scores over symbol embeddings.

        Args:
            coverage: (batch, S) cumulative attention over symbols.
                      Used to penalize re-attending to the same symbols.
        """
        query = self.pointer_query(h).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.pointer_key(symbol_embeds)       # (batch, S, hidden)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (batch, S)
        scores = scores / (self.hidden_dim ** 0.5)
        if coverage is not None:
            # Learned penalty for repeated attention (coverage_weight is negative)
            scores = scores + self.coverage_weight * torch.log1p(coverage)
        if symbol_mask is not None:
            scores = scores.masked_fill(~symbol_mask, float('-inf'))
        return scores

    def forward(self, x_dict: dict[str, torch.Tensor],
                target_actions: torch.Tensor,
                target_arguments: torch.Tensor,
                target_lengths: torch.Tensor,
                num_symbols: torch.Tensor,
                batch_data=None) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass for training.

        Args:
            x_dict: encoder outputs, dict of node_type -> (N_total, hidden)
            target_actions: (batch, max_seq_len) action types
            target_arguments: (batch, max_seq_len) arguments
            target_lengths: (batch,) sequence lengths
            num_symbols: (batch,) number of symbols per problem
            batch_data: the batched HeteroData (for batch assignments)

        Returns:
            dict with 'action_logits', 'pointer_logits', 'var_logits'
        """
        batch_size = target_actions.shape[0]
        max_len = target_actions.shape[1]
        device = target_actions.device

        # --- Prepare padded per-sample symbol and node embeddings ---
        # We need per-sample embeddings for pointer attention
        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'], batch_data['symbol'].batch
            if 'symbol' in batch_data and hasattr(batch_data['symbol'], 'batch')
            else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device),
            batch_size,
        )
        # Concatenate all node embeddings for context attention
        all_embeds_list = []
        all_batch_list = []
        for ntype in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if ntype in x_dict and x_dict[ntype].shape[0] > 0:
                all_embeds_list.append(x_dict[ntype])
                if hasattr(batch_data[ntype], 'batch'):
                    all_batch_list.append(batch_data[ntype].batch)
                else:
                    all_batch_list.append(
                        torch.zeros(x_dict[ntype].shape[0], dtype=torch.long, device=device)
                    )

        all_embeds_cat = torch.cat(all_embeds_list, dim=0)
        all_batch_cat = torch.cat(all_batch_list, dim=0)
        all_node_embeds, _ = self._pad_per_sample(
            all_embeds_cat, all_batch_cat, batch_size,
        )

        # --- Initial hidden state from global graph embedding ---
        # Global pooling: scatter mean of all node embeddings per sample
        global_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        global_embed.scatter_add_(0, all_batch_cat.unsqueeze(1).expand_as(all_embeds_cat), all_embeds_cat)
        counts = torch.bincount(all_batch_cat, minlength=batch_size).float().clamp(min=1).unsqueeze(1)
        global_embed = global_embed / counts
        h = torch.tanh(self.init_hidden(global_embed))  # (batch, hidden)

        # --- Autoregressive decoding with teacher forcing ---
        all_action_logits = []
        all_pointer_logits = []
        all_var_logits = []

        # Coverage: cumulative pointer attention for anti-repetition
        n_symbols = symbol_embeds.shape[1]
        coverage = torch.zeros(batch_size, n_symbols, device=device)
        # Literal counter per sample
        lit_count = torch.zeros(batch_size, 1, device=device)

        # First input: start token (use END_CLAUSE embedding as BOS)
        action_input = self.action_embed(
            torch.full((batch_size,), END_CLAUSE, dtype=torch.long, device=device)
        )
        arg_input = torch.zeros(batch_size, self.hidden_dim, device=device)

        for t in range(max_len):
            # Context via attention
            context = self._compute_context(h, all_node_embeds)

            # GRU step
            gru_input = torch.cat([action_input, context, arg_input], dim=-1)
            h = self.gru(gru_input, h)

            # Predict action type — include normalized literal count
            lit_feat = lit_count / self.max_literals
            action_logits = self.action_head(
                torch.cat([h, lit_feat], dim=-1)
            )  # (batch, NUM_ACTION_TYPES)
            all_action_logits.append(action_logits)

            # Predict pointer with coverage penalty
            ptr_logits = self._pointer_scores(
                h, symbol_embeds, symbol_mask=None, coverage=coverage,
            )
            all_pointer_logits.append(ptr_logits)

            # Update coverage with soft attention from this step
            ptr_attn = F.softmax(ptr_logits, dim=-1)
            coverage = coverage + ptr_attn

            # Predict variable slot
            var_logits = self.var_head(h)  # (batch, max_vars)
            all_var_logits.append(var_logits)

            # Teacher forcing: use ground truth for next input
            if t < max_len - 1:
                next_action = target_actions[:, t]
                next_arg = target_arguments[:, t]
                action_input = self.action_embed(next_action)

                # Track literal count
                is_new_lit = (next_action == NEW_LIT_POS) | (next_action == NEW_LIT_NEG)
                lit_count = lit_count + is_new_lit.float().unsqueeze(1)

                # Build argument embedding based on action type (vectorized)
                arg_input = torch.zeros(batch_size, self.hidden_dim, device=device)

                # Symbol pointer args (PRED or ARG_FUNC)
                ptr_mask = (next_action == PRED) | (next_action == ARG_FUNC)
                if ptr_mask.any():
                    ptr_idx = next_arg[ptr_mask].clamp(0, symbol_embeds.shape[1] - 1)
                    batch_idx = torch.arange(batch_size, device=device)[ptr_mask]
                    sym_vecs = symbol_embeds[batch_idx, ptr_idx]  # (n_ptr, hidden)
                    arg_input[ptr_mask] = self.arg_sym_proj(sym_vecs)

                # Variable args
                var_mask = next_action == ARG_VAR
                if var_mask.any():
                    var_slots = next_arg[var_mask].clamp(0, self.max_vars - 1)
                    arg_input[var_mask] = self.var_slot_embed(var_slots)

        action_logits = torch.stack(all_action_logits, dim=1)   # (batch, seq, 7)
        pointer_logits = torch.stack(all_pointer_logits, dim=1)  # (batch, seq, max_sym)
        var_logits = torch.stack(all_var_logits, dim=1)          # (batch, seq, max_vars)

        return {
            'action_logits': action_logits,
            'pointer_logits': pointer_logits,
            'var_logits': var_logits,
        }

    def _pad_per_sample(self, embeds: torch.Tensor, batch_assign: torch.Tensor,
                        batch_size: int):
        """Pad variable-length per-sample embeddings to (batch, max_N, hidden).

        Returns (padded_embeds, mask).
        """
        device = embeds.device
        counts = torch.bincount(batch_assign, minlength=batch_size)
        max_n = counts.max().item() if counts.numel() > 0 else 0
        if max_n == 0:
            max_n = 1

        padded = torch.zeros(batch_size, max_n, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

        # Compute per-sample offsets for scatter
        sorted_idx = torch.argsort(batch_assign, stable=True)
        sorted_batch = batch_assign[sorted_idx]
        sorted_embeds = embeds[sorted_idx]

        # Position within each sample
        offsets = torch.zeros_like(batch_assign)
        for i in range(batch_size):
            sample_mask = sorted_batch == i
            n = sample_mask.sum().item()
            if n > 0:
                positions = torch.arange(n, device=device)
                offsets[sample_mask] = positions
                padded[i, :n] = sorted_embeds[sample_mask]
                mask[i, :n] = True

        return padded, mask

    @torch.no_grad()
    def generate(self, x_dict: dict[str, torch.Tensor],
                 batch_data=None,
                 max_steps: int = 80,
                 temperature: float = 1.0) -> list[list[tuple[int, int]]]:
        """Autoregressive generation (greedy or with temperature).

        Returns list of sequences, one per sample in the batch.
        """
        device = next(self.parameters()).device

        # Handle single sample (no batch dimension)
        if batch_data is None:
            batch_size = 1
            # Assume all nodes belong to sample 0
            symbol_batch = torch.zeros(
                x_dict['symbol'].shape[0], dtype=torch.long, device=device
            )
        else:
            symbol_batch = (
                batch_data['symbol'].batch
                if hasattr(batch_data['symbol'], 'batch')
                else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device)
            )
            batch_size = symbol_batch.max().item() + 1 if symbol_batch.numel() > 0 else 1

        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'], symbol_batch, batch_size,
        )

        # All node embeddings for context
        all_embeds_list = []
        all_batch_list = []
        for ntype in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if ntype in x_dict and x_dict[ntype].shape[0] > 0:
                all_embeds_list.append(x_dict[ntype])
                if batch_data is not None and hasattr(batch_data[ntype], 'batch'):
                    all_batch_list.append(batch_data[ntype].batch)
                else:
                    all_batch_list.append(
                        torch.zeros(x_dict[ntype].shape[0], dtype=torch.long, device=device)
                    )
        all_embeds_cat = torch.cat(all_embeds_list, dim=0)
        all_batch_cat = torch.cat(all_batch_list, dim=0)
        all_node_embeds, _ = self._pad_per_sample(
            all_embeds_cat, all_batch_cat, batch_size,
        )

        # Initial hidden state
        global_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        for i in range(batch_size):
            mask = all_batch_cat == i
            if mask.any():
                global_embed[i] = all_embeds_cat[mask].mean(dim=0)
        h = torch.tanh(self.init_hidden(global_embed))

        # Generate
        sequences = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        lit_counts = [0] * batch_size

        # Coverage and literal tracking
        n_symbols = symbol_embeds.shape[1]
        coverage = torch.zeros(batch_size, n_symbols, device=device)
        lit_count_t = torch.zeros(batch_size, 1, device=device)

        # Track recently generated (action, arg) pairs for repetition penalty
        recent_preds = [[] for _ in range(batch_size)]  # list of recent predicate indices

        action_input = self.action_embed(
            torch.full((batch_size,), END_CLAUSE, dtype=torch.long, device=device)
        )
        arg_input = torch.zeros(batch_size, self.hidden_dim, device=device)

        for step in range(max_steps):
            context = self._compute_context(h, all_node_embeds)
            gru_input = torch.cat([action_input, context, arg_input], dim=-1)
            h = self.gru(gru_input, h)

            # Predict action with literal count feature
            lit_feat = lit_count_t / self.max_literals
            action_logits = self.action_head(
                torch.cat([h, lit_feat], dim=-1)
            ) / temperature

            # Force END_CLAUSE if max literals reached
            for i in range(batch_size):
                if not done[i] and lit_counts[i] >= self.max_literals:
                    action_logits[i, NEW_LIT_POS] = float('-inf')
                    action_logits[i, NEW_LIT_NEG] = float('-inf')
                    action_logits[i, END_CLAUSE] += 5.0

            actions = torch.argmax(action_logits, dim=-1)

            # Predict pointer with coverage
            ptr_logits = self._pointer_scores(
                h, symbol_embeds, symbol_mask, coverage=coverage,
            )

            # Repetition penalty: reduce score for predicates used in recent literals
            for i in range(batch_size):
                if not done[i] and recent_preds[i]:
                    for pred_idx in recent_preds[i][-3:]:  # penalize last 3
                        if pred_idx < ptr_logits.shape[1]:
                            ptr_logits[i, pred_idx] -= 2.0

            var_logits = self.var_head(h)

            # Update coverage
            ptr_attn = F.softmax(ptr_logits, dim=-1)
            coverage = coverage + ptr_attn

            # Record and prepare next input
            arg_input = torch.zeros(batch_size, self.hidden_dim, device=device)

            for i in range(batch_size):
                if done[i]:
                    continue
                act = actions[i].item()

                # Track literal count
                if act in (NEW_LIT_POS, NEW_LIT_NEG):
                    lit_counts[i] += 1
                    lit_count_t[i, 0] = lit_counts[i]

                if act in (PRED, ARG_FUNC):
                    arg_val = torch.argmax(ptr_logits[i]).item()
                    if arg_val < symbol_embeds.shape[1] and symbol_mask[i, arg_val]:
                        arg_input[i] = self.arg_sym_proj(symbol_embeds[i, arg_val])
                    # Track predicate for repetition penalty
                    if act == PRED:
                        recent_preds[i].append(arg_val)
                elif act == ARG_VAR:
                    arg_val = torch.argmax(var_logits[i]).item()
                    arg_val = min(arg_val, self.max_vars - 1)
                    arg_input[i] = self.var_slot_embed(
                        torch.tensor(arg_val, device=device)
                    )
                else:
                    arg_val = 0

                sequences[i].append((act, arg_val))
                if act == END_CLAUSE:
                    done[i] = True

            action_input = self.action_embed(actions)

            if all(done):
                break

        return sequences


class ConjectureModel(nn.Module):
    """Full model: GNN encoder + pointer tree decoder."""

    def __init__(self, hidden_dim: int = 128, num_gnn_layers: int = 6,
                 max_vars: int = 20, max_literals: int = 8):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = PointerTreeDecoder(hidden_dim, max_vars, max_literals)
        self.hidden_dim = hidden_dim

    def forward(self, batch_data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing."""
        # Encode
        x_dict = self.encoder(batch_data)

        # Decode
        return self.decoder(
            x_dict,
            batch_data.target_actions,
            batch_data.target_arguments,
            batch_data.target_length,
            batch_data.num_symbols,
            batch_data=batch_data,
        )

    @torch.no_grad()
    def generate(self, data: HeteroData, max_steps: int = 80,
                 temperature: float = 1.0) -> list[list[tuple[int, int]]]:
        """Generate conjectures for given problem(s)."""
        self.eval()
        x_dict = self.encoder(data)
        return self.decoder.generate(
            x_dict, batch_data=data,
            max_steps=max_steps, temperature=temperature,
        )


if __name__ == '__main__':
    # Quick smoke test
    from conjecture_gen.tptp_parser import parse_problem_file
    from conjecture_gen.graph_builder import clauses_to_graph

    clauses = parse_problem_file('problems/l100_fomodel0')
    graph = clauses_to_graph(clauses)

    model = ConjectureModel(hidden_dim=64, num_gnn_layers=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test encoding
    x_dict = model.encoder(graph)
    for ntype, emb in x_dict.items():
        print(f"  {ntype}: {emb.shape}")

    # Test generation on single sample
    seqs = model.generate(graph)
    print(f"\nGenerated sequence (untrained): {seqs[0][:20]}...")
