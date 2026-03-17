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
    """Transformer-based autoregressive tree decoder with pointer mechanism.

    Uses causal self-attention over the generated sequence so the decoder
    can see its full history (preventing repetition), plus cross-attention
    over GNN encoder outputs for context and pointer-based symbol selection.
    """

    def __init__(self, hidden_dim: int = 128, max_vars: int = 20,
                 max_literals: int = 8, num_layers: int = 3,
                 nhead: int = 4, max_seq_len: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals
        self.max_seq_len = max_seq_len

        # Input embeddings: action + argument combined into one token embedding
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)
        self.arg_sym_proj = nn.Linear(hidden_dim, hidden_dim)
        self.var_slot_embed = nn.Embedding(max_vars, hidden_dim)
        self.input_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer decoder layers (self-attn + cross-attn + FFN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
        )

        # Output heads
        self.action_head = nn.Linear(hidden_dim, NUM_ACTION_TYPES)

        # Pointer head: attention over symbol embeddings
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)

        # Variable slot prediction
        self.var_head = nn.Linear(hidden_dim, max_vars)

    def _pointer_scores(self, h: torch.Tensor,
                        symbol_embeds: torch.Tensor,
                        symbol_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute pointer attention scores over symbol embeddings.

        h: (batch, hidden) or (batch, seq, hidden)
        symbol_embeds: (batch, S, hidden)
        """
        if h.dim() == 2:
            h = h.unsqueeze(1)  # (batch, 1, hidden)
        query = self.pointer_query(h)             # (batch, seq, hidden)
        keys = self.pointer_key(symbol_embeds)    # (batch, S, hidden)
        scores = torch.bmm(query, keys.transpose(1, 2))  # (batch, seq, S)
        scores = scores / (self.hidden_dim ** 0.5)
        if symbol_mask is not None:
            scores = scores.masked_fill(~symbol_mask.unsqueeze(1), float('-inf'))
        if h.shape[1] == 1:
            scores = scores.squeeze(1)  # back to (batch, S)
        return scores

    def _build_input_embeds(self, actions: torch.Tensor,
                            arguments: torch.Tensor,
                            symbol_embeds: torch.Tensor) -> torch.Tensor:
        """Build input token embeddings from (action, argument) pairs.

        actions: (batch, seq) action types
        arguments: (batch, seq) argument values
        symbol_embeds: (batch, S, hidden) for looking up symbol embeddings

        Returns: (batch, seq, hidden)
        """
        B, T = actions.shape
        device = actions.device

        # Action embeddings
        act_emb = self.action_embed(actions)  # (B, T, hidden)

        # Argument embeddings (depends on action type)
        arg_emb = torch.zeros(B, T, self.hidden_dim, device=device)

        # Symbol pointer args (PRED or ARG_FUNC)
        ptr_mask = (actions == PRED) | (actions == ARG_FUNC)
        if ptr_mask.any():
            ptr_idx = arguments[ptr_mask].clamp(0, symbol_embeds.shape[1] - 1)
            # Gather per-sample symbol embeddings
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(actions)[ptr_mask]
            sym_vecs = symbol_embeds[batch_indices, ptr_idx]
            arg_emb[ptr_mask] = self.arg_sym_proj(sym_vecs)

        # Variable args
        var_mask = actions == ARG_VAR
        if var_mask.any():
            var_slots = arguments[var_mask].clamp(0, self.max_vars - 1)
            arg_emb[var_mask] = self.var_slot_embed(var_slots)

        # Combine action + argument
        combined = self.input_combine(torch.cat([act_emb, arg_emb], dim=-1))
        return combined

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for self-attention."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x_dict: dict[str, torch.Tensor],
                target_actions: torch.Tensor,
                target_arguments: torch.Tensor,
                target_lengths: torch.Tensor,
                num_symbols: torch.Tensor,
                batch_data=None) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass. Processes all timesteps in parallel."""
        batch_size = target_actions.shape[0]
        max_len = target_actions.shape[1]
        device = target_actions.device

        # --- Prepare encoder outputs ---
        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'], batch_data['symbol'].batch
            if hasattr(batch_data['symbol'], 'batch')
            else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device),
            batch_size,
        )

        # All node embeddings for cross-attention memory
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
        memory, memory_mask = self._pad_per_sample(
            all_embeds_cat, all_batch_cat, batch_size,
        )
        # memory: (batch, N_max, hidden), memory_mask: (batch, N_max) bool

        # --- Build decoder input sequence ---
        # Shift right: prepend BOS token, drop last target token
        bos_action = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        bos_arg = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        dec_actions = torch.cat([bos_action, target_actions[:, :-1]], dim=1)
        dec_args = torch.cat([bos_arg, target_arguments[:, :-1]], dim=1)

        # Token embeddings + positional encoding
        tok_embeds = self._build_input_embeds(dec_actions, dec_args, symbol_embeds)
        positions = torch.arange(max_len, device=device).unsqueeze(0).clamp(max=self.max_seq_len - 1)
        tok_embeds = tok_embeds + self.pos_embed(positions)

        # --- Transformer decoder ---
        causal_mask = self._get_causal_mask(max_len, device)
        # Padding mask for decoder: positions beyond target length
        tgt_key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)
        # Padding mask for memory
        memory_key_padding_mask = ~memory_mask

        hidden = self.transformer_decoder(
            tgt=tok_embeds,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (batch, seq, hidden)

        # --- Output heads ---
        action_logits = self.action_head(hidden)  # (batch, seq, 7)

        # Pointer logits: each position attends to symbols
        pointer_logits = self._pointer_scores(hidden, symbol_embeds, symbol_mask)  # (batch, seq, S)

        var_logits = self.var_head(hidden)  # (batch, seq, max_vars)

        return {
            'action_logits': action_logits,
            'pointer_logits': pointer_logits,
            'var_logits': var_logits,
        }

    def _pad_per_sample(self, embeds: torch.Tensor, batch_assign: torch.Tensor,
                        batch_size: int):
        """Pad variable-length per-sample embeddings to (batch, max_N, hidden)."""
        device = embeds.device
        counts = torch.bincount(batch_assign, minlength=batch_size)
        max_n = counts.max().item() if counts.numel() > 0 else 0
        if max_n == 0:
            max_n = 1

        padded = torch.zeros(batch_size, max_n, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)

        sorted_idx = torch.argsort(batch_assign, stable=True)
        sorted_batch = batch_assign[sorted_idx]
        sorted_embeds = embeds[sorted_idx]

        for i in range(batch_size):
            sample_mask = sorted_batch == i
            n = sample_mask.sum().item()
            if n > 0:
                padded[i, :n] = sorted_embeds[sample_mask]
                mask[i, :n] = True

        return padded, mask

    @torch.no_grad()
    def generate(self, x_dict: dict[str, torch.Tensor],
                 batch_data=None,
                 max_steps: int = 80,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0) -> list[list[tuple[int, int]]]:
        """Autoregressive generation with top-k/nucleus sampling and arity constraints."""
        from conjecture_gen.sampling import sample_from_logits, ArityConstraint
        device = next(self.parameters()).device

        # Handle single sample (no batch dimension)
        if batch_data is None:
            batch_size = 1
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

        # Build memory (all encoder node embeddings)
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
        memory, memory_mask = self._pad_per_sample(
            all_embeds_cat, all_batch_cat, batch_size,
        )
        memory_key_padding_mask = ~memory_mask

        # Generate autoregressively with arity constraints
        sym_arities = getattr(batch_data, 'symbol_arities', None)
        if sym_arities is None:
            sym_arities = [0] * symbol_embeds.shape[1]
        elif isinstance(sym_arities, list) and sym_arities and isinstance(sym_arities[0], list):
            sym_arities = sym_arities[0]
        elif not isinstance(sym_arities, list):
            sym_arities = [0] * symbol_embeds.shape[1]
        arity_con = ArityConstraint(sym_arities, batch_size)

        sequences = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        lit_counts = [0] * batch_size

        all_actions = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        all_args = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        for step in range(max_steps):
            seq_len = all_actions.shape[1]

            tok_embeds = self._build_input_embeds(all_actions, all_args, symbol_embeds)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).clamp(max=self.max_seq_len - 1)
            tok_embeds = tok_embeds + self.pos_embed(positions)

            causal_mask = self._get_causal_mask(seq_len, device)
            hidden = self.transformer_decoder(
                tgt=tok_embeds,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            h_last = hidden[:, -1, :]

            # Predict action with arity constraints
            action_logits = self.action_head(h_last)

            for i in range(batch_size):
                if done[i]:
                    continue
                if lit_counts[i] >= self.max_literals:
                    action_logits[i, NEW_LIT_POS] = float('-inf')
                    action_logits[i, NEW_LIT_NEG] = float('-inf')
                    if not arity_con.stacks[i]:
                        action_logits[i, END_CLAUSE] += 5.0
                arity_con.constrain_actions(i, action_logits[i])

            actions = sample_from_logits(action_logits, temperature, top_k, top_p)

            ptr_logits = self._pointer_scores(h_last, symbol_embeds, symbol_mask)
            var_logits = self.var_head(h_last)

            new_args = torch.zeros(batch_size, dtype=torch.long, device=device)
            ptr_sampled = sample_from_logits(ptr_logits, temperature, top_k, top_p)
            var_sampled = sample_from_logits(var_logits, temperature, top_k, top_p)

            for i in range(batch_size):
                if done[i]:
                    continue
                act = actions[i].item()

                if act in (NEW_LIT_POS, NEW_LIT_NEG):
                    lit_counts[i] += 1

                if act in (PRED, ARG_FUNC):
                    arg_val = ptr_sampled[i].item()
                elif act == ARG_VAR:
                    arg_val = min(var_sampled[i].item(), self.max_vars - 1)
                else:
                    arg_val = 0

                new_args[i] = arg_val
                sequences[i].append((act, arg_val))
                arity_con.notify_action(i, act, arg_val)
                if act == END_CLAUSE:
                    done[i] = True

            if all(done):
                break

            # Append to sequence for next step
            all_actions = torch.cat([all_actions, actions.unsqueeze(1)], dim=1)
            all_args = torch.cat([all_args, new_args.unsqueeze(1)], dim=1)

        return sequences


class ConjectureModel(nn.Module):
    """Full model: GNN encoder + pointer tree decoder."""

    def __init__(self, hidden_dim: int = 128, num_gnn_layers: int = 6,
                 max_vars: int = 20, max_literals: int = 8,
                 dec_layers: int = 3, dec_nhead: int = 4):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = PointerTreeDecoder(
            hidden_dim, max_vars, max_literals,
            num_layers=dec_layers, nhead=dec_nhead,
        )
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
                 temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 0.0) -> list[list[tuple[int, int]]]:
        """Generate conjectures for given problem(s)."""
        self.eval()
        x_dict = self.encoder(data)
        return self.decoder.generate(
            x_dict, batch_data=data,
            max_steps=max_steps, temperature=temperature,
            top_k=top_k, top_p=top_p,
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
