"""
Plan D: GNN Encoder + SSM (State-Space Model) Sequence Decoder

Uses a lightweight SSM-style decoder inspired by Mamba: selective state
updates with input-dependent gating, giving RNN-like linear-time inference
with better long-range memory than a vanilla GRU.

Works on CPU (no CUDA-only Mamba dependency).

The decoder:
1. Embeds (action, argument) pairs as tokens
2. Processes them through SSM blocks with cross-attention to GNN memory
3. Outputs action/pointer/variable predictions

Key vs GRU: the SSM has a structured state matrix that selectively
retains or forgets information, plus input-dependent gating for
selective state updates (the core Mamba idea).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from conjecture_gen.model import HeteroGNNEncoder
from conjecture_gen.target_encoder import (
    NUM_ACTION_TYPES, NEW_LIT_POS, NEW_LIT_NEG, PRED,
    ARG_VAR, ARG_FUNC, END_ARGS, END_CLAUSE,
    decode_sequence,
)


class SSMBlock(nn.Module):
    """Selective State Space Model block (Mamba-style, pure PyTorch).

    Implements the core S6 selective scan: input-dependent A, B, C matrices
    that allow the model to selectively remember/forget based on input content.

    For training: parallel scan via associative scan (or sequential for simplicity).
    For inference: simple recurrence.
    """

    def __init__(self, hidden_dim: int, state_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Input projection (expand then contract, like Mamba)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        # SSM parameters: input-dependent discretization
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)  # delta (step size)
        self.A_log = nn.Parameter(torch.randn(hidden_dim, state_dim))  # log of A matrix
        self.B_proj = nn.Linear(hidden_dim, state_dim)
        self.C_proj = nn.Linear(hidden_dim, state_dim)
        self.D = nn.Parameter(torch.ones(hidden_dim))  # skip connection

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence.
        x: (batch, seq_len, hidden_dim)
        """
        B, L, D = x.shape
        residual = x

        # Input projection and split into two paths
        xz = self.in_proj(x)  # (B, L, 2*D)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, D)
        z = F.silu(z)  # gating

        # SSM parameters (input-dependent)
        dt = F.softplus(self.dt_proj(x_ssm))  # (B, L, D) - step sizes
        B_mat = self.B_proj(x_ssm)  # (B, L, N) - input matrix
        C_mat = self.C_proj(x_ssm)  # (B, L, N) - output matrix
        A = -torch.exp(self.A_log)  # (D, N) - state matrix (negative for stability)

        # Selective scan (sequential for simplicity & CPU compatibility)
        y = self._selective_scan(x_ssm, dt, A, B_mat, C_mat)

        # Skip connection + gating
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        y = y * z  # gating

        # Output
        y = self.out_proj(y)
        return self.norm(y + residual)

    def _selective_scan(self, x, dt, A, B, C):
        """Memory-efficient sequential selective scan.
        Computes discretization per-step to avoid materializing full (B,L,D,N) tensors.
        """
        batch, seq_len, D = x.shape
        N = A.shape[1]
        device = x.device

        # A is (D, N) — shared, small
        h = torch.zeros(batch, D, N, device=device)
        ys = []

        for t in range(seq_len):
            # Discretize per-step: only (B, D, N) not (B, L, D, N)
            dt_t = dt[:, t]           # (B, D)
            dA_t = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, D, N)
            dB_t = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1)       # (B, D, N)

            h = dA_t * h + dB_t * x[:, t].unsqueeze(-1)  # (B, D, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)


class SSMDecoder(nn.Module):
    """SSM-based autoregressive decoder with pointer mechanism."""

    def __init__(self, hidden_dim: int = 128, max_vars: int = 20,
                 max_literals: int = 8, num_layers: int = 3,
                 state_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals

        # Input embeddings
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)
        self.arg_sym_proj = nn.Linear(hidden_dim, hidden_dim)
        self.var_slot_embed = nn.Embedding(max_vars, hidden_dim)
        self.input_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # SSM layers
        self.ssm_layers = nn.ModuleList([
            SSMBlock(hidden_dim, state_dim) for _ in range(num_layers)
        ])

        # Linear cross-attention to encoder (memory-efficient)
        # Instead of full (seq×mem) attention, pre-computes a memory summary
        # then gates the decoder hidden state with it.
        self.mem_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.gate_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Output heads
        self.action_head = nn.Linear(hidden_dim, NUM_ACTION_TYPES)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.var_head = nn.Linear(hidden_dim, max_vars)

    def _pad_per_sample(self, embeds, batch_assign, batch_size):
        device = embeds.device
        counts = torch.bincount(batch_assign, minlength=batch_size)
        max_n = max(counts.max().item(), 1)
        padded = torch.zeros(batch_size, max_n, self.hidden_dim, device=device)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool, device=device)
        sorted_idx = torch.argsort(batch_assign, stable=True)
        sorted_batch = batch_assign[sorted_idx]
        sorted_embeds = embeds[sorted_idx]
        for i in range(batch_size):
            sm = sorted_batch == i
            n = sm.sum().item()
            if n > 0:
                padded[i, :n] = sorted_embeds[sm]
                mask[i, :n] = True
        return padded, mask

    def _build_input_embeds(self, actions, arguments, symbol_embeds):
        B, T = actions.shape
        device = actions.device
        act_emb = self.action_embed(actions)
        arg_emb = torch.zeros(B, T, self.hidden_dim, device=device)

        ptr_mask = (actions == PRED) | (actions == ARG_FUNC)
        if ptr_mask.any():
            idx = arguments[ptr_mask].clamp(0, symbol_embeds.shape[1] - 1)
            bi = torch.arange(B, device=device).unsqueeze(1).expand_as(actions)[ptr_mask]
            arg_emb[ptr_mask] = self.arg_sym_proj(symbol_embeds[bi, idx])

        var_mask = actions == ARG_VAR
        if var_mask.any():
            arg_emb[var_mask] = self.var_slot_embed(
                arguments[var_mask].clamp(0, self.max_vars - 1)
            )

        return self.input_combine(torch.cat([act_emb, arg_emb], dim=-1))

    def _pointer_scores(self, h, symbol_embeds, symbol_mask=None):
        if h.dim() == 2:
            h = h.unsqueeze(1)
        q = self.pointer_query(h)
        k = self.pointer_key(symbol_embeds)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        if symbol_mask is not None:
            scores = scores.masked_fill(~symbol_mask.unsqueeze(1), float('-inf'))
        if h.shape[1] == 1:
            scores = scores.squeeze(1)
        return scores

    def forward(self, x_dict, target_actions, target_arguments,
                target_lengths, num_symbols, batch_data=None):
        batch_size = target_actions.shape[0]
        max_len = target_actions.shape[1]
        device = target_actions.device

        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'],
            batch_data['symbol'].batch if hasattr(batch_data['symbol'], 'batch')
            else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device),
            batch_size,
        )

        all_e, all_b = [], []
        for nt in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if nt in x_dict and x_dict[nt].shape[0] > 0:
                all_e.append(x_dict[nt])
                if hasattr(batch_data[nt], 'batch'):
                    all_b.append(batch_data[nt].batch)
                else:
                    all_b.append(torch.zeros(x_dict[nt].shape[0], dtype=torch.long, device=device))
        memory, mem_mask = self._pad_per_sample(
            torch.cat(all_e), torch.cat(all_b), batch_size,
        )

        # Shift right
        bos_a = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        bos_arg = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        dec_actions = torch.cat([bos_a, target_actions[:, :-1]], dim=1)
        dec_args = torch.cat([bos_arg, target_arguments[:, :-1]], dim=1)

        tok_embeds = self._build_input_embeds(dec_actions, dec_args, symbol_embeds)

        # SSM layers with gated linear cross-attention
        # Pre-compute memory summaries (one per layer, no seq_len dimension)
        h = tok_embeds
        for ssm, mp, gp, cn in zip(self.ssm_layers, self.mem_projs, self.gate_projs, self.cross_norms):
            h = ssm(h)
            # Memory summary: mean-pool encoder memory (B, H), project, broadcast
            mem_summary = memory.mean(dim=1)  # (B, H)
            mem_ctx = mp(mem_summary).unsqueeze(1).expand_as(h)  # (B, L, H)
            # Gated fusion
            gate = torch.sigmoid(gp(torch.cat([h, mem_ctx], dim=-1)))
            h = cn(h + gate * mem_ctx)

        action_logits = self.action_head(h)
        pointer_logits = self._pointer_scores(h, symbol_embeds, symbol_mask)
        var_logits = self.var_head(h)

        return {
            'action_logits': action_logits,
            'pointer_logits': pointer_logits,
            'var_logits': var_logits,
        }

    @torch.no_grad()
    def generate(self, x_dict, batch_data=None, max_steps=80, temperature=1.0):
        device = next(self.parameters()).device

        if batch_data is None:
            batch_size = 1
            sb = torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device)
        else:
            sb = (batch_data['symbol'].batch if hasattr(batch_data['symbol'], 'batch')
                  else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device))
            batch_size = sb.max().item() + 1 if sb.numel() > 0 else 1

        symbol_embeds, symbol_mask = self._pad_per_sample(x_dict['symbol'], sb, batch_size)

        all_e, all_b = [], []
        for nt in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if nt in x_dict and x_dict[nt].shape[0] > 0:
                all_e.append(x_dict[nt])
                if batch_data is not None and hasattr(batch_data[nt], 'batch'):
                    all_b.append(batch_data[nt].batch)
                else:
                    all_b.append(torch.zeros(x_dict[nt].shape[0], dtype=torch.long, device=device))
        memory, mem_mask = self._pad_per_sample(
            torch.cat(all_e), torch.cat(all_b), batch_size,
        )

        sequences = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        lit_counts = [0] * batch_size

        all_actions = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        all_args = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        for step in range(max_steps):
            tok = self._build_input_embeds(all_actions, all_args, symbol_embeds)
            h = tok
            for ssm, mp, gp, cn in zip(self.ssm_layers, self.mem_projs, self.gate_projs, self.cross_norms):
                h = ssm(h)
                mem_summary = memory.mean(dim=1)
                mem_ctx = mp(mem_summary).unsqueeze(1).expand_as(h)
                gate = torch.sigmoid(gp(torch.cat([h, mem_ctx], dim=-1)))
                h = cn(h + gate * mem_ctx)

            h_last = h[:, -1, :]
            action_logits = self.action_head(h_last) / temperature

            for i in range(batch_size):
                if not done[i] and lit_counts[i] >= self.max_literals:
                    action_logits[i, NEW_LIT_POS] = float('-inf')
                    action_logits[i, NEW_LIT_NEG] = float('-inf')
                    action_logits[i, END_CLAUSE] += 5.0

            actions = torch.argmax(action_logits, dim=-1)
            ptr_logits = self._pointer_scores(h_last, symbol_embeds, symbol_mask)
            var_logits = self.var_head(h_last)

            new_args = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                if done[i]:
                    continue
                act = actions[i].item()
                if act in (NEW_LIT_POS, NEW_LIT_NEG):
                    lit_counts[i] += 1
                if act in (PRED, ARG_FUNC):
                    arg_val = ptr_logits[i].argmax().item()
                elif act == ARG_VAR:
                    arg_val = min(var_logits[i].argmax().item(), self.max_vars - 1)
                else:
                    arg_val = 0
                new_args[i] = arg_val
                sequences[i].append((act, arg_val))
                if act == END_CLAUSE:
                    done[i] = True

            if all(done):
                break
            all_actions = torch.cat([all_actions, actions.unsqueeze(1)], dim=1)
            all_args = torch.cat([all_args, new_args.unsqueeze(1)], dim=1)

        return sequences


class ConjectureModelD(nn.Module):
    """Plan D: GNN encoder + SSM decoder."""

    def __init__(self, hidden_dim=128, num_gnn_layers=6, max_vars=20, **kwargs):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = SSMDecoder(hidden_dim, max_vars)
        self.hidden_dim = hidden_dim

    def forward(self, batch_data):
        x_dict = self.encoder(batch_data)
        return self.decoder(
            x_dict, batch_data.target_actions, batch_data.target_arguments,
            batch_data.target_length, batch_data.num_symbols,
            batch_data=batch_data,
        )

    @torch.no_grad()
    def generate(self, data, max_steps=80, temperature=1.0):
        self.eval()
        x_dict = self.encoder(data)
        return self.decoder.generate(x_dict, batch_data=data,
                                     max_steps=max_steps, temperature=temperature)


if __name__ == '__main__':
    from conjecture_gen.tptp_parser import parse_problem_file
    from conjecture_gen.graph_builder import clauses_to_graph

    clauses = parse_problem_file('problems/l100_fomodel0')
    graph = clauses_to_graph(clauses)

    model = ConjectureModelD(hidden_dim=64, num_gnn_layers=4)
    print(f"Model D parameters: {sum(p.numel() for p in model.parameters()):,}")

    seqs = model.generate(graph)
    decoded = decode_sequence(seqs[0], graph.symbol_names)
    print(f"Generated (untrained): {decoded}")
