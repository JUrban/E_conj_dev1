"""
Plan E: Subgraph Completion Model

Instead of autoregressive generation, pre-instantiate K "blank literal slots"
with potential connections to all symbols, then predict which edges to
activate. One-shot generation — no sequential decoding.

Architecture:
  1. Encode problem graph with GNN (same encoder as Plan A)
  2. Add K blank literal slot nodes + M blank term slots per literal
  3. Run additional GNN rounds over the expanded graph
  4. For each slot, predict:
     - Is this slot active? (binary)
     - If literal slot: polarity, which predicate (pointer)
     - If term slot: variable or function (pointer), argument connections

This is trained as a standard supervised classification problem.
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


class SubgraphCompletionDecoder(nn.Module):
    """One-shot conjecture generation via slot prediction.

    Pre-allocates K literal slots. For each slot, predicts:
    - active (binary): is this literal part of the conjecture?
    - polarity (binary): positive or negative literal
    - predicate (pointer): which predicate symbol
    - arguments: for each of M arg positions, predict symbol/variable/none

    All predictions are made in parallel after GNN message passing.
    """

    def __init__(self, hidden_dim: int = 128, max_vars: int = 20,
                 max_literals: int = 6, max_args: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals
        self.max_args = max_args

        # Learnable slot embeddings
        self.literal_slot_embed = nn.Parameter(
            torch.randn(max_literals, hidden_dim) * 0.02
        )
        self.arg_slot_embed = nn.Parameter(
            torch.randn(max_literals, max_args, hidden_dim) * 0.02
        )

        # Cross-attention: slots attend to encoder graph
        self.slot_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True,
        )
        self.slot_norm1 = nn.LayerNorm(hidden_dim)

        # Self-attention among slots (so they coordinate)
        self.slot_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True,
        )
        self.slot_norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.slot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.slot_norm3 = nn.LayerNorm(hidden_dim)

        # Repeat the attention block
        self.slot_cross_attn2 = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True,
        )
        self.slot_norm4 = nn.LayerNorm(hidden_dim)
        self.slot_self_attn2 = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True,
        )
        self.slot_norm5 = nn.LayerNorm(hidden_dim)
        self.slot_ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.slot_norm6 = nn.LayerNorm(hidden_dim)

        # Output heads per literal slot
        self.active_head = nn.Linear(hidden_dim, 1)  # is slot active?
        self.polarity_head = nn.Linear(hidden_dim, 1)  # positive(0) or negative(1)

        # Predicate pointer
        self.pred_query = nn.Linear(hidden_dim, hidden_dim)
        self.pred_key = nn.Linear(hidden_dim, hidden_dim)

        # Argument heads per position
        # For each arg position: predict type (none/variable/function_pointer)
        self.arg_type_head = nn.Linear(hidden_dim, 3)  # none, variable, function
        self.arg_var_head = nn.Linear(hidden_dim, max_vars)
        self.arg_func_query = nn.Linear(hidden_dim, hidden_dim)
        self.arg_func_key = nn.Linear(hidden_dim, hidden_dim)

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
            sample_mask = sorted_batch == i
            n = sample_mask.sum().item()
            if n > 0:
                padded[i, :n] = sorted_embeds[sample_mask]
                mask[i, :n] = True
        return padded, mask

    def forward(self, x_dict, target_actions, target_arguments,
                target_lengths, num_symbols, batch_data=None):
        """Forward pass: predict slot activations and contents.

        We convert the sequential target into a slot-based target for loss.
        """
        batch_size = target_actions.shape[0]
        device = target_actions.device

        # Get symbol embeddings and all-node memory
        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'],
            batch_data['symbol'].batch if hasattr(batch_data['symbol'], 'batch')
            else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device),
            batch_size,
        )

        all_embeds_list, all_batch_list = [], []
        for ntype in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if ntype in x_dict and x_dict[ntype].shape[0] > 0:
                all_embeds_list.append(x_dict[ntype])
                if hasattr(batch_data[ntype], 'batch'):
                    all_batch_list.append(batch_data[ntype].batch)
                else:
                    all_batch_list.append(torch.zeros(x_dict[ntype].shape[0], dtype=torch.long, device=device))
        memory, memory_mask = self._pad_per_sample(
            torch.cat(all_embeds_list), torch.cat(all_batch_list), batch_size,
        )

        # Initialize literal slots
        slots = self.literal_slot_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, H)

        # Cross-attention to encoder
        residual = slots
        slots2, _ = self.slot_cross_attn(
            slots, memory, memory,
            key_padding_mask=~memory_mask,
        )
        slots = self.slot_norm1(residual + slots2)

        # Self-attention among slots
        residual = slots
        slots2, _ = self.slot_self_attn(slots, slots, slots)
        slots = self.slot_norm2(residual + slots2)

        # FFN
        residual = slots
        slots = self.slot_norm3(residual + self.slot_ffn(slots))

        # Second round
        residual = slots
        slots2, _ = self.slot_cross_attn2(slots, memory, memory, key_padding_mask=~memory_mask)
        slots = self.slot_norm4(residual + slots2)
        residual = slots
        slots2, _ = self.slot_self_attn2(slots, slots, slots)
        slots = self.slot_norm5(residual + slots2)
        residual = slots
        slots = self.slot_norm6(residual + self.slot_ffn2(slots))

        # Predict per slot
        active_logits = self.active_head(slots).squeeze(-1)  # (B, K)
        polarity_logits = self.polarity_head(slots).squeeze(-1)  # (B, K)

        # Predicate pointer: slots attend to symbol embeddings
        pred_q = self.pred_query(slots)  # (B, K, H)
        pred_k = self.pred_key(symbol_embeds)  # (B, S, H)
        pred_logits = torch.bmm(pred_q, pred_k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        # (B, K, S)

        # Argument predictions: for each literal slot, for each arg position
        arg_slots = self.arg_slot_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # (B, K, M, H) — add slot context
        arg_slots = arg_slots + slots.unsqueeze(2)

        B, K, M, H = arg_slots.shape
        arg_flat = arg_slots.reshape(B, K * M, H)

        arg_type_logits = self.arg_type_head(arg_flat).reshape(B, K, M, 3)
        arg_var_logits = self.arg_var_head(arg_flat).reshape(B, K, M, self.max_vars)

        arg_func_q = self.arg_func_query(arg_flat)  # (B, K*M, H)
        arg_func_k = self.arg_func_key(symbol_embeds)  # (B, S, H)
        arg_func_logits = torch.bmm(arg_func_q, arg_func_k.transpose(1, 2)) / (H ** 0.5)
        arg_func_logits = arg_func_logits.reshape(B, K, M, -1)  # (B, K, M, S)

        # Convert sequential target to slot-based target for loss computation
        slot_targets = self._seq_to_slots(
            target_actions, target_arguments, target_lengths,
            num_symbols, device,
        )

        return {
            'active_logits': active_logits,
            'polarity_logits': polarity_logits,
            'pred_logits': pred_logits,
            'arg_type_logits': arg_type_logits,
            'arg_var_logits': arg_var_logits,
            'arg_func_logits': arg_func_logits,
            'slot_targets': slot_targets,
        }

    def _seq_to_slots(self, target_actions, target_arguments, target_lengths,
                      num_symbols, device):
        """Convert sequential target to per-slot targets.

        Returns dict with tensors for each slot prediction.
        """
        B = target_actions.shape[0]
        K = self.max_literals
        M = self.max_args

        active = torch.zeros(B, K, device=device)
        polarity = torch.zeros(B, K, device=device)  # 0=pos, 1=neg
        pred_idx = torch.zeros(B, K, dtype=torch.long, device=device)
        arg_type = torch.zeros(B, K, M, dtype=torch.long, device=device)  # 0=none,1=var,2=func
        arg_var = torch.zeros(B, K, M, dtype=torch.long, device=device)
        arg_func = torch.zeros(B, K, M, dtype=torch.long, device=device)

        for b in range(B):
            lit_idx = -1
            arg_idx = 0
            depth = 0
            for t in range(target_lengths[b].item()):
                act = target_actions[b, t].item()
                arg = target_arguments[b, t].item()

                if act in (NEW_LIT_POS, NEW_LIT_NEG):
                    lit_idx += 1
                    arg_idx = 0
                    depth = 0
                    if lit_idx < K:
                        active[b, lit_idx] = 1.0
                        polarity[b, lit_idx] = 1.0 if act == NEW_LIT_NEG else 0.0
                elif act == PRED and lit_idx >= 0 and lit_idx < K:
                    pred_idx[b, lit_idx] = arg
                elif act == ARG_VAR and lit_idx >= 0 and lit_idx < K and arg_idx < M and depth == 0:
                    arg_type[b, lit_idx, arg_idx] = 1
                    arg_var[b, lit_idx, arg_idx] = min(arg, self.max_vars - 1)
                    arg_idx += 1
                elif act == ARG_FUNC and lit_idx >= 0 and lit_idx < K and arg_idx < M and depth == 0:
                    arg_type[b, lit_idx, arg_idx] = 2
                    arg_func[b, lit_idx, arg_idx] = arg
                    depth += 1
                elif act == END_ARGS:
                    if depth > 0:
                        depth -= 1
                    else:
                        # End of literal args — reset for next
                        pass

        return {
            'active': active,
            'polarity': polarity,
            'pred_idx': pred_idx,
            'arg_type': arg_type,
            'arg_var': arg_var,
            'arg_func': arg_func,
        }

    @torch.no_grad()
    def generate(self, x_dict, batch_data=None, max_steps=80,
                 temperature=1.0, top_k=0, top_p=0.0):
        """Generate conjectures by predicting slot contents."""
        device = next(self.parameters()).device

        if batch_data is None:
            batch_size = 1
            symbol_batch = torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device)
        else:
            symbol_batch = (
                batch_data['symbol'].batch if hasattr(batch_data['symbol'], 'batch')
                else torch.zeros(x_dict['symbol'].shape[0], dtype=torch.long, device=device)
            )
            batch_size = symbol_batch.max().item() + 1 if symbol_batch.numel() > 0 else 1

        symbol_embeds, symbol_mask = self._pad_per_sample(
            x_dict['symbol'], symbol_batch, batch_size,
        )

        all_embeds_list, all_batch_list = [], []
        for ntype in ['clause', 'literal', 'symbol', 'term', 'variable']:
            if ntype in x_dict and x_dict[ntype].shape[0] > 0:
                all_embeds_list.append(x_dict[ntype])
                if batch_data is not None and hasattr(batch_data[ntype], 'batch'):
                    all_batch_list.append(batch_data[ntype].batch)
                else:
                    all_batch_list.append(torch.zeros(x_dict[ntype].shape[0], dtype=torch.long, device=device))
        memory, memory_mask = self._pad_per_sample(
            torch.cat(all_embeds_list), torch.cat(all_batch_list), batch_size,
        )

        # Run slot prediction
        slots = self.literal_slot_embed.unsqueeze(0).expand(batch_size, -1, -1)
        residual = slots
        slots2, _ = self.slot_cross_attn(slots, memory, memory, key_padding_mask=~memory_mask)
        slots = self.slot_norm1(residual + slots2)
        residual = slots
        slots2, _ = self.slot_self_attn(slots, slots, slots)
        slots = self.slot_norm2(residual + slots2)
        residual = slots
        slots = self.slot_norm3(residual + self.slot_ffn(slots))
        residual = slots
        slots2, _ = self.slot_cross_attn2(slots, memory, memory, key_padding_mask=~memory_mask)
        slots = self.slot_norm4(residual + slots2)
        residual = slots
        slots2, _ = self.slot_self_attn2(slots, slots, slots)
        slots = self.slot_norm5(residual + slots2)
        residual = slots
        slots = self.slot_norm6(residual + self.slot_ffn2(slots))

        # Decode slots into sequences
        active_probs = torch.sigmoid(self.active_head(slots).squeeze(-1))
        polarity_probs = torch.sigmoid(self.polarity_head(slots).squeeze(-1))

        pred_q = self.pred_query(slots)
        pred_k = self.pred_key(symbol_embeds)
        pred_scores = torch.bmm(pred_q, pred_k.transpose(1, 2)) / (self.hidden_dim ** 0.5)

        arg_slots = self.arg_slot_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        arg_slots = arg_slots + slots.unsqueeze(2)
        B, K, M, H = arg_slots.shape
        arg_flat = arg_slots.reshape(B, K * M, H)

        arg_type_logits = self.arg_type_head(arg_flat).reshape(B, K, M, 3)
        arg_var_logits = self.arg_var_head(arg_flat).reshape(B, K, M, self.max_vars)
        arg_func_q = self.arg_func_query(arg_flat)
        arg_func_k = self.arg_func_key(symbol_embeds)
        arg_func_scores = torch.bmm(arg_func_q, arg_func_k.transpose(1, 2)) / (H ** 0.5)
        arg_func_scores = arg_func_scores.reshape(B, K, M, -1)

        # Convert to sequences
        sequences = []
        for b in range(batch_size):
            seq = []
            for k in range(self.max_literals):
                if active_probs[b, k] < 0.5:
                    continue
                # New literal
                if polarity_probs[b, k] > 0.5:
                    seq.append((NEW_LIT_NEG, 0))
                else:
                    seq.append((NEW_LIT_POS, 0))
                # Predicate
                pred = pred_scores[b, k].argmax().item()
                seq.append((PRED, pred))
                # Arguments
                for m in range(self.max_args):
                    atype = arg_type_logits[b, k, m].argmax().item()
                    if atype == 0:  # none
                        break
                    elif atype == 1:  # variable
                        var = arg_var_logits[b, k, m].argmax().item()
                        seq.append((ARG_VAR, var))
                    elif atype == 2:  # function
                        func = arg_func_scores[b, k, m].argmax().item()
                        seq.append((ARG_FUNC, func))
                        seq.append((END_ARGS, 0))
                seq.append((END_ARGS, 0))
            seq.append((END_CLAUSE, 0))
            sequences.append(seq)
        return sequences


def compute_slot_loss(model_output, batch):
    """Compute loss for subgraph completion model."""
    targets = model_output['slot_targets']
    weights = batch.quality_weight

    # Active slot loss (binary cross-entropy)
    active_loss = F.binary_cross_entropy_with_logits(
        model_output['active_logits'], targets['active'], reduction='none',
    ).mean(dim=1)  # (B,)
    active_loss = (active_loss * weights).mean()

    # Polarity loss (only for active slots)
    active_mask = targets['active'] > 0.5
    if active_mask.any():
        pol_loss = F.binary_cross_entropy_with_logits(
            model_output['polarity_logits'][active_mask],
            targets['polarity'][active_mask],
        )
    else:
        pol_loss = torch.tensor(0.0, device=active_mask.device)

    # Predicate pointer loss (only for active slots)
    if active_mask.any():
        B, K, S = model_output['pred_logits'].shape
        pred_targets = targets['pred_idx'].clamp(0, S - 1)
        pred_loss = F.cross_entropy(
            model_output['pred_logits'][active_mask],
            pred_targets[active_mask],
        )
    else:
        pred_loss = torch.tensor(0.0, device=active_mask.device)

    # Argument type loss (for active slots)
    if active_mask.any():
        # Expand active_mask to cover arg positions
        active_expanded = active_mask.unsqueeze(-1).expand_as(targets['arg_type'])
        atype_logits = model_output['arg_type_logits'][active_expanded].reshape(-1, 3)
        atype_targets = targets['arg_type'][active_expanded].reshape(-1)
        arg_type_loss = F.cross_entropy(atype_logits, atype_targets)
    else:
        arg_type_loss = torch.tensor(0.0, device=active_mask.device)

    # Variable loss (for var-type args)
    var_mask = targets['arg_type'] == 1
    combined_mask = active_mask.unsqueeze(-1) & var_mask
    if combined_mask.any():
        var_logits = model_output['arg_var_logits'][combined_mask]
        var_targets = targets['arg_var'][combined_mask]
        var_loss = F.cross_entropy(var_logits, var_targets)
    else:
        var_loss = torch.tensor(0.0, device=active_mask.device)

    # Function pointer loss (for func-type args)
    func_mask = targets['arg_type'] == 2
    combined_func = active_mask.unsqueeze(-1) & func_mask
    if combined_func.any():
        S = model_output['arg_func_logits'].shape[-1]
        func_logits = model_output['arg_func_logits'][combined_func]
        func_targets = targets['arg_func'][combined_func].clamp(0, S - 1)
        func_loss = F.cross_entropy(func_logits, func_targets)
    else:
        func_loss = torch.tensor(0.0, device=active_mask.device)

    total = active_loss + pol_loss + pred_loss + arg_type_loss + var_loss + func_loss

    return {
        'total': total,
        'action': active_loss.item() + pol_loss.item() + arg_type_loss.item(),
        'pointer': pred_loss.item() + func_loss.item(),
        'variable': var_loss.item(),
    }


class ConjectureModelE(nn.Module):
    """Plan E: GNN encoder + subgraph completion decoder."""

    def __init__(self, hidden_dim=128, num_gnn_layers=6, max_vars=20,
                 max_literals=6, max_args=4, **kwargs):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = SubgraphCompletionDecoder(
            hidden_dim, max_vars, max_literals, max_args,
        )
        self.hidden_dim = hidden_dim

    def forward(self, batch_data):
        x_dict = self.encoder(batch_data)
        return self.decoder(
            x_dict, batch_data.target_actions, batch_data.target_arguments,
            batch_data.target_length, batch_data.num_symbols,
            batch_data=batch_data,
        )

    @torch.no_grad()
    def generate(self, data, max_steps=80, temperature=1.0, top_k=0, top_p=0.0):
        self.eval()
        x_dict = self.encoder(data)
        return self.decoder.generate(x_dict, batch_data=data,
                                     max_steps=max_steps, temperature=temperature,
                                     top_k=top_k, top_p=top_p)


if __name__ == '__main__':
    from conjecture_gen.tptp_parser import parse_problem_file
    from conjecture_gen.graph_builder import clauses_to_graph

    clauses = parse_problem_file('problems/l100_fomodel0')
    graph = clauses_to_graph(clauses)

    model = ConjectureModelE(hidden_dim=64, num_gnn_layers=4)
    print(f"Model E parameters: {sum(p.numel() for p in model.parameters()):,}")

    seqs = model.generate(graph)
    decoded = decode_sequence(seqs[0], graph.symbol_names)
    print(f"Generated (untrained): {decoded}")
