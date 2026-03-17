"""
Plan B: Graph-Internal Generation ("Graph Growing GNN")

The most novel approach: generate the conjecture by growing the problem
graph with new nodes and edges, all within the GNN framework.

Architecture:
  1. Encode problem graph with GNN -> node embeddings
  2. Compute global "generation context" embedding
  3. Iteratively predict:
     a. Should we add another literal? (stop vs continue)
     b. If yes: predict polarity, predicate (pointer), arguments
  4. Each new literal is represented as new nodes added to a running
     "generation state" that is cross-attended to the problem graph

The key insight: we use the same GNN-style processing for both
understanding the problem and generating the conjecture. Everything
stays in the graph domain.

Training uses teacher forcing: at each step, we provide the correct
next literal and train the model to predict it.
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


class LiteralPredictor(nn.Module):
    """Predicts one literal at a time given problem context and generation history.

    Uses cross-attention over problem graph + self-attention over previously
    generated literals. Predicts: continue/stop, polarity, predicate, arguments.
    """

    def __init__(self, hidden_dim=128, max_vars=20, max_args=4, nhead=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_args = max_args

        # Cross-attention: query from generation state, keys from problem
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, nhead, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Self-attention over generated literals so far
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, nhead, batch_first=True,
        )
        self.self_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.continue_head = nn.Linear(hidden_dim, 1)  # continue generating?
        self.polarity_head = nn.Linear(hidden_dim, 1)  # 0=pos, 1=neg

        # Predicate pointer
        self.pred_query = nn.Linear(hidden_dim, hidden_dim)
        self.pred_key = nn.Linear(hidden_dim, hidden_dim)

        # Argument prediction (simplified: predict top-level args only)
        self.n_args_head = nn.Linear(hidden_dim, max_args + 1)  # 0..max_args
        self.arg_type_head = nn.Linear(hidden_dim, 3)  # none/var/func per position
        self.arg_var_head = nn.Linear(hidden_dim, max_vars)
        self.arg_func_query = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, memory, memory_mask, symbol_embeds, symbol_mask,
                history=None):
        """Predict one literal.

        query: (B, 1, H) - current generation query
        memory: (B, N, H) - problem graph node embeddings
        history: (B, K, H) or None - previously generated literal embeddings
        """
        # Self-attention over history + current query
        if history is not None and history.shape[1] > 0:
            combined = torch.cat([history, query], dim=1)
        else:
            combined = query

        residual = combined
        sa_out, _ = self.self_attn(combined, combined, combined)
        combined = self.self_norm(residual + sa_out)

        # Take the last position (current query)
        current = combined[:, -1:, :]

        # Cross-attention to problem
        residual = current
        ca_out, _ = self.cross_attn(
            current, memory, memory,
            key_padding_mask=~memory_mask if memory_mask is not None else None,
        )
        current = self.cross_norm(residual + ca_out)

        # FFN
        residual = current
        current = self.ffn_norm(residual + self.ffn(current))

        h = current.squeeze(1)  # (B, H)

        # Predictions
        continue_logit = self.continue_head(h).squeeze(-1)  # (B,)
        polarity_logit = self.polarity_head(h).squeeze(-1)  # (B,)

        # Predicate pointer
        pq = self.pred_query(h).unsqueeze(1)  # (B, 1, H)
        pk = self.pred_key(symbol_embeds)  # (B, S, H)
        pred_scores = torch.bmm(pq, pk.transpose(1, 2)).squeeze(1)  # (B, S)
        pred_scores = pred_scores / (self.hidden_dim ** 0.5)
        if symbol_mask is not None:
            pred_scores = pred_scores.masked_fill(~symbol_mask, float('-inf'))

        # Arguments
        n_args_logits = self.n_args_head(h)  # (B, max_args+1)

        # Per-argument predictions (using the literal embedding)
        arg_type_logits = self.arg_type_head(h).unsqueeze(1).expand(-1, self.max_args, -1)  # simplified
        arg_var_logits = self.arg_var_head(h).unsqueeze(1).expand(-1, self.max_args, -1)
        afq = self.arg_func_query(h).unsqueeze(1)  # (B, 1, H)
        arg_func_scores = torch.bmm(afq, pk.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)
        arg_func_scores = arg_func_scores.unsqueeze(1).expand(-1, self.max_args, -1)

        return {
            'continue_logit': continue_logit,
            'polarity_logit': polarity_logit,
            'pred_scores': pred_scores,
            'n_args_logits': n_args_logits,
            'arg_type_logits': arg_type_logits,
            'arg_var_logits': arg_var_logits,
            'arg_func_scores': arg_func_scores,
            'literal_embed': current,  # (B, 1, H) for history
        }


class GraphGrowingDecoder(nn.Module):
    """Iteratively grows the graph by predicting literals one at a time."""

    def __init__(self, hidden_dim=128, max_vars=20, max_literals=8, max_args=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals
        self.max_args = max_args

        # Initial query embedding (learned "what should I generate?" seed)
        self.query_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Literal predictor
        self.lit_predictor = LiteralPredictor(
            hidden_dim, max_vars, max_args,
        )

        # Project generated literal info back to embedding for history
        self.history_proj = nn.Linear(hidden_dim, hidden_dim)

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

    def forward(self, x_dict, target_actions, target_arguments,
                target_lengths, num_symbols, batch_data=None):
        """Training: predict each literal with teacher forcing."""
        batch_size = target_actions.shape[0]
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

        # Parse targets into per-literal structure
        lit_targets = self._parse_literal_targets(
            target_actions, target_arguments, target_lengths, device,
        )

        # Predict each literal iteratively
        n_lits = lit_targets['n_literals']  # (B,)
        max_n_lits = min(n_lits.max().item(), self.max_literals)

        all_continue_logits = []
        all_polarity_logits = []
        all_pred_scores = []
        all_n_args_logits = []
        all_arg_type_logits = []
        all_arg_var_logits = []
        all_arg_func_scores = []

        history = None
        query = self.query_embed.expand(batch_size, -1, -1)

        # Predict max_n_lits+1 steps (including the "stop" prediction)
        for k in range(max_n_lits + 1):
            out = self.lit_predictor(
                query, memory, mem_mask, symbol_embeds, symbol_mask,
                history=history,
            )

            all_continue_logits.append(out['continue_logit'])
            all_polarity_logits.append(out['polarity_logit'])
            all_pred_scores.append(out['pred_scores'])
            all_n_args_logits.append(out['n_args_logits'])
            all_arg_type_logits.append(out['arg_type_logits'])
            all_arg_var_logits.append(out['arg_var_logits'])
            all_arg_func_scores.append(out['arg_func_scores'])

            # Update history with generated literal embedding
            lit_emb = self.history_proj(out['literal_embed'])
            if history is None:
                history = lit_emb
            else:
                history = torch.cat([history, lit_emb], dim=1)

        return {
            'continue_logits': torch.stack(all_continue_logits, dim=1),  # (B, K+1)
            'polarity_logits': torch.stack(all_polarity_logits, dim=1),
            'pred_scores': torch.stack(all_pred_scores, dim=1),  # (B, K+1, S)
            'n_args_logits': torch.stack(all_n_args_logits, dim=1),
            'arg_type_logits': torch.stack(all_arg_type_logits, dim=1),
            'arg_var_logits': torch.stack(all_arg_var_logits, dim=1),
            'arg_func_scores': torch.stack(all_arg_func_scores, dim=1),
            'lit_targets': lit_targets,
        }

    def _parse_literal_targets(self, actions, arguments, lengths, device):
        """Parse sequential targets into per-literal structure."""
        B = actions.shape[0]
        K = self.max_literals
        M = self.max_args

        n_lits = torch.zeros(B, dtype=torch.long, device=device)
        polarity = torch.zeros(B, K, device=device)
        pred_idx = torch.zeros(B, K, dtype=torch.long, device=device)
        n_args = torch.zeros(B, K, dtype=torch.long, device=device)
        arg_type = torch.zeros(B, K, M, dtype=torch.long, device=device)
        arg_var = torch.zeros(B, K, M, dtype=torch.long, device=device)
        arg_func = torch.zeros(B, K, M, dtype=torch.long, device=device)

        for b in range(B):
            li = -1
            ai = 0
            depth = 0
            for t in range(lengths[b].item()):
                act = actions[b, t].item()
                arg = arguments[b, t].item()
                if act in (NEW_LIT_POS, NEW_LIT_NEG):
                    li += 1
                    ai = 0
                    depth = 0
                    if li < K:
                        n_lits[b] = li + 1
                        polarity[b, li] = 1.0 if act == NEW_LIT_NEG else 0.0
                elif act == PRED and 0 <= li < K:
                    pred_idx[b, li] = arg
                elif act == ARG_VAR and 0 <= li < K and ai < M and depth == 0:
                    arg_type[b, li, ai] = 1
                    arg_var[b, li, ai] = min(arg, self.max_vars - 1)
                    n_args[b, li] = ai + 1
                    ai += 1
                elif act == ARG_FUNC and 0 <= li < K and ai < M and depth == 0:
                    arg_type[b, li, ai] = 2
                    arg_func[b, li, ai] = arg
                    n_args[b, li] = ai + 1
                    ai += 1
                    depth += 1
                elif act == END_ARGS:
                    if depth > 0:
                        depth -= 1

        return {
            'n_literals': n_lits,
            'polarity': polarity,
            'pred_idx': pred_idx,
            'n_args': n_args,
            'arg_type': arg_type,
            'arg_var': arg_var,
            'arg_func': arg_func,
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
        history = None
        query = self.query_embed.expand(batch_size, -1, -1)

        sym_names_len = symbol_embeds.shape[1]

        for k in range(self.max_literals):
            out = self.lit_predictor(
                query, memory, mem_mask, symbol_embeds, symbol_mask,
                history=history,
            )

            # Should we continue?
            continue_prob = torch.sigmoid(out['continue_logit'])

            for i in range(batch_size):
                if continue_prob[i] < 0.5:
                    continue

                # Polarity
                neg = torch.sigmoid(out['polarity_logit'][i]) > 0.5
                sequences[i].append((NEW_LIT_NEG if neg else NEW_LIT_POS, 0))

                # Predicate
                pred = out['pred_scores'][i].argmax().item()
                sequences[i].append((PRED, pred))

                # Arguments
                n_args = out['n_args_logits'][i].argmax().item()
                for m in range(min(n_args, self.max_args)):
                    atype = out['arg_type_logits'][i, m].argmax().item()
                    if atype == 1:  # var
                        v = out['arg_var_logits'][i, m].argmax().item()
                        sequences[i].append((ARG_VAR, min(v, self.max_vars - 1)))
                    elif atype == 2:  # func
                        f = out['arg_func_scores'][i, m].argmax().item()
                        sequences[i].append((ARG_FUNC, min(f, sym_names_len - 1)))
                        sequences[i].append((END_ARGS, 0))

                sequences[i].append((END_ARGS, 0))

            # Update history
            lit_emb = self.history_proj(out['literal_embed'])
            if history is None:
                history = lit_emb
            else:
                history = torch.cat([history, lit_emb], dim=1)

        for i in range(batch_size):
            sequences[i].append((END_CLAUSE, 0))

        return sequences


def compute_graph_grow_loss(model_output, batch):
    """Loss for graph growing model."""
    targets = model_output['lit_targets']
    weights = batch.quality_weight
    device = weights.device

    n_lits = targets['n_literals']  # (B,)
    B = n_lits.shape[0]
    K_pred = model_output['continue_logits'].shape[1]  # K+1

    # Continue/stop loss
    # For each step k: should_continue = 1 if k < n_lits, 0 otherwise
    continue_targets = torch.zeros(B, K_pred, device=device)
    for b in range(B):
        nl = min(n_lits[b].item(), K_pred)
        continue_targets[b, :nl] = 1.0

    cont_loss = F.binary_cross_entropy_with_logits(
        model_output['continue_logits'], continue_targets, reduction='none',
    ).mean(dim=1)
    cont_loss = (cont_loss * weights).mean()

    # For active literals only
    max_k = min(n_lits.max().item(), K_pred - 1)
    if max_k == 0:
        return {'total': cont_loss, 'action': cont_loss.item(), 'pointer': 0.0, 'variable': 0.0}

    # Slice predictions to match target size (K) — drop the extra "stop" step for content losses
    K_tgt = targets['polarity'].shape[1]
    K_use = min(K_pred, K_tgt)

    # Create mask for active literal positions
    active = torch.zeros(B, K_use, dtype=torch.bool, device=device)
    for b in range(B):
        nl = min(n_lits[b].item(), K_use)
        active[b, :nl] = True

    # Polarity loss
    pol_loss = F.binary_cross_entropy_with_logits(
        model_output['polarity_logits'][:, :K_use][active],
        targets['polarity'][:, :K_use][active],
    ) if active.any() else torch.tensor(0.0, device=device)

    # Predicate loss
    S = model_output['pred_scores'].shape[-1]
    if active.any():
        pred_loss = F.cross_entropy(
            model_output['pred_scores'][:, :K_use][active],
            targets['pred_idx'][:, :K_use][active].clamp(0, S - 1),
        )
    else:
        pred_loss = torch.tensor(0.0, device=device)

    # Number of args loss
    if active.any():
        M = model_output['n_args_logits'].shape[-1] - 1
        nargs_loss = F.cross_entropy(
            model_output['n_args_logits'][:, :K_use][active],
            targets['n_args'][:, :K_use][active].clamp(0, M),
        )
    else:
        nargs_loss = torch.tensor(0.0, device=device)

    total = cont_loss + pol_loss + pred_loss + nargs_loss

    return {
        'total': total,
        'action': cont_loss.item() + pol_loss.item() + nargs_loss.item(),
        'pointer': pred_loss.item(),
        'variable': 0.0,
    }


class ConjectureModelB(nn.Module):
    """Plan B: GNN encoder + graph growing decoder."""

    def __init__(self, hidden_dim=128, num_gnn_layers=6, max_vars=20, **kwargs):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = GraphGrowingDecoder(hidden_dim, max_vars)
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

    model = ConjectureModelB(hidden_dim=64, num_gnn_layers=4)
    print(f"Model B parameters: {sum(p.numel() for p in model.parameters()):,}")

    seqs = model.generate(graph)
    decoded = decode_sequence(seqs[0], graph.symbol_names)
    print(f"Generated (untrained): {decoded}")
