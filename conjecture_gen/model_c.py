"""
Plan C: GNN Encoder + Conditional VAE Decoder

Architecture:
  1. GNN encoder processes problem graph -> node embeddings
  2. A clause encoder maps the target clause to a latent vector z (training only)
  3. A conditional prior predicts the latent distribution from the problem embedding
  4. A Transformer decoder generates the clause conditioned on z + problem memory
  5. At inference: sample z from the prior, decode to get diverse conjectures

The VAE naturally produces diverse candidates (different z samples -> different
conjectures), which is ideal for conjecturing where we want many candidates.
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


class ClauseEncoder(nn.Module):
    """Encodes a target clause sequence into a latent vector."""

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, actions, lengths):
        """Encode target actions into latent distribution.
        actions: (B, T) action type indices
        lengths: (B,) sequence lengths
        Returns: mu, logvar each (B, latent_dim)
        """
        emb = self.action_embed(actions)  # (B, T, H)
        # Pack and run RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        _, h = self.rnn(packed)  # h: (1, B, H)
        h = h.squeeze(0)  # (B, H)
        return self.mu_proj(h), self.logvar_proj(h)


class ConditionalPrior(nn.Module):
    """Predicts latent distribution from problem graph embedding."""

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, graph_embed):
        """graph_embed: (B, H) -> mu, logvar each (B, latent_dim)"""
        h = self.net(graph_embed)
        return self.mu_proj(h), self.logvar_proj(h)


class VAETransformerDecoder(nn.Module):
    """Transformer decoder conditioned on latent z + GNN memory."""

    def __init__(self, hidden_dim=128, max_vars=20, latent_dim=32,
                 max_literals=8, num_layers=3, nhead=4, max_seq_len=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_vars = max_vars
        self.max_literals = max_literals
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len

        # Clause encoder (training only)
        self.clause_encoder = ClauseEncoder(hidden_dim, latent_dim)

        # Conditional prior
        self.prior = ConditionalPrior(hidden_dim, latent_dim)

        # Latent -> initial decoder context
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Input embeddings
        self.action_embed = nn.Embedding(NUM_ACTION_TYPES, hidden_dim)
        self.arg_sym_proj = nn.Linear(hidden_dim, hidden_dim)
        self.var_slot_embed = nn.Embedding(max_vars, hidden_dim)
        self.input_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer decoder
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

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

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        all_ec = torch.cat(all_e)
        all_bc = torch.cat(all_b)
        memory, mem_mask = self._pad_per_sample(all_ec, all_bc, batch_size)

        # Global graph embedding for prior
        global_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        global_embed.scatter_add_(0, all_bc.unsqueeze(1).expand_as(all_ec), all_ec)
        counts = torch.bincount(all_bc, minlength=batch_size).float().clamp(min=1).unsqueeze(1)
        global_embed = global_embed / counts

        # Encode target clause -> posterior q(z|x, clause)
        post_mu, post_logvar = self.clause_encoder(target_actions, target_lengths)

        # Prior p(z|x) from graph
        prior_mu, prior_logvar = self.prior(global_embed)

        # Sample z from posterior (training)
        z = self._reparameterize(post_mu, post_logvar)

        # Project z and prepend to memory as conditioning
        z_embed = self.z_proj(z).unsqueeze(1)  # (B, 1, H)
        memory = torch.cat([z_embed, memory], dim=1)
        z_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        mem_mask = torch.cat([z_mask, mem_mask], dim=1)

        # Decoder input (shift right)
        bos_a = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        bos_arg = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        dec_actions = torch.cat([bos_a, target_actions[:, :-1]], dim=1)
        dec_args = torch.cat([bos_arg, target_arguments[:, :-1]], dim=1)

        tok_embeds = self._build_input_embeds(dec_actions, dec_args, symbol_embeds)
        positions = torch.arange(max_len, device=device).unsqueeze(0).clamp(max=self.max_seq_len - 1)
        tok_embeds = tok_embeds + self.pos_embed(positions)

        causal_mask = torch.triu(torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1)
        tgt_pad = torch.arange(max_len, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)

        hidden = self.transformer(
            tgt=tok_embeds, memory=memory,
            tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=~mem_mask,
        )

        action_logits = self.action_head(hidden)
        pointer_logits = self._pointer_scores(hidden, symbol_embeds, symbol_mask)
        var_logits = self.var_head(hidden)

        return {
            'action_logits': action_logits,
            'pointer_logits': pointer_logits,
            'var_logits': var_logits,
            'post_mu': post_mu,
            'post_logvar': post_logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
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
        all_ec, all_bc = torch.cat(all_e), torch.cat(all_b)
        memory, mem_mask = self._pad_per_sample(all_ec, all_bc, batch_size)

        global_embed = torch.zeros(batch_size, self.hidden_dim, device=device)
        global_embed.scatter_add_(0, all_bc.unsqueeze(1).expand_as(all_ec), all_ec)
        counts = torch.bincount(all_bc, minlength=batch_size).float().clamp(min=1).unsqueeze(1)
        global_embed = global_embed / counts

        # Sample z from PRIOR (no target available at inference)
        prior_mu, prior_logvar = self.prior(global_embed)
        z = self._reparameterize(prior_mu, prior_logvar)

        z_embed = self.z_proj(z).unsqueeze(1)
        memory = torch.cat([z_embed, memory], dim=1)
        z_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        mem_mask = torch.cat([z_mask, mem_mask], dim=1)

        # Autoregressive decode
        sequences = [[] for _ in range(batch_size)]
        done = [False] * batch_size
        lit_counts = [0] * batch_size

        all_actions = torch.full((batch_size, 1), END_CLAUSE, dtype=torch.long, device=device)
        all_args = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

        for step in range(max_steps):
            seq_len = all_actions.shape[1]
            tok = self._build_input_embeds(all_actions, all_args, symbol_embeds)
            pos = torch.arange(seq_len, device=device).unsqueeze(0).clamp(max=self.max_seq_len - 1)
            tok = tok + self.pos_embed(pos)

            cmask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            hidden = self.transformer(
                tgt=tok, memory=memory,
                tgt_mask=cmask, memory_key_padding_mask=~mem_mask,
            )

            h_last = hidden[:, -1]
            action_logits = self.action_head(h_last) / temperature

            for i in range(batch_size):
                if not done[i] and lit_counts[i] >= self.max_literals:
                    action_logits[i, NEW_LIT_POS] = float('-inf')
                    action_logits[i, NEW_LIT_NEG] = float('-inf')
                    action_logits[i, END_CLAUSE] += 5.0

            actions = action_logits.argmax(dim=-1)
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


def compute_vae_loss(model_output, batch):
    """Loss = reconstruction + KL divergence."""
    from conjecture_gen.train import compute_loss

    # Reconstruction loss (same as Plan A)
    recon = compute_loss(model_output, batch)

    # KL divergence between posterior and prior
    post_mu = model_output['post_mu']
    post_logvar = model_output['post_logvar']
    prior_mu = model_output['prior_mu']
    prior_logvar = model_output['prior_logvar']

    # KL(q || p) = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q-mu_p)^2)/var_p - 1)
    kl = 0.5 * (
        prior_logvar - post_logvar
        + (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp().clamp(min=1e-8)
        - 1
    ).sum(dim=-1).mean()

    # KL weight (annealing would help but keep it simple)
    kl_weight = 0.1

    total = recon['total'] + kl_weight * kl

    return {
        'total': total,
        'action': recon['action'],
        'pointer': recon['pointer'],
        'variable': recon['variable'] + kl_weight * kl.item(),
    }


class ConjectureModelC(nn.Module):
    """Plan C: GNN encoder + conditional VAE decoder."""

    def __init__(self, hidden_dim=128, num_gnn_layers=6, max_vars=20, **kwargs):
        super().__init__()
        self.encoder = HeteroGNNEncoder(hidden_dim, num_gnn_layers)
        self.decoder = VAETransformerDecoder(
            hidden_dim, max_vars, latent_dim=hidden_dim // 4,
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

    model = ConjectureModelC(hidden_dim=64, num_gnn_layers=4)
    print(f"Model C parameters: {sum(p.numel() for p in model.parameters()):,}")

    seqs = model.generate(graph)
    decoded = decode_sequence(seqs[0], graph.symbol_names)
    print(f"Generated (untrained): {decoded}")
