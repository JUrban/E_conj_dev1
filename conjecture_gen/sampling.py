"""
Sampling utilities for conjecture generation: top-k, nucleus (top-p),
temperature scaling, and arity-constrained decoding.
"""

import torch
import torch.nn.functional as F

from conjecture_gen.target_encoder import (
    NEW_LIT_POS, NEW_LIT_NEG, PRED, ARG_VAR, ARG_FUNC, END_ARGS, END_CLAUSE,
    NUM_ACTION_TYPES,
)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and/or nucleus (top-p) filtering.

    Args:
        logits: (batch, vocab) raw logits
        temperature: temperature scaling (1.0 = no change, <1 = sharper, >1 = flatter)
        top_k: keep only top-k logits (0 = disabled)
        top_p: keep smallest set of logits with cumulative prob >= top_p (0.0 = disabled)

    Returns:
        (batch,) sampled indices
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Nucleus (top-p) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold
        # Shift right so that the first token above threshold is kept
        remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        # Scatter back to original order
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ArityConstraint:
    """Tracks arity obligations during autoregressive generation.

    After selecting a predicate or function symbol, enforces the correct
    number of arguments before allowing END_ARGS. This prevents the most
    common source of syntactically invalid output.

    Usage:
        ac = ArityConstraint(symbol_arities)  # list of int arities per symbol
        for step in generation:
            action_logits = ac.constrain_actions(i, action_logits)
            if action is PRED or ARG_FUNC:
                ac.push_symbol(i, symbol_idx)
            elif action is ARG_VAR or ARG_FUNC:
                ac.count_arg(i)
            elif action is END_ARGS:
                ac.pop(i)
    """

    def __init__(self, symbol_arities: list[int], batch_size: int):
        """
        Args:
            symbol_arities: list mapping symbol index -> arity (int)
            batch_size: number of samples in the batch
        """
        self.arities = symbol_arities
        self.batch_size = batch_size
        # Stack per sample: each entry is (expected_args, args_so_far)
        self.stacks = [[] for _ in range(batch_size)]
        # Whether we're expecting a PRED after NEW_LIT
        self.expecting_pred = [False] * batch_size

    def push_symbol(self, i: int, symbol_idx: int):
        """Called after PRED or ARG_FUNC action: push arity obligation."""
        arity = self.arities[symbol_idx] if symbol_idx < len(self.arities) else 0
        self.stacks[i].append({'expected': arity, 'got': 0})
        self.expecting_pred[i] = False

    def count_arg(self, i: int):
        """Called after ARG_VAR (a complete argument was provided)."""
        if self.stacks[i]:
            self.stacks[i][-1]['got'] += 1

    def pop(self, i: int):
        """Called after END_ARGS: pop the current obligation."""
        if self.stacks[i]:
            self.stacks[i].pop()
            # The popped function was itself an argument to the parent
            if self.stacks[i]:
                self.stacks[i][-1]['got'] += 1

    def constrain_actions(self, i: int, action_logits: torch.Tensor) -> torch.Tensor:
        """Apply arity constraints to action logits for sample i.

        Modifies logits in-place and returns them.
        """
        if self.expecting_pred[i]:
            # After NEW_LIT, must emit PRED next
            action_logits[NEW_LIT_POS] = float('-inf')
            action_logits[NEW_LIT_NEG] = float('-inf')
            action_logits[ARG_VAR] = float('-inf')
            action_logits[ARG_FUNC] = float('-inf')
            action_logits[END_ARGS] = float('-inf')
            action_logits[END_CLAUSE] = float('-inf')
            # Only PRED allowed
            return action_logits

        if not self.stacks[i]:
            # No obligation: can start new literal or end clause
            # Cannot emit args or END_ARGS without a context
            action_logits[ARG_VAR] = float('-inf')
            action_logits[ARG_FUNC] = float('-inf')
            action_logits[END_ARGS] = float('-inf')
            action_logits[PRED] = float('-inf')
            return action_logits

        top = self.stacks[i][-1]
        remaining = top['expected'] - top['got']

        if remaining > 0:
            # Still need more arguments — cannot END_ARGS yet
            action_logits[END_ARGS] = float('-inf')
            action_logits[END_CLAUSE] = float('-inf')
            action_logits[NEW_LIT_POS] = float('-inf')
            action_logits[NEW_LIT_NEG] = float('-inf')
            action_logits[PRED] = float('-inf')
            # ARG_VAR and ARG_FUNC are allowed
        elif remaining == 0:
            # All args satisfied — must END_ARGS
            action_logits[ARG_VAR] = float('-inf')
            action_logits[ARG_FUNC] = float('-inf')
            action_logits[NEW_LIT_POS] = float('-inf')
            action_logits[NEW_LIT_NEG] = float('-inf')
            action_logits[PRED] = float('-inf')
            action_logits[END_CLAUSE] = float('-inf')
            # Only END_ARGS allowed

        return action_logits

    def notify_action(self, i: int, action: int, arg_val: int):
        """Update state after an action is taken. Call this for every step."""
        if action in (NEW_LIT_POS, NEW_LIT_NEG):
            self.expecting_pred[i] = True
        elif action == PRED:
            self.push_symbol(i, arg_val)
        elif action == ARG_FUNC:
            # ARG_FUNC is both "count as arg to parent" and "push new obligation"
            self.push_symbol(i, arg_val)
        elif action == ARG_VAR:
            self.count_arg(i)
        elif action == END_ARGS:
            self.pop(i)
