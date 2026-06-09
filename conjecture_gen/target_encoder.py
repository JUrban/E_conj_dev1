"""
Encodes a target conjecture clause as a generation sequence for the
tree decoder.

The decoder generates clauses as a sequence of actions:
  - NEW_LIT_POS / NEW_LIT_NEG  (start a new positive/negative literal)
  - PRED <symbol_idx>           (select predicate from problem symbols)
  - ARG_VAR <var_slot>          (argument is a variable, identified by slot 0,1,2...)
  - ARG_FUNC <symbol_idx>      (argument starts a function application)
  - END_ARGS                    (close current function/literal arguments)
  - END_CLAUSE                  (done)

For equality literals, predicate is the special $eq symbol.

Variable slots: we assign variables canonical IDs by order of first
occurrence in the conjecture (0, 1, 2, ...). This is symbol-independent.

Symbol indices: these refer to the *problem graph's* symbol node indices.
If a conjecture uses a symbol not in the problem, we map it to a special
UNK_SYMBOL index.
"""

from conjecture_gen.tptp_parser import Clause, Literal, Term

# Action tokens
NEW_LIT_POS = 0
NEW_LIT_NEG = 1
PRED = 2        # followed by symbol index
ARG_VAR = 3     # followed by variable slot
ARG_FUNC = 4    # followed by symbol index, then args, then END_ARGS
END_ARGS = 5
END_CLAUSE = 6
NUM_ACTION_TYPES = 7

ACTION_NAMES = [
    'NEW_LIT_POS', 'NEW_LIT_NEG', 'PRED',
    'ARG_VAR', 'ARG_FUNC', 'END_ARGS', 'END_CLAUSE',
]


def encode_conjecture(clause: Clause, symbol_names: list[str],
                      symbol_is_pred: list[bool] = None,
                      symbol_arities: list[int] = None,
                      strict: bool = False) -> 'list[tuple[int, int]] | tuple[list[tuple[int, int]], dict]':
    """Encode a conjecture clause as a sequence of (action_type, argument) pairs.

    Args:
        clause: The conjecture clause to encode.
        symbol_names: List of symbol names from the problem graph, where
                      index i corresponds to symbol node i.
        symbol_is_pred: Optional list of booleans indicating whether each
                        symbol is a predicate (True) or function (False).
                        Enables role-aware lookup to avoid collisions when
                        the same name is used as both predicate and function.
        symbol_arities: Optional list of arities (unused currently, reserved
                        for future arity-aware encoding).
        strict: If True, enable strict validation mode:
                - Validate metadata length consistency
                - Detect duplicate (name, is_pred, arity) keys
                - No cascading fallback: require exact match or UNK
                - Return (sequence, stats) tuple instead of just sequence

    Returns:
        If strict=False (default): List of (action_type, argument) tuples.
        If strict=True: Tuple of (sequence, stats) where stats is a dict with
            'exact_hits', 'role_fallback_hits', 'name_fallback_hits', 'unk_hits'.
    """
    # Encoding stats tracking
    stats = {'exact_hits': 0, 'role_fallback_hits': 0, 'name_fallback_hits': 0, 'unk_hits': 0}

    # Strict mode: validate metadata consistency
    if strict:
        if symbol_is_pred is not None and len(symbol_names) != len(symbol_is_pred):
            raise ValueError(
                f"Metadata length mismatch: len(symbol_names)={len(symbol_names)} "
                f"!= len(symbol_is_pred)={len(symbol_is_pred)}"
            )
        if symbol_arities is not None and len(symbol_names) != len(symbol_arities):
            raise ValueError(
                f"Metadata length mismatch: len(symbol_names)={len(symbol_names)} "
                f"!= len(symbol_arities)={len(symbol_arities)}"
            )
        if symbol_is_pred is not None and symbol_arities is not None:
            if len(symbol_is_pred) != len(symbol_arities):
                raise ValueError(
                    f"Metadata length mismatch: len(symbol_is_pred)={len(symbol_is_pred)} "
                    f"!= len(symbol_arities)={len(symbol_arities)}"
                )

    # Build symbol name -> index mapping
    if symbol_is_pred is not None and symbol_arities is not None:
        # Full lookup: (name, is_pred, arity) -> index with fallbacks
        sym_to_idx = {}

        if strict:
            # Check for duplicate keys
            seen_keys = set()
            for i, (name, is_pred, arity) in enumerate(zip(symbol_names, symbol_is_pred, symbol_arities)):
                key = (name, is_pred, arity)
                if key in seen_keys:
                    raise ValueError(f"Duplicate symbol key: {key}")
                seen_keys.add(key)
                sym_to_idx[key] = i
            # In strict mode: NO fallback entries
        else:
            for i, (name, is_pred, arity) in enumerate(zip(symbol_names, symbol_is_pred, symbol_arities)):
                sym_to_idx[(name, is_pred, arity)] = i
                # Fallback without arity for backward compat
                sym_to_idx.setdefault((name, is_pred), i)
                sym_to_idx.setdefault(name, i)
    elif symbol_is_pred is not None:
        # Role-aware lookup: (name, is_pred) -> index
        sym_to_idx = {}

        if strict:
            seen_keys = set()
            for i, (name, is_pred) in enumerate(zip(symbol_names, symbol_is_pred)):
                key = (name, is_pred)
                if key in seen_keys:
                    raise ValueError(f"Duplicate symbol key: {key}")
                seen_keys.add(key)
                sym_to_idx[key] = i
        else:
            for i, (name, is_pred) in enumerate(zip(symbol_names, symbol_is_pred)):
                sym_to_idx[(name, is_pred)] = i
                sym_to_idx.setdefault(name, i)
    else:
        sym_to_idx = {name: i for i, name in enumerate(symbol_names)}
    unk_idx = len(symbol_names)  # UNK symbol index = one past the end

    # Track variable canonical ordering
    var_to_slot = {}
    next_var_slot = 0

    sequence = []

    def _get_var_slot(var_name: str) -> int:
        nonlocal next_var_slot
        if var_name not in var_to_slot:
            var_to_slot[var_name] = next_var_slot
            next_var_slot += 1
        return var_to_slot[var_name]

    def _get_pred_idx(predicate_name: str, arity: int) -> int:
        """Look up predicate index with arity-aware, role-aware, then name fallbacks."""
        if strict:
            # Strict: exact match only
            if symbol_arities is not None:
                result = sym_to_idx.get((predicate_name, True, arity))
            elif symbol_is_pred is not None:
                result = sym_to_idx.get((predicate_name, True))
            else:
                result = sym_to_idx.get(predicate_name)
            if result is not None:
                stats['exact_hits'] += 1
                return result
            stats['unk_hits'] += 1
            return unk_idx

        # Non-strict: cascading fallback with stats tracking
        exact = sym_to_idx.get((predicate_name, True, arity))
        if exact is not None:
            stats['exact_hits'] += 1
            return exact
        role_fb = sym_to_idx.get((predicate_name, True))
        if role_fb is not None:
            stats['role_fallback_hits'] += 1
            return role_fb
        name_fb = sym_to_idx.get(predicate_name)
        if name_fb is not None:
            stats['name_fallback_hits'] += 1
            return name_fb
        stats['unk_hits'] += 1
        return unk_idx

    def _get_func_idx(func_name: str, arity: int) -> int:
        """Look up function index with arity-aware, role-aware, then name fallbacks."""
        if strict:
            if symbol_arities is not None:
                result = sym_to_idx.get((func_name, False, arity))
            elif symbol_is_pred is not None:
                result = sym_to_idx.get((func_name, False))
            else:
                result = sym_to_idx.get(func_name)
            if result is not None:
                stats['exact_hits'] += 1
                return result
            stats['unk_hits'] += 1
            return unk_idx

        exact = sym_to_idx.get((func_name, False, arity))
        if exact is not None:
            stats['exact_hits'] += 1
            return exact
        role_fb = sym_to_idx.get((func_name, False))
        if role_fb is not None:
            stats['role_fallback_hits'] += 1
            return role_fb
        name_fb = sym_to_idx.get(func_name)
        if name_fb is not None:
            stats['name_fallback_hits'] += 1
            return name_fb
        stats['unk_hits'] += 1
        return unk_idx

    def _get_sym_idx(name: str, is_pred: bool = None, arity: int = None) -> int:
        if is_pred is not None and arity is not None and symbol_arities is not None:
            if is_pred:
                return _get_pred_idx(name, arity)
            else:
                return _get_func_idx(name, arity)
        if symbol_is_pred is not None and is_pred is not None:
            if strict:
                result = sym_to_idx.get((name, is_pred))
                if result is not None:
                    stats['exact_hits'] += 1
                    return result
                stats['unk_hits'] += 1
                return unk_idx
            result = sym_to_idx.get((name, is_pred))
            if result is not None:
                stats['exact_hits'] += 1
                return result
            fb = sym_to_idx.get(name)
            if fb is not None:
                stats['name_fallback_hits'] += 1
                return fb
            stats['unk_hits'] += 1
            return unk_idx
        result = sym_to_idx.get(name)
        if result is not None:
            stats['exact_hits'] += 1
            return result
        stats['unk_hits'] += 1
        return unk_idx

    def _encode_term(term: Term):
        if term.is_variable:
            slot = _get_var_slot(term.name)
            sequence.append((ARG_VAR, slot))
        else:
            sym_idx = _get_sym_idx(term.name, is_pred=False, arity=len(term.args))
            sequence.append((ARG_FUNC, sym_idx))
            for arg in term.args:
                _encode_term(arg)
            sequence.append((END_ARGS, 0))

    for lit in clause.literals:
        # Start literal
        if lit.negated:
            sequence.append((NEW_LIT_NEG, 0))
        else:
            sequence.append((NEW_LIT_POS, 0))

        # Predicate
        pred_idx = _get_sym_idx(lit.predicate, is_pred=True, arity=len(lit.args))
        sequence.append((PRED, pred_idx))

        # Arguments
        for arg in lit.args:
            _encode_term(arg)

        sequence.append((END_ARGS, 0))

    sequence.append((END_CLAUSE, 0))

    if strict:
        return (sequence, stats)
    return sequence


def _split_top_level_args(s: str) -> list[str]:
    """Split a string by commas only at parenthesis depth 0.

    Correctly handles nested terms like 'f(a,b),g(c,d)' -> ['f(a,b)', 'g(c,d)'].
    """
    args = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            args.append(s[start:i])
            start = i + 1
    args.append(s[start:])
    return args


def decode_sequence(sequence: list[tuple[int, int]],
                    symbol_names: list[str]) -> str:
    """Decode a generation sequence back into a human-readable clause string.

    Useful for debugging and evaluation.
    """
    parts = []
    literals = []
    current_lit = None
    depth = 0
    var_names = {}  # slot -> name

    def _var_name(slot: int) -> str:
        if slot not in var_names:
            var_names[slot] = f"X{slot + 1}"
        return var_names[slot]

    def _sym_name(idx: int) -> str:
        if idx < len(symbol_names):
            return symbol_names[idx]
        return f"?UNK{idx}"

    i = 0
    result_parts = []
    stack = []  # stack of partial strings being built

    for action, arg in sequence:
        try:
            if action == NEW_LIT_POS:
                stack = ['']
            elif action == NEW_LIT_NEG:
                stack = ['~']
            elif action == PRED:
                if not stack:
                    stack = ['']
                stack[-1] += _sym_name(arg) + '('
            elif action == ARG_VAR:
                if not stack:
                    continue
                if stack[-1] and stack[-1][-1] not in '(':
                    stack[-1] += ','
                stack[-1] += _var_name(arg)
            elif action == ARG_FUNC:
                if not stack:
                    stack = ['']
                if stack[-1] and stack[-1][-1] not in '(':
                    stack[-1] += ','
                stack.append(_sym_name(arg) + '(')
            elif action == END_ARGS:
                if not stack:
                    continue
                top = stack.pop()
                # If top ends with '(' it's a 0-arity constant — drop the '('
                if top.endswith('('):
                    closed = top[:-1]  # e.g. "esk3_0(" -> "esk3_0"
                else:
                    closed = top + ')'
                if stack:
                    stack[-1] += closed
                else:
                    result_parts.append(closed)
            elif action == END_CLAUSE:
                break
        except (IndexError, TypeError):
            continue

    # Flush anything remaining on stack
    while stack:
        result_parts.append(stack.pop() + '...')

    if not result_parts:
        return '<empty>'

    # Post-process: convert $eq(A,B) to A=B and ~$eq(A,B) to A!=B
    processed = []
    for part in result_parts:
        # ~$eq(A,B) -> A!=B
        if part.startswith('~$eq(') and part.endswith(')'):
            inner = part[5:-1]  # strip '~$eq(' and ')'
            args = _split_top_level_args(inner)
            if len(args) == 2:
                processed.append(f'{args[0]}!={args[1]}')
                continue
        # $eq(A,B) -> A=B
        if part.startswith('$eq(') and part.endswith(')'):
            inner = part[4:-1]  # strip '$eq(' and ')'
            args = _split_top_level_args(inner)
            if len(args) == 2:
                processed.append(f'{args[0]}={args[1]}')
                continue
        processed.append(part)

    return ' | '.join(processed)


if __name__ == '__main__':
    from conjecture_gen.tptp_parser import parse_clause

    # Test encoding and decoding
    clause = parse_clause(
        'cnf(test,axiom, (v1_finseq_1(k12_finseq_1(X1,X2))|v1_xboole_0(X1)|~m1_subset_1(X2,X1))).'
    )
    symbol_names = [
        'v1_finseq_1', 'k12_finseq_1', 'v1_xboole_0', 'm1_subset_1',
        'k9_finseq_1', 'esk1_0',
    ]

    seq = encode_conjecture(clause, symbol_names)
    print("Original:", clause)
    print("\nEncoded sequence:")
    for action, arg in seq:
        print(f"  {ACTION_NAMES[action]:15s} {arg}")

    decoded = decode_sequence(seq, symbol_names)
    print(f"\nDecoded: {decoded}")

    # Test with equality
    clause2 = parse_clause(
        'cnf(test2,axiom, (esk1_0=X1|~v1_xboole_0(X1))).'
    )
    seq2 = encode_conjecture(clause2, symbol_names)
    print(f"\nEquality clause: {clause2}")
    print("Encoded:")
    for action, arg in seq2:
        print(f"  {ACTION_NAMES[action]:15s} {arg}")
    decoded2 = decode_sequence(seq2, symbol_names)
    print(f"Decoded: {decoded2}")
