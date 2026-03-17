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


def encode_conjecture(clause: Clause, symbol_names: list[str]) -> list[tuple[int, int]]:
    """Encode a conjecture clause as a sequence of (action_type, argument) pairs.

    Args:
        clause: The conjecture clause to encode.
        symbol_names: List of symbol names from the problem graph, where
                      index i corresponds to symbol node i.

    Returns:
        List of (action_type, argument) tuples.
        For actions without a meaningful argument, argument is 0.
        For PRED/ARG_FUNC, argument is the symbol index in the problem graph.
        For ARG_VAR, argument is the canonical variable slot.
    """
    # Build symbol name -> index mapping
    sym_to_idx = {}
    for i, name in enumerate(symbol_names):
        sym_to_idx[name] = i
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

    def _get_sym_idx(name: str) -> int:
        return sym_to_idx.get(name, unk_idx)

    def _encode_term(term: Term):
        if term.is_variable:
            slot = _get_var_slot(term.name)
            sequence.append((ARG_VAR, slot))
        else:
            sym_idx = _get_sym_idx(term.name)
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
        pred_idx = _get_sym_idx(lit.predicate)
        sequence.append((PRED, pred_idx))

        # Arguments
        for arg in lit.args:
            _encode_term(arg)

        sequence.append((END_ARGS, 0))

    sequence.append((END_CLAUSE, 0))
    return sequence


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
                closed = stack.pop() + ')'
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

    return ' | '.join(result_parts) if result_parts else '<empty>'


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
