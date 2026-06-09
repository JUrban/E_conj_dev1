"""
Safe validation helper for generated clause text.

Wraps the strict TPTP parser to provide a non-throwing validation
interface suitable for bulk generation and evaluation pipelines.
"""

from conjecture_gen.tptp_parser import parse_clause, TPTPParseError


def validate_clause_text(body: str) -> dict:
    """Validate a clause body string without raising exceptions.

    Args:
        body: The clause body text (e.g. "p(a) | ~q(X)"), without the
              surrounding cnf(...) wrapper.

    Returns:
        dict with at least 'valid' (bool) and, if invalid, 'reason' (str).
        On success, includes parsed clause info.
    """
    # Empty or placeholder
    if not body or body.strip() == '' or body.strip() == '<empty>':
        return {'valid': False, 'reason': 'empty'}

    # Truncated output (decoder ran out of steps)
    if '...' in body:
        return {'valid': False, 'reason': 'truncated'}

    # Unknown symbol marker from decoder
    if '?UNK' in body:
        return {'valid': False, 'reason': 'unknown_symbol'}

    # Try strict parsing
    tptp_str = f"cnf(gen, axiom, ({body}))."
    try:
        parsed = parse_clause(tptp_str, strict=True)
    except TPTPParseError as e:
        return {'valid': False, 'reason': f'parse_error:{e}'}

    if parsed is None:
        return {'valid': False, 'reason': 'parse_returned_none'}

    return {
        'valid': True,
        'n_literals': len(parsed.literals),
        'clause': parsed,
    }
