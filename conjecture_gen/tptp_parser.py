"""
TPTP CNF parser: converts CNF problem files and lemma clauses into
structured Python objects suitable for graph construction.

A CNF clause looks like:
  cnf(name, type, (lit1 | lit2 | ~lit3)).

Where each literal is:
  predicate(term1, term2, ...)  or  ~predicate(term1, term2, ...)
  or  term1 = term2  /  term1 != term2

Terms are:
  Variable (uppercase start: X1, X2)
  constant  (lowercase: esk1_0, k1_xboole_0)
  function(term, term, ...)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Term:
    """A first-order term: variable, constant, or function application."""
    name: str
    args: list  # list of Term; empty for variables/constants
    is_variable: bool = False

    def __repr__(self):
        if not self.args:
            return self.name
        return f"{self.name}({','.join(repr(a) for a in self.args)})"


@dataclass
class Literal:
    """A literal: possibly negated predicate applied to terms, or equality."""
    predicate: str
    args: list  # list of Term
    negated: bool = False
    is_equality: bool = False  # for =/!=

    def __repr__(self):
        neg = "~" if self.negated else ""
        if self.is_equality:
            op = "!=" if self.negated else "="
            return f"{repr(self.args[0])}{op}{repr(self.args[1])}"
        return f"{neg}{self.predicate}({','.join(repr(a) for a in self.args)})"


@dataclass
class Clause:
    """A CNF clause: disjunction of literals."""
    name: str
    role: str  # plain, negated_conjecture, axiom, etc.
    literals: list  # list of Literal

    def __repr__(self):
        return f"cnf({self.name},{self.role}, {' | '.join(repr(l) for l in self.literals)})"


class TPTPParseError(Exception):
    pass


def _is_variable(name: str) -> bool:
    """Variables start with uppercase letter in TPTP."""
    return len(name) > 0 and name[0].isupper()


def _tokenize(s: str, strict: bool = True) -> list[str]:
    """Tokenize a TPTP formula string into meaningful tokens.

    Args:
        s: The formula string to tokenize.
        strict: If True (default), raise TPTPParseError on unknown characters
                or ambiguous operators like '=>'. If False, skip unknown chars
                (backward-compatible permissive mode).
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '(,)|~':
            tokens.append(c)
            i += 1
        elif c == '!' and i + 1 < len(s) and s[i + 1] == '=':
            tokens.append('!=')
            i += 2
        elif c == '=' and i + 1 < len(s) and s[i + 1] == '>':
            if strict:
                raise TPTPParseError(
                    f"Unsupported operator '=>' at position {i} in: {s!r}. "
                    f"TPTP CNF uses '|' for disjunction, not '=>' for implication."
                )
            # permissive: treat '=' only, skip '>'
            tokens.append('=')
            i += 1
        elif c == '=':
            tokens.append('=')
            i += 1
        elif c.isalnum() or c == '_' or c == '$':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_' or s[j] == '$'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            if strict:
                raise TPTPParseError(
                    f"Unknown character {c!r} at position {i} in: {s!r}"
                )
            # permissive: skip unexpected chars
            i += 1
    return tokens


class _Parser:
    """Recursive descent parser for tokenized TPTP clause body."""

    _PUNCTUATION = frozenset('(,)|~=!=')

    def __init__(self, tokens: list[str], strict: bool = True):
        self.tokens = tokens
        self.pos = 0
        self.strict = strict

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise TPTPParseError(f"Unexpected end of tokens, expected {expected}")
        if expected is not None and tok != expected:
            raise TPTPParseError(
                f"Expected '{expected}' but got '{tok}' at pos {self.pos}, "
                f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
            )
        self.pos += 1
        return tok

    def consume_identifier(self) -> str:
        """Consume and return a token that is a valid identifier.

        An identifier is an alphanumeric word token (letters, digits,
        underscores, '$').  Punctuation tokens ('(', ')', ',', '|', '~',
        '=', '!=') are rejected.

        Raises TPTPParseError if the next token is not a valid identifier.
        """
        tok = self.peek()
        if tok is None:
            raise TPTPParseError(
                f"Unexpected end of tokens, expected identifier"
            )
        if tok in ('(', ')', ',', '|', '~', '=', '!='):
            raise TPTPParseError(
                f"Expected identifier but got punctuation '{tok}' at pos {self.pos}, "
                f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
            )
        # Validate: must consist of alphanumeric, underscore, or '$' characters
        if not all(c.isalnum() or c in ('_', '$') for c in tok):
            raise TPTPParseError(
                f"Invalid identifier '{tok}' at pos {self.pos}: "
                f"contains non-alphanumeric characters"
            )
        self.pos += 1
        return tok

    def parse_term(self) -> Term:
        name = self.consume_identifier()
        if self.peek() == '(':
            # function application
            self.consume('(')
            # Reject empty argument lists like p()
            if self.strict and self.peek() == ')':
                raise TPTPParseError(
                    f"Empty argument list for '{name}' at pos {self.pos}, "
                    f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
                )
            args = [self.parse_term()]
            while self.peek() == ',':
                self.consume(',')
                # Reject trailing comma like p(a,)
                if self.strict and self.peek() == ')':
                    raise TPTPParseError(
                        f"Trailing comma in argument list for '{name}' at pos {self.pos}, "
                        f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
                    )
                args.append(self.parse_term())
            self.consume(')')
            return Term(name=name, args=args, is_variable=False)
        else:
            return Term(name=name, args=[], is_variable=_is_variable(name))

    def parse_literal(self) -> Literal:
        negated = False
        if self.peek() == '~':
            self.consume('~')
            negated = True

        # Parse the first term/predicate
        first = self.parse_term()

        # Check for equality/inequality
        if self.peek() in ('=', '!='):
            op = self.consume()
            second = self.parse_term()
            return Literal(
                predicate='$eq',
                args=[first, second],
                negated=(negated != (op == '!=')),  # ~(a=b) same as a!=b
                is_equality=True,
            )

        # Regular predicate literal
        return Literal(
            predicate=first.name,
            args=first.args if first.args else [],
            negated=negated,
            is_equality=False,
        )

    def parse_clause_body(self) -> list[Literal]:
        """Parse: (lit1 | lit2 | ...) or lit1 | lit2 | ..."""
        has_paren = False
        if self.peek() == '(':
            self.consume('(')
            has_paren = True

        # Reject empty clause body like ()
        if self.strict and has_paren and self.peek() == ')':
            raise TPTPParseError(
                f"Empty clause body at pos {self.pos}, "
                f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
            )

        literals = [self.parse_literal()]
        while self.peek() == '|':
            self.consume('|')
            # Reject trailing pipe like (p(a)|)
            if self.strict and self.peek() in (')', None):
                raise TPTPParseError(
                    f"Trailing '|' with no following literal at pos {self.pos}, "
                    f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
                )
            literals.append(self.parse_literal())

        if has_paren:
            if self.peek() == ')':
                self.consume(')')
            elif self.strict:
                raise TPTPParseError(
                    f"Missing closing ')' for clause body at pos {self.pos}, "
                    f"tokens: {self.tokens[max(0,self.pos-3):self.pos+3]}"
                )

        return literals


def parse_clause(line: str, strict: bool = True) -> Optional[Clause]:
    """Parse a single cnf(...) line into a Clause object.

    Args:
        line: The cnf(...) line to parse.
        strict: If True (default), raise TPTPParseError on tokenizer errors
                and unconsumed trailing tokens. If False, use permissive mode
                (backward-compatible: returns None on errors, ignores trailing tokens).

    Returns:
        Clause object, or None if the line is not a cnf clause (or parse fails
        in permissive mode).

    Raises:
        TPTPParseError: In strict mode, if the clause contains unknown characters,
                        ambiguous operators, or trailing unparsed tokens.
    """
    line = line.strip()
    if not line.startswith('cnf('):
        return None

    # Remove trailing ). and the leading cnf(
    if line.endswith(').'):
        line = line[:-2]
    elif line.endswith(')'):
        line = line[:-1]
    body = line[4:]  # remove 'cnf('

    # Split into name, role, formula
    # We need to find the first two commas that are not inside parentheses
    depth = 0
    comma_positions = []
    for i, c in enumerate(body):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            comma_positions.append(i)
            if len(comma_positions) == 2:
                break

    if len(comma_positions) < 2:
        return None

    name = body[:comma_positions[0]].strip()
    role = body[comma_positions[0] + 1:comma_positions[1]].strip()
    formula_str = body[comma_positions[1] + 1:].strip()

    # Tokenize and parse the formula
    try:
        tokens = _tokenize(formula_str, strict=strict)
    except TPTPParseError:
        if strict:
            raise
        return None

    if not tokens:
        return None

    parser = _Parser(tokens, strict=strict)
    try:
        literals = parser.parse_clause_body()
    except TPTPParseError:
        if strict:
            raise
        return None

    # In strict mode, check that all tokens were consumed
    if strict and parser.pos < len(parser.tokens):
        trailing = parser.tokens[parser.pos:]
        raise TPTPParseError(
            f"Trailing unparsed tokens after clause body: {trailing!r} "
            f"in clause: {line!r}"
        )

    return Clause(name=name, role=role, literals=literals)


def parse_problem_file(filepath: str, strict: bool = True,
                       collect_errors: bool = False) -> list[Clause]:
    """Parse a TPTP CNF problem file, returning all clauses.

    Args:
        filepath: Path to the TPTP problem file.
        strict: If True (default), use strict tokenization and parsing on
                each cnf(...) clause line. Lines starting with '#' or '%'
                are always skipped as comments before tokenization.
        collect_errors: If True, collect parse errors and issue warnings
                        instead of raising. If False (default) and strict=True,
                        raise on first error.

    Returns:
        List of successfully parsed Clause objects.

    Raises:
        TPTPParseError: In strict mode with collect_errors=False, on first
                        parse error.
    """
    import warnings
    clauses = []
    errors = []
    with open(filepath, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            try:
                clause = parse_clause(line, strict=strict)
                if clause is not None:
                    clauses.append(clause)
            except TPTPParseError as e:
                if collect_errors:
                    errors.append((line_no, line, str(e)))
                else:
                    raise TPTPParseError(
                        f"{filepath}:{line_no}: {e}"
                    ) from e

    if errors:
        warnings.warn(
            f"parse_problem_file({filepath!r}): {len(errors)} parse error(s):\n" +
            "\n".join(f"  line {ln}: {msg}" for ln, _, msg in errors[:5])
        )

    return clauses


def parse_lemma_line(line: str) -> Optional[tuple[str, str, Clause]]:
    """Parse a lemma file line like:
    ./problem_name/lemma_id: cnf(...)

    Returns (problem_name, lemma_id, clause) or None.
    """
    line = line.strip()
    if not line:
        return None

    colon_idx = line.find(': cnf(')
    if colon_idx == -1:
        return None

    path_part = line[:colon_idx]
    clause_str = line[colon_idx + 2:]  # skip ': '

    # Parse path: ./problem_name/lemma_id
    parts = path_part.split('/')
    if len(parts) < 3:
        return None
    problem_name = parts[-2]
    lemma_id = parts[-1]

    clause = parse_clause(clause_str)
    if clause is None:
        return None

    return (problem_name, lemma_id, clause)


def parse_statistics_line(line: str) -> Optional[dict]:
    """Parse a statistics line like:
    ratio:problem:cut:L1:L2:L1+L2:# Processed clauses : L

    Returns dict with parsed fields or None.
    """
    line = line.strip()
    if not line:
        return None

    # Split on ':'
    parts = line.split(':')
    if len(parts) < 8:
        return None

    try:
        ratio = float(parts[0])
        problem = parts[1]
        cut_id = parts[2].replace('.res', '')
        l1 = int(parts[3])
        l2 = int(parts[4])
        l1_plus_l2 = int(parts[5])
        # parts[6] is "# Processed clauses    " (skip)
        l_original = int(parts[7].strip())
    except (ValueError, IndexError):
        return None

    return {
        'ratio': ratio,
        'problem': problem,
        'cut_id': cut_id,
        'l1': l1,
        'l2': l2,
        'l1_plus_l2': l1_plus_l2,
        'l_original': l_original,
    }


# --- Quick self-test ---
if __name__ == '__main__':
    # Test parsing a clause
    c = parse_clause(
        'cnf(ac_0_22,axiom, (v1_finseq_1(k12_finseq_1(X1,X2))|v1_xboole_0(X1)|~m1_subset_1(X2,X1))).'
    )
    print("Parsed clause:", c)
    for lit in c.literals:
        print(f"  Literal: negated={lit.negated} pred={lit.predicate} args={lit.args}")

    # Test parsing a problem file
    import sys
    if len(sys.argv) > 1:
        clauses = parse_problem_file(sys.argv[1])
        print(f"\nParsed {len(clauses)} clauses from {sys.argv[1]}")
        for cl in clauses[:3]:
            print(f"  {cl}")

    # Test parsing an equality clause
    c2 = parse_clause(
        'cnf(i_0_3, negated_conjecture, (esk3_0=esk2_0|k1_funct_1(X1)!=k12_finseq_1(X2))).'
    )
    print("\nEquality clause:", c2)
    for lit in c2.literals:
        print(f"  Literal: negated={lit.negated} pred={lit.predicate} "
              f"eq={lit.is_equality} args={lit.args}")
