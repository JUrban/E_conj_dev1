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


def _tokenize(s: str) -> list[str]:
    """Tokenize a TPTP formula string into meaningful tokens."""
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
            # skip unexpected chars
            i += 1
    return tokens


class _Parser:
    """Recursive descent parser for tokenized TPTP clause body."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

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

    def parse_term(self) -> Term:
        name = self.consume()
        if self.peek() == '(':
            # function application
            self.consume('(')
            args = [self.parse_term()]
            while self.peek() == ',':
                self.consume(',')
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

        literals = [self.parse_literal()]
        while self.peek() == '|':
            self.consume('|')
            literals.append(self.parse_literal())

        if has_paren and self.peek() == ')':
            self.consume(')')

        return literals


def parse_clause(line: str) -> Optional[Clause]:
    """Parse a single cnf(...) line into a Clause object.

    Returns None if the line is not a cnf clause.
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
    tokens = _tokenize(formula_str)
    if not tokens:
        return None

    parser = _Parser(tokens)
    try:
        literals = parser.parse_clause_body()
    except TPTPParseError:
        return None

    return Clause(name=name, role=role, literals=literals)


def parse_problem_file(filepath: str) -> list[Clause]:
    """Parse a TPTP CNF problem file, returning all clauses."""
    clauses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            clause = parse_clause(line)
            if clause is not None:
                clauses.append(clause)
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
