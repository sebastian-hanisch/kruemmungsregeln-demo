"""Ausdrucksbaum und Parser (nur Standardbibliothek, kein `eval`).

Erlaubt sind die Variablen x und y, Zahlen, die vier Operationen mit den Einschränkungen unten, Klammern und die Atome aus ATOMS.
Der Parser kennt nichts anderes: kein Attributzugriff, keine Namen außerhalb der Liste, Länge und Tiefe sind begrenzt.

    Summe, Differenz     a + b, a - b
    Skalieren            Konstante * Term, Term * Konstante, Term / Konstante
    Produkt zweier Terme mit Variablen: erlaubt, ergibt den Knoten `mul` (hat keine DCP-Regel, der Prüfer sagt „unbekannt“)
    Potenz               nur  u^2  (gleich square(u))
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_LENGTH = 200
MAX_DEPTH = 30
MAX_NODES = 150
VARIABLES = ("x", "y")

# Name -> (kleinste Argumentzahl, größte Argumentzahl)
ATOMS = {
    "square": (1, 1), "abs": (1, 1), "exp": (1, 1), "log": (1, 1), "sqrt": (1, 1), "inv_pos": (1, 1),
    "maximum": (2, 4), "minimum": (2, 4), "norm2": (2, 4), "log_sum_exp": (2, 4), "quad_over_lin": (2, 2),
}
CONE_ATOMS = ("square", "abs", "sqrt", "inv_pos", "maximum", "minimum", "norm2", "quad_over_lin")     # mit Kegelübersetzung (Orthant, Lorentz)
EXP_ATOMS = ("exp", "log", "log_sum_exp")                                                             # brauchen den Exponentialkegel (Stück 7)


class ParseError(ValueError):
    """Eingabefehler mit einer Meldung, die dem Nutzer sagt, was erlaubt ist."""


@dataclass(frozen=True)
class Node:
    op: str                       # var, const, neg, scale, add, mul oder ein Name aus ATOMS
    args: tuple = ()
    k: float | None = None        # const: Wert; scale: Faktor
    name: str | None = None       # var: x oder y

    def size(self) -> int:
        """Zahl der Knoten (iterativ, ohne Rekursionsgrenze)."""
        total, stack = 0, [self]
        while stack:
            n = stack.pop()
            total += 1
            stack.extend(n.args)
        return total

    def depth(self) -> int:
        best, stack = 0, [(self, 1)]
        while stack:
            n, d = stack.pop()
            best = max(best, d)
            stack.extend((a, d + 1) for a in n.args)
        return best

    def variables(self) -> tuple:
        seen, stack = set(), [self]
        while stack:
            n = stack.pop()
            if n.op == "var":
                seen.add(n.name)
            stack.extend(n.args)
        return tuple(v for v in VARIABLES if v in seen)

    def has_var(self) -> bool:
        return bool(self.variables())


def var(name: str) -> Node:
    return Node("var", name=name)


def const(k: float) -> Node:
    return Node("const", k=float(k))


def neg(a: Node) -> Node:
    return const(-a.k) if a.op == "const" else Node("neg", (a,))


def scale(k: float, a: Node) -> Node:
    if a.op == "const":
        return const(k * a.k)
    return Node("scale", (a,), k=float(k))


def add(*a: Node) -> Node:
    return Node("add", tuple(a))


def atom(name: str, *a: Node) -> Node:
    return Node(name, tuple(a))


# ---------------------------------------------------------------------------------------------------------------- Auswertung

def evaluate(root: Node, values: dict):
    """Wert des Baums an Arrays (oder Zahlen) `values = {"x": ..., "y": ...}`; nan, wo ein Atom nicht definiert ist. Iterativ von unten nach oben."""
    import numpy as np

    sample = next(iter(values.values())) if values else np.zeros(1)
    order, stack = [], [(root, False)]
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
        else:
            stack.append((n, True))
            stack.extend((a, False) for a in n.args)
    memo: dict[int, object] = {}
    with np.errstate(all="ignore"):
        for n in order:
            a = [memo[id(c)] for c in n.args]
            op = n.op
            if op == "var":
                v = values[n.name]
            elif op == "const":
                v = np.full_like(np.asarray(sample, dtype=float), n.k)
            elif op == "neg":
                v = -a[0]
            elif op == "scale":
                v = n.k * a[0]
            elif op == "add":
                v = sum(a[1:], a[0])
            elif op == "mul":
                v = a[0] * a[1]
            elif op == "square":
                v = a[0] ** 2
            elif op == "abs":
                v = np.abs(a[0])
            elif op == "exp":
                v = np.exp(a[0])
            elif op == "maximum":
                v = np.max(a, axis=0)
            elif op == "minimum":
                v = np.min(a, axis=0)
            elif op == "log":
                v = np.where(a[0] > 0, np.log(np.where(a[0] > 0, a[0], 1.0)), np.nan)
            elif op == "sqrt":
                v = np.where(a[0] >= 0, np.sqrt(np.where(a[0] >= 0, a[0], 0.0)), np.nan)
            elif op == "inv_pos":
                v = np.where(a[0] > 0, 1.0 / np.where(a[0] > 0, a[0], 1.0), np.nan)
            elif op == "norm2":
                v = np.sqrt(sum(x ** 2 for x in a))
            elif op == "log_sum_exp":
                m = np.max(a, axis=0)
                v = m + np.log(sum(np.exp(x - m) for x in a))
            elif op == "quad_over_lin":
                v = np.where(a[1] > 0, a[0] ** 2 / np.where(a[1] > 0, a[1], 1.0), np.nan)
            else:
                raise ValueError(f"unbekannter Knoten {op}")
            memo[id(n)] = v
    return memo[id(root)]


def constant_value(n: Node) -> float:
    """Wert eines Teilbaums ohne Variablen."""
    return float(evaluate(n, {"x": 0.0}))


# ---------------------------------------------------------------------------------------------------------------- Ausgabe

_PREC = {"add": 1, "mul": 2, "scale": 2, "neg": 3}


def _num(k: float) -> str:
    return f"{k:.12g}"


def to_text(root: Node) -> str:
    """Lesbarer, wieder einlesbarer Text des Baums (iterativ über eine explizite Funktion mit Rekursion begrenzter Tiefe: Bäume sind höchstens MAX_DEPTH tief)."""

    def go(n: Node, parent_prec: int, right: bool = False) -> str:
        if n.op == "var":
            return n.name
        if n.op == "const":
            return _num(n.k) if n.k >= 0 else f"({_num(n.k)})"
        if n.op in ATOMS:
            return f"{n.op}(" + ", ".join(go(a, 0) for a in n.args) + ")"
        if n.op == "neg":
            s = "-" + go(n.args[0], 3)
            prec = 3
        elif n.op == "scale":
            s = f"{_num(n.k)}*" + go(n.args[0], 2, True)
            prec = 2
        elif n.op == "mul":
            s = go(n.args[0], 2) + "*" + go(n.args[1], 2, True)
            prec = 2
        elif n.op == "add":
            parts = [go(n.args[0], 1)]
            for a in n.args[1:]:
                t = go(a, 1, True)
                parts.append(t)
            s = " + ".join(parts)
            prec = 1
        else:
            raise ValueError(n.op)
        return f"({s})" if prec < parent_prec or (prec == parent_prec and right and prec in (1, 2)) else s

    return go(root, 0)


# ---------------------------------------------------------------------------------------------------------------- Parser

_TOKEN = re.compile(r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>\*\*|[-+*/^(),]))")


def _tokens(text: str) -> list:
    text = text.replace("−", "-").replace("·", "*").replace("×", "*").replace("²", "^2")
    out, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            bad = text[pos:].lstrip()[:1]
            raise ParseError(f"Unerlaubtes Zeichen „{bad}“. Erlaubt: x, y, Zahlen, + − * / ^2, Klammern, Komma und die Atome {', '.join(ATOMS)}.")
        if m.group("num") is not None:
            out.append(("num", float(m.group("num"))))
        elif m.group("name") is not None:
            out.append(("name", m.group("name")))
        else:
            out.append(("op", "^" if m.group("op") == "**" else m.group("op")))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0
        self.nodes = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, op):
        kind, val = self.take()
        if (kind, val) != ("op", op):
            raise ParseError(f"Erwartet „{op}“" + (f", gefunden „{val}“." if val is not None else ", aber der Ausdruck endet."))

    def count(self):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ParseError(f"Der Ausdruck ist zu groß (mehr als {MAX_NODES} Knoten).")

    def expr(self, depth):
        if depth > MAX_DEPTH:
            raise ParseError(f"Zu tief verschachtelt (mehr als {MAX_DEPTH} Ebenen).")
        node = self.term(depth)
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            rhs = self.term(depth)
            self.count()
            if op == "-":
                rhs = neg(rhs)
            node = _fold_add(node, rhs)
        return node

    def term(self, depth):
        node = self.unary(depth)
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            rhs = self.unary(depth)
            self.count()
            if op == "*":
                node = _mul(node, rhs)
            else:
                if rhs.has_var():
                    raise ParseError("Division durch einen Ausdruck mit Variablen: schreiben Sie inv_pos(...) oder quad_over_lin(...).")
                v = constant_value(rhs)
                if v == 0 or not math.isfinite(v):
                    raise ParseError("Division durch null.")
                node = _mul(node, const(1.0 / v))
        return node

    def unary(self, depth):
        if self.peek() == ("op", "-"):
            self.take()
            self.count()
            return neg(self.unary(depth))
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary(depth)
        return self.power(depth)

    def power(self, depth):
        base = self.primary(depth)
        if self.peek() == ("op", "^"):
            self.take()
            kind, val = self.take()
            if kind != "num" or val != 2.0:
                raise ParseError("Potenzen: nur ^2 (gleich square(...)). Für andere Exponenten gibt es in dieser Demo kein Atom.")
            self.count()
            return atom("square", base) if base.has_var() else const(constant_value(base) ** 2)
        return base

    def primary(self, depth):
        kind, val = self.take()
        self.count()
        if kind == "num":
            return const(val)
        if kind == "op" and val == "(":
            node = self.expr(depth + 1)
            self.expect(")")
            return node
        if kind == "name":
            if val in VARIABLES:
                return var(val)
            if val in ATOMS:
                self.expect("(")
                args = [self.expr(depth + 1)]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.expr(depth + 1))
                self.expect(")")
                lo, hi = ATOMS[val]
                if not lo <= len(args) <= hi:
                    raise ParseError(f"{val} braucht " + (f"{lo}" if lo == hi else f"{lo} bis {hi}") + f" Argumente, nicht {len(args)}.")
                if not any(a.has_var() for a in args):
                    return const(constant_value(Node(val, tuple(args))))
                return Node(val, tuple(args))
            raise ParseError(f"Unbekannter Name „{val}“. Erlaubt sind x, y und die Atome {', '.join(ATOMS)}.")
        if kind is None:
            raise ParseError("Der Ausdruck endet zu früh.")
        raise ParseError(f"Unerwartetes Zeichen „{val}“.")


def _fold_add(a: Node, b: Node) -> Node:
    if a.op == "const" and b.op == "const":
        return const(a.k + b.k)
    parts = list(a.args) if a.op == "add" else [a]
    parts.append(b)
    return Node("add", tuple(parts))


def _mul(a: Node, b: Node) -> Node:
    if a.op == "const":
        return scale(a.k, b) if b.has_var() else const(a.k * constant_value(b))
    if b.op == "const":
        return scale(b.k, a) if a.has_var() else const(b.k * constant_value(a))
    if not a.has_var():
        return scale(constant_value(a), b)
    if not b.has_var():
        return scale(constant_value(b), a)
    return Node("mul", (a, b))


def parse(text: str) -> Node:
    """Text -> Baum; wirft ParseError mit einer verständlichen Meldung."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Der Ausdruck ist leer.")
    if len(text) > MAX_LENGTH:
        raise ParseError(f"Der Ausdruck ist zu lang (mehr als {MAX_LENGTH} Zeichen).")
    p = _Parser(_tokens(text))
    node = p.expr(1)
    if p.i != len(p.t):
        kind, val = p.peek()
        raise ParseError(f"Unerwartet „{val}“ nach dem Ausdruck.")
    if not node.has_var():
        raise ParseError("Der Ausdruck enthält keine Variable (x oder y).")
    return node
