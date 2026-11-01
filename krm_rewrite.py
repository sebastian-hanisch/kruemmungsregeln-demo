"""Umformungen, die einen „unbekannten“ Ausdruck in einen bewiesenen verwandeln, ohne die Funktion zu ändern.

Sieben kleine Regeln, jede eine bekannte Identität. Eine Umformung gilt erst, wenn sie gerechnet bestätigt ist: auf zufälligen Punkten des Kastens, an denen beide Fassungen
definiert sind, weichen sie höchstens um 1e-9 (relativ) ab. Gesucht wird in höchstens drei Schritten; die Suche ist keine Vollständigkeit: findet sie nichts, heißt das nur, dass
diese Regeln nicht reichen."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import krm_dcp as D
import krm_expr as X

MAX_STEPS = 3
MAX_CANDIDATES = 400
DEV_TOL = 1e-9
IDENTITY_POINTS = 4000
MIN_COMMON = 50

RULES = {
    "sqrt_squares": "sqrt(Summe von Quadraten) = norm2(…)",
    "sqrt_square": "sqrt(u²) = abs(u)",
    "log_exps": "log(Summe von Exponentialfunktionen) = log_sum_exp(…)",
    "self_product": "u·u = square(u)",
    "exp_product": "exp(a)·exp(b) = exp(a + b)",
    "quad_over_lin": "u²·(1/v) = quad_over_lin(u, v)",
    "inv_square": "1/u² = (1/u)²  für u > 0",
}


@dataclass(frozen=True)
class Rewrite:
    steps: tuple                 # Namen der angewandten Regeln (Schlüssel aus RULES)
    tree: X.Node
    text: str
    curvature: str               # Krümmung der neuen Fassung
    deviation: float             # größte relative Abweichung zur ursprünglichen Funktion auf den Prüfpunkten


def _same(a: X.Node, b: X.Node) -> bool:
    return a == b


def _local(n: X.Node, var_sign: str):
    """Alle Umformungen, die genau an diesem Knoten ansetzen: (Regel, neuer Knoten)."""
    out = []
    if n.op == "sqrt":
        u = n.args[0]
        if u.op == "square":
            out.append(("sqrt_square", X.atom("abs", u.args[0])))
        if u.op == "add":
            parts, ok, squares = [], True, 0
            for s in u.args:
                if s.op == "square":
                    parts.append(s.args[0])
                    squares += 1
                elif s.op == "scale" and s.k > 0 and s.args[0].op == "square":
                    parts.append(X.scale(math.sqrt(s.k), s.args[0].args[0]))
                    squares += 1
                elif s.op == "const" and s.k >= 0:
                    parts.append(X.const(math.sqrt(s.k)))
                else:
                    ok = False
                    break
            if ok and squares >= 1 and 2 <= len(parts) <= 4:
                out.append(("sqrt_squares", X.atom("norm2", *parts)))
    elif n.op == "log" and n.args[0].op == "add":
        parts, ok, exps = [], True, 0
        for s in n.args[0].args:
            if s.op == "exp":
                parts.append(s.args[0])
                exps += 1
            elif s.op == "scale" and s.k > 0 and s.args[0].op == "exp":
                parts.append(X.add(s.args[0].args[0], X.const(math.log(s.k))))
                exps += 1
            elif s.op == "const" and s.k > 0:
                parts.append(X.const(math.log(s.k)))
            else:
                ok = False
                break
        if ok and exps >= 1 and 2 <= len(parts) <= 4:
            out.append(("log_exps", X.atom("log_sum_exp", *parts)))
    elif n.op == "mul":
        a, b = n.args
        if _same(a, b):
            out.append(("self_product", X.atom("square", a)))
        if a.op == "exp" and b.op == "exp":
            out.append(("exp_product", X.atom("exp", X.add(a.args[0], b.args[0]))))
        for p, q in ((a, b), (b, a)):
            if p.op == "square" and q.op == "inv_pos":
                out.append(("quad_over_lin", X.atom("quad_over_lin", p.args[0], q.args[0])))
                break
    elif n.op == "inv_pos" and n.args[0].op == "square":
        u = n.args[0].args[0]
        if D.check(u, var_sign).sign == D.POS:
            out.append(("inv_square", X.atom("square", X.atom("inv_pos", u))))
    return out


def _variants(n: X.Node, var_sign: str):
    """Alle Bäume, die entstehen, wenn an genau einer Stelle eine Regel angewandt wird."""
    for rule, new in _local(n, var_sign):
        yield rule, new
    for i, child in enumerate(n.args):
        for rule, new_child in _variants(child, var_sign):
            args = n.args[:i] + (new_child,) + n.args[i + 1:]
            yield rule, X.Node(n.op, args, k=n.k, name=n.name)


def deviation(a: X.Node, b: X.Node, lo: float, hi: float, seed: int = 20261101) -> float | None:
    """Größte relative Abweichung beider Funktionen auf zufälligen Punkten, an denen beide definiert sind; None, wenn zu wenige Punkte übrig bleiben."""
    rng = np.random.default_rng(seed)
    pts = {k: rng.uniform(lo, hi, IDENTITY_POINTS) for k in X.VARIABLES}
    fa, fb = X.evaluate(a, pts), X.evaluate(b, pts)
    ok = np.isfinite(fa) & np.isfinite(fb)
    if int(ok.sum()) < MIN_COMMON:
        return None
    return float(np.max(np.abs(fa[ok] - fb[ok]) / (1.0 + np.abs(fa[ok]))))


def find_rewrites(root: X.Node, var_sign: str, lo: float, hi: float) -> list:
    """Bestätigte Umformungen, die den Prüfer zufriedenstellen (konvex, affin oder konkav), kürzeste zuerst."""
    start = D.check(root, var_sign)
    if start.curvature != D.UNK:
        return []
    seen = {X.to_text(root)}
    frontier = [((), root)]
    found: list[Rewrite] = []
    budget = MAX_CANDIDATES
    for _ in range(MAX_STEPS):
        nxt = []
        for steps, tree in frontier:
            for rule, new in _variants(tree, var_sign):
                text = X.to_text(new)
                if text in seen:
                    continue
                seen.add(text)
                budget -= 1
                if budget < 0:
                    return found
                dev = deviation(root, new, lo, hi)
                if dev is None or dev > DEV_TOL:
                    continue
                rep = D.check(new, var_sign)
                path = steps + (rule,)
                if rep.curvature != D.UNK:
                    found.append(Rewrite(path, new, text, rep.curvature, dev))
                else:
                    nxt.append((path, new))
        if found:
            break
        frontier = nxt
    return found
