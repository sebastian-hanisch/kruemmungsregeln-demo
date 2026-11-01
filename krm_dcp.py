"""Der Krümmungsprüfer (DCP): jeder Knoten wird genau einmal besucht, von unten nach oben.

Jeder Knoten bekommt Krümmung (affin, konvex, konkav oder unbekannt) und Vorzeichen (≥ 0, ≤ 0 oder offen). Ein Atom hat eine feste Krümmung und je Argument eine
Monotonie, die vom Vorzeichen des Arguments abhängen darf. Die Verkettungsregel: Ein konvexes Atom braucht für ein wachsendes Argument ein konvexes (oder affines),
für ein fallendes ein konkaves, bei unbekannter Monotonie ein affines Argument; ein konkaves Atom spiegelbildlich. Die Regeln sind hinreichend, nicht notwendig:
„unbekannt“ heißt, dass kein Beweis gefunden wurde, nicht, dass die Funktion nicht konvex wäre.

Die Regeln und Vorzeichenregeln entsprechen denen von cvxpy 1.9.3 (tests/test_oracle_kruemmungsregeln.py vergleicht sie auf festen Zufallsbäumen)."""

from __future__ import annotations

from dataclasses import dataclass

import krm_expr as X

AFF, CVX, CCV, UNK = "affin", "konvex", "konkav", "unbekannt"
POS, NEG, ANY = "≥ 0", "≤ 0", "offen"
INC, DEC = "wachsend", "fallend"

_FLIP = {AFF: AFF, CVX: CCV, CCV: CVX, UNK: UNK}
_FLIP_SIGN = {POS: NEG, NEG: POS, ANY: ANY}


def _by_sign(sign):
    """Monotonie von square, abs und der Vektornorm: wachsend ab null, fallend bis null, sonst unbestimmt."""
    return INC if sign == POS else (DEC if sign == NEG else None)


# op -> (Krümmung des Atoms, Monotonie je Argument (Vorzeichen, Index) -> INC / DEC / None, Vorzeichen des Ergebnisses aus den Vorzeichen der Argumente, Kurzbeschreibung)
ATOMS = {
    "square": (CVX, lambda s, i: _by_sign(s), lambda ss: POS, "konvex; wachsend für Argumente ≥ 0, fallend für Argumente ≤ 0"),
    "abs": (CVX, lambda s, i: _by_sign(s), lambda ss: POS, "konvex; wachsend für Argumente ≥ 0, fallend für Argumente ≤ 0"),
    "exp": (CVX, lambda s, i: INC, lambda ss: POS, "konvex und wachsend"),
    "maximum": (CVX, lambda s, i: INC, lambda ss: POS if any(v == POS for v in ss) else ANY, "konvex und wachsend"),
    "minimum": (CCV, lambda s, i: INC, lambda ss: POS if all(v == POS for v in ss) else (NEG if any(v == NEG for v in ss) else ANY), "konkav und wachsend"),
    "log": (CCV, lambda s, i: INC, lambda ss: ANY, "konkav und wachsend"),
    "sqrt": (CCV, lambda s, i: INC, lambda ss: POS, "konkav und wachsend"),
    "inv_pos": (CVX, lambda s, i: DEC, lambda ss: POS, "konvex und fallend"),
    "norm2": (CVX, lambda s, i: _by_sign(s), lambda ss: POS, "konvex; wachsend für Vektoren ≥ 0, fallend für Vektoren ≤ 0"),
    "log_sum_exp": (CVX, lambda s, i: INC, lambda ss: POS if all(v == POS for v in ss) else ANY, "konvex und wachsend"),
    "quad_over_lin": (CVX, lambda s, i: _by_sign(s) if i == 0 else DEC, lambda ss: POS, "konvex; im ersten Argument wachsend für ≥ 0 und fallend für ≤ 0, im zweiten fallend"),
}
assert set(ATOMS) == set(X.ATOMS)


@dataclass(frozen=True)
class Info:
    curv: str
    sign: str
    reason: str | None = None      # nur bei „unbekannt“: welche Regel verletzt ist


@dataclass(frozen=True)
class Report:
    """Ergebnis der Prüfung. `nodes` ist in der Reihenfolge von unten nach oben (die Wurzel zuletzt), `infos` läuft parallel dazu.

    Bewusst keine Schlüssel aus `id(Knoten)`: Streamlits `cache_data` kopiert Ergebnisse (pickle), und die Kopien haben andere Adressen."""
    root: X.Node
    nodes: tuple
    infos: tuple
    var_sign: str

    def by_id(self) -> dict:
        """Zuordnung id(Knoten) -> Info für die Knoten dieses Objekts (nach jedem Kopieren neu zu bilden)."""
        return {id(n): i for n, i in zip(self.nodes, self.infos)}

    @property
    def curvature(self) -> str:
        return self.infos[-1].curv

    @property
    def sign(self) -> str:
        return self.infos[-1].sign

    @property
    def proven_convex(self) -> bool:
        return self.curvature in (CVX, AFF)

    @property
    def proven_concave(self) -> bool:
        return self.curvature in (CCV, AFF)


def _postorder(root: X.Node) -> list:
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
            continue
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.append((n, True))
        stack.extend((a, False) for a in n.args)
    return order


def _sum_sign(ss):
    return POS if all(v == POS for v in ss) else (NEG if all(v == NEG for v in ss) else ANY)


def check(root: X.Node, var_sign: str = POS) -> Report:
    """Krümmung und Vorzeichen jedes Knotens. `var_sign` ist POS (x, y ≥ 0) oder ANY (frei)."""
    if var_sign not in (POS, ANY):
        raise ValueError("var_sign: POS oder ANY")
    infos: dict[int, Info] = {}
    has_var: dict[int, bool] = {}
    order = _postorder(root)
    for n in order:
        op = n.op
        if op == "var":
            infos[id(n)], has_var[id(n)] = Info(AFF, var_sign), True
            continue
        if op == "const":
            infos[id(n)], has_var[id(n)] = Info(AFF, POS if n.k >= 0 else NEG), False
            continue
        kids = [infos[id(a)] for a in n.args]
        hv = any(has_var[id(a)] for a in n.args)
        has_var[id(n)] = hv
        if not hv:
            v = X.constant_value(n)
            infos[id(n)] = Info(AFF, POS if v >= 0 else NEG)
            continue
        cs, ss = [k.curv for k in kids], [k.sign for k in kids]
        if op == "neg":
            infos[id(n)] = Info(_FLIP[cs[0]], _FLIP_SIGN[ss[0]], None if cs[0] != UNK else "Argument ist unbekannt")
        elif op == "scale":
            if n.k >= 0:
                infos[id(n)] = Info(cs[0], ss[0], None if cs[0] != UNK else "Argument ist unbekannt")
            else:
                infos[id(n)] = Info(_FLIP[cs[0]], _FLIP_SIGN[ss[0]], None if cs[0] != UNK else "Argument ist unbekannt")
        elif op == "add":
            sign = _sum_sign(ss)
            if UNK in cs:
                infos[id(n)] = Info(UNK, sign, "ein Summand ist unbekannt")
            elif all(c == AFF for c in cs):
                infos[id(n)] = Info(AFF, sign)
            elif all(c in (AFF, CVX) for c in cs):
                infos[id(n)] = Info(CVX, sign)
            elif all(c in (AFF, CCV) for c in cs):
                infos[id(n)] = Info(CCV, sign)
            else:
                infos[id(n)] = Info(UNK, sign, "Summe aus einem konvexen und einem konkaven Summanden")
        elif op == "mul":
            if ss[0] == ss[1] != ANY:
                sign = POS
            elif ANY not in ss:
                sign = NEG
            else:
                sign = ANY
            infos[id(n)] = Info(UNK, sign, "Produkt zweier Terme mit Variablen: dafür gibt es keine Regel (Konstante · Term wäre ein Skalieren)")
        else:
            curv, mono, sign_of, _ = ATOMS[op]
            sign = sign_of(ss)
            if op == "norm2":
                vec = POS if all(v == POS for v in ss) else (NEG if all(v == NEG for v in ss) else ANY)
                arg_signs = [vec] * len(ss)
            else:
                arg_signs = ss
            reason = None
            for i, c in enumerate(cs):
                if c == AFF:
                    continue
                m = mono(arg_signs[i], i)
                ok = (c != UNK) and ((curv == CVX and ((m == INC and c == CVX) or (m == DEC and c == CCV))) or (curv == CCV and ((m == INC and c == CCV) or (m == DEC and c == CVX))))
                if ok:
                    continue
                where = f"Argument {i + 1} von {op}" if len(cs) > 1 else f"Argument von {op}"
                if c == UNK:
                    reason = f"{where} ist selbst unbekannt"
                elif m is None:
                    reason = f"{where}: die Monotonie von {op} hängt vom Vorzeichen ab, das Argument ist {c} und sein Vorzeichen ist nicht bekannt"
                else:
                    need = CVX if (curv == CVX) == (m == INC) else CCV
                    reason = f"{where}: {op} ist {curv} und {m}, verlangt wird deshalb ein {need}es Argument, es ist aber {c}"
                break
            infos[id(n)] = Info(curv if reason is None else UNK, sign, reason)
    return Report(root, tuple(order), tuple(infos[id(n)] for n in order), var_sign)


def annotate(report: Report) -> list:
    """Zeilen für die Anzeige im Vorlauf (Wurzel zuerst): (Tiefe, Knoten-Text, Krümmung, Vorzeichen, Grund)."""
    rows, stack = [], [(report.root, 0)]
    by_id = report.by_id()
    while stack:
        n, d = stack.pop()
        info = by_id[id(n)]
        if n.op == "var":
            label = n.name
        elif n.op == "const":
            label = f"{n.k:g}"
        elif n.op == "scale":
            label = f"{n.k:g} · …"
        elif n.op == "add":
            label = "Summe"
        elif n.op == "neg":
            label = "−(…)"
        elif n.op == "mul":
            label = "Produkt"
        else:
            label = n.op
        rows.append((d, label, info.curv, info.sign, info.reason))
        stack.extend((a, d + 1) for a in reversed(n.args))
    return rows


def first_failure(report: Report):
    """Der unterste Knoten, dessen Regel verletzt ist (dort liegt die Ursache): (Knoten, Grund) oder None."""
    by_id = report.by_id()
    for n in report.nodes:
        info = by_id[id(n)]
        if info.reason is not None and not any(by_id[id(a)].curv == UNK for a in n.args):
            return n, info.reason
    for n in report.nodes:
        if by_id[id(n)].reason is not None:
            return n, by_id[id(n)].reason
    return None
