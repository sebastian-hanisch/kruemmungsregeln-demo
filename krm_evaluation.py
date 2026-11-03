"""Auswertung: eine Analyse für den eingestellten Ausdruck, die Experimente 2 bis 4 und die Verdikte (reine Funktionen, an Hand gesetzten Werten getestet)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from statistics import median

import numpy as np

import krm_canon as CN
import krm_catalog as CAT
import krm_chords as CH
import krm_constants as C
import krm_dcp as D
import krm_expr as X
import krm_rewrite as RW


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num(x: float) -> str:
    """Wert für Texte: Dezimalkomma, drei Nachkommastellen, kleine und große Beträge in Zehnerpotenz."""
    return sci(x, 2) if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5) else de(x, 3)


def pct(part: int, whole: int) -> str:
    return f"{de(100.0 * part / whole, 1)} %" if whole else "–"


# ----------------------------------------------------------------------------- Analyse des eingestellten Ausdrucks

@dataclass(frozen=True)
class Settings:
    expr: str = C.DEFAULT_EXPR
    sign: str = C.DEFAULT_SIGN
    chords: int = C.DEFAULT_CHORDS
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    error: str | None = None
    tree: X.Node | None = None
    text: str = ""
    var_sign: str = D.ANY
    box: tuple = (-3.0, 3.0)
    report: D.Report | None = None
    annotation: list = field(default_factory=list)
    failure: tuple | None = None                 # (Knoten, Grund) der untersten verletzten Regel
    chords: CH.ChordResult | None = None
    judgement: CH.Judgement | None = None
    rewrites: list = field(default_factory=list)
    canon: CN.CanonResult | None = None          # Kegelprogramm des eingegebenen Ausdrucks (wenn bewiesen konvex)
    canon_rewrite: CN.CanonResult | None = None  # Kegelprogramm der ersten Umformung
    grid_min: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def var_sign_of(sign: str) -> str:
    return D.POS if sign == C.SIGN_NONNEG else D.ANY


def analyse(s: Settings) -> Analysis:
    try:
        tree = X.parse(s.expr)
    except X.ParseError as e:
        return Analysis(s, error=str(e))
    vs = var_sign_of(s.sign)
    lo, hi = C.BOX[s.sign]
    report = D.check(tree, vs)
    chords = CH.chord_test(tree, lo, hi, s.chords, s.seed)
    judgement = CH.judge(report, chords)
    rewrites = RW.find_rewrites(tree, vs, lo, hi) if report.curvature == D.UNK else []
    canon = CN.solve(tree, report, lo, hi) if report.proven_convex else None
    canon_rw = None
    grid = None
    if rewrites and rewrites[0].curvature in (D.CVX, D.AFF):
        canon_rw = CN.solve(rewrites[0].tree, D.check(rewrites[0].tree, vs), lo, hi)
    if report.proven_convex or (rewrites and rewrites[0].curvature in (D.CVX, D.AFF)):
        grid = CN.grid_minimum(tree, lo, hi)
    return Analysis(s, None, tree, X.to_text(tree), vs, (lo, hi), report, D.annotate(report), D.first_failure(report), chords, judgement, rewrites, canon, canon_rw, grid)


def chord_text(c: CH.Chord) -> str:
    return (f"von a = ({num(c.a[0])}; {num(c.a[1])}) nach b = ({num(c.b[0])}; {num(c.b[1])}) bei λ = {de(c.lam, 3)}: f(a) = {num(c.f_a)}, f(b) = {num(c.f_b)}, "
            f"f(Mitte) = {num(c.f_mid)}, Sehnenmittel {num(c.average)}")


# ----------------------------------------------------------------------------- Experiment 2: Katalog

def catalog_rows() -> list:
    rows = []
    for e in CAT.CATALOG:
        tree = X.parse(e.text)
        rep = D.check(tree, e.sign)
        ch = CH.chord_test(tree, e.lo, e.hi, C.CATALOG_CHORDS, C.CORPUS_SEED)
        j = CH.judge(rep, ch)
        rw = RW.find_rewrites(tree, e.sign, e.lo, e.hi)[:1] if rep.curvature == D.UNK else []
        rows.append({"entry": e, "curvature": rep.curvature, "judgement": j, "chords": ch,
                     "rewrite": rw[0] if rw else None})
    return rows


# ----------------------------------------------------------------------------- Zufallsbäume

_ALL_OPS = ["add", "scale", "neg", "square", "abs", "exp", "maximum", "minimum", "norm2", "log_sum_exp", "mul", "quad_over_lin"]
_POSITIVE_OPS = ["log", "sqrt", "inv_pos"]
_CONE_OPS = ["add", "scale", "neg", "square", "abs", "maximum", "minimum", "norm2", "mul", "quad_over_lin", "sqrt", "inv_pos"]
_COMBINING = ("add", "scale", "maximum", "minimum", "mul", "quad_over_lin", "norm2", "log_sum_exp")


def random_tree(rng: np.random.Generator, depth: int, ops: list) -> X.Node:
    """Zufallsbaum aus den Operationen `ops`; Blätter sind x, y oder eine Konstante. Atome ohne Variable im Argument werden mit einer Variablen aufgefüllt."""
    if depth == 0 or rng.random() < 0.15:
        if rng.random() < 0.75:
            return X.var(X.VARIABLES[int(rng.integers(2))])
        return X.const(float(rng.choice([1.0, 2.0, 0.5])))
    op = ops[int(rng.integers(len(ops)))]

    def sub():
        t = random_tree(rng, depth - 1, ops)
        if t.has_var() or op in _COMBINING:
            return t
        return X.var(X.VARIABLES[int(rng.integers(2))])

    if op == "scale":
        return X.scale(float(rng.choice([-2.0, -1.0, 0.5, 1.0, 3.0])), _with_var(sub(), rng))
    if op == "neg":
        return X.neg(_with_var(sub(), rng))
    if op in ("add", "maximum", "minimum", "norm2", "log_sum_exp", "mul", "quad_over_lin"):
        a, b = sub(), sub()
        if op == "add":
            return X.add(a, b) if (a.has_var() or b.has_var()) else X.add(X.var("x"), b)
        if op == "mul":
            return X.Node("mul", (_with_var(a, rng), _with_var(b, rng)))
        return X.atom(op, _with_var(a, rng), b)
    return X.atom(op, _with_var(sub(), rng))


def _with_var(t: X.Node, rng) -> X.Node:
    return t if t.has_var() else X.var(X.VARIABLES[int(rng.integers(2))])


def corpus_summary(sign: str, trees: int = C.CORPUS_TREES, chords: int = C.CORPUS_CHORDS, seed: int = C.CORPUS_SEED) -> dict:
    """Zufallsbäume (Tiefe 3) gegen Prüfer und Sehnen: Urteil des Prüfers, Urteil der Sehnen, Widersprüche."""
    rng = np.random.default_rng(seed + (0 if sign == C.SIGN_NONNEG else 1))
    ops = _ALL_OPS + (_POSITIVE_OPS if sign == C.SIGN_NONNEG else [])
    lo, hi = C.BOX[sign]
    vs = var_sign_of(sign)
    out = dict(sign=sign, trees=0, excluded=0, proven_convex=0, proven_concave=0, unknown=0, unknown_refuted=0, unknown_neither=0, unknown_open_convex=0,
               unknown_open_concave=0, contradictions=0)
    i = 0
    while out["trees"] + out["excluded"] < trees:
        t = random_tree(rng, 3, ops)
        if not t.has_var() or t.op in ("var", "const"):
            continue
        rep = D.check(t, vs)
        ch = CH.chord_test(t, lo, hi, chords, seed + i)
        i += 1
        if not ch.conclusive:
            out["excluded"] += 1
            continue
        out["trees"] += 1
        j = CH.judge(rep, ch)
        if j.status == CH.CONTRADICTION:
            out["contradictions"] += 1
        elif rep.proven_convex:
            out["proven_convex"] += 1
        elif rep.proven_concave:
            out["proven_concave"] += 1
        else:
            out["unknown"] += 1
            if ch.convex_refuted and ch.concave_refuted:
                out["unknown_neither"] += 1
                out["unknown_refuted"] += 1
            elif ch.convex_refuted:
                out["unknown_refuted"] += 1
                out["unknown_open_concave"] += 1
            else:
                out["unknown_open_convex"] += 1
    return out


# ----------------------------------------------------------------------------- Experiment 3: Trefferquote der Sehnen

def chord_power(trees: int = C.POWER_TREES, steps=C.POWER_STEPS, reference: int = C.POWER_REFERENCE, bump_runs: int = C.BUMP_RUNS) -> dict:
    """Wie viele Sehnen braucht die Widerlegung? Zufallsbäume (x, y > 0), die der Prüfer nicht beweist und die bei `reference` Sehnen als nicht konvex widerlegt sind,
    und die kleine Beule x² + 0,8 e^(−2000 (x−2)²)."""
    lo, hi = C.BOX[C.SIGN_NONNEG]
    rng = np.random.default_rng(C.CORPUS_SEED + 7)
    pool = []
    seed = 0
    while len(pool) < trees:
        t = random_tree(rng, 3, _ALL_OPS + _POSITIVE_OPS)
        if not t.has_var() or t.op in ("var", "const") or D.check(t, D.POS).curvature != D.UNK:
            continue
        seed += 1
        if CH.chord_test(t, lo, hi, reference, 10_000_000 + seed).convex_refuted:
            pool.append(t)
    corpus = {}
    for n in steps:
        corpus[n] = sum(CH.chord_test(t, lo, hi, n, 20_000_000 + k).convex_refuted for k, t in enumerate(pool))
    bump_tree = X.parse(C.BUMP_EXPR)
    flo, fhi = C.BOX[C.SIGN_FREE]
    bump = {n: sum(CH.chord_test(bump_tree, flo, fhi, n, 30_000_000 + r).convex_refuted for r in range(bump_runs)) for n in steps}
    return {"trees": len(pool), "steps": tuple(steps), "corpus": corpus, "bump": bump, "bump_runs": bump_runs, "reference": reference,
            "bump_verdict": D.check(bump_tree, D.ANY).curvature}


# ----------------------------------------------------------------------------- Zeit des Prüfers

def timing_rows(chains=C.TIMING_CHAINS, repeats: int = 15) -> list:
    """Zeit des Prüfers an Ketten wachsender Größe (jeder Knoten einmal besucht)."""
    rows = []
    for reps in chains:
        t = X.var("x")
        for _ in range(reps):
            t = X.scale(0.3, X.add(X.atom("exp", t), X.var("y")))
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            D.check(t, D.ANY)
            best = min(best, time.perf_counter() - t0)
        n = t.size()
        rows.append({"nodes": n, "seconds": best, "per_node": best / n})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Kegelprogramme

def canon_corpus(trees: int = C.CANON_TREES, seed: int = C.CANON_SEED) -> dict:
    """DCP-konvexe Zufallsbäume mit Kegelatomen (x, y ∈ [0,1; 3]) als Kegelprogramm gelöst: gelöst, ohne Start, nicht zentriert, Abweichung zur Gittersuche."""
    rng = np.random.default_rng(seed)
    lo, hi = C.BOX[C.SIGN_NONNEG]
    out = dict(drawn=0, convex=0, no_start=0, solved=0, not_centered=0, consistent=0, aux=[], rows=[], soc=[], newton=[])
    while out["convex"] < trees:
        t = random_tree(rng, 3, _CONE_OPS)
        if not t.has_var() or t.op in ("var", "const"):
            continue
        out["drawn"] += 1
        rep = D.check(t, D.POS)
        if not rep.proven_convex:
            continue
        out["convex"] += 1
        res = CN.solve(t, rep, lo, hi)
        if res.status == CN.NO_START:
            out["no_start"] += 1
        elif res.status == CN.NOT_CENTERED:
            out["not_centered"] += 1
        elif res.status == CN.SOLVED:
            out["solved"] += 1
            grid = CN.grid_minimum(t, lo, hi, 81)
            tol = res.gap_bound + 1e-6 * (1.0 + abs(res.value))
            if res.value <= grid + res.gap_bound + 1e-9 and abs(res.direct_value - res.value) <= tol:
                out["consistent"] += 1
            out["aux"].append(res.aux)
            out["rows"].append(res.rows)
            out["soc"].append(len(res.soc))
            out["newton"].append(res.newton_steps)
    return out


def outlier_experiment() -> dict:
    """Der Fall, in dem der Kern ohne Absicherung eine falsche Lückenschranke meldet (mit und ohne Absicherung)."""
    lo, hi = C.BOX[C.SIGN_NONNEG]
    t = X.parse(C.OUTLIER_EXPR)
    rep = D.check(t, D.POS)
    guarded, bare = CN.solve(t, rep, lo, hi, guard=True), CN.solve(t, rep, lo, hi, guard=False)
    return {"expr": C.OUTLIER_EXPR, "guarded": guarded, "bare": bare, "grid": CN.grid_minimum(t, lo, hi, 401)}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_analysis(a: Analysis) -> tuple:
    j, rep, ch = a.judgement, a.report, a.chords
    nodes = a.tree.size()
    if j.status == CH.CONTRADICTION:
        return "error", ("Widerspruch: Der Prüfer beweist die Krümmung, aber eine Sehne verletzt sie. Das darf nicht vorkommen und ist ein Fehler im Prüfer: " + chord_text(ch.convex_example or ch.concave_example))
    if j.status == CH.PROVEN_CONVEX:
        return "success", (f"**Bewiesen konvex.** Der Prüfer hat {nodes} Knoten je einmal besucht, die Wurzel ist {rep.curvature}. {ch.valid} gültige Sehnen von {ch.count} finden keine Verletzung, "
                           "wie es bei einem Beweis sein muss.")
    if j.status == CH.PROVEN_CONCAVE:
        return "info", (f"**Bewiesen konkav.** Zum Minimieren taugt die Funktion nicht (sie ist nicht konvex, außer sie wäre affin), zum Maximieren wohl: −f wäre bewiesen konvex. {ch.valid} gültige Sehnen von {ch.count} finden keine Verletzung der Konkavität.")
    if j.status == CH.REFUTED:
        c = ch.convex_example
        extra = " Auch die Konkavität ist widerlegt: weder konvex noch konkav." if j.neither else ""
        return "warning", (f"**Widerlegt: nicht konvex.** Der Prüfer schweigt, und eine Sehne zeigt, warum: {chord_text(c)}; die Funktion liegt in der Mitte um {num(c.violation)} über der Sehne, "
                           f"und das ist ein Gegenbeispiel, also ein Beweis der Nichtkonvexität.{extra}")
    if not ch.conclusive:
        return "info", (f"**Offen.** Der Prüfer schweigt, und nur {ch.valid} von {ch.count} Sehnen liegen im Definitionsbereich (mindestens {CH.MIN_VALID} nötig): die Probe sagt hier nichts.")
    text = (f"**Offen.** Der Prüfer schweigt („unbekannt“ heißt nicht „nicht konvex“), und {ch.valid} gültige Sehnen von {ch.count} finden keine Verletzung. "
            "Das ist weder ein Beweis der Konvexität noch einer der Nichtkonvexität: endlich viele Sehnen können eine kleine Beule übersehen.")
    if a.rewrites:
        r = a.rewrites[0]
        text += f" Eine bestätigte Umformung beweist es: `{r.text}` ist {r.curvature} (Abweichung der beiden Funktionen höchstens {sci(r.deviation)})."
    return "warning", text


def verdict_catalog(rows: list) -> tuple:
    unknown = [r for r in rows if r["curvature"] == D.UNK]
    open_ = [r for r in unknown if r["judgement"].status == CH.OPEN]
    refuted = [r for r in unknown if r["judgement"].status == CH.REFUTED]
    rescued = [r for r in open_ if r["rewrite"] is not None]
    contra = sum(r["judgement"].status == CH.CONTRADICTION for r in rows)
    proven = sum(r["curvature"] != D.UNK for r in rows)
    text = (f"Von {len(rows)} Einträgen beweist der Prüfer {proven}. Bei {len(unknown)} schweigt er: {len(refuted)} davon sind widerlegt (zu Recht keine Aussage), bei {len(open_)} finden die Sehnen nichts. "
            f"Von diesen {len(open_)} rettet bei **{len(rescued)}** eine Umformung den Beweis, ohne die Funktion zu ändern (eine Identität, numerisch auf 4000 Punkten bestätigt); bei den übrigen {len(open_) - len(rescued)} fehlt ein Atom oder die Regeln der Umformung reichen nicht. "
            f"Widersprüche zwischen Prüfer und Sehnen: {contra}.")
    return "info", text


def verdict_corpus(rows: list) -> tuple:
    """rows: corpus_summary je Vorzeichenfall."""
    parts = []
    for r in rows:
        possible = r["proven_convex"] + r["unknown_open_convex"]
        parts.append(f"{r['sign']}: {r['trees']} Bäume, bewiesen konvex oder affin {r['proven_convex']}, bewiesen konkav {r['proven_concave']}, „unbekannt“ {r['unknown']} "
                     f"(davon bei den Sehnen als nicht konvex widerlegt {r['unknown_refuted']}, nicht widerlegt {r['unknown_open_convex']}); bewiesen sind {pct(r['proven_convex'], possible)} der nicht widerlegten konvexen")
    contra = sum(r["contradictions"] for r in rows)
    return "info", ("; ".join(parts) + f". Widersprüche zwischen Prüfer und Sehnen: **{contra}**. Die Anteile gelten für diese Zufallsbäume, nicht für Modelle aus der Praxis.")


def verdict_power(p: dict) -> tuple:
    s0, s1 = p["steps"][0], p["steps"][-1]
    t = p["trees"]
    mid = [n for n in p["steps"] if n in (200, 4000)]
    corpus = ", ".join(f"{n} Sehnen {pct(p['corpus'][n], t)}" for n in p["steps"])
    bump = ", ".join(f"{n}: {p['bump'][n]} von {p['bump_runs']}" for n in p["steps"])
    return "info", (f"Von {t} Zufallsbäumen, die der Prüfer nicht beweist und die bei {p['reference']} Sehnen widerlegt sind, widerlegen {corpus}. Bei der kleinen Beule (nichtkonvex nur für |x − 2| < 0,016) "
                    f"finden die Sehnen die Verletzung in so vielen von {p['bump_runs']} Läufen: {bump}. **Eine verletzte Sehne beweist, eine unverletzte beweist nichts.**")


def verdict_timing(rows: list) -> tuple:
    a, b = rows[0], rows[-1]
    return "info", (f"Der Prüfer braucht für {a['nodes']} Knoten {sci(a['seconds'])} s, für {b['nodes']} Knoten {sci(b['seconds'])} s: je Knoten {de(a['per_node'] * 1e6, 1)} und {de(b['per_node'] * 1e6, 1)} µs, "
                    "also linear in der Größe (jeder Knoten wird einmal besucht, Zeiten hängen vom Rechner ab).")


def verdict_canon(a: Analysis) -> tuple:
    r = a.canon
    if r is None:
        return "info", "Ohne bewiesene Konvexität gibt es keine Übersetzung: der Beweis ist der Eintritt."
    if r.status == CN.EXP_CONE:
        return "info", r.detail
    if r.status == CN.NO_START:
        return "warning", "Bewiesen konvex, aber nicht lösbar ohne Weiteres: " + r.detail
    if r.status == CN.NOT_CENTERED:
        return "error", r.detail
    if r.status == CN.NUMERIC:
        return "warning", "Bewiesen konvex, aber die Rechnung scheitert: " + r.detail
    probs = "; ".join(r.problems) if r.problems else ""
    text = (f"Das Kegelprogramm hat {r.variables + r.aux} Variablen ({r.variables} eigene, {r.aux} Hilfsvariablen), {r.rows} Zeilen in {r.orthant} Orthant-Zeilen und {len(r.soc)} Lorentz-Kegeln. "
            f"Der Kern löst es in {r.rounds} Runden mit {r.newton_steps} Newton-Schritten, Wert {num(r.value)} bei Lücke ≤ {sci(r.gap_bound)}. "
            f"Der Ausdruck am gelösten Punkt direkt ausgewertet: {num(r.direct_value)}; das Gitter des Kastens hat sein Minimum bei {num(a.grid_min)} (nie unter dem Optimum).")
    if probs:
        return "error", text + " Die Rückumformung besteht nicht: " + probs
    return "success", text + " Die Rückumformung besteht."


def verdict_canon_corpus(c: dict) -> tuple:
    med = lambda v: de(median(v), 0) if v else "–"
    return "info", (f"{c['convex']} bewiesen konvexe Zufallsbäume mit Kegelatomen: **{c['solved']} gelöst** ({c['consistent']} davon stimmen mit Direktauswertung und Gitter überein), "
                    f"{c['no_start']} ({pct(c['no_start'], c['convex'])}) ohne streng zulässigen Start im Kastenmittelpunkt, {c['not_centered']} nicht zentriert. Im Median {med(c['aux'])} Hilfsvariablen, "
                    f"{med(c['rows'])} Zeilen, {med(c['soc'])} Lorentz-Kegel, {med(c['newton'])} Newton-Schritte.")


def verdict_outlier(o: dict) -> tuple:
    b, g = o["bare"], o["guarded"]
    return "warning", (f"`{o['expr']}` auf [0,1; 3]²: Ohne Absicherung meldet der Kern den Wert {num(b.value)} mit Lückenschranke {sci(b.gap_bound)}, das Gitter findet aber {num(o['grid'])}; "
                       f"der Kern meldet dabei `far_from_path = {b.far_from_path}`, und das halbe Dekrement-Quadrat am Endpunkt ist {sci(b.centrality)} (nicht zentriert, die Schranke gilt dort nicht). "
                       f"Mit der Absicherung (Startwert t₀ = ν/max(1, |cᵀx₀|), Prüfung der Zentriertheit) ist der Wert {num(g.value)}, Dekrement {sci(g.centrality)}.")


def verdict_tree(a: Analysis) -> tuple:
    rep = a.report
    nodes = a.tree.size()
    if rep.curvature == D.UNK:
        node, reason = a.failure
        label = node.name if node.op == "var" else node.op
        text = (f"Die Regel ist zuerst am Knoten **{label}** verletzt: {reason}. Das heißt nur, dass kein Beweis gefunden wurde, nicht dass die Funktion nicht konvex wäre.")
        if a.rewrites:
            r = a.rewrites[0]
            text += f" Nach der Umformung `{r.text}` ist die Wurzel {r.curvature}."
        return "warning", text
    return "success", f"Alle {nodes} Knoten erfüllen die Regeln: die Wurzel ist **{rep.curvature}** (Vorzeichen {rep.sign})."


def verdict_canon_rewrite(a: Analysis) -> tuple:
    r, rw = a.canon_rewrite, a.rewrites[0]
    if r.status == CN.SOLVED:
        return "success", (f"Nach der Umformung `{rw.text}`: Kegelprogramm mit {r.aux} Hilfsvariablen, {r.rows} Zeilen und {len(r.soc)} Lorentz-Kegeln, Wert {num(r.value)} bei Lücke ≤ {sci(r.gap_bound)}; "
                           f"das Gitter des Kastens hat sein Minimum bei {num(a.grid_min)} (nie unter dem Optimum).")
    return "info", f"Nach der Umformung `{rw.text}`: {r.detail}"
