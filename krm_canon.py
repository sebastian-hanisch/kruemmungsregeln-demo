"""Vom Beweis zum Kegelprogramm: jedes Atom bekommt eine Hilfsvariable und eine Kegelbedingung (Epigraph bzw. Hypograph).

Konvexe Atome (abs, square, maximum, norm2, inv_pos, quad_over_lin): t ≥ f(Argumente); konkave Atome (minimum, sqrt): t ≤ f(Argumente). Welche Seite gilt, bestimmt die Krümmung des
Atoms, und dass die Verkettung aufgeht, hat der Prüfer bewiesen. Das Ziel ist die Wurzel (konvex, also Epigraph). Die Kegel sind Orthant und Lorentz (Stück 2); der Kern ist der Barrierlöser aus Stück 5 (wortgleich in Stück 2, 4 und 5).

Zwei Absicherungen, die der Kern nur teilweise leistet (er meldet `far_from_path`, mehr nicht):
  * Der Startwert t₀ ist ν / max(1, |cᵀx₀|): bei Startwerten, die weit über dem Optimum liegen, braucht die gedämpfte Phase sonst hunderte Schritte.
  * Am Ende wird die Zentriertheit unabhängig geprüft (0 ≤ halbes Dekrement-Quadrat ≤ 1e-3 bei t = ν/Lücke). Die Schranke ν/t gilt nur für einen zentrierten Punkt; ist er es nicht, gibt es keinen Wert."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import kegel_kern as K
import krm_dcp as D
import krm_expr as X

SOLVED, NO_START, NOT_CENTERED, EXP_CONE, NOT_PROVEN, NUMERIC = "gelöst", "kein Start", "nicht zentriert", "Exponentialkegel", "nicht konvex bewiesen", "Rechengrenze"
CENTERED_MAX = 1e-3
DEFAULT_TOL = 1e-9


class NoStart(Exception):
    pass


class NeedsExpCone(Exception):
    pass


@dataclass(frozen=True)
class CanonResult:
    status: str
    detail: str = ""
    value: float | None = None
    point: dict = field(default_factory=dict)        # Variablenwerte im Optimum
    gap_bound: float | None = None
    converged: bool = True                           # falsch, wenn die Rechengenauigkeit vor der geforderten Lücke endete (die gemeldete Lücke gilt trotzdem)
    far_from_path: bool = False                      # Meldung des Kerns: eine Zentrierung endete nach 100 Schritten weit vom Pfad (die Schranke ν/t gilt dort nicht)
    rounds: int = 0
    newton_steps: int = 0
    centrality: float | None = None
    variables: int = 0                               # Eingabevariablen
    aux: int = 0                                     # Hilfsvariablen
    rows: int = 0
    orthant: int = 0
    soc: tuple = ()
    program: tuple = ()                              # (Bedingung, Kegel, Zeilen) je Atom und Kastenrand
    problems: tuple = ()                             # Rückumformung: Kegel, in denen s nicht liegt
    direct_value: float | None = None                # Ausdruck an dem gelösten Punkt direkt ausgewertet


class _Canon:
    def __init__(self, names, lo, hi):
        self.names, self.lo, self.hi = list(names), lo, hi
        self.x0 = [(lo + hi) / 2.0] * len(names)
        self.cones = []                              # (kind, [affin ...], Beschriftung)
        self.memo: dict[int, tuple] = {}

    def new_var(self, value):
        self.x0.append(float(value))
        return len(self.x0) - 1

    @staticmethod
    def comb(*terms):
        coef, const = {}, 0.0
        for f, (c, k) in terms:
            for j, v in c.items():
                coef[j] = coef.get(j, 0.0) + f * v
            const += f * k
        return coef, const

    def value(self, a):
        return sum(v * self.x0[j] for j, v in a[0].items()) + a[1]

    def build(self, n: X.Node):
        key = id(n)
        if key in self.memo:
            return self.memo[key]
        out = self._build(n)
        self.memo[key] = out
        return out

    def _build(self, n):
        op = n.op
        if op == "var":
            return {self.names.index(n.name): 1.0}, 0.0
        if op == "const":
            return {}, float(n.k)
        if not n.has_var():
            return {}, X.constant_value(n)
        if op == "neg":
            return self.comb((-1.0, self.build(n.args[0])))
        if op == "scale":
            return self.comb((n.k, self.build(n.args[0])))
        if op == "add":
            return self.comb(*[(1.0, self.build(a)) for a in n.args])
        if op in X.EXP_ATOMS:
            raise NeedsExpCone(op)
        if op not in X.CONE_ATOMS:
            raise ValueError(f"keine Kegelübersetzung für {op}")
        args = [self.build(a) for a in n.args]
        vals = [self.value(a) for a in args]
        one = ({}, 1.0)
        text = ", ".join(X.to_text(a) for a in n.args)
        j = len(self.x0)
        if op == "abs":
            t = self.new_var(abs(vals[0]) + 1.0)
            T = ({t: 1.0}, 0.0)
            self.cones.append(("l", [self.comb((1, T), (-1, args[0]))], f"t{j} ≥ {text}"))
            self.cones.append(("l", [self.comb((1, T), (1, args[0]))], f"t{j} ≥ −({text})"))
        elif op == "maximum":
            t = self.new_var(max(vals) + 1.0)
            T = ({t: 1.0}, 0.0)
            for a, arg in zip(args, n.args):
                self.cones.append(("l", [self.comb((1, T), (-1, a))], f"t{j} ≥ {X.to_text(arg)}"))
        elif op == "minimum":
            t = self.new_var(min(vals) - 1.0)
            T = ({t: 1.0}, 0.0)
            for a, arg in zip(args, n.args):
                self.cones.append(("l", [self.comb((1, a), (-1, T))], f"t{j} ≤ {X.to_text(arg)}"))
        elif op == "square":
            t = self.new_var(vals[0] ** 2 + 1.0)
            T = ({t: 1.0}, 0.0)
            self.cones.append(("q", [self.comb((1, T), (1, one)), self.comb((1, T), (-1, one)), self.comb((2, args[0]))], f"t{j} ≥ ({text})²   [(t+1, t−1, 2u) ∈ Lorentz]"))
        elif op == "norm2":
            t = self.new_var(float(np.sqrt(sum(v * v for v in vals))) + 1.0)
            T = ({t: 1.0}, 0.0)
            self.cones.append(("q", [T] + args, f"t{j} ≥ ‖({text})‖   [(t, u₁, …) ∈ Lorentz]"))
        elif op == "sqrt":
            if vals[0] <= 0:
                raise NoStart(f"sqrt: das Argument ist im Startpunkt {vals[0]:.3g} und nicht größer als null")
            t = self.new_var(0.5 * float(np.sqrt(vals[0])))
            T = ({t: 1.0}, 0.0)
            self.cones.append(("q", [self.comb((1, args[0]), (1, one)), self.comb((1, args[0]), (-1, one)), self.comb((2, T))], f"t{j} ≤ √({text})   [(u+1, u−1, 2t) ∈ Lorentz]"))
        elif op == "inv_pos":
            if vals[0] <= 0:
                raise NoStart(f"inv_pos: das Argument ist im Startpunkt {vals[0]:.3g} und nicht größer als null")
            t = self.new_var(1.0 / vals[0] + 1.0)
            T = ({t: 1.0}, 0.0)
            self.cones.append(("q", [self.comb((1, T), (1, args[0])), self.comb((1, T), (-1, args[0])), ({}, 2.0)], f"t{j} ≥ 1/({text})   [(t+u, t−u, 2) ∈ Lorentz]"))
        elif op == "quad_over_lin":
            if vals[1] <= 0:
                raise NoStart(f"quad_over_lin: das zweite Argument ist im Startpunkt {vals[1]:.3g} und nicht größer als null")
            t = self.new_var(vals[0] ** 2 / vals[1] + 1.0)
            T = ({t: 1.0}, 0.0)
            self.cones.append(("q", [self.comb((1, T), (1, args[1])), self.comb((1, T), (-1, args[1])), self.comb((2, args[0]))], f"t{j} ≥ u²/v   [(t+v, t−v, 2u) ∈ Lorentz]"))
        return T

    def problem(self, root):
        obj = self.build(root)
        nv = len(self.x0)
        rows, kinds, program = [], [], []
        for j, name in enumerate(self.names):
            rows.append(({j: 1.0}, -self.lo))
            kinds.append(("l", 1))
            rows.append(({j: -1.0}, self.hi))
            kinds.append(("l", 1))
            program.append((f"{self.lo:g} ≤ {name} ≤ {self.hi:g}", "Orthant", 2))
        for kind, comps, label in self.cones:
            rows.extend(comps)
            kinds.append((kind, len(comps)))
            program.append((label, "Orthant" if kind == "l" else "Lorentz", len(comps)))
        A = np.zeros((len(rows), nv))
        b = np.zeros(len(rows))
        for i, (coef, const) in enumerate(rows):
            for j, v in coef.items():
                A[i, j] = -v
            b[i] = const
        c = np.zeros(nv)
        for j, v in obj[0].items():
            c[j] = v
        return K.ConeProblem(c, A, b, tuple(kinds), np.array(self.x0, float)), obj[1], tuple(program)


def centrality(prob: K.ConeProblem, x: np.ndarray, gap_bound: float) -> float:
    """Halbes Dekrement-Quadrat am Punkt x bei t = ν/Lücke. Klein und nicht negativ heißt: x ist zentriert, die Schranke gilt."""
    nu = K.barrier_parameter(prob.cones)
    t = nu / gap_bound
    g, h = K.newton_system(prob, x, t)
    _, d2 = K.newton_step(g, h)
    return float(d2 / 2.0)


def is_centered(value: float) -> bool:
    return 0.0 <= value <= CENTERED_MAX


def solve(root: X.Node, report: D.Report, lo: float, hi: float, tol: float = DEFAULT_TOL, guard: bool = True) -> CanonResult:
    """Kegelprogramm bauen und lösen. `guard=False` lässt den Startwert t₀ des Kerns und die Zentriertheitsprüfung weg (nur für die Gegenprobe der Falschmeldung)."""
    if not report.proven_convex:
        return CanonResult(NOT_PROVEN, "Der Prüfer hat keine Konvexität bewiesen: ohne Beweis gibt es keine Übersetzung.")
    names = root.variables()
    cn = _Canon(names, lo, hi)
    try:
        prob, offset, program = cn.problem(root)
    except NeedsExpCone as e:
        return CanonResult(EXP_CONE, f"{e} braucht den Exponentialkegel (Stück 7 der Reihe); der Kern kennt nur Orthant, Lorentz-Kegel und PSD-Kegel.")
    except NoStart as e:
        return CanonResult(NO_START, f"{e}. Ein anderer Kasten oder Startpunkt (Phase I, Stück 6) wäre nötig.")
    s0 = prob.b - prob.A @ prob.x0
    if not np.isfinite(K.barrier_value(prob.cones, s0)):
        return CanonResult(NO_START, "Der Kastenmittelpunkt ist kein streng zulässiger Punkt.")
    nu = K.barrier_parameter(prob.cones)
    t0 = nu / max(1.0, abs(float(prob.c @ prob.x0) + offset)) if guard else None
    sizes = prob.sizes()
    try:
        res = K.solve_barrier(prob, tol=tol, t0=t0)
    except np.linalg.LinAlgError:
        return CanonResult(NUMERIC, "Die Rechnung des Kerns scheitert schon in der ersten Runde (die Hesse-Matrix wird numerisch singulär, etwa bei sehr verschieden großen Koeffizienten); es gibt keinen Wert.",
                           variables=len(names), aux=len(prob.c) - len(names), rows=sizes["rows"], orthant=sizes["orthant"], soc=tuple(sizes["soc"]), program=program)
    common = dict(variables=len(names), aux=len(prob.c) - len(names), rows=sizes["rows"], orthant=sizes["orthant"], soc=tuple(sizes["soc"]), program=program)
    cen = centrality(prob, res.x, res.gap_bound)
    if guard and not is_centered(cen):
        return CanonResult(NOT_CENTERED, "Die letzte Zentrierung ist nicht erreicht (Dekrement " + f"{cen:.2g}" + "): die Schranke ν/t gilt dort nicht, es gibt keinen Wert.",
                           rounds=res.rounds, newton_steps=res.newton_steps, centrality=cen, far_from_path=res.far_from_path, **common)
    point = {n: float(res.x[i]) for i, n in enumerate(names)}
    full = {k: np.array([point.get(k, 0.0)]) for k in X.VARIABLES}
    direct = float(X.evaluate(root, full)[0])
    return CanonResult(SOLVED, "", value=res.value + offset, point=point, gap_bound=res.gap_bound, converged=res.converged, far_from_path=res.far_from_path, rounds=res.rounds, newton_steps=res.newton_steps, centrality=cen,
                       problems=tuple(K.check_solution(prob, res.x)), direct_value=direct, **common)


def grid_minimum(root: X.Node, lo: float, hi: float, points: int = 201) -> float:
    """Kleinster Wert auf einem Gitter des Kastens: eine unabhängige, grobe Gegenprobe (jeder Gitterpunkt ist zulässig, also liegt das Minimum nie über dem Optimum)."""
    names = root.variables()
    axis = np.linspace(lo, hi, points)
    if len(names) == 1:
        grid = {names[0]: axis, **{k: np.zeros_like(axis) for k in X.VARIABLES if k != names[0]}}
    else:
        gx, gy = np.meshgrid(axis, axis)
        grid = {"x": gx.ravel(), "y": gy.ravel()}
    f = X.evaluate(root, grid)
    f = f[np.isfinite(f)]
    return float(f.min())
