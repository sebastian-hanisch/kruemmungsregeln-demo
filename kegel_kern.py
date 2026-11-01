"""Gemeinsamer Kern der Reihe „Konvexe und konische Optimierung“ (wortgleich in allen Repos der Reihe; nur numpy).

Kegelform:  min c^T x  mit  s = b - A x  im Kegel K.   K ist ein Produkt aus
  'l'  Orthant (LP):                      s >= 0                       (m Komponenten)
  'q'  Lorentz-Kegel (SOCP):              s0 >= ||s_1..s_m-1||          (m Komponenten)
  's'  PSD-Kegel (SDP):                   S = smat(s) ist positiv semidefinit (k x k, svec mit Faktor sqrt 2: k(k+1)/2 Komponenten)

Löser: primale Barrierenmethode. Für wachsendes t wird  t c^T x + phi(b - A x)  mit Newton minimiert (phi = Summe der Barrieren der Kegel);
zu jedem x gibt nu / t eine Schranke für die Lücke zum Optimum (nu = Barriere-Parameter). Der Start x0 muss streng zulässig sein.
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben; `solve_barrier` ruft nur der Reihe nach auf."""

import math
from dataclasses import dataclass, field

import numpy as np

SQ2 = math.sqrt(2.0)
MIN_TOL = 1e-13          # kleinere Lückenschranken sind in doppelter Genauigkeit sinnlos: nu / t läge unter der Rundung von c^T x
FAR_FROM_PATH = 1.0      # halbes Dekrement-Quadrat, ab dem eine erschöpfte Zentrierung als „weit vom Pfad“ gilt (siehe BarrierResult.far_from_path)


@dataclass(frozen=True, eq=False)
class ConeProblem:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    cones: tuple             # ((art, größe), ...) in der Zeilenreihenfolge von A; art 'l', 'q' oder 's'
    x0: np.ndarray           # streng zulässiger Start

    def sizes(self) -> dict:
        return {"variables": len(self.c), "rows": self.A.shape[0],
                "orthant": sum(m for t, m in self.cones if t == "l"),
                "soc": [m for t, m in self.cones if t == "q"],
                "psd": [m for t, m in self.cones if t == "s"]}


@dataclass(frozen=True, eq=False)
class BarrierResult:
    x: np.ndarray
    value: float             # c^T x
    gap_bound: float         # nu / t am Ende: das Optimum liegt höchstens so weit unter `value`
    newton_steps: int
    rounds: int
    history: list = field(default_factory=list)   # je äußere Runde (t, Newton-Schritte bisher, Lückenschranke nu/t, Zielwert)
    converged: bool = True                        # falsch, wenn die Rechengenauigkeit vor der geforderten Lücke endete (gap_bound ist trotzdem bewiesen)
    far_from_path: bool = False                   # wahr, wenn irgendeine Zentrierung nach max_iter Schritten mit einem halben Dekrement-Quadrat über FAR_FROM_PATH endete: der Punkt stand dann weit vom
                                                  # zentralen Pfad, die Lückenschranke nu / t gilt für ihn nicht (Beispiel: Wert 190,4 statt 1e-6 bei gemeldeter Lücke 1,7e-10). Falsch heißt nicht „zentriert“.


# ----------------------------------------------------------------------------- Kegel: Größen und Zugehörigkeit

def cone_dim(kind: str, m: int) -> int:
    """Zahl der Komponenten von s im Kegel: m, m oder k(k+1)/2 für einen PSD-Kegel der Matrixgröße k."""
    if kind not in ("l", "q", "s"):
        raise ValueError(f"unbekannter Kegel {kind!r}")
    return m * (m + 1) // 2 if kind == "s" else m


def barrier_parameter(cones) -> int:
    """nu: Orthant m, Lorentz-Kegel 2, PSD-Kegel k (die Lücke zum Optimum ist am zentralen Punkt höchstens nu / t)."""
    return sum({"l": m, "q": 2, "s": m}[t] for t, m in cones)


def svec(S: np.ndarray) -> np.ndarray:
    """Obere Dreiecksmatrix zeilenweise als Vektor, Außerdiagonale mit sqrt 2: <S, T> = svec(S) . svec(T)."""
    iu = np.triu_indices(S.shape[0])
    v = S[iu].astype(float).copy()
    v[iu[0] != iu[1]] *= SQ2
    return v


def smat(v: np.ndarray, k: int) -> np.ndarray:
    """Umkehrung von svec."""
    iu = np.triu_indices(k)
    w = np.array(v, dtype=float)
    w[iu[0] != iu[1]] /= SQ2
    S = np.zeros((k, k))
    S[iu] = w
    return S + S.T - np.diag(np.diag(S))


def in_interior(kind: str, s: np.ndarray, m: int) -> bool:
    """Streng im Inneren des Kegels (dort ist die Barriere definiert)."""
    if kind == "l":
        return bool(np.all(s > 0))
    if kind == "q":
        return bool(s[0] > 0 and s[0] ** 2 - s[1:] @ s[1:] > 0)
    try:
        np.linalg.cholesky(smat(s, m))
        return True
    except np.linalg.LinAlgError:
        return False


def cone_margin(kind: str, s: np.ndarray, m: int) -> float:
    """Wie weit s im Kegel liegt (>= 0 heißt im Kegel): kleinste Komponente, s0 - ||rest||, kleinster Eigenwert. Unabhängig von der Barriere."""
    if kind == "l":
        return float(np.min(s))
    if kind == "q":
        return float(s[0] - np.linalg.norm(s[1:]))
    return float(np.linalg.eigvalsh(smat(s, m)).min())


# ----------------------------------------------------------------------------- Barrieren

def groups(cones):
    """Fasst aufeinanderfolgende gleiche Kegel zusammen: (art, größe, Anzahl, Offset in s). Gleiche Kegel werden gebündelt gerechnet (np.linalg arbeitet auf Stapeln)."""
    out, off, i = [], 0, 0
    while i < len(cones):
        kind, m = cones[i]
        j = i
        while j < len(cones) and cones[j] == (kind, m):
            j += 1
        out.append((kind, m, j - i, off))
        off += (j - i) * cone_dim(kind, m)
        i = j
    return out


def _smat_stack(blocks: np.ndarray, k: int) -> np.ndarray:
    """svec-Vektoren (nb, k(k+1)/2) -> Matrizen (nb, k, k)."""
    iu = np.triu_indices(k)
    w = blocks.copy()
    w[:, iu[0] != iu[1]] /= SQ2
    S = np.zeros((len(blocks), k, k))
    S[:, iu[0], iu[1]] = w
    return S + S.transpose(0, 2, 1) - S * np.eye(k)


def barrier_value(cones, s: np.ndarray) -> float:
    """phi(s) = Summe der Barrieren; unendlich außerhalb des Inneren."""
    total = 0.0
    for kind, m, nb, off in groups(cones):
        d = cone_dim(kind, m)
        blk = s[off:off + nb * d]
        if kind == "l":
            if not np.all(blk > 0):
                return math.inf
            total -= float(np.log(blk).sum())
        elif kind == "q":
            B = blk.reshape(nb, d)
            q = B[:, 0] ** 2 - (B[:, 1:] ** 2).sum(axis=1)
            if not (np.all(B[:, 0] > 0) and np.all(q > 0)):
                return math.inf
            total -= float(np.log(q).sum())
        else:
            try:
                L = np.linalg.cholesky(_smat_stack(blk.reshape(nb, d), m))
            except np.linalg.LinAlgError:
                return math.inf
            total -= 2.0 * float(np.log(np.diagonal(L, axis1=1, axis2=2)).sum())
    return total


_BASIS_CACHE: dict = {}


def _basis_stack(k: int) -> np.ndarray:
    """Die d svec-Basismatrizen (d, k, k), d = k(k+1)/2."""
    if k not in _BASIS_CACHE:
        d = k * (k + 1) // 2
        _BASIS_CACHE[k] = _smat_stack(np.eye(d), k)
    return _BASIS_CACHE[k]


def group_grad_hess(kind: str, s: np.ndarray, m: int, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Gradient (nb, d) und Hesse-Matrizen (nb, d, d) der Barriere für nb gleiche Kegel (s: (nb, d), im Inneren)."""
    d = cone_dim(kind, m)
    if kind == "l":
        return -1.0 / s, np.stack([np.diag(1.0 / v ** 2) for v in s])
    if kind == "q":
        q = s[:, 0] ** 2 - (s[:, 1:] ** 2).sum(axis=1)
        J = np.ones(d)
        J[1:] = -1.0
        Js = s * J
        H = -2.0 * np.diag(J)[None] / q[:, None, None] + 4.0 * Js[:, :, None] * Js[:, None, :] / (q ** 2)[:, None, None]
        return -2.0 * Js / q[:, None], H
    k = m
    Si = np.linalg.inv(_smat_stack(s, k))
    iu = np.triu_indices(k)
    f = np.where(iu[0] == iu[1], 1.0, SQ2)
    g = -Si[:, iu[0], iu[1]] * f
    R = Si[:, None] @ _basis_stack(k)[None] @ Si[:, None]                          # Si E_x Si, gestapelt (nb, d, k, k)
    H = R[:, :, iu[0], iu[1]] * f                                                  # H[b, x, y] = svec(Si E_x Si)_y
    return g, H


def block_grad_hess(kind: str, s: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Gradient und Hesse-Matrix der Barriere eines einzelnen Kegels im Punkt s (im Inneren)."""
    g, H = group_grad_hess(kind, s[None, :], m, 1)
    return g[0], H[0]


def newton_system(problem: ConeProblem, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient und Hesse-Matrix von f_t(x) = t c^T x + phi(b - A x), gruppenweise über die Kegel (A^T H A ohne die große Block-Matrix H)."""
    s = problem.b - problem.A @ x
    grad = t * problem.c.copy()
    hess = np.zeros((len(x), len(x)))
    for kind, m, nb, off in groups(problem.cones):
        d = cone_dim(kind, m)
        Ab = problem.A[off:off + nb * d]
        sb = s[off:off + nb * d]
        if kind == "l":                                  # Hesse-Matrix des Orthanten ist diagonal: nie als (Zeilen x Zeilen)-Matrix aufbauen (LP mit Zehntausenden Zeilen)
            inv = 1.0 / sb
            grad += Ab.T @ inv
            hess += Ab.T @ (Ab * (inv ** 2)[:, None])
        else:
            g, H = group_grad_hess(kind, sb.reshape(nb, d), m, nb)
            A3 = Ab.reshape(nb, d, -1)
            grad -= np.einsum("bdi,bd->i", A3, g)
            hess += np.einsum("bdi,bdj->ij", A3, H @ A3)
    return grad, hess


def newton_step(grad: np.ndarray, hess: np.ndarray) -> tuple[np.ndarray, float]:
    """Newton-Richtung und das Quadrat des Newton-Dekrements. Wirft np.linalg.LinAlgError, wenn die Hesse-Matrix numerisch singulär ist (sehr großes t)."""
    if not (np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))):
        raise np.linalg.LinAlgError("nicht endlicher Gradient oder Hesse-Matrix")
    dx = -np.linalg.solve(hess + 1e-14 * np.eye(len(grad)), grad)
    return dx, float(-grad @ dx)


def backtrack(problem: ConeProblem, x: np.ndarray, dx: np.ndarray, t: float, dec2: float) -> tuple[np.ndarray, float]:
    """Armijo-Rücksetzen; hält den Punkt im Inneren (Barriere ist außerhalb unendlich). Gibt (neuer Punkt, Abnahme von f_t) zurück; Abnahme 0, wenn kein Schritt gelingt."""
    f0 = t * problem.c @ x + barrier_value(problem.cones, problem.b - problem.A @ x)
    step = 1.0
    while step > 1e-14:
        xn = x + step * dx
        fn = t * problem.c @ xn + barrier_value(problem.cones, problem.b - problem.A @ xn)
        if math.isfinite(fn) and fn <= f0 - 0.25 * step * dec2:
            return xn, f0 - fn
        step *= 0.5
    return x, 0.0


def _center(problem: ConeProblem, x: np.ndarray, t: float, tol: float, max_iter: int) -> tuple[np.ndarray, int, bool, bool]:
    """Wie `center`, mit einem vierten Wert: ob die Zentrierung nach max_iter Schritten weit vom Pfad endete (halbes Dekrement-Quadrat über FAR_FROM_PATH)."""
    steps = 0
    last = 0.0
    for _ in range(max_iter):
        grad, hess = newton_system(problem, x, t)
        try:
            dx, dec2 = newton_step(grad, hess)
        except np.linalg.LinAlgError:
            return x, steps, False, False
        if dec2 / 2.0 <= tol:
            break
        last = dec2 / 2.0
        x, gain = backtrack(problem, x, dx, t, dec2)
        steps += 1
        if gain <= 1e-13 * (1.0 + abs(t * float(problem.c @ x))):
            break
    else:
        return x, steps, True, last > FAR_FROM_PATH
    return x, steps, True, False


def center(problem: ConeProblem, x: np.ndarray, t: float, tol: float = 1e-6, max_iter: int = 100) -> tuple[np.ndarray, int, bool]:
    """Zentrieren für festes t: Newton-Schritte bis das halbe Dekrement-Quadrat unter `tol` liegt. Gibt (x, Zahl der Schritte, ok) zurück.

    **ok = True heißt nur: die Rechnung ist nicht zusammengebrochen. Es heißt nicht, dass x zentriert ist.** Auch diese Enden gelten als ok:
      * max_iter Schritte erschöpft, ohne tol zu erreichen (x kann weit vom Pfad stehen: siehe BarrierResult.far_from_path);
      * ein negatives Dekrement-Quadrat (die Hesse-Matrix ist numerisch nicht positiv definit; tritt bei sehr großem t auf) zählt als „unter tol“;
      * der Rundungsstopp: bei sehr großem t verhindert Rundung, die Toleranz zu erreichen; dann bricht das Zentrieren ab, sobald ein Schritt f_t kaum noch senkt (relativ 1e-13).
    Die Schranke nu / t gilt streng nur für einen zentrierten Punkt; bei t über etwa 1e10 ist das Dekrement durch Rundung nicht mehr verlässlich zu bestimmen.
    ok ist falsch, wenn die Hesse-Matrix numerisch singulär wurde: das Ergebnis dieser Runde gilt dann nicht als zentriert."""
    x, steps, ok, _ = _center(problem, x, t, tol, max_iter)
    return x, steps, ok


def initial_t(problem: ConeProblem) -> float:
    """Startwert für t: minimiert die Norm des Gradienten von f_t im Start x0 (so liegt x0 schon nahe am zentralen Punkt dieses t)."""
    grad0, _ = newton_system(problem, problem.x0, 0.0)          # = -A^T grad phi(s0)
    cc = float(problem.c @ problem.c)
    return 1.0 if cc == 0.0 else max(-float(problem.c @ grad0) / cc, 1e-6)


def solve_barrier(problem: ConeProblem, tol: float = 1e-8, mu: float = 8.0, t0: float | None = None, max_newton: int = 800) -> BarrierResult:
    """Barrierenmethode: zentrieren, t mit mu multiplizieren, bis nu / t <= tol. Der Start `problem.x0` muss streng zulässig sein.

    Ohne t0 wird `initial_t` benutzt. `tol` wird nach unten auf MIN_TOL begrenzt. Gezentrierte Punkte sind bis auf ein Dekrement-Quadrat von 2e-6 genau; das ändert die Lückenschranke nu / t nicht merklich.
    Das Ergebnis meldet `far_from_path`, wenn eine Zentrierung weit vom Pfad endete; dann gilt gap_bound nicht (siehe `center` und BarrierResult). Ein zu großer Startwert von cᵀx0 gegenüber
    dem Optimum lässt die gedämpfte Phase hunderte Schritte brauchen: t0 = nu / max(1, |cᵀx0|) ist dann der bessere Start."""
    s0 = problem.b - problem.A @ problem.x0
    if not math.isfinite(barrier_value(problem.cones, s0)):
        raise ValueError("Start nicht streng zulässig")
    tol = max(tol, MIN_TOL)
    nu = barrier_parameter(problem.cones)
    x, t, total, history = problem.x0.copy(), (initial_t(problem) if t0 is None else t0), 0, []
    t_good, x_good, far = None, x, False
    while True:
        x_new, steps, ok, far_round = _center(problem, x, t, 1e-6, 100)
        total += steps
        far = far or far_round
        if not ok:                                       # Rechengenauigkeit erschöpft: die zuletzt zentrierte Runde ist das Ergebnis, ihre Lücke ist bewiesen
            if t_good is None:
                raise np.linalg.LinAlgError("schon die erste Runde konnte nicht zentriert werden")
            return BarrierResult(x_good, float(problem.c @ x_good), nu / t_good, total, len(history), history, False, far)
        x, t_good, x_good = x_new, t, x_new
        history.append((t, total, nu / t, float(problem.c @ x)))
        if nu / t <= tol or total > max_newton:
            return BarrierResult(x, float(problem.c @ x), nu / t, total, len(history), history, nu / t <= tol, far)
        t *= mu


# ----------------------------------------------------------------------------- Prüfer (unabhängig von der Barriere)

def check_solution(problem: ConeProblem, x: np.ndarray, tol: float = 1e-7) -> list[str]:
    """Rückumformung: nennt jeden Kegel, in dem s = b - A x nicht liegt (Rundungstoleranz tol). Leere Liste = zulässig."""
    s = problem.b - problem.A @ x
    problems, off = [], 0
    names = {"l": "Orthant", "q": "Lorentz-Kegel", "s": "PSD-Kegel"}
    for idx, (kind, m) in enumerate(problem.cones):
        d = cone_dim(kind, m)
        margin = cone_margin(kind, s[off:off + d], m)
        if margin < -tol:
            problems.append(f"{names[kind]} {idx + 1} verletzt um {-margin:.2e}")
        off += d
    return problems
