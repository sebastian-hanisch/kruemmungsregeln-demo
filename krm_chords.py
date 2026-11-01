"""Die Sehnenprobe: eine zweite, unabhängige Sicht auf Konvexität.

Für zufällige Punkte a, b im Kasten und λ ∈ (0, 1) gilt bei einer konvexen Funktion f(λa + (1−λ)b) ≤ λ f(a) + (1−λ) f(b). Eine einzige Sehne, die das verletzt, **widerlegt**
die Konvexität (Gegenbeispiel mit Zahlen). Keine verletzte Sehne beweist nichts: endlich viele Sehnen können eine kleine Beule übersehen.
Gerechnet wird nur an Sehnen, bei denen f in beiden Endpunkten und in der Mitte definiert ist (log, sqrt, inv_pos und quad_over_lin haben einen Definitionsbereich)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import krm_expr as X

TOL = 1e-8             # Verletzung, relativ zu 1 + größtem Funktionswert; kleiner gilt als Rundung
MIN_VALID = 20         # so viele auswertbare Sehnen braucht ein Urteil
CHUNK = 20000


@dataclass(frozen=True)
class Chord:
    a: tuple               # Endpunkt (x, y)
    b: tuple
    lam: float
    f_a: float
    f_b: float
    f_mid: float           # f(λa + (1−λ)b)
    average: float         # λ f(a) + (1−λ) f(b)
    violation: float       # f_mid − average (konvex) bzw. average − f_mid (konkav); positiv = verletzt


@dataclass(frozen=True)
class ChordResult:
    count: int
    valid: int
    convex_refuted: bool
    concave_refuted: bool
    convex_example: Chord | None
    concave_example: Chord | None

    @property
    def conclusive(self) -> bool:
        return self.valid >= MIN_VALID


def chord_test(root: X.Node, lo: float, hi: float, count: int, seed: int) -> ChordResult:
    """`count` zufällige Sehnen im Kasten [lo, hi]² (nur die Variablen, die der Ausdruck braucht, wirken)."""
    rng = np.random.default_rng(seed)
    valid = 0
    best = {"cvx": (TOL, None), "ccv": (TOL, None)}
    done = 0
    while done < count:
        m = min(CHUNK, count - done)
        done += m
        A = {k: rng.uniform(lo, hi, m) for k in X.VARIABLES}
        B = {k: rng.uniform(lo, hi, m) for k in X.VARIABLES}
        lam = rng.uniform(0.0, 1.0, m)
        M = {k: lam * A[k] + (1.0 - lam) * B[k] for k in X.VARIABLES}
        fa, fb, fm = X.evaluate(root, A), X.evaluate(root, B), X.evaluate(root, M)
        ok = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(fm)
        if not ok.any():
            continue
        valid += int(ok.sum())
        avg = lam * fa + (1.0 - lam) * fb
        scale = 1.0 + np.maximum(np.abs(fa), np.abs(fb))
        with np.errstate(all="ignore"):
            d = np.where(ok, (fm - avg) / scale, 0.0)
        for key, vals in (("cvx", d), ("ccv", -d)):
            i = int(np.argmax(vals))
            if ok[i] and vals[i] > best[key][0]:
                sgn = 1.0 if key == "cvx" else -1.0
                best[key] = (float(vals[i]), Chord((float(A["x"][i]), float(A["y"][i])), (float(B["x"][i]), float(B["y"][i])), float(lam[i]),
                                                   float(fa[i]), float(fb[i]), float(fm[i]), float(avg[i]), sgn * float(fm[i] - avg[i])))
    return ChordResult(count, valid, best["cvx"][1] is not None, best["ccv"][1] is not None, best["cvx"][1], best["ccv"][1])


# ---------------------------------------------------------------------------------------------------------------- dreiwertiges Urteil

PROVEN_CONVEX, PROVEN_CONCAVE, REFUTED, OPEN, CONTRADICTION = "bewiesen konvex", "bewiesen konkav", "widerlegt", "offen", "Widerspruch"


@dataclass(frozen=True)
class Judgement:
    status: str
    neither: bool = False        # nur bei REFUTED: auch die Konkavität ist widerlegt
    concave_refuted: bool = False


def judge(report, chords: ChordResult) -> Judgement:
    """Prüfer und Sehnen zusammen: bewiesen konvex / bewiesen konkav / widerlegt / offen.

    Ein Widerspruch (der Prüfer beweist, die Sehne widerlegt) darf nie vorkommen; er wird nicht verschluckt, sondern als eigener Zustand gemeldet."""
    if report.proven_convex:
        return Judgement(CONTRADICTION if chords.convex_refuted else PROVEN_CONVEX)
    if report.proven_concave:
        return Judgement(CONTRADICTION if chords.concave_refuted else PROVEN_CONCAVE, concave_refuted=False)
    if chords.convex_refuted:
        return Judgement(REFUTED, neither=chords.concave_refuted, concave_refuted=chords.concave_refuted)
    return Judgement(OPEN, concave_refuted=chords.concave_refuted)
