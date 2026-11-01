"""Der Katalog: bekannte Ausdrücke, bei denen der Prüfer beweist, schweigt oder zu Recht nichts sagen kann.

Jeder Eintrag trägt Text, Vorzeichen der Variablen und Kasten. `note` sagt in einem Satz, was daran lehrt. Die Urteile selbst werden gerechnet (krm_evaluation.catalog_rows), nicht hier hineingeschrieben."""

from __future__ import annotations

from dataclasses import dataclass

import krm_dcp as D


@dataclass(frozen=True)
class Entry:
    name: str
    text: str
    sign: str               # D.POS (x, y ≥ 0) oder D.ANY (frei)
    lo: float
    hi: float
    note: str


FREE = (D.ANY, -3.0, 3.0)
POSITIVE = (D.POS, 0.1, 3.0)

CATALOG = (
    Entry("√(x²+1)", "sqrt(square(x) + 1)", *FREE, "Konvex (es ist eine Norm), aber sqrt ist konkav: konkav ∘ konvex hat keine Regel."),
    Entry("log(eˣ+eʸ)", "log(exp(x) + exp(y))", *FREE, "Konvex (log-sum-exp), aber als log einer Summe von Exponentialfunktionen geschrieben."),
    Entry("x·x", "x*x", *FREE, "Ein Produkt zweier Terme mit Variablen hat keine Regel, auch wenn beide gleich sind."),
    Entry("eˣ·eʸ", "exp(x)*exp(y)", *FREE, "Gleich exp(x+y), aber als Produkt geschrieben."),
    Entry("log(1+e⁻ˣ)", "log(1 + exp(-x))", *FREE, "Softplus, die Zielfunktion der logistischen Regression; als log-sum-exp(0, −x) bewiesen."),
    Entry("√(x²)", "sqrt(square(x))", *FREE, "Gleich |x|."),
    Entry("1/x²  (x > 0)", "inv_pos(square(x))", *POSITIVE, "Konvex auf x > 0; inv_pos braucht ein konkaves Argument, x² ist konvex."),
    Entry("x²·(1/y)  (y > 0)", "square(x)*inv_pos(y)", *POSITIVE, "Gleich quad_over_lin(x, y), dem Atom für x²/y."),
    Entry("(max(x,y))²  frei", "square(maximum(x, y))", *FREE, "Nicht konvex: das Quadrat ist nur ab null wachsend, und max(x,y) kann negativ sein."),
    Entry("(max(x,y))²  x, y ≥ 0", "square(maximum(x, y))", *POSITIVE, "Derselbe Ausdruck, aber mit bekanntem Vorzeichen bewiesen konvex."),
    Entry("x² − y²", "square(x) - square(y)", *FREE, "Weder konvex noch konkav: zu Recht keine Aussage."),
    Entry("e^(−x²)", "exp(-square(x))", *FREE, "Die Gauß-Glocke: weder konvex noch konkav."),
    Entry("Sigmoid 1/(1+e⁻ˣ)", "inv_pos(1 + exp(-x))", *FREE, "Weder konvex noch konkav (S-Kurve)."),
    Entry("e^(x²)", "exp(square(x))", *FREE, "Exponentialfunktion einer konvexen Funktion: bewiesen konvex."),
    Entry("−log x", "-log(x)", *POSITIVE, "Konkav, mit Minus konvex: bewiesen."),
    Entry("‖(x,y)‖²", "square(norm2(x, y))", *FREE, "Quadrat einer Norm: die Norm ist ≥ 0, damit ist square wachsend."),
    Entry("max(x,y) − min(x,y)", "maximum(x, y) - minimum(x, y)", *FREE, "Gleich |x−y|: konvex plus konvex."),
    Entry("x·log x", "x*log(x)", *POSITIVE, "Konvex (negative Entropie), aber ein Produkt; ohne Entropie-Atom nicht zu beweisen."),
    Entry("√(x·y)", "sqrt(x*y)", *POSITIVE, "Konkav (geometrisches Mittel); beweisbar nur mit einem eigenen Atom."),
    Entry("log(x+y)", "log(x + y)", *POSITIVE, "Konkav: bewiesen."),
    Entry("1/(x·y)", "inv_pos(x*y)", *POSITIVE, "Konvex auf x, y > 0; ein Produkt im Argument."),
)

BY_NAME = {e.name: e for e in CATALOG}
