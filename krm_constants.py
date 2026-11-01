"""Konstanten: Reglergrenzen, Kästen, feste Stichproben und Presets."""

SIGN_FREE = "x, y frei"
SIGN_NONNEG = "x, y ≥ 0"
SIGNS = (SIGN_FREE, SIGN_NONNEG)
DEFAULT_SIGN = SIGN_FREE
BOX = {SIGN_FREE: (-3.0, 3.0), SIGN_NONNEG: (0.1, 3.0)}          # Kasten für Sehnenprobe, Umformungsprüfung und Kegelprogramm

CHORD_STEPS = (50, 200, 1000, 4000, 20000, 100000)
DEFAULT_CHORDS = 4000
SEED_MAX, DEFAULT_SEED = 999999, 35
DEFAULT_EXPR = "sqrt(square(x) + 1)"

CATALOG_CHORDS = 20000                 # Sehnen je Katalogeintrag
CORPUS_SEED = 20261101                 # feste Zufallsbäume für Experiment 3 und 4: nie ein einzelner Baum als Befund
CORPUS_TREES = 1500                    # Zufallsbäume je Vorzeichenfall (Experiment 3)
CORPUS_CHORDS = 4000
POWER_TREES = 150                      # DCP-unbekannte, bei der Referenz widerlegte Bäume für die Trefferquote
POWER_REFERENCE = 20000
POWER_STEPS = (50, 200, 1000, 4000, 20000)
BUMP_RUNS = 40
BUMP_EXPR = "square(x) + 0.8*exp(-2000*square(x - 2))"          # lint-ok: dezimalpunkt (Ausdruckssprache mit Dezimalpunkt)
CANON_TREES = 120                      # DCP-konvexe Zufallsbäume mit Kegelatomen (Experiment 4)
CANON_SEED = 20261101
OUTLIER_EXPR = "square(quad_over_lin(square(x), minimum(x, y)))"
TIMING_CHAINS = (10, 50, 150)

COLOR_CONVEX = "#2F6B65"
COLOR_UNKNOWN = "#b08a2e"
COLOR_BAD = "#b34a4a"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Standardfall: √(x²+1)": {"expr": DEFAULT_EXPR, "sign": SIGN_FREE, "chords": DEFAULT_CHORDS, "seed": DEFAULT_SEED},
    "Bewiesen: Norm plus Betrag": {"expr": "norm2(x, y) + abs(x - 2*y)", "sign": SIGN_NONNEG, "chords": DEFAULT_CHORDS, "seed": DEFAULT_SEED},
    "Das Vorzeichen entscheidet": {"expr": "square(maximum(x, y))", "sign": SIGN_FREE, "chords": DEFAULT_CHORDS, "seed": DEFAULT_SEED},
    "Nichtkonvex: x² − y²": {"expr": "square(x) - square(y)", "sign": SIGN_FREE, "chords": DEFAULT_CHORDS, "seed": DEFAULT_SEED},
    "Kleine Beule: Sehnen übersehen": {"expr": BUMP_EXPR, "sign": SIGN_FREE, "chords": 50, "seed": 3},
    "Exponentialatom": {"expr": "log_sum_exp(x, y)", "sign": SIGN_FREE, "chords": DEFAULT_CHORDS, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall: √(x²+1)": "Konvex (es ist die Norm von (x, 1)), aber der Prüfer schweigt: sqrt ist konkav, x² konvex. Die Umformung zu norm2(x, 1) beweist es.",
    "Bewiesen: Norm plus Betrag": "Ein Ausdruck, den der Prüfer beweist; daraus wird ein Kegelprogramm (Orthant und Lorentz), das der Kern löst.",
    "Das Vorzeichen entscheidet": "Dasselbe max(x, y)² ist für freie x, y nicht konvex (die Sehne zeigt es) und für x, y ≥ 0 bewiesen konvex: Schalter links umlegen.",
    "Nichtkonvex: x² − y²": "Weder konvex noch konkav: der Prüfer schweigt zu Recht, und eine Sehne widerlegt die Konvexität mit Zahlen.",
    "Kleine Beule: Sehnen übersehen": "Nichtkonvex nur in einem Streifen der Breite etwa 0,03 um x = 2. Mit 50 Sehnen findet die Probe nichts: Nicht widerlegt ist nicht bewiesen.",
    "Exponentialatom": "Bewiesen konvex, aber die Übersetzung braucht den Exponentialkegel, den der Kern nicht kennt (Stück 7 der Reihe).",
}
