"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses

import pytest

import krm_canon as CN
import krm_chords as CH
import krm_constants as C
import krm_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1"
    assert ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich" and ev.sci(0.0) == "0,0"
    assert ev.num(2.0) == "2,000" and ev.num(1e-6) == "1,00·10⁻⁶" and ev.num(190.38639) == "190,386" and ev.num(-0.5) == "−0,500" and ev.num(2e6) == "2,00·10⁶" and ev.num(0.0) == "0,000"
    assert ev.pct(1, 4) == "25,0 %" and ev.pct(0, 0) == "–"


def analysis(text, sign=C.SIGN_FREE, chords=4000, seed=35):
    return ev.analyse(ev.Settings(text, sign, chords, seed))


def test_analysis_verdict_proven_convex():
    kind, text = ev.verdict_analysis(analysis("norm2(x, y) + abs(x - 2*y)", C.SIGN_NONNEG))
    assert kind == "success" and "**Bewiesen konvex.**" in text and "10 Knoten" in text and "die Wurzel ist konvex" in text and "4000 gültige Sehnen von 4000 finden keine Verletzung" in text


def test_analysis_verdict_proven_concave():
    kind, text = ev.verdict_analysis(analysis("log(x + y)", C.SIGN_NONNEG))
    assert kind == "info" and "**Bewiesen konkav.**" in text and "−f wäre bewiesen konvex" in text and "Konkavität" in text


def test_analysis_verdict_refuted_names_the_chord_and_the_violation():
    a = analysis("square(x) - square(y)")
    kind, text = ev.verdict_analysis(a)
    c = a.chords.convex_example
    assert kind == "warning" and "**Widerlegt: nicht konvex.**" in text and ev.chord_text(c) in text and f"um {ev.num(c.violation)} über der Sehne" in text
    assert "Gegenbeispiel, also ein Beweis der Nichtkonvexität" in text and "Auch die Konkavität ist widerlegt" in text
    kind, text = ev.verdict_analysis(analysis("sqrt(x*y)", C.SIGN_NONNEG))
    assert "**Widerlegt: nicht konvex.**" in text and "Auch die Konkavität" not in text


def test_analysis_verdict_open_names_the_rewrite():
    kind, text = ev.verdict_analysis(analysis("sqrt(square(x) + 1)"))
    assert kind == "warning" and "**Offen.**" in text and "„unbekannt“ heißt nicht „nicht konvex“" in text and "4000 gültige Sehnen von 4000" in text
    assert "`norm2(x, 1)` ist konvex (Abweichung der beiden Funktionen höchstens 0,0)" in text
    kind, text = ev.verdict_analysis(analysis("x*log(x)", C.SIGN_NONNEG))
    assert "**Offen.**" in text and "Umformung" not in text


def test_analysis_verdict_open_without_enough_chords():
    kind, text = ev.verdict_analysis(analysis("log(-1 - square(x)) + sqrt(square(y))", chords=200))
    assert kind == "info" and "nur 0 von 200 Sehnen liegen im Definitionsbereich (mindestens 20 nötig)" in text


def test_analysis_verdict_contradiction_is_never_swallowed():
    a = analysis("square(x)")
    forged = dataclasses.replace(a, chords=CH.ChordResult(10, 10, True, False, a.chords.convex_example or CH.Chord((0, 0), (1, 1), 0.5, 0.0, 2.0, 5.0, 1.0, 4.0), None))
    forged = dataclasses.replace(forged, judgement=CH.judge(a.report, forged.chords))
    kind, text = ev.verdict_analysis(forged)
    assert kind == "error" and "Widerspruch" in text and "Fehler im Prüfer" in text


def test_tree_verdict():
    kind, text = ev.verdict_tree(analysis("norm2(x, y)"))
    assert kind == "success" and "Alle 3 Knoten erfüllen die Regeln" in text and "**konvex**" in text and "Vorzeichen ≤ 0" not in text
    kind, text = ev.verdict_tree(analysis("sqrt(square(x) + 1)"))
    assert kind == "warning" and "zuerst am Knoten **sqrt** verletzt" in text and "konkaves Argument, es ist aber konvex" in text and "Nach der Umformung `norm2(x, 1)` ist die Wurzel konvex" in text
    kind, text = ev.verdict_tree(analysis("x*y", C.SIGN_NONNEG))
    assert "Knoten **mul**" in text and "Produkt zweier Terme" in text and "Nach der Umformung" not in text


def _rows(**kw):
    base = {"curvature": "unbekannt", "judgement": CH.Judgement(CH.OPEN), "rewrite": None}
    return {**base, **kw}


def test_catalog_verdict_counts():
    rw = object()
    rows = ([_rows(curvature="konvex", judgement=CH.Judgement(CH.PROVEN_CONVEX))] * 2 + [_rows(judgement=CH.Judgement(CH.REFUTED))] * 3
            + [_rows(rewrite=rw)] * 4 + [_rows()] * 2)
    kind, text = ev.verdict_catalog(rows)
    assert kind == "info" and "Von 11 Einträgen beweist der Prüfer 2." in text and "Bei 9 schweigt er: 3 davon sind widerlegt" in text and "bei 6 finden die Sehnen nichts" in text
    assert "rettet bei **4** eine Umformung" in text and "bei den übrigen 2 fehlt ein Atom" in text and "Widersprüche zwischen Prüfer und Sehnen: 0." in text


def test_corpus_verdict():
    row = lambda sign, pc: dict(sign=sign, trees=100, proven_convex=pc, proven_concave=10, unknown=100 - pc - 10, unknown_refuted=40, unknown_open_convex=100 - pc - 10 - 40,
                                contradictions=0)
    kind, text = ev.verdict_corpus([row("x, y ≥ 0", 30), row("x, y frei", 40)])
    assert kind == "info" and "x, y ≥ 0: 100 Bäume, bewiesen konvex oder affin 30, bewiesen konkav 10, „unbekannt“ 60 (davon bei den Sehnen als nicht konvex widerlegt 40, nicht widerlegt 20)" in text
    assert "bewiesen sind 60,0 % der nicht widerlegten konvexen" in text and "x, y frei: 100 Bäume" in text and "bewiesen sind 80,0 %" in text
    assert "Widersprüche zwischen Prüfer und Sehnen: **0**" in text and "nicht für Modelle aus der Praxis" in text


def test_power_verdict():
    p = {"trees": 100, "steps": (50, 200, 1000), "corpus": {50: 90, 200: 96, 1000: 100}, "bump": {50: 10, 200: 30, 1000: 40}, "bump_runs": 40, "reference": 20000}
    kind, text = ev.verdict_power(p)
    assert kind == "info" and "Von 100 Zufallsbäumen" in text and "bei 20000 Sehnen widerlegt sind" in text and "50 Sehnen 90,0 %, 200 Sehnen 96,0 %, 1000 Sehnen 100,0 %" in text
    assert "für |x − 2| < 0,016" in text and "50: 10 von 40, 200: 30 von 40, 1000: 40 von 40" in text and "**Eine verletzte Sehne beweist, eine unverletzte beweist nichts.**" in text


def test_timing_verdict():
    rows = [{"nodes": 41, "seconds": 0.0002, "per_node": 4.9e-6}, {"nodes": 601, "seconds": 0.003, "per_node": 5.0e-6}]
    kind, text = ev.verdict_timing(rows)
    assert kind == "info" and "für 41 Knoten 2,0·10⁻⁴ s, für 601 Knoten 3,0·10⁻³ s" in text and "je Knoten 4,9 und 5,0 µs" in text and "linear" in text


def test_canon_verdicts():
    a = analysis("norm2(x, y) + abs(x - 2*y)", C.SIGN_NONNEG)
    kind, text = ev.verdict_canon(a)
    r = a.canon
    assert kind == "success" and f"{r.variables + r.aux} Variablen (2 eigene, 2 Hilfsvariablen)" in text and f"{r.rows} Zeilen" in text and f"{len(r.soc)} Lorentz-Kegeln" in text
    assert "Der Kern löst es" in text and f"Wert {ev.num(r.value)}" in text and "Die Rückumformung besteht." in text and "nie unter dem Optimum" in text
    kind, text = ev.verdict_canon(analysis("sqrt(square(x) + 1)"))
    assert kind == "info" and "Ohne bewiesene Konvexität gibt es keine Übersetzung" in text
    kind, text = ev.verdict_canon(analysis("log_sum_exp(x, y)"))
    assert kind == "info" and "log_sum_exp braucht den Exponentialkegel (Stück 7" in text
    kind, text = ev.verdict_canon(analysis("inv_pos(x - y)", C.SIGN_NONNEG))
    assert kind == "warning" and "Bewiesen konvex, aber nicht lösbar ohne Weiteres" in text and "Phase I, Stück 6" in text
    forged = dataclasses.replace(analysis("square(x)"), canon=CN.CanonResult(CN.NOT_CENTERED, "nicht zentriert (Dekrement 5)"))
    assert ev.verdict_canon(forged) == ("error", "nicht zentriert (Dekrement 5)")


def test_canon_rewrite_verdict():
    a = analysis("sqrt(square(x) + 1)")
    kind, text = ev.verdict_canon_rewrite(a)
    assert kind == "success" and "Nach der Umformung `norm2(x, 1)`" in text and "Wert 1,000" in text and "nie unter dem Optimum" in text
    a2 = analysis("log(exp(x) + exp(y))")
    kind, text = ev.verdict_canon_rewrite(a2)
    assert kind == "info" and "Nach der Umformung `log_sum_exp(x, y)`" in text and "Exponentialkegel" in text


def test_canon_corpus_verdict():
    c = dict(convex=120, solved=109, consistent=109, no_start=11, not_centered=0, aux=[1, 1, 2], rows=[6, 6, 8], soc=[1, 1, 2], newton=[30, 32, 40])
    kind, text = ev.verdict_canon_corpus(c)
    assert kind == "info" and "120 bewiesen konvexe Zufallsbäume" in text and "**109 gelöst** (109 davon stimmen mit Direktauswertung und Gitter überein)" in text
    assert "11 (9,2 %) ohne streng zulässigen Start im Kastenmittelpunkt, 0 nicht zentriert" in text and "Im Median 1 Hilfsvariablen, 6 Zeilen, 1 Lorentz-Kegel, 32 Newton-Schritte" in text


def test_outlier_verdict():
    o = ev.outlier_experiment()
    kind, text = ev.verdict_outlier(o)
    b = o["bare"]                                                        # die Zahlen des Kerns hängen von der CPU ab: Text gegen die gerechneten Werte prüfen
    assert kind == "warning" and f"Ohne Absicherung meldet der Kern den Wert {ev.num(b.value)} mit Lückenschranke {ev.sci(b.gap_bound)}" in text and "das Gitter findet aber 1,00·10⁻⁶" in text
    assert "der Kern meldet dabei `far_from_path = True`" in text and "nicht zentriert, die Schranke gilt dort nicht" in text and "Mit der Absicherung" in text and "ist der Wert 1,00·10⁻⁶" in text
