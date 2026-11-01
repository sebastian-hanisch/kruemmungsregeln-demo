"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Stichproben laufen über feste Seeds (C.CORPUS_SEED); Bänder statt Punktwerte, wo eine Bibliotheksversion oder die Rundung auf Linux die letzte Stelle bewegen kann (CI installiert ungepinnt)."""

import statistics

import pytest

import krm_canon as CN
import krm_chords as CH
import krm_constants as C
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X
import krm_rewrite as RW


@pytest.fixture(scope="module")
def corpus():
    return {s: ev.corpus_summary(s) for s in (C.SIGN_NONNEG, C.SIGN_FREE)}


@pytest.fixture(scope="module")
def power():
    return ev.chord_power()


@pytest.fixture(scope="module")
def canon():
    return ev.canon_corpus()


def test_standard_case():
    """README: sqrt(square(x) + 1) auf [−3; 3]: Prüfer „unbekannt“, 4000 gültige Sehnen ohne Verletzung, Umformung norm2(x, 1), identisch."""
    a = ev.analyse(ev.Settings())
    assert a.report.curvature == D.UNK and a.judgement.status == CH.OPEN and a.chords.valid == 4000 and not a.chords.convex_refuted
    assert a.rewrites[0].text == "norm2(x, 1)" and a.rewrites[0].deviation < 1e-12 and a.rewrites[0].curvature == D.CVX
    assert a.canon is None and a.canon_rewrite.status == CN.SOLVED and a.canon_rewrite.value == pytest.approx(1.0, abs=1e-8)


def test_catalog_counts():
    """README: 21 Einträge; der Prüfer beweist 6, schweigt bei 15 (5 widerlegt, 10 nicht widerlegt, bei 8 rettet eine Umformung); 0 Widersprüche."""
    rows = ev.catalog_rows()
    unknown = [r for r in rows if r["curvature"] == D.UNK]
    assert (len(rows), len(rows) - len(unknown), len(unknown)) == (21, 6, 15)
    assert sum(r["judgement"].status == CH.REFUTED for r in unknown) == 5 and sum(r["judgement"].status == CH.OPEN for r in unknown) == 10
    assert sum(r["rewrite"] is not None for r in unknown) == 8 and sum(r["judgement"].status == CH.CONTRADICTION for r in rows) == 0
    assert max(r["rewrite"].deviation for r in rows if r["rewrite"]) < 1e-9


def test_random_trees_have_no_contradiction_and_the_documented_shares(corpus):
    """README: je Vorzeichenfall rund 1500 Zufallsbäume; 0 Widersprüche; bewiesen sind etwa 54 % (x, y ≥ 0) und 63 % (frei) der „bewiesen oder nicht widerlegt konvexen“."""
    for sign, c in corpus.items():
        assert c["contradictions"] == 0 and c["trees"] + c["excluded"] == C.CORPUS_TREES, sign
        assert c["proven_convex"] + c["proven_concave"] + c["unknown"] == c["trees"]
        assert c["unknown_refuted"] + c["unknown_open_convex"] == c["unknown"]
    nonneg, free = corpus[C.SIGN_NONNEG], corpus[C.SIGN_FREE]
    share = lambda c: 100.0 * c["proven_convex"] / (c["proven_convex"] + c["unknown_open_convex"])
    assert 50.0 <= share(nonneg) <= 58.0 and 59.0 <= share(free) <= 67.0
    assert 330 <= nonneg["proven_convex"] <= 430 and 400 <= free["proven_convex"] <= 520
    assert 250 <= nonneg["unknown_open_convex"] <= 400 and 200 <= free["unknown_open_convex"] <= 340
    assert nonneg["excluded"] > free["excluded"]                          # log, sqrt und inv_pos brauchen einen Definitionsbereich


def test_chord_detection_rates(power):
    """README: von 150 Zufallsbäumen widerlegen 50 Sehnen etwa 91 %, 200 etwa 97 %, 4000 alle; kleine Beule: 50 Sehnen 13 von 40, 200: 31 von 40, ab 1000 alle."""
    t = power["trees"]
    assert t == 150 and power["bump_verdict"] == D.UNK
    rate = {n: 100.0 * power["corpus"][n] / t for n in power["steps"]}
    assert 86.0 <= rate[50] <= 96.0 and 93.0 <= rate[200] <= 99.0 and rate[4000] >= 98.0
    assert all(b >= a - 3.0 for a, b in zip(rate.values(), list(rate.values())[1:]))                  # mehr Sehnen widerlegen nicht weniger (bis auf Stichprobenschwankung)
    bump = power["bump"]
    assert 6 <= bump[50] <= 22 and 22 <= bump[200] <= 37 and bump[1000] == 40 and bump[4000] == 40
    assert bump[50] < bump[200] < bump[1000]


def test_the_small_bump_preset_is_missed_with_few_chords_and_found_with_more():
    """Preset „Kleine Beule“: 50 Sehnen, Seed 3 finden nichts (Urteil offen); mit 200 Sehnen findet es die Verletzung."""
    p = C.PRESETS["Kleine Beule: Sehnen übersehen"]
    a = ev.analyse(ev.Settings(**p))
    assert p["chords"] == 50 and a.judgement.status == CH.OPEN and not a.chords.convex_refuted
    a = ev.analyse(ev.Settings(p["expr"], p["sign"], 200, p["seed"]))
    assert a.judgement.status == CH.REFUTED


def test_the_bump_is_nonconvex_only_in_a_narrow_strip():
    """README: nichtkonvex nur für |x − 2| < 0,016 (zweite Ableitung von Hand: 2 + 0,8·(−4000 + 1,6·10⁷ u²)·e^(−2000u²), u = x − 2)."""
    import numpy as np

    u = np.linspace(-0.2, 0.2, 40001)
    d2 = 2.0 + 0.8 * (-4000.0 + 16_000_000.0 * u ** 2) * np.exp(-2000.0 * u ** 2)
    neg = u[d2 < 0]
    assert neg.min() == pytest.approx(-0.0158, abs=0.0005) and neg.max() == pytest.approx(0.0158, abs=0.0005)
    assert d2[u == 0.0][0] == pytest.approx(2.0 - 3200.0)


def test_the_checker_is_linear():
    """README: je Knoten rund 4 bis 6 µs, bei 41, 201, 601 Knoten (hier nur: die Zeit je Knoten wächst nicht mit der Größe)."""
    rows = ev.timing_rows(repeats=25)
    assert [r["nodes"] for r in rows] == [41, 201, 601]
    assert rows[-1]["per_node"] < 3.0 * rows[0]["per_node"] + 5e-6


def test_cone_programs_from_random_convex_trees(canon):
    """README: 120 bewiesen konvexe Zufallsbäume mit Kegelatomen; 109 gelöst, 11 ohne Start, 0 nicht zentriert; alle gelösten stimmen mit Direktauswertung und Gitter überein."""
    assert canon["convex"] == C.CANON_TREES == 120
    assert canon["solved"] + canon["no_start"] + canon["not_centered"] == 120
    assert 100 <= canon["solved"] <= 118 and 5 <= canon["no_start"] <= 20 and canon["not_centered"] == 0
    assert canon["consistent"] == canon["solved"]
    assert statistics.median(canon["aux"]) == 1 and 5 <= statistics.median(canon["rows"]) <= 8 and statistics.median(canon["soc"]) == 1 and 24 <= statistics.median(canon["newton"]) <= 42
    assert max(canon["aux"]) <= 8 and max(canon["rows"]) <= 24


def test_the_outlier_of_the_kernel():
    """README: ohne Absicherung 190,386 mit Lückenschranke 1,7·10⁻¹⁰ und halbem Dekrement-Quadrat von etwa −4,8·10¹⁴; das Optimum ist 1,0·10⁻⁶; mit Absicherung stimmt es."""
    o = ev.outlier_experiment()
    b, g = o["bare"], o["guarded"]
    assert b.value > 100.0 and b.gap_bound < 1e-9 and b.centrality < 0.0                 # rund 190,4 und etwa −5·10¹⁴ auf dem Entwicklungsrechner: die genauen Zahlen hängen von der CPU ab
    assert abs(b.direct_value - b.value) > 100.0
    assert b.far_from_path and not g.far_from_path                                      # die Rückmeldung des Kerns (siehe tests/test_kern.py)
    assert o["grid"] == pytest.approx(1e-6, rel=1e-5) and g.value == pytest.approx(1e-6, abs=2e-9) and CN.is_centered(g.centrality)


def test_atom_and_rule_counts():
    """README: elf Atome im Prüfer, davon acht mit Kegelübersetzung (Orthant, Lorentz) und drei mit Exponentialkegel; sieben Umformungsregeln."""
    assert len(X.ATOMS) == 11 and len(D.ATOMS) == 11 and len(X.CONE_ATOMS) == 8 and len(X.EXP_ATOMS) == 3 and set(X.CONE_ATOMS) | set(X.EXP_ATOMS) == set(X.ATOMS)
    assert len(RW.RULES) == 7
    assert (X.MAX_LENGTH, X.MAX_DEPTH, X.MAX_NODES) == (200, 30, 150)


def test_the_sign_example_of_the_documentation():
    """App/README: max(x, y)² ist für x, y ≥ 0 bewiesen konvex, für freie nicht (Sehne (1, −5) – (−5, 1))."""
    free = ev.analyse(ev.Settings("square(maximum(x, y))", C.SIGN_FREE))
    nonneg = ev.analyse(ev.Settings("square(maximum(x, y))", C.SIGN_NONNEG))
    assert free.judgement.status == CH.REFUTED and nonneg.judgement.status == CH.PROVEN_CONVEX
