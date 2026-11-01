"""Abbildungen: Achsen gesperrt, Diagramme bauen an den Rändern."""

import pytest

import krm_constants as C
import krm_evaluation as ev
import krm_expr as X
import krm_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def _fig(text, sign=C.SIGN_FREE, chords=500):
    a = ev.analyse(ev.Settings(text, sign, chords, 1))
    ch = a.chords.convex_example if a.chords.convex_refuted else V.typical_chord(a.tree, *a.box, 1)
    return a, V.build_function(a.tree, a.box, ch, a.chords.convex_refuted, a.canon.point if a.canon and a.canon.status == "gelöst" else None)


def test_function_figure_for_one_variable_has_curve_chord_and_optimum():
    a, fig = _fig("square(x - 1)")
    names = [t.name for t in fig.data]
    assert _locked(fig) and "f" in names and "Sehne" in names and "Optimum (Kegelprogramm)" in names and "gerade Verbindung der Endwerte" in names
    opt = next(t for t in fig.data if t.name == "Optimum (Kegelprogramm)")
    assert opt.x[0] == pytest.approx(1.0, abs=1e-3) and opt.y[0] == pytest.approx(0.0, abs=1e-6)


def test_function_figure_for_two_variables_uses_contours():
    a, fig = _fig("square(x) - square(y)")
    assert _locked(fig) and any(t.type == "contour" for t in fig.data)
    chord = next(t for t in fig.data if t.name == "Sehne")
    assert chord.line.color == C.COLOR_BAD                                           # verletzte Sehne rot
    _, fig2 = _fig("norm2(x, y)")
    assert next(t for t in fig2.data if t.name == "Sehne").line.color == C.COLOR_LINE


def test_chord_profile_matches_the_end_values():
    a, fig = _fig("square(x) - square(y)")
    c = a.chords.convex_example
    along = next(t for t in fig.data if t.name == "f entlang der Sehne")
    line = next(t for t in fig.data if t.name == "gerade Verbindung der Endwerte")
    assert along.y[0] == pytest.approx(c.f_a) and along.y[-1] == pytest.approx(c.f_b) and line.y[0] == pytest.approx(c.f_a) and line.y[-1] == pytest.approx(c.f_b)
    assert max(p - q for p, q in zip(along.y, line.y)) >= c.violation * 0.99


def test_typical_chord_exists_where_defined_and_not_where_undefined():
    assert V.typical_chord(X.parse("square(x)"), -3.0, 3.0, 1) is not None
    assert V.typical_chord(X.parse("log(-1 - square(x)) + sqrt(square(y))"), -3.0, 3.0, 1) is None


def test_function_figure_without_a_chord():
    fig = V.build_function(X.parse("log(-1 - square(x)) + sqrt(square(y))"), (-3.0, 3.0), None, False)
    assert _locked(fig)


def test_catalog_figure_has_one_row_per_entry_and_three_columns():
    rows = ev.catalog_rows()
    fig = V.build_catalog(rows)
    assert _locked(fig) and len(fig.data[0].y) == 21 and list(fig.data[0].x) == ["Prüfer", "Sehnen", "nach Umformung"]
    z = fig.data[0].z
    assert z[0][0] == 0 and z[0][2] == 2 and z[13][0] == 2                           # √(x²+1) unbekannt, gerettet; e^(x²) bewiesen
    assert fig.data[0].text[0][2] == "norm2(x, 1)"


def test_power_corpus_timing_and_canon_figures():
    power = {"trees": 100, "steps": (50, 200), "corpus": {50: 90, 200: 96}, "bump": {50: 10, 200: 30}, "bump_runs": 40}
    fig = V.build_power(power)
    assert _locked(fig) and list(fig.data[0].y) == [90.0, 96.0] and list(fig.data[1].y) == [25.0, 75.0]
    rows = [dict(sign="a", trees=10, proven_convex=4, proven_concave=1, unknown_open_convex=2, unknown_open_concave=1, unknown_neither=2)]
    fig = V.build_corpus(rows)
    assert _locked(fig) and sum(t.y[0] for t in fig.data) == pytest.approx(100.0)
    assert _locked(V.build_timing([{"nodes": 41, "seconds": 1e-4}, {"nodes": 201, "seconds": 5e-4}]))
    fig = V.build_canon_corpus(dict(solved=109, no_start=11, not_centered=0))
    assert _locked(fig) and list(fig.data[0].y) == [109, 11, 0]
