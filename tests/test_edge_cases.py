"""Randfälle, an denen Parser, Prüfer, Sehnen und Übersetzung erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import pytest

import krm_canon as CN
import krm_chords as CH
import krm_constants as C
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X


def a_(text, sign=C.SIGN_FREE, chords=500, seed=1):
    return ev.analyse(ev.Settings(text, sign, chords, seed))


def test_a_parse_error_becomes_a_message_not_an_exception():
    a = a_("sqrt(")
    assert not a.ok and "endet zu früh" in a.error and a.tree is None and a.report is None
    assert not a_("").ok and not a_("x**3").ok and not a_("z").ok


def test_expressions_in_one_variable_only():
    for text in ("square(y)", "abs(x - 1)", "exp(y)"):
        a = a_(text)
        assert a.ok and a.judgement.status == CH.PROVEN_CONVEX and len(a.tree.variables()) == 1


def test_affine_and_constant_offsets():
    a = a_("2*x + 3 - y")
    assert a.report.curvature == D.AFF and a.judgement.status == CH.PROVEN_CONVEX and a.canon.status == CN.SOLVED and a.canon.value == pytest.approx(-6 + 3 - 3, abs=1e-6)
    assert not a.chords.convex_refuted and not a.chords.concave_refuted and a.canon.aux == 0


def test_overflow_and_infinity_do_not_break_the_chord_test():
    a = a_("exp(exp(exp(x)))", chords=4000)                  # e^(e^20) = inf: diese Sehnen fallen weg, der Rest zählt
    assert a.ok and 0 < a.chords.valid < 4000 and a.judgement.status == CH.PROVEN_CONVEX
    big = a_("exp(100*x) + exp(100*y)")
    assert big.ok and not big.chords.convex_refuted


def test_extreme_scales_are_not_mistaken_for_violations_and_never_crash_the_kernel():
    for text in ("1e12*square(x)", "1e-12*square(x)", "1e8*x - 1e8*y + 1e-8*square(x)"):
        a = a_(text, chords=4000)                                        # kein Absturz, auch wenn der Kern an den Koeffizienten scheitert
        assert a.judgement.status == CH.PROVEN_CONVEX and not a.chords.convex_refuted, text
        assert a.canon.status in (CN.SOLVED, CN.NUMERIC, CN.NOT_CENTERED), text          # nie ein falscher Wert mit „gelöst“:
        if a.canon.status == CN.SOLVED:
            assert a.canon.value <= a.grid_min + a.canon.gap_bound + 1e-6 * (1.0 + abs(a.grid_min)), text
    bad = a_("1e8*x - 1e8*y + 1e-8*square(x)").canon
    assert bad.status == CN.NUMERIC and bad.value is None and "numerisch singulär" in bad.detail
    assert ev.verdict_canon(a_("1e8*x - 1e8*y + 1e-8*square(x)"))[0] == "warning"


def test_deep_but_legal_expressions():
    text = "(" * 25 + "x" + ")" * 25
    assert a_(text).judgement.status == CH.PROVEN_CONVEX
    chain = "x"
    for _ in range(8):
        chain = f"abs({chain})"
    assert a_(chain).report.curvature == D.CVX


def test_unicode_and_whitespace():
    assert a_("  sqrt( square( x )  +  1 )  ").ok and a_("x − y + 2·x").ok and a_("x²").ok and a_("\tx\n+ y").ok


def test_solutions_at_the_box_boundary_and_in_the_middle():
    r = ev.analyse(ev.Settings("square(x - 5) + y", C.SIGN_FREE, 100, 1)).canon          # Minimum liegt außerhalb des Kastens: am Rand x = 3, y = −3
    assert r.status == CN.SOLVED and r.point["x"] == pytest.approx(3.0, abs=1e-4) and r.point["y"] == pytest.approx(-3.0, abs=1e-4) and r.value == pytest.approx(4.0 - 3.0, abs=1e-6)
    r = ev.analyse(ev.Settings("square(x) + square(y)", C.SIGN_FREE, 100, 1)).canon
    assert r.value == pytest.approx(0.0, abs=1e-8)


def test_the_box_is_the_documented_one():
    assert C.BOX == {C.SIGN_FREE: (-3.0, 3.0), C.SIGN_NONNEG: (0.1, 3.0)}
    a = a_("norm2(x, y)", C.SIGN_NONNEG)
    assert a.box == (0.1, 3.0) and a.canon.value == pytest.approx(0.1 * 2 ** 0.5, abs=1e-8)


def test_chord_count_extremes():
    for n in (1, 50, 100000):
        res = CH.chord_test(X.parse("square(x)"), -3.0, 3.0, n, 1)
        assert res.count == n and res.valid == n and not res.convex_refuted
    assert not CH.chord_test(X.parse("square(x)"), -3.0, 3.0, 1, 1).conclusive            # eine Sehne reicht nicht für ein Urteil
    assert CH.chord_test(X.parse("square(x) - square(y)"), -3.0, 3.0, 100000, 1).convex_refuted


def test_the_largest_legal_input_runs():
    big = "+".join(["x"] * 49) + "+y"
    a = a_(big, chords=100000)
    assert a.ok and a.judgement.status == CH.PROVEN_CONVEX and a.canon.status == CN.SOLVED


def test_a_whole_analysis_survives_copying_like_the_streamlit_cache_does_it():
    import pickle

    for text in ("sqrt(square(x) + 1)", "norm2(x, y) + abs(x - 2*y)", "square(x) - square(y)", "log_sum_exp(x, y)"):
        a = a_(text)
        b = pickle.loads(pickle.dumps(a))
        assert ev.verdict_analysis(b) == ev.verdict_analysis(a) and ev.verdict_tree(b) == ev.verdict_tree(a) and ev.verdict_canon(b) == ev.verdict_canon(a)
        assert b.annotation == a.annotation and b.report.curvature == a.report.curvature
