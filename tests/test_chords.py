"""Die Sehnenprobe und das dreiwertige Urteil (krm_chords)."""

import dataclasses

import numpy as np
import pytest

import krm_chords as CH
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X


def run(text, lo=-3.0, hi=3.0, count=4000, seed=1):
    return CH.chord_test(X.parse(text), lo, hi, count, seed)


def test_hand_chord_of_the_square_of_a_maximum():
    """Von Hand: f = max(x, y)², a = (1, −5), b = (−5, 1), λ = ½: f(a) = f(b) = 1, die Mitte (−2, −2) liegt bei 4 > 1."""
    n = X.parse("square(maximum(x, y))")
    f = lambda x, y: float(X.evaluate(n, {"x": np.array([x]), "y": np.array([y])})[0])
    assert (f(1, -5), f(-5, 1), f(-2, -2)) == (1.0, 1.0, 4.0)
    res = CH.chord_test(n, -3.0, 3.0, 4000, 1)
    assert res.convex_refuted and res.concave_refuted


def test_a_stored_chord_is_self_consistent():
    n = X.parse("square(x) - square(y)")
    res = CH.chord_test(n, -3.0, 3.0, 4000, 1)
    c = res.convex_example
    pts = lambda p: {"x": np.array([p[0]]), "y": np.array([p[1]])}
    mid = (c.lam * c.a[0] + (1 - c.lam) * c.b[0], c.lam * c.a[1] + (1 - c.lam) * c.b[1])
    assert float(X.evaluate(n, pts(c.a))[0]) == pytest.approx(c.f_a) and float(X.evaluate(n, pts(c.b))[0]) == pytest.approx(c.f_b)
    assert float(X.evaluate(n, pts(mid))[0]) == pytest.approx(c.f_mid)
    assert c.average == pytest.approx(c.lam * c.f_a + (1 - c.lam) * c.f_b) and c.violation == pytest.approx(c.f_mid - c.average) and c.violation > 0
    k = res.concave_example
    assert k.violation == pytest.approx(k.average - k.f_mid) and k.violation > 0


def test_convex_functions_are_never_refuted_and_concave_ones_not_the_other_way():
    for text in ("square(x) + abs(y)", "norm2(x, y)", "maximum(x, y, 0)", "exp(x - y)", "log_sum_exp(x, y)", "2*x - 3*y + 1"):
        res = run(text)
        assert not res.convex_refuted and res.valid == 4000, text
    assert run("log_sum_exp(x, y)").concave_refuted
    res = run("log(x + y)", 0.1, 3.0)
    assert not res.concave_refuted and res.convex_refuted
    aff = run("2*x - y")
    assert not aff.convex_refuted and not aff.concave_refuted


def test_nonconvex_and_non_concave_functions_are_refuted_both_ways():
    for text in ("square(x) - square(y)", "exp(-square(x))", "inv_pos(1 + exp(-x))"):
        res = run(text)
        assert res.convex_refuted and res.concave_refuted, text


def test_the_same_seed_gives_the_same_chords_and_another_seed_other_ones():
    a, b, c = run("square(x) - square(y)", seed=5), run("square(x) - square(y)", seed=5), run("square(x) - square(y)", seed=6)
    assert a == b and a.convex_example != c.convex_example


def test_counts_above_one_chunk_are_all_used():
    res = run("square(x) + y", count=45000)
    assert res.count == 45000 and res.valid == 45000


def test_undefined_chords_are_skipped_and_counted():
    res = run("log(x)", count=4000)                                        # nur für x > 0 definiert: ein Viertel der Sehnen hat zwei Endpunkte mit x > 0 (die Mitte liegt dann auch dort)
    assert 800 < res.valid < 1200 and res.conclusive
    text = "log(-1 - square(x)) + sqrt(square(y))"                          # nirgends definiert, und der Prüfer schweigt (sqrt über einer Quadratsumme)
    nowhere = run(text, count=500)
    assert nowhere.valid == 0 and not nowhere.conclusive
    rep = D.check(X.parse(text), D.ANY)
    assert CH.judge(rep, nowhere).status == CH.OPEN


def test_judgement_covers_all_five_states():
    def rep(text, sign=D.ANY):
        return D.check(X.parse(text), sign)

    none = CH.ChordResult(100, 100, False, False, None, None)
    cvx_only = CH.ChordResult(100, 100, True, False, None, None)
    both = CH.ChordResult(100, 100, True, True, None, None)
    assert CH.judge(rep("square(x)"), none).status == CH.PROVEN_CONVEX
    assert CH.judge(rep("log(x)", D.POS), none).status == CH.PROVEN_CONCAVE
    assert CH.judge(rep("sqrt(square(x) + 1)"), none).status == CH.OPEN
    j = CH.judge(rep("square(x) - square(y)"), both)
    assert j.status == CH.REFUTED and j.neither
    j = CH.judge(rep("sqrt(x*y)", D.POS), cvx_only)
    assert j.status == CH.REFUTED and not j.neither
    assert CH.judge(rep("square(x)"), cvx_only).status == CH.CONTRADICTION               # Prüfer beweist, Sehne widerlegt: nie verschlucken
    assert CH.judge(rep("log(x)", D.POS), CH.ChordResult(100, 100, False, True, None, None)).status == CH.CONTRADICTION


def test_the_checker_never_contradicts_the_chords_on_fixed_random_trees():
    rng = np.random.default_rng(20261101)
    contradictions = proven = 0
    for i in range(400):
        t = ev.random_tree(rng, 3, ev._ALL_OPS + ev._POSITIVE_OPS)
        if not t.has_var() or t.op in ("var", "const"):
            continue
        rep = D.check(t, D.POS)
        ch = CH.chord_test(t, 0.1, 3.0, 2000, i)
        if not ch.conclusive:
            continue
        j = CH.judge(rep, ch)
        proven += rep.curvature != D.UNK
        contradictions += j.status == CH.CONTRADICTION
    assert contradictions == 0 and proven > 80


def test_scale_invariance_of_the_verdict():
    """f und 1000·f sind gleich konvex: die relative Schwelle darf das Urteil nicht kippen."""
    for text in ("square(x) + abs(y)", "1000*square(x) + 1000*abs(y)", "0.001*square(x)"):
        assert not run(text).convex_refuted, text
    assert run("1000*(square(x) - square(y))").convex_refuted and run("0.001*(square(x) - square(y))").convex_refuted
