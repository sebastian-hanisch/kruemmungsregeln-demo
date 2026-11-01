"""Umformungen (krm_rewrite) und Katalog (krm_catalog)."""

import numpy as np
import pytest

import krm_catalog as CAT
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X
import krm_rewrite as RW


def find(text, sign=D.ANY, lo=-3.0, hi=3.0):
    return RW.find_rewrites(X.parse(text), sign, lo, hi)


@pytest.mark.parametrize("text,sign,lo,steps,new", [
    ("sqrt(square(x) + 1)", D.ANY, -3.0, ("sqrt_squares",), "norm2(x, 1)"),
    ("sqrt(square(x) + square(y))", D.ANY, -3.0, ("sqrt_squares",), "norm2(x, y)"),
    ("sqrt(4*square(x) + 9)", D.ANY, -3.0, ("sqrt_squares",), "norm2(2*x, 3)"),
    ("sqrt(square(x))", D.ANY, -3.0, ("sqrt_square",), "abs(x)"),
    ("log(exp(x) + exp(y))", D.ANY, -3.0, ("log_exps",), "log_sum_exp(x, y)"),
    ("log(1 + exp(-x))", D.ANY, -3.0, ("log_exps",), "log_sum_exp(0, -x)"),
    ("log(exp(x) + 2*exp(y))", D.ANY, -3.0, ("log_exps",), "log_sum_exp(x, y + 0.69314718056)"),
    ("x*x", D.ANY, -3.0, ("self_product",), "square(x)"),
    ("exp(x)*exp(y)", D.ANY, -3.0, ("exp_product",), "exp(x + y)"),
    ("square(x)*inv_pos(y)", D.POS, 0.1, ("quad_over_lin",), "quad_over_lin(x, y)"),
    ("inv_pos(y)*square(x)", D.POS, 0.1, ("quad_over_lin",), "quad_over_lin(x, y)"),
    ("inv_pos(square(x))", D.POS, 0.1, ("inv_square",), "square(inv_pos(x))"),
])
def test_each_rule_rescues_its_pattern(text, sign, lo, steps, new):
    r = find(text, sign, lo)[0]
    assert r.steps == steps and r.curvature == D.CVX and r.deviation < 1e-9
    assert r.text == new


def test_the_identity_is_checked_not_assumed():
    """Die Abweichung wird auf Zufallspunkten gerechnet; eine falsche Umformung würde verworfen."""
    a, b = X.parse("sqrt(square(x) + 1)"), X.parse("norm2(x, 1)")
    assert RW.deviation(a, b, -3.0, 3.0) < 1e-12
    assert RW.deviation(a, X.parse("norm2(x, 2)"), -3.0, 3.0) > 0.1
    assert RW.deviation(X.parse("log(x)"), X.parse("log(x - 10)"), -3.0, 3.0) is None        # fast nirgends gemeinsam definiert
    assert len(RW.RULES) == 7


def test_inv_square_needs_a_known_positive_sign():
    """1/u² = (1/u)² gilt für u > 0; für freie Variablen wäre die Umformung ein anderer Definitionsbereich."""
    assert find("inv_pos(square(x))", D.ANY) == []
    assert find("inv_pos(square(x))", D.POS, 0.1)[0].steps == ("inv_square",)


def test_nothing_found_for_things_that_cannot_be_rescued_with_these_rules():
    assert find("x*log(x)", D.POS, 0.1) == []                    # kein Entropie-Atom
    assert find("inv_pos(x*y)", D.POS, 0.1) == []
    assert find("sqrt(x*y)", D.POS, 0.1) == []                   # kein geometrisches Mittel
    assert find("square(x) - square(y)") == []                   # nicht konvex
    assert find("square(x) + abs(y)") == []                      # schon bewiesen: nichts zu retten


def test_rewrites_apply_inside_a_larger_expression():
    r = find("3*sqrt(square(x) + 1) + abs(y)")[0]
    assert r.text == "3*norm2(x, 1) + abs(y)" and r.curvature == D.CVX


def test_two_steps_are_found_when_one_is_not_enough():
    r = find("sqrt(square(x)) + exp(x)*exp(y)")
    assert len(r) == 1 and r[0].steps == ("sqrt_square", "exp_product") and r[0].curvature == D.CVX      # die Suche merkt sich schon gesehene Fassungen: die andere Reihenfolge ergibt dieselbe
    assert r[0].text == "abs(x) + exp(x + y)"


def test_catalog_has_the_documented_counts():
    assert len(CAT.CATALOG) == 21 and len({e.name for e in CAT.CATALOG}) == 21
    rows = ev.catalog_rows()
    proven = [r for r in rows if r["curvature"] != D.UNK]
    unknown = [r for r in rows if r["curvature"] == D.UNK]
    refuted = [r for r in unknown if r["judgement"].status == "widerlegt"]
    open_ = [r for r in unknown if r["judgement"].status == "offen"]
    rescued = [r for r in open_ if r["rewrite"] is not None]
    assert (len(proven), len(unknown), len(refuted), len(open_), len(rescued)) == (6, 15, 5, 10, 8)
    assert all(r["judgement"].status != "Widerspruch" for r in rows)
    assert {r["entry"].name for r in open_ if r["rewrite"] is None} == {"x·log x", "1/(x·y)"}
    assert {r["entry"].name for r in refuted} == {"(max(x,y))²  frei", "x² − y²", "e^(−x²)", "Sigmoid 1/(1+e⁻ˣ)", "√(x·y)"}


def test_rescued_catalog_functions_are_the_same_function():
    for r in ev.catalog_rows():
        if r["rewrite"]:
            assert r["rewrite"].deviation < 1e-9 and r["rewrite"].curvature == D.CVX, r["entry"].name


def test_the_sign_decides_for_the_square_of_a_maximum():
    rows = {r["entry"].name: r for r in ev.catalog_rows()}
    assert rows["(max(x,y))²  frei"]["curvature"] == D.UNK and rows["(max(x,y))²  frei"]["judgement"].status == "widerlegt"
    assert rows["(max(x,y))²  x, y ≥ 0"]["curvature"] == D.CVX
    assert rows["√(x·y)"]["judgement"].concave_refuted is False and rows["log(x+y)"]["curvature"] == D.CCV
