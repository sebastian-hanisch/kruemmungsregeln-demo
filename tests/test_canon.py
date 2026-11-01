"""Vom Beweis zum Kegelprogramm (krm_canon): Epigraph-Formeln als Eigenschaft, Minima von Hand, Zustände, Absicherung des Kerns."""

import math

import numpy as np
import pytest

import kegel_kern as K
import krm_canon as CN
import krm_dcp as D
import krm_expr as X

BOX_FREE, BOX_POS = (-3.0, 3.0), (0.1, 3.0)


def solve(text, sign=D.ANY, box=BOX_FREE, **kw):
    n = X.parse(text)
    return CN.solve(n, D.check(n, sign), *box, **kw)


def in_lorentz(s, slack=1e-12):
    s = np.asarray(s, dtype=float)
    return s[0] >= np.linalg.norm(s[1:]) - slack


# ----------------------------------------------------------------------------- die Epigraph-Formeln (unabhängig vom Baumcode)

def test_epigraph_formulas_describe_exactly_the_epigraphs():
    rng = np.random.default_rng(3)
    for _ in range(4000):
        t, u, v = rng.uniform(-4, 4, 3)
        assert in_lorentz([t + 1, t - 1, 2 * u], 0.0) == (t >= u * u) or abs(t - u * u) < 1e-9                    # t ≥ u²
        if u > 0.05:
            assert in_lorentz([t + u, t - u, 2.0], 0.0) == (t >= 1.0 / u) or abs(t - 1.0 / u) < 1e-9              # t ≥ 1/u
        if v > 0.05:
            assert in_lorentz([t + v, t - v, 2 * u], 0.0) == (t >= u * u / v) or abs(t - u * u / v) < 1e-9        # t ≥ u²/v
        if u >= 0.0:
            assert in_lorentz([u + 1, u - 1, 2 * t], 0.0) == (t * t <= u) or abs(t * t - u) < 1e-9               # t² ≤ u, also t ≤ √u für t ≥ 0
        w = rng.uniform(-4, 4, 2)
        assert in_lorentz([t, *w], 0.0) == (t >= math.hypot(*w)) or abs(t - math.hypot(*w)) < 1e-9


# ----------------------------------------------------------------------------- Minima von Hand

@pytest.mark.parametrize("text,sign,box,expected,point", [
    ("norm2(x, 1)", D.ANY, BOX_FREE, 1.0, {"x": 0.0}),
    ("norm2(x, 1)", D.POS, BOX_POS, math.sqrt(1.01), {"x": 0.1}),
    ("square(x - 1) + square(y + 2)", D.ANY, BOX_FREE, 0.0, {"x": 1.0, "y": -2.0}),
    ("abs(x - 1) + abs(y + 1)", D.ANY, BOX_FREE, 0.0, {"x": 1.0, "y": -1.0}),
    ("inv_pos(x) + x", D.POS, BOX_POS, 2.0, {"x": 1.0}),
    ("quad_over_lin(x, y)", D.POS, BOX_POS, 0.1 ** 2 / 3.0, {"x": 0.1, "y": 3.0}),
    ("-sqrt(x) + x/4", D.POS, BOX_POS, -math.sqrt(3.0) + 0.75, {"x": 3.0}),
    ("-minimum(x, y)", D.POS, BOX_POS, -3.0, {"x": 3.0, "y": 3.0}),
    ("maximum(x, 2*y) + abs(x)", D.ANY, BOX_FREE, 0.0, {}),                           # Minimum auf einer ganzen Fläche: kein eindeutiger Punkt
    ("norm2(x, y) + abs(x - 2*y)", D.POS, BOX_POS, math.sqrt(0.05), {"x": 0.2, "y": 0.1}),
    ("2*x + y", D.ANY, BOX_FREE, -9.0, {"x": -3.0, "y": -3.0}),
    ("square(maximum(x, y))", D.POS, BOX_POS, 0.01, {"x": 0.1, "y": 0.1}),
])
def test_the_kernel_finds_the_hand_minimum(text, sign, box, expected, point):
    r = solve(text, sign, box)
    assert r.status == CN.SOLVED and not r.problems
    assert -1e-9 <= r.value - expected <= r.gap_bound + 1e-9                         # die bewiesene Lücke deckt den Abstand zum Handwert
    assert r.gap_bound <= 1e-6
    assert r.direct_value == pytest.approx(expected, abs=r.gap_bound + 1e-8)
    for k, v in point.items():
        assert r.point[k] == pytest.approx(v, abs=1e-3)
    assert CN.is_centered(r.centrality)


def test_the_hand_instance_sizes():
    """norm2(x, 1) im Kasten: zwei Orthant-Zeilen für den Kasten, ein Lorentz-Kegel der Größe 3, eine Hilfsvariable."""
    r = solve("norm2(x, 1)")
    assert (r.variables, r.aux, r.rows, r.orthant, r.soc) == (1, 1, 5, 2, (3,))
    r = solve("abs(x)")
    assert (r.aux, r.rows, r.orthant, r.soc) == (1, 4, 4, ())
    r = solve("square(x)")
    assert (r.aux, r.rows, r.orthant, r.soc) == (1, 5, 2, (3,))
    r = solve("maximum(x, y, 0)")
    assert (r.variables, r.aux, r.orthant) == (2, 1, 4 + 3)
    r = solve("2*x + y")
    assert (r.aux, r.soc, r.rows) == (0, (), 4)                                     # affin: nur der Kasten
    program = solve("square(x)").program
    assert program[-1][1] == "Lorentz" and program[-1][2] == 3 and "t1 ≥ (x)²" in program[-1][0]


def test_a_whole_optimal_face_ends_the_kernel_early_with_a_larger_but_valid_gap():
    """max(x, 2y) + |x| ist null auf der ganzen Fläche x ≤ 0, 2y ≤ x: die Hesse-Matrix wird schlecht konditioniert, der Kern endet bei Lücke ~1e-6 statt 1e-9 und sagt es."""
    r = solve("maximum(x, 2*y) + abs(x)")
    assert r.status == CN.SOLVED and not r.converged and 1e-9 < r.gap_bound < 1e-5 and 0.0 <= r.value <= r.gap_bound
    assert solve("norm2(x, 1)").converged


def test_only_the_variables_in_use_become_variables():
    assert solve("square(y)").variables == 1 and solve("x + y").variables == 2


# ----------------------------------------------------------------------------- Zustände

def test_unproven_expressions_have_no_translation():
    r = solve("sqrt(square(x) + 1)")
    assert r.status == CN.NOT_PROVEN and r.value is None and "ohne Beweis" in r.detail


@pytest.mark.parametrize("text", ["exp(x)", "log_sum_exp(x, y)", "square(x) + exp(y)", "-log(x)"])
def test_exponential_atoms_are_judged_but_not_translated(text):
    r = solve(text, D.POS, BOX_POS)
    assert r.status == CN.EXP_CONE and "Exponentialkegel" in r.detail and "Stück 7" in r.detail


def test_no_start_is_reported_not_crashed():
    r = solve("inv_pos(x - y)", D.POS, BOX_POS)                 # im Kastenmittelpunkt x = y: Argument 0
    assert r.status == CN.NO_START and "inv_pos" in r.detail and "Phase I" in r.detail
    assert solve("-sqrt(x)", D.POS, BOX_FREE).status == CN.NO_START         # Kasten [−3; 3]: Mittelpunkt 0
    assert solve("quad_over_lin(x, y - x)", D.POS, BOX_POS).status == CN.NO_START
    assert solve("-sqrt(x)", D.POS, BOX_POS).status == CN.SOLVED
    assert solve("sqrt(x)", D.POS, BOX_POS).status == CN.NOT_PROVEN                                      # sqrt allein ist konkav


# ----------------------------------------------------------------------------- Absicherung des Kerns

OUTLIER = "square(quad_over_lin(square(x), minimum(x, y)))"


def test_the_guard_gives_the_right_value_where_the_bare_kernel_does_not():
    n = X.parse(OUTLIER)
    rep = D.check(n, D.POS)
    grid = CN.grid_minimum(n, *BOX_POS, 401)
    assert grid == pytest.approx(1e-6, rel=1e-6)
    good = CN.solve(n, rep, *BOX_POS)
    assert good.status == CN.SOLVED and good.value == pytest.approx(1e-6, abs=2e-9) and CN.is_centered(good.centrality) and not good.far_from_path
    bare = CN.solve(n, rep, *BOX_POS, guard=False)
    # Verhalten des Kerns ohne Absicherung (dokumentiert in README und Experiment 4): falscher Wert, winzige Lückenschranke, negatives Dekrement-Quadrat; der Kern meldet far_from_path.
    # Wird der Kern strenger gehärtet (siehe PLAN-KERN-HAERTUNG.md: gemessen und verworfen), wird dieser Test hinfällig und die Aussage in der README geändert.
    assert bare.far_from_path
    assert bare.status == CN.SOLVED and bare.value > 100.0 and bare.gap_bound < 1e-9 and bare.centrality < 0.0 and not CN.is_centered(bare.centrality)
    assert abs(bare.direct_value - bare.value) > 100.0                      # auch die Direktauswertung entlarvt die Zahl


def test_is_centered_rejects_negative_and_large_decrements():
    assert CN.is_centered(0.0) and CN.is_centered(1e-3) and not CN.is_centered(1.0001e-3) and not CN.is_centered(-1e-12) and not CN.is_centered(float("nan"))


def test_centrality_is_small_at_a_solved_point_and_large_at_the_start():
    n = X.parse("norm2(x, y) + abs(x - 2*y)")
    cn = CN._Canon(["x", "y"], *BOX_POS)
    prob, offset, _ = cn.problem(n)
    res = K.solve_barrier(prob, tol=1e-9)
    assert CN.centrality(prob, res.x, res.gap_bound) < 1e-3
    assert CN.centrality(prob, prob.x0, res.gap_bound) > 1.0


def test_solutions_pass_the_independent_cone_check():
    for text in ("norm2(x, y) + abs(x - 2*y)", "square(x) + inv_pos(y)", "maximum(square(x), abs(y)) - minimum(x, y)"):
        sign = D.POS
        r = solve(text, sign, BOX_POS)
        assert r.status == CN.SOLVED and r.problems == ()


# ----------------------------------------------------------------------------- Gitter

def test_grid_minimum_is_an_upper_bound_of_the_optimum():
    assert CN.grid_minimum(X.parse("square(x)"), -1.0, 1.0) == 0.0                         # das Gitter enthält 0
    assert CN.grid_minimum(X.parse("square(x - 0.0123) + y"), -1.0, 1.0, 51) >= -1.0
    for text, sign, box in (("norm2(x, y) + abs(x - 2*y)", D.POS, BOX_POS), ("inv_pos(x) + x", D.POS, BOX_POS), ("square(x - 1) + square(y + 2)", D.ANY, BOX_FREE)):
        n = X.parse(text)
        r = CN.solve(n, D.check(n, sign), *box)
        assert CN.grid_minimum(n, *box) >= r.value - r.gap_bound - 1e-9
