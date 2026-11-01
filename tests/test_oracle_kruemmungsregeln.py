"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Prüfer und Kegelprogramm gegen einen anderen Rechenweg.

- cvxpy: Urteil `is_convex` / `is_concave` / `is_affine` auf festen Zufallsbäumen (beide Vorzeichenfälle) muss vollständig übereinstimmen; der Optimalwert des Kegelprogramms gegen Clarabel auf demselben Kasten
- zweite Differenzen auf einem feinen Gitter für alle bewiesenen Ausdrücke einer Variablen (unabhängig von Sehnen und Prüfer)
- jede bestätigte Umformung des Katalogs: cvxpy beweist die neue Fassung und die alte nicht

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import krm_canon as CN
import krm_catalog as CAT
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X
import krm_rewrite as RW

cp = pytest.importorskip("cvxpy")


def to_cvxpy(n, cvs):
    if n.op == "var":
        return cvs[n.name]
    if n.op == "const":
        return cp.Constant(n.k)
    a = [to_cvxpy(c, cvs) for c in n.args]
    if n.op == "neg":
        return -a[0]
    if n.op == "scale":
        return n.k * a[0]
    if n.op == "add":
        out = a[0]
        for t in a[1:]:
            out = out + t
        return out
    if n.op == "mul":
        return cp.multiply(a[0], a[1])
    simple = {"square": cp.square, "abs": cp.abs, "exp": cp.exp, "log": cp.log, "sqrt": cp.sqrt, "inv_pos": cp.inv_pos}
    if n.op in simple:
        return simple[n.op](a[0])
    if n.op == "maximum":
        return cp.maximum(*a)
    if n.op == "minimum":
        return cp.minimum(*a)
    if n.op == "norm2":
        return cp.norm(cp.hstack(a), 2)
    if n.op == "log_sum_exp":
        return cp.log_sum_exp(cp.hstack(a))
    if n.op == "quad_over_lin":
        return cp.quad_over_lin(a[0], a[1])
    raise ValueError(n.op)


def cvxpy_curvature(tree, nonneg):
    cvs = {k: cp.Variable(nonneg=nonneg) for k in X.VARIABLES}
    e = to_cvxpy(tree, cvs)
    if e.is_affine():
        return D.AFF
    if e.is_convex():
        return D.CVX
    if e.is_concave():
        return D.CCV
    return D.UNK


@pytest.mark.parametrize("nonneg", [True, False])
def test_curvature_agrees_with_cvxpy_on_fixed_random_trees(nonneg):
    rng = np.random.default_rng(20261101 + int(nonneg))
    ops = ev._ALL_OPS + (ev._POSITIVE_OPS if nonneg else [])
    seen = {D.AFF: 0, D.CVX: 0, D.CCV: 0, D.UNK: 0}
    for _ in range(700):
        t = ev.random_tree(rng, 3, ops)
        if not t.has_var():
            continue
        mine = D.check(t, D.POS if nonneg else D.ANY).curvature
        theirs = cvxpy_curvature(t, nonneg)
        assert mine == theirs, (X.to_text(t), mine, theirs)
        seen[mine] += 1
    assert all(v > 15 for v in seen.values()), seen             # alle vier Urteile kommen vor: der Vergleich ist nicht leer


def test_kernel_optimum_equals_clarabel_on_random_convex_trees():
    rng = np.random.default_rng(20261101)
    lo, hi = 0.1, 3.0
    solved = 0
    for _ in range(400):
        t = ev.random_tree(rng, 3, ev._CONE_OPS)
        if not t.has_var() or t.op in ("var", "const"):
            continue
        rep = D.check(t, D.POS)
        if not rep.proven_convex:
            continue
        res = CN.solve(t, rep, lo, hi)
        if res.status != CN.SOLVED:
            continue
        cvs = {k: cp.Variable(nonneg=True) for k in X.VARIABLES}
        prob = cp.Problem(cp.Minimize(to_cvxpy(t, cvs)), [v >= lo for v in cvs.values()] + [v <= hi for v in cvs.values()])
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        assert abs(res.value - prob.value) <= 1e-6 * max(1.0, abs(prob.value)), (X.to_text(t), res.value, prob.value)
        assert res.value - prob.value <= res.gap_bound + 1e-7, X.to_text(t)           # cvxpys eigene Genauigkeit liegt bei etwa 1e-8
        solved += 1
        if solved >= 60:
            break
    assert solved >= 60


def test_second_differences_agree_with_every_proof_for_one_variable():
    """Unabhängig von Sehnen und Prüfer: ein bewiesen konvexer Ausdruck hat auf einem feinen Gitter nirgends eine negative zweite Differenz (und ein konkaver keine positive)."""
    rng = np.random.default_rng(7)
    xs = np.linspace(0.1, 3.0, 2001)
    checked = 0
    for _ in range(1500):
        t = ev.random_tree(rng, 3, ev._ALL_OPS + ev._POSITIVE_OPS)
        if t.variables() != ("x",):
            continue
        rep = D.check(t, D.POS)
        if rep.curvature == D.UNK:
            continue
        f = X.evaluate(t, {"x": xs, "y": np.zeros_like(xs)})
        if not np.all(np.isfinite(f)):
            continue
        d2 = f[:-2] - 2 * f[1:-1] + f[2:]
        tol = 1e-9 * (1.0 + np.abs(f[1:-1]).max())
        if rep.proven_convex:
            assert d2.min() >= -tol, X.to_text(t)
        if rep.proven_concave:
            assert d2.max() <= tol, X.to_text(t)
        checked += 1
    assert checked > 60


def test_cvxpy_proves_every_rescue_and_not_the_original():
    for e in CAT.CATALOG:
        tree = X.parse(e.text)
        if D.check(tree, e.sign).curvature != D.UNK:
            continue
        nonneg = e.sign == D.POS
        rw = RW.find_rewrites(tree, e.sign, e.lo, e.hi)
        if not rw:
            continue
        assert cvxpy_curvature(tree, nonneg) == D.UNK, e.name
        assert cvxpy_curvature(rw[0].tree, nonneg) in (D.CVX, D.CCV, D.AFF), e.name
