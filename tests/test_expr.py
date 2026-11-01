"""Parser und Baum (krm_expr): Werte von Hand, Fehlermeldungen, Einschleusungen, Grenzen, Rundlauf des Textes."""

import math

import numpy as np
import pytest

import krm_expr as X


def val(text, x=2.0, y=3.0):
    return float(X.evaluate(X.parse(text), {"x": np.array([x]), "y": np.array([y])})[0])


@pytest.mark.parametrize("text,expected", [
    ("x + y", 5.0),
    ("2*x - y/3", 3.0),
    ("-(x+1)*2", -6.0),
    ("x^2 - y^2", -5.0),
    ("x**2", 4.0),
    ("−x² + y", -1.0),                                   # typografisches Minus und Hochzahl
    ("1/2*y", 1.5),
    ("2*3*x", 12.0),
    ("x*y", 6.0),
    ("1e-3*x", 0.002),
    ("sqrt(square(x) + 1)", math.sqrt(5.0)),
    ("norm2(x, y, 1)", math.sqrt(14.0)),
    ("maximum(x, y, 1)", 3.0),
    ("minimum(x, y, 1)", 1.0),
    ("log_sum_exp(x, y)", math.log(math.exp(2) + math.exp(3))),
    ("quad_over_lin(x, y)", 4.0 / 3.0),
    ("inv_pos(x) + abs(x - y)", 1.5),
    ("exp(x)*exp(y)", math.exp(5.0)),
    ("log(x) + sqrt(y)", math.log(2) + math.sqrt(3)),
    ("+x", 2.0),
])
def test_values_by_hand(text, expected):
    assert val(text) == pytest.approx(expected, rel=1e-12)


def test_constants_fold_and_scaling_is_a_node_of_its_own():
    n = X.parse("2*3*x")
    assert n.op == "scale" and n.k == 6.0 and n.args[0].op == "var"
    n = X.parse("sqrt(4) + x")
    assert n.op == "add" and n.args[0].op == "const" and n.args[0].k == 2.0
    assert X.parse("x*y").op == "mul"
    assert X.parse("x/4").op == "scale" and X.parse("x/4").k == 0.25


def test_variables_size_depth():
    n = X.parse("sqrt(square(x) + 1)")
    assert n.variables() == ("x",) and n.size() == 5 and n.depth() == 4
    assert X.parse("y + x").variables() == ("x", "y")


@pytest.mark.parametrize("text,part", [
    ("", "leer"), ("   ", "leer"), ("1 + 2", "keine Variable"), ("z + 1", "Unbekannter Name"), ("sqrt(", "endet zu früh"), ("x**3", "nur ^2"), ("x^y", "nur ^2"),
    ("x/y", "Division durch einen Ausdruck mit Variablen"), ("x/0", "Division durch null"), ("x/(1-1)", "Division durch null"),
    ("exp(x, y)", "braucht 1 Argumente"), ("maximum(x)", "braucht 2 bis 4"), ("norm2(x, y, x, y, x)", "braucht 2 bis 4"), ("x y", "Unerwartet"),
    ("(x", "Erwartet"), ("x)", "Unerwartet"), ("x.real", "Unerlaubtes Zeichen"), ("__import__('os')", "Unerlaubtes Zeichen"), ("__import__", "Unbekannter Name"),
    ("x; y", "Unerlaubtes Zeichen"), ("lambda", "Unbekannter Name"), ("abs(x)(y)", "Unerwartet"), ("$", "Unerlaubtes Zeichen"), ("x == y", "Unerlaubtes Zeichen"),
])
def test_parse_errors_say_what_is_allowed(text, part):
    with pytest.raises(X.ParseError) as e:
        X.parse(text)
    assert part in str(e.value)


def test_limits_on_length_depth_and_nodes():
    with pytest.raises(X.ParseError, match="zu lang"):
        X.parse("x+" * 101 + "x")
    with pytest.raises(X.ParseError, match="Zu tief"):
        X.parse("(" * 31 + "x" + ")" * 31)
    X.parse("(" * 20 + "x" + ")" * 20)                            # unter der Grenze
    with pytest.raises(X.ParseError, match="zu groß"):
        X.parse("+".join(["x"] * 100))
    with pytest.raises(X.ParseError):
        X.parse(None)


def test_nothing_but_the_atom_list_is_reachable():
    """Der Parser ruft nie `eval`: jeder Python-Name außerhalb der Liste ist ein Fehler."""
    for name in ("open", "print", "os", "sys", "__builtins__", "globals", "eval", "exec", "True", "None", "len", "pi"):
        with pytest.raises(X.ParseError, match="Unbekannter Name"):
            X.parse(f"{name}(x)" if name not in ("True", "None", "pi") else f"{name} + x")


@pytest.mark.parametrize("text", ["sqrt(square(x) + 1)", "2*x - y/3", "-(x+1)*2 + 3*abs(x - y)/2", "x*y - x^2", "log(1 + exp(-x))", "maximum(x, y, 0.5) - minimum(x, 2)",
                                  "quad_over_lin(square(x), y + 1)", "norm2(x, -y, 3)", "x - (y - x)", "(x + y) - (x + y)*2", "-(-x)", "2*(x + 1)"])
def test_text_round_trip_keeps_the_function(text):
    n = X.parse(text)
    again = X.parse(X.to_text(n))
    assert X.to_text(again) == X.to_text(n)
    pts = {"x": np.linspace(0.3, 2.7, 9), "y": np.linspace(0.5, 2.5, 9)}
    a, b = X.evaluate(n, pts), X.evaluate(again, pts)
    assert np.allclose(a, b, equal_nan=True)


def test_atoms_outside_their_domain_give_nan_not_errors():
    pts = {"x": np.array([-1.0, 0.0, 1.0]), "y": np.array([1.0, 1.0, 1.0])}
    assert np.isnan(X.evaluate(X.parse("log(x)"), pts)).tolist() == [True, True, False]
    assert np.isnan(X.evaluate(X.parse("sqrt(x)"), pts)).tolist() == [True, False, False]
    assert np.isnan(X.evaluate(X.parse("inv_pos(x)"), pts)).tolist() == [True, True, False]
    assert np.isnan(X.evaluate(X.parse("quad_over_lin(y, x)"), pts)).tolist() == [True, True, False]


def test_very_deep_hand_built_trees_need_no_recursion():
    t = X.var("x")
    for _ in range(4000):
        t = X.scale(0.999, t)
    assert t.size() == 4001 and t.depth() == 4001
    assert float(X.evaluate(t, {"x": np.array([1.0])})[0]) == pytest.approx(0.999 ** 4000)
    assert t.variables() == ("x",)


def test_shared_subtrees_are_evaluated_correctly():
    u = X.atom("square", X.var("x"))
    t = X.add(u, u)
    assert float(X.evaluate(t, {"x": np.array([3.0])})[0]) == 18.0
