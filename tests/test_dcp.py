"""Der Prüfer (krm_dcp): jede Regel an einem Mini-Baum von Hand, Vorzeichenregeln, Gründe, Eigenschaften."""

import pytest

import krm_dcp as D
import krm_expr as X

AFF, CVX, CCV, UNK = D.AFF, D.CVX, D.CCV, D.UNK


def curv(text, sign=D.POS):
    return D.check(X.parse(text), sign).curvature


@pytest.mark.parametrize("text,sign,expected", [
    # Summen, Skalieren, Negation
    ("2*x - y/3", D.ANY, AFF), ("square(x) + abs(y)", D.ANY, CVX), ("sqrt(x) + log(y)", D.POS, CCV), ("square(x) + sqrt(y)", D.POS, UNK),
    ("-square(x)", D.ANY, CCV), ("-sqrt(x)", D.POS, CVX), ("-3*square(x)", D.ANY, CCV), ("0.5*square(x)", D.ANY, CVX),
    # Produkt: nur Konstante mal Term ist eine Regel
    ("x*y", D.POS, UNK), ("x*x", D.ANY, UNK), ("3*x", D.ANY, AFF), ("exp(x)*exp(y)", D.ANY, UNK),
    # affines Argument: jedes Atom darf
    ("square(x - y)", D.ANY, CVX), ("sqrt(x + y)", D.POS, CCV), ("log(2*x + 1)", D.POS, CCV), ("inv_pos(x + 1)", D.POS, CVX), ("exp(x - y)", D.ANY, CVX),
    # square / abs: wachsend nur für Argument >= 0
    ("square(abs(x))", D.ANY, CVX), ("square(maximum(x, y))", D.ANY, UNK), ("square(maximum(x, y))", D.POS, CVX), ("square(minimum(x, y))", D.POS, UNK),
    ("abs(-abs(x))", D.ANY, CVX), ("abs(square(x) - 1)", D.ANY, UNK), ("square(norm2(x, y))", D.ANY, CVX),
    # exp wachsend, log und sqrt konkav wachsend, inv_pos konvex fallend
    ("exp(square(x))", D.ANY, CVX), ("exp(minimum(x, y))", D.POS, UNK), ("sqrt(minimum(x, y))", D.POS, CCV), ("sqrt(square(x))", D.ANY, UNK), ("sqrt(square(x) + 1)", D.ANY, UNK),
    ("log(minimum(x, y))", D.POS, CCV), ("log(exp(x) + exp(y))", D.ANY, UNK), ("inv_pos(minimum(x, y))", D.POS, CVX), ("inv_pos(square(x))", D.POS, UNK), ("inv_pos(sqrt(x))", D.POS, CVX),
    # maximum, minimum, Normen, log_sum_exp
    ("maximum(square(x), abs(y))", D.ANY, CVX), ("maximum(square(x), sqrt(y))", D.POS, UNK), ("minimum(sqrt(x), sqrt(y))", D.POS, CCV), ("minimum(square(x), y)", D.POS, UNK),
    ("norm2(x, y)", D.ANY, CVX), ("norm2(abs(x), y)", D.ANY, UNK), ("norm2(abs(x), y)", D.POS, CVX), ("norm2(abs(x), -y)", D.POS, UNK),
    ("log_sum_exp(square(x), y)", D.ANY, CVX), ("log_sum_exp(square(x), abs(y))", D.ANY, CVX), ("log_sum_exp(sqrt(x), y)", D.POS, UNK),
    # quad_over_lin: erstes Argument nach Vorzeichen, zweites fallend
    ("quad_over_lin(x, y)", D.POS, CVX), ("quad_over_lin(square(x), y)", D.POS, CVX), ("quad_over_lin(maximum(x, y), y)", D.ANY, UNK), ("quad_over_lin(maximum(x, y), y)", D.POS, CVX), ("quad_over_lin(x, sqrt(y))", D.POS, CVX), ("quad_over_lin(x, square(y))", D.POS, UNK),
])
def test_rules_by_hand(text, sign, expected):
    assert curv(text, sign) == expected


def test_signs_by_hand():
    def sign(text, vs=D.ANY):
        return D.check(X.parse(text), vs).sign

    assert sign("x", D.POS) == D.POS and sign("x", D.ANY) == D.ANY and sign("-x", D.POS) == D.NEG and sign("-3*x", D.POS) == D.NEG
    assert sign("square(x)") == D.POS and sign("abs(x)") == D.POS and sign("exp(x)") == D.POS and sign("sqrt(x)", D.POS) == D.POS and sign("inv_pos(x)", D.POS) == D.POS
    assert sign("x + y", D.POS) == D.POS and sign("x - y", D.POS) == D.ANY and sign("-x - y", D.POS) == D.NEG
    assert sign("maximum(x, -y)", D.POS) == D.POS                         # ein nichtnegatives Argument genügt
    assert sign("minimum(x, -y)", D.POS) == D.NEG                         # ein nichtpositives genügt
    assert sign("minimum(x, y)", D.POS) == D.POS                          # nur wenn alle
    assert sign("log_sum_exp(x, y)", D.POS) == D.POS and sign("log_sum_exp(x, -y)", D.POS) == D.ANY    # cvxpy: nur wenn alle Argumente nichtnegativ sind
    assert sign("log(x)", D.POS) == D.ANY
    assert sign("x*y", D.POS) == D.POS and sign("x*(-y)", D.POS) == D.NEG
    assert sign("2 + x", D.POS) == D.POS


def test_constant_subtrees_are_affine():
    """Auch von Hand gebaute konstante Teilbäume (der Parser faltet sie) gelten als affin und tragen ihr Vorzeichen."""
    c = X.atom("sqrt", X.const(4.0))
    t = X.add(X.var("x"), X.atom("square", X.add(c, X.const(-3.0))))
    rep = D.check(t, D.ANY)
    assert rep.curvature == AFF
    assert rep.by_id()[id(c)] == D.Info(AFF, D.POS)


def test_reasons_name_the_rule_that_fails():
    rep = D.check(X.parse("sqrt(square(x) + 1)"), D.ANY)
    node, reason = D.first_failure(rep)
    assert node.op == "sqrt"
    assert reason == "Argument von sqrt: sqrt ist konkav und wachsend, verlangt wird deshalb ein konkaves Argument, es ist aber konvex"
    rep = D.check(X.parse("square(maximum(x, y))"), D.ANY)
    node, reason = D.first_failure(rep)
    assert node.op == "square" and "hängt vom Vorzeichen ab" in reason and "Argument von square" in reason
    rep = D.check(X.parse("x*y"), D.POS)
    assert "Produkt zweier Terme mit Variablen" in D.first_failure(rep)[1]
    rep = D.check(X.parse("square(x) + sqrt(y)"), D.POS)
    assert "konvexen und einem konkaven" in D.first_failure(rep)[1]
    rep = D.check(X.parse("quad_over_lin(x, square(y))"), D.POS)
    assert D.first_failure(rep)[1].startswith("Argument 2 von quad_over_lin")
    rep = D.check(X.parse("exp(sqrt(square(x) + 1))"), D.ANY)
    assert D.first_failure(rep)[0].op == "sqrt"                           # die Ursache liegt unten, nicht im unbekannten Elternknoten
    assert D.check(X.parse("square(x)"), D.ANY).nodes and D.first_failure(D.check(X.parse("square(x)"), D.ANY)) is None


def test_annotation_lists_root_first_with_depths():
    rows = D.annotate(D.check(X.parse("sqrt(square(x) + 1)"), D.ANY))
    assert [(d, label) for d, label, *_ in rows] == [(0, "sqrt"), (1, "Summe"), (2, "square"), (3, "x"), (2, "1")]
    assert rows[0][2] == UNK and rows[1][2] == CVX and rows[2][2] == CVX and rows[3][2] == AFF
    assert rows[0][4] and rows[1][4] is None


def test_the_catalog_cases_stay_unknown():
    """Unvollständigkeit als Test: wer den Prüfer „verbessert“, soll es merken (siehe krm_catalog)."""
    for t in ("sqrt(square(x) + 1)", "log(exp(x) + exp(y))", "x*x", "exp(x)*exp(y)", "log(1 + exp(-x))", "sqrt(square(x))"):
        assert curv(t, D.ANY) == UNK, t
    for t in ("inv_pos(square(x))", "square(x)*inv_pos(y)", "x*log(x)", "inv_pos(x*y)"):
        assert curv(t, D.POS) == UNK, t
    assert curv("sqrt(x*y)", D.POS) == UNK                              # konkav, aber ohne geometrisches Mittel nicht zu beweisen


def test_proof_helpers_and_invalid_sign():
    assert D.check(X.parse("square(x)"), D.ANY).proven_convex and not D.check(X.parse("square(x)"), D.ANY).proven_concave
    assert D.check(X.parse("2*x"), D.ANY).proven_convex and D.check(X.parse("2*x"), D.ANY).proven_concave
    assert D.check(X.parse("log(x)"), D.POS).proven_concave
    with pytest.raises(ValueError):
        D.check(X.parse("x"), "egal")


def test_checker_is_iterative_and_visits_each_node_once():
    t = X.var("x")
    for _ in range(3000):
        t = X.scale(0.5, X.add(X.atom("exp", t), X.var("y")))
    rep = D.check(t, D.ANY)                                             # kein Rekursionsfehler, jeder Knoten genau einmal
    assert rep.curvature == CVX and len(rep.nodes) == t.size() == len(rep.infos)
    shared = X.atom("square", X.var("x"))
    rep = D.check(X.add(shared, shared, shared), D.ANY)
    assert rep.curvature == CVX and len(rep.nodes) == 3                 # add, square, x: das geteilte Teil einmal


def test_negation_and_scaling_mirror_the_curvature():
    for text in ("square(x)", "sqrt(x)", "x + y", "maximum(x, y)", "log(x)", "inv_pos(x)", "x*y"):
        c = curv(text)
        mirrored = {AFF: AFF, CVX: CCV, CCV: CVX, UNK: UNK}[c]
        assert curv(f"-({text})") == mirrored and curv(f"-2*({text})") == mirrored and curv(f"3*({text})") == c


def test_sum_of_two_convex_is_convex_and_chain_of_affine_stays_affine():
    atoms = ["square(x)", "abs(y)", "exp(x)", "maximum(x, y)", "norm2(x, y)", "inv_pos(x)", "log_sum_exp(x, y)"]
    for a in atoms:
        for b in atoms:
            assert curv(f"{a} + {b}", D.POS) == CVX, (a, b)
    assert curv("3*(2*x - y) + 1", D.ANY) == AFF


def test_a_report_survives_copying_with_pickle():
    """Streamlit kopiert zwischengespeicherte Ergebnisse: die Auswertung darf nicht an Speicheradressen hängen (Fehler beim ersten Bau gefunden)."""
    import pickle

    rep = D.check(X.parse("sqrt(square(x) + 1)"), D.ANY)
    copy = pickle.loads(pickle.dumps(rep))
    assert copy.curvature == rep.curvature == UNK and copy.sign == rep.sign
    assert D.annotate(copy) == D.annotate(rep) and D.first_failure(copy)[1] == D.first_failure(rep)[1] and D.first_failure(copy)[0].op == "sqrt"
