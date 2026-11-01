"""Pfade der App, die der Rauchtest nicht erreicht: jedes Preset, der Vorzeichenschalter, Eingabefehler, alle Katalogtexte, die vier Experimente im Seiteninhalt, PDF (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import krm_catalog as CAT
import krm_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    _set(at, **state)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _set(at, expr=None, sign=None, chords=None, seed=None):
    if expr is not None:
        at.sidebar.text_input(key="expr_input").set_value(expr)
    if sign is not None:
        at.sidebar.radio(key="sign_radio").set_value(sign)
    if chords is not None:
        at.sidebar.select_slider(key="chords_select").set_value(chords)
    if seed is not None:
        at.sidebar.number_input(key="seed_input").set_value(seed)
    at.run()


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Prüfer"] == "unbekannt" and values["Sehnen"] == "nicht widerlegt" and values["Urteil"] == "offen" and values["Knoten"] == "5" and values["Kegelprogramm"] == "–"
    text = _texts(at)
    for part in ("**Offen.**", "zuerst am Knoten **sqrt** verletzt", "Von 21 Einträgen beweist der Prüfer 6.", "**Eine verletzte Sehne beweist, eine unverletzte beweist nichts.**", "Widersprüche zwischen Prüfer und Sehnen: **0**",
                 "linear", "bewiesen konvexe Zufallsbäume mit Kegelatomen", "Ohne Absicherung meldet der Kern den Wert 190,386", "Nach der Umformung `norm2(x, 1)`: Kegelprogramm"):
        assert part in text, part


@pytest.mark.parametrize("name,part", [
    ("Standardfall: √(x²+1)", "**Offen.**"),
    ("Bewiesen: Norm plus Betrag", "**Bewiesen konvex.**"),
    ("Das Vorzeichen entscheidet", "**Widerlegt: nicht konvex.**"),
    ("Nichtkonvex: x² − y²", "Auch die Konkavität ist widerlegt"),
    ("Kleine Beule: Sehnen übersehen", "**Offen.**"),
    ("Exponentialatom", "braucht den Exponentialkegel"),
])
def test_every_preset_renders_and_gives_its_verdict(name, part):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert part in _texts(at)


def test_the_sign_switch_turns_the_square_of_a_maximum_from_refuted_to_proven():
    at = _app()
    at.sidebar.button(key="preset_Das Vorzeichen entscheidet").click().run()
    assert any("Widerlegt: nicht konvex" in w.value for w in at.warning)
    _set(at, sign=C.SIGN_NONNEG)
    assert not at.exception and any("**Bewiesen konvex.**" in s.value for s in at.success)


def test_more_chords_find_the_small_bump():
    at = _app()
    at.sidebar.button(key="preset_Kleine Beule: Sehnen übersehen").click().run()
    assert any("**Offen.**" in w.value for w in at.warning)
    _set(at, chords=200)
    assert any("Widerlegt: nicht konvex" in w.value for w in at.warning)


def test_a_bad_expression_shows_a_message_and_stops_without_an_exception():
    at = _app()
    for bad in ("x**3", "", "sqrt(", "__import__('os').system('x')", "<script>alert(1)</script>", "z + 1", "x/y", "1+1"):
        _set(at, expr=bad)
        assert not at.exception, bad
        assert any("Der Ausdruck lässt sich nicht lesen" in e.value for e in at.error), bad
        assert not at.metric and not at.dataframe                       # nach dem Fehler stoppt die Seite (kein halbes Ergebnis)
    _set(at, expr="x + y")
    assert not at.error and at.metric[0].label == "Prüfer" and len(at.metric) >= 5


def test_every_catalog_text_runs_through_the_page():
    at = _app()
    for e in CAT.CATALOG:
        _set(at, expr=e.text, sign=C.SIGN_NONNEG if e.sign == "≥ 0" else C.SIGN_FREE)
        assert not at.exception, (e.name, [str(x) for x in at.exception])
        assert at.metric[0].label == "Prüfer" and len(at.metric) >= 5, e.name


def test_every_chord_count_works():
    at = _app()
    for n in C.CHORD_STEPS:
        _set(at, chords=n)
        assert not at.exception, n


def test_the_longest_legal_expression_works():
    at = _app()
    _set(at, expr="+".join(["x"] * 49) + "+y")
    assert not at.exception and not at.error and at.metric[0].value == "affin"


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    pdf = at.session_state["pdf_made"][1]
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500


def test_pdf_builds_for_every_state():
    import krm_evaluation as ev
    from krm_pdf_export import build_pdf

    for expr in ("sqrt(square(x) + 1)", "norm2(x, y) + abs(x - 2*y)", "square(x) - square(y)", "log(x + y)", "log_sum_exp(x, y)", "inv_pos(x - y)", "2*x + y", "sqrt(x*y)"):
        for sign in C.SIGNS:
            a = ev.analyse(ev.Settings(expr, sign, 200, 1))
            assert build_pdf(a, ev.verdict_analysis(a)[1])[:4] == b"%PDF", (expr, sign)
