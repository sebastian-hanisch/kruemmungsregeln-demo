"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import krm_chords as CH
import krm_constants as C
import krm_dcp as D
import krm_evaluation as ev
import krm_expr as X
import krm_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds_and_parse():
    for name, preset in C.PRESETS.items():
        assert preset["sign"] in C.SIGNS and preset["chords"] in C.CHORD_STEPS and 0 <= preset["seed"] <= C.SEED_MAX, name
        assert len(preset["expr"]) <= X.MAX_LENGTH and X.parse(preset["expr"]), name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall: √(x²+1)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_each_help_text_names_what_the_preset_produces():
    a = {n: ev.analyse(ev.Settings(**p)) for n, p in C.PRESETS.items()}
    std = a["Standardfall: √(x²+1)"]
    assert std.report.curvature == D.UNK and std.rewrites[0].text == "norm2(x, 1)" and "norm2(x, 1)" in C.PRESET_HELP["Standardfall: √(x²+1)"]
    proven = a["Bewiesen: Norm plus Betrag"]
    assert proven.judgement.status == CH.PROVEN_CONVEX and proven.canon.status == "gelöst"
    sign = C.PRESETS["Das Vorzeichen entscheidet"]
    assert a["Das Vorzeichen entscheidet"].judgement.status == CH.REFUTED
    assert ev.analyse(ev.Settings(sign["expr"], C.SIGN_NONNEG, sign["chords"], sign["seed"])).judgement.status == CH.PROVEN_CONVEX
    assert a["Nichtkonvex: x² − y²"].judgement.neither
    bump = C.PRESETS["Kleine Beule: Sehnen übersehen"]
    assert a["Kleine Beule: Sehnen übersehen"].judgement.status == CH.OPEN and "50 Sehnen" in C.PRESET_HELP["Kleine Beule: Sehnen übersehen"] and bump["chords"] == 50
    exp = a["Exponentialatom"]
    assert exp.judgement.status == CH.PROVEN_CONVEX and exp.canon.status == "Exponentialkegel" and "Stück 7" in C.PRESET_HELP["Exponentialatom"]


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["expr"] = "norm2(x, y)"
    at.query_params["sign"] = "x, y ≥ 0"
    at.query_params["chords"] = "123"           # nicht in der Liste erlaubter Werte: bleibt Standard
    at.query_params["seed"] = "99999999"        # über der Grenze: wird begrenzt
    at.run()
    assert not at.exception
    assert at.text_input(key="expr_input").value == "norm2(x, y)" and at.radio(key="sign_radio").value == "x, y ≥ 0"
    assert at.select_slider(key="chords_select").value == C.DEFAULT_CHORDS and at.number_input(key="seed_input").value == C.SEED_MAX


def test_an_overlong_permalink_expression_is_cut_to_the_limit():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["expr"] = "x+" * 400
    at.run()
    assert not at.exception and len(at.text_input(key="expr_input").value) <= X.MAX_LENGTH
