"""Krümmungsregeln – wann ein Prüfer Konvexität beweist und wo er schweigt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Reihe „Konvexe und konische Optimierung“: Ein Prüfer nach den Regeln des „Disciplined Convex Programming“ (DCP) sagt „konvex“, „konkav“ oder „unbekannt“. Er beweist oder schweigt;
„unbekannt“ heißt nicht „nicht konvex“. Eine Sehnenprobe widerlegt Nichtkonvexität (Gegenbeispiel mit Zahlen), beweist aber nie Konvexität. Der Beweis liefert das Kegelprogramm: jedes Atom bekommt
eine Hilfsvariable und eine Kegelbedingung (Orthant, Lorentz), und der gemeinsame Kern (kegel_kern.py, wortgleich mit Stück 2, 4 und 5) löst es.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import krm_catalog as CAT
import krm_constants as C
import krm_dcp as D
import krm_expr as X
import krm_rewrite as RW

from krm_evaluation import (
    Settings,
    analyse,
    canon_corpus,
    catalog_rows,
    chord_power,
    corpus_summary,
    de,
    num,
    outlier_experiment,
    sci,
    timing_rows,
    verdict_analysis,
    verdict_canon,
    verdict_canon_corpus,
    verdict_canon_rewrite,
    verdict_catalog,
    verdict_corpus,
    verdict_outlier,
    verdict_power,
    verdict_timing,
    verdict_tree,
)
from krm_layout import dunkles_design, kopf_css
from krm_pdf_export import build_pdf
from krm_presets import init_session_state_defaults, load_permalink_settings, randomize_seed, bounds, sidebar_presets, sync_query_params
from krm_visualization import build_canon_corpus, build_catalog, build_corpus, build_function, build_power, build_timing, typical_chord

st.set_page_config(page_title="Krümmungsregeln – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _catalog():
    return catalog_rows()


@st.cache_data(show_spinner=False)
def _corpus():
    return [corpus_summary(C.SIGN_NONNEG), corpus_summary(C.SIGN_FREE)]


@st.cache_data(show_spinner=False)
def _power():
    return chord_power()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_rows()


@st.cache_data(show_spinner=False)
def _canon_corpus():
    return canon_corpus()


@st.cache_data(show_spinner=False)
def _outlier():
    return outlier_experiment()


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in krm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📐 Krümmungsregeln – wann ein Prüfer Konvexität beweist und wo er schweigt")
st.markdown(
    """
**Ein Prüfer für Konvexität kann beweisen, aber nicht widerlegen.** Er folgt einem festen Regelwerk (*Disciplined Convex Programming*, DCP): jedes Atom (`square`, `sqrt`, `maximum`, …) hat eine Krümmung und eine Monotonie,
und eine Verkettung ist erlaubt, wenn beides zusammenpasst. Das Ergebnis ist „konvex“, „konkav“ oder **„unbekannt“** – und „unbekannt“ heißt nicht „nicht konvex“: `sqrt(x² + 1)` ist konvex, und der Prüfer schweigt trotzdem.
Eine Sehne, die die Konvexität verletzt, **widerlegt** sie mit Zahlen; keine verletzte Sehne beweist nichts. Zusammen ergeben beide ein dreiwertiges Urteil: bewiesen, widerlegt oder offen.
Der Beweis ist nützlich: **Jedes bewiesene Atom lässt sich in ein Kegelprogramm übersetzen** (Stück 2), und der Kern aus Stück 5 löst es.
Stück 3 der Reihe „Konvexe und konische Optimierung“.

Weiter unten: **vier Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So arbeitet der Prüfer", expanded=True):
    st.markdown(
        """
1. **Baum:** Der Ausdruck wird zu einem Baum aus Variablen, Konstanten und Atomen. Erlaubt sind nur `x`, `y`, Zahlen, `+ − *` (Konstante mal Term), `/` durch eine Konstante, `^2` und die Atome
   `square, abs, exp, log, sqrt, inv_pos, maximum, minimum, norm2, log_sum_exp, quad_over_lin`.
2. **Von unten nach oben:** Jeder Knoten bekommt eine Krümmung (affin, konvex, konkav, unbekannt) und ein Vorzeichen (≥ 0, ≤ 0, offen). Jeder Knoten wird genau einmal besucht.
3. **Verkettung:** Ein konvexes Atom braucht für ein wachsendes Argument ein konvexes (oder affines), für ein fallendes ein konkaves, bei unbekannter Monotonie ein affines Argument. Ob `square` oder `abs` wachsend oder fallend ist, hängt vom Vorzeichen des Arguments ab – darum zählt das Vorzeichen der Variablen.
4. **Sehnen:** Zufällige Sehnen zwischen zwei Punkten des Kastens prüfen, ob die Funktion in der Mitte höchstens so hoch liegt wie die Sehne. Eine einzige Verletzung widerlegt die Konvexität.
5. **Umformung:** Sieben kleine Identitäten (`sqrt(u² + 1)` → `norm2(u, 1)` usw.) werden ausprobiert und nur behalten, wenn beide Fassungen auf Zufallspunkten übereinstimmen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Ausdruck**")
    expr = st.text_input("Ausdruck in x und y", key="expr_input", max_chars=X.MAX_LENGTH,
                         help="Erlaubt: x, y, Zahlen, + − *, / durch eine Zahl, ^2, Klammern und die Atome square, abs, exp, log, sqrt, inv_pos, maximum, minimum, norm2, log_sum_exp, quad_over_lin. Beispiel: sqrt(square(x) + 1)")
    sign = st.radio("Vorzeichen der Variablen", C.SIGNS, key="sign_radio",
                    help="Der Prüfer darf nur benutzen, was er weiß. Sind x und y als nichtnegativ vereinbart, sind square, abs und die Norm wachsend und mehr Verkettungen erlaubt. Der Kasten für Sehnen und Kegelprogramm ist [0,1; 3] bzw. [−3; 3].")
    chords = st.select_slider("Zahl der Sehnen", options=C.CHORD_STEPS, key="chords_select", help="Wie viele zufällige Sehnen die Probe prüft. Mehr Sehnen finden kleinere Beulen, beweisen aber nie etwas.")
    seed = st.number_input("Zufalls-Seed der Sehnen", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Sehnen ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"expr_input": expr, "sign_radio": sign, "chords_select": int(chords), "seed_input": int(seed)})

settings = Settings(expr, sign, int(chords), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if not a.ok:
    st.error(f"Der Ausdruck lässt sich nicht lesen: {a.error}")
    st.stop()

canon_state = "–" if a.canon is None else a.canon.status
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Prüfer", a.report.curvature, help="Krümmung der Wurzel nach den DCP-Regeln: nur ein Beweis, wenn „konvex“, „konkav“ oder „affin“.")
m2.metric("Sehnen", "widerlegt" if a.chords.convex_refuted else "nicht widerlegt", help="Ob eine der Sehnen die Konvexität verletzt.")
m3.metric("Urteil", a.judgement.status, help="Dreiwertig: bewiesen konvex, bewiesen konkav, widerlegt oder offen (Prüfer schweigt, Sehnen finden nichts).")
m4.metric("Knoten", a.tree.size(), help="Größe des Ausdrucksbaums.")
m5.metric("Kegelprogramm", canon_state, help="Ob aus dem Beweis ein Kegelprogramm entsteht und der Kern es löst.")
chord_shown = a.chords.convex_example if a.chords.convex_refuted else typical_chord(a.tree, *a.box, a.settings.seed)
st.plotly_chart(build_function(a.tree, a.box, chord_shown, a.chords.convex_refuted, a.canon.point if a.canon and a.canon.status == "gelöst" else None), width="stretch", key="main_chart")
st.caption("Links die Funktion im Kasten mit der Sehne (rot: verletzt, blau: nicht verletzt); rechts die Funktion entlang dieser Sehne gegen die gerade Verbindung der Endwerte. Bei einer konvexen Funktion liegt die Kurve nie über der Geraden.")

if st.button("📄 PDF der aktuellen Analyse erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a, verdict_analysis(a)[1]))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kruemmungsregeln.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was beweist ein Prüfer, und wo schweigt er?")
verdict(verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der Prüfer am Ausdruck")
st.caption("Jeder Knoten mit Krümmung, Vorzeichen und, wo die Regel verletzt ist, dem Grund. Darunter die bestätigten Umformungen.")
st.dataframe(pd.DataFrame([{"Knoten": "· " * d + label, "Krümmung": curv, "Vorzeichen": sg, "Regel verletzt, weil": reason or ""} for d, label, curv, sg, reason in a.annotation]), hide_index=True, width="stretch")
if a.report.curvature == D.UNK:
    if a.rewrites:
        st.dataframe(pd.DataFrame([{"Umformung": " → ".join(RW.RULES[k] for k in r.steps), "neue Fassung": r.text, "Krümmung": r.curvature, "größte Abweichung": sci(r.deviation)} for r in a.rewrites]),
                     hide_index=True, width="stretch")
    else:
        st.caption("Keine der sieben Umformungen führt zu einem Beweis (oder keine ist auf diesem Ausdruck bestätigt).")
verdict(verdict_tree(a))

st.markdown("---")

st.subheader("🔬 Experiment 2: Der Katalog der Täuscher")
st.caption(f"Alle {len(CAT.CATALOG)} Katalogeinträge auf einmal ({C.CATALOG_CHORDS} Sehnen je Eintrag). Spalten: was der Prüfer sagt, was die Sehnen finden, und ob eine bestätigte Umformung den Beweis liefert. Die Texte stehen in der Tabelle darunter zum Einfügen in das Textfeld.")
crows = _catalog()
st.plotly_chart(build_catalog(crows), width="stretch", key="catalog_chart")
st.dataframe(pd.DataFrame([{"Beispiel": r["entry"].name, "Text zum Einfügen": r["entry"].text, "Vorzeichen": "x, y ≥ 0" if r["entry"].sign == D.POS else "frei", "Lehre": r["entry"].note,
                            "Abweichung der Umformung": sci(r["rewrite"].deviation) if r["rewrite"] else ""} for r in crows]), hide_index=True, width="stretch")
verdict(verdict_catalog(crows))

st.markdown("---")

st.subheader("🔬 Experiment 3: Beweisen gegen Widerlegen")
st.caption("Wie viele Sehnen braucht die Widerlegung? Dazu Zufallsbäume gegen Prüfer und Sehnen, und die Zeit des Prüfers.")
power = _power()
st.plotly_chart(build_power(power), width="stretch", key="power_chart")
verdict(verdict_power(power))
corpus = _corpus()
st.plotly_chart(build_corpus(corpus), width="stretch", key="corpus_chart")
verdict(verdict_corpus(corpus))
timing = _timing()
st.plotly_chart(build_timing(timing), width="stretch", key="timing_chart")
verdict(verdict_timing(timing))

st.markdown("---")

st.subheader("🔬 Experiment 4: Vom Beweis zum Kegelprogramm")
st.caption("Jedes Atom bekommt eine Hilfsvariable und eine Kegelbedingung; der Kern löst das Programm, und die Gegenprobe ist der direkt ausgewertete Ausdruck und ein Gitter des Kastens.")
if a.canon is not None and a.canon.program:
    c = a.canon
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Variablen", c.variables + c.aux, help="Eigene Variablen plus eine Hilfsvariable je nichtaffinem Atom.")
    k2.metric("Zeilen", c.rows, help="Zeilen der Kegelbedingungen s = b − A x ∈ K.")
    k3.metric("Orthant-Zeilen", c.orthant)
    k4.metric("Lorentz-Kegel", len(c.soc))
    st.dataframe(pd.DataFrame([{"Bedingung": label, "Kegel": kind, "Zeilen": rows} for label, kind, rows in c.program]), hide_index=True, width="stretch")
verdict(verdict_canon(a))
if a.canon_rewrite is not None:
    verdict(verdict_canon_rewrite(a))
ccorp = _canon_corpus()
st.plotly_chart(build_canon_corpus(ccorp), width="stretch", key="canon_chart")
verdict(verdict_canon_corpus(ccorp))
st.markdown("**Ein Ausreißer im Kern.** Der gemeinsame Löser meldet in einem Fall eine Lückenschranke, die nicht gilt (und dabei `far_from_path = True`). Die Absicherung in dieser Demo fängt ihn ab:")
verdict(verdict_outlier(_outlier()))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Prüfer beweist nur, was seine Regeln zulassen** | Konvexe Funktionen außerhalb der Regeln bleiben „unbekannt“ (Katalog). Die Umformungen dieser Demo sind sieben Identitäten, kein vollständiges Verfahren. | – |
| **Atome sind die Bausteine** | Was kein Atom hat (geometrisches Mittel, Entropie `x·log x`), lässt sich nicht beweisen, auch wenn es konvex bzw. konkav ist. | – |
| **Die Sehnenprobe widerlegt, beweist nicht** | Endlich viele Sehnen übersehen eine kleine Beule (Experiment 3); „nicht widerlegt“ ist kein Beweis. | – |
| **Die Zufallsbäume stehen für Modelle** | Die Anteile in Experiment 3 gelten für zufällige Bäume dieser Atomliste, nicht für Modelle aus der Praxis. | – |
| **Der Kastenmittelpunkt ist ein streng zulässiger Start** | Bei `sqrt`, `inv_pos` und `quad_over_lin` kann er außerhalb des Definitionsbereichs liegen; dann gibt es keinen Start, obwohl die Funktion bewiesen konvex ist. | Stück 6 (Phase I) |
| **Die Schranke ν/t des Kerns gilt für einen zentrierten Punkt** | Der Kern meldet `ok = True` auch ohne Zentrierung und warnt nur vor der gefährlichsten Klasse (`far_from_path`, Experiment 4, Ausreißer); diese Demo prüft die Zentriertheit zusätzlich am Endpunkt. | – |
| **Exponential- und Logarithmus-Atome** | `exp`, `log`, `log_sum_exp` bekommen ein Urteil, aber kein Kegelprogramm: der Exponentialkegel kommt in Stück 7. | Stück 7 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Verkettungsregel.** Sei $f(g_1,\dots,g_k)$ mit konvexem $f$, das in jedem Argument $i$ wachsend ist, wenn $g_i$ konvex ist (fallend, wenn $g_i$ konkav ist; beliebig, wenn $g_i$ affin ist). Dann ist die Verkettung konvex:
für konvexe $g$ und wachsendes $f$ gilt $f(g(\theta x + (1-\theta)y)) \le f(\theta g(x) + (1-\theta) g(y)) \le \theta f(g(x)) + (1-\theta) f(g(y))$. Das ist eine **hinreichende** Bedingung: $\sqrt{x^2+1}$ ist konvex, aber
$\sqrt{\cdot}$ ist konkav, also bleibt der Prüfer stumm; $\lVert (x,1)\rVert$ beschreibt dieselbe Funktion und ist ein Atom.

**Vorzeichen.** $u \mapsto u^2$, $|u|$ und $\lVert u\rVert$ sind für $u \ge 0$ wachsend und für $u \le 0$ fallend. Eine Verkettung $\mathrm{square}(g)$ mit konvexem $g$ ist deshalb nur bewiesen, wenn $g \ge 0$ bekannt ist.
Beispiel: $\max(x,y)^2$ ist für $x,y \ge 0$ konvex, für freie $x,y$ nicht (die Sehne von $(1,-5)$ nach $(-5,1)$ endet bei Wert 1, ihre Mitte $(-2,-2)$ liegt bei Wert 4).

**Sehne.** Für konvexes $f$ gilt $f(\lambda a + (1-\lambda) b) \le \lambda f(a) + (1-\lambda) f(b)$ für alle $a, b$ und $\lambda \in [0,1]$. Eine Verletzung widerlegt die Konvexität; keine Verletzung auf endlich vielen Sehnen beweist nichts.
Gerechnet wird nur an Sehnen, bei denen $f$ in beiden Endpunkten und in der Mitte definiert ist.

**Epigraph je Atom (Orthant und Lorentz-Kegel $\mathcal{Q} = \{(s_0, \bar s) : s_0 \ge \lVert \bar s\rVert\}$).**
$t \ge |u|$: $t - u \ge 0,\ t + u \ge 0$. $\quad t \ge \max_i u_i$: $t - u_i \ge 0$. $\quad t \le \min_i u_i$: $u_i - t \ge 0$. $\quad t \ge \lVert (u_1, \dots, u_k)\rVert$: $(t, u_1,\dots,u_k) \in \mathcal{Q}$.
$t \ge u^2$: $(t+1,\ t-1,\ 2u) \in \mathcal{Q}$, denn $(t-1)^2 + 4u^2 \le (t+1)^2 \iff u^2 \le t$.
$t \ge 1/u$ ($u > 0$): $(t+u,\ t-u,\ 2) \in \mathcal{Q} \iff 4 \le 4tu$.
$t \ge u^2/v$ ($v > 0$): $(t+v,\ t-v,\ 2u) \in \mathcal{Q} \iff u^2 \le tv$.
$t \le \sqrt{u}$: $(u+1,\ u-1,\ 2t) \in \mathcal{Q} \iff t^2 \le u$.
Konvexe Atome bekommen die Epigraph-Seite ($t \ge f$), konkave die Hypograph-Seite ($t \le f$); welche gilt, bestimmt die Krümmung, und dass die Verkettung aufgeht, hat der Prüfer bewiesen. Das Ziel ist die Wurzel, minimiert.

**Barriere-Parameter.** Orthant: eine Einheit je Zeile; Lorentz-Kegel: 2 (Stück 5).

**Literatur.** M. Grant, S. Boyd und Y. Ye: *Disciplined Convex Programming*, in L. Liberti und N. Maculan (Hrsg.): *Global Optimization: From Theory to Implementation*, Springer 2006, S. 155–210 (die Regeln: Atome, Monotonie, Verkettung).
A. Agrawal, R. Verschueren, S. Diamond und S. Boyd: *A rewriting system for convex optimization problems*, Journal of Control and Decision 5 (1), 2018, S. 42–60, doi 10.1080/23307706.2017.1397554 (Umformung eines DCP-Problems in ein Kegelprogramm).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
