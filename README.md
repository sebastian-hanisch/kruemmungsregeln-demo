# Krümmungsregeln – wann ein Prüfer Konvexität beweist und wo er schweigt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kruemmungsregeln-demo.streamlit.app/)** -->

Stück 3 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dieses Stück sitzt vor dem Löser: es fragt, **woran man mechanisch erkennt, dass ein Ausdruck ein Kegelprogramm ergibt**. Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1), [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2: dieselbe Aufgabe in mehreren Kegelformen), [kegel-dualitaet-demo](https://github.com/sebastian-hanisch/kegel-dualitaet-demo) (Stück 4) und [kegel-barriere-demo](https://github.com/sebastian-hanisch/kegel-barriere-demo) (Stück 5: der Löser, den dieses Stück benutzt). Stück 3 war in der Gliederung optional und ist gebaut.

## Kernfrage

**Was beweist ein Prüfer für Konvexität, und wo schweigt er?** Ein Prüfer nach den Regeln des *Disciplined Convex Programming* (DCP) gibt jedem Knoten eines Ausdrucksbaums eine Krümmung (affin, konvex, konkav, unbekannt) und ein Vorzeichen. Jedes Atom hat eine feste Krümmung und je Argument eine Monotonie, die vom Vorzeichen des Arguments abhängen darf; eine Verkettung ist erlaubt, wenn beides zusammenpasst. Die Regeln sind **hinreichend, nicht notwendig**: „unbekannt“ heißt, dass kein Beweis gefunden wurde, nicht, dass die Funktion nicht konvex wäre. `sqrt(x² + 1)` ist konvex (es ist die Norm von `(x, 1)`), der Prüfer schweigt trotzdem, weil `sqrt` konkav ist und `x² + 1` konvex.

Dazu die Gegenprobe: **Sehnen widerlegen, beweisen nie.** Eine einzige Sehne, die `f(λa + (1−λ)b) ≤ λ f(a) + (1−λ) f(b)` verletzt, ist ein Gegenbeispiel mit Zahlen. Keine verletzte Sehne beweist nichts: endlich viele Sehnen übersehen eine kleine Beule. Zusammen ergeben beide ein **dreiwertiges Urteil: bewiesen, widerlegt oder offen.**

Und der Beweis ist nützlich: **Jedes bewiesene Atom mit Orthant- oder Lorentz-Übersetzung (acht der elf) wird zu einem Kegelprogramm** (eine Hilfsvariable und eine Orthant- oder Lorentz-Bedingung je Atom, wie in Stück 2), und der Kern aus Stück 5 löst es; `exp`, `log` und `log_sum_exp` brauchen den Exponentialkegel (Stück 7).

## Modell

- **Ausdrücke** (`krm_expr.py`): Variablen `x`, `y`, Zahlen, `+ − *` (Konstante mal Term), `/` durch eine Konstante, `^2` und **elf Atome**: `square, abs, exp, log, sqrt, inv_pos, maximum, minimum, norm2, log_sum_exp, quad_over_lin`. Ein Produkt zweier Terme mit Variablen ist erlaubt und hat keine Regel (der Prüfer sagt „unbekannt“). Eigener Parser (rekursiver Abstieg, **kein `eval`**), begrenzt auf 200 Zeichen, 30 Ebenen, 150 Knoten.
- **Prüfer** (`krm_dcp.py`): ein Durchlauf von unten nach oben, jeder Knoten genau einmal, iterativ; Vorzeichenregeln wie in cvxpy 1.9.3. Gibt je Knoten Krümmung, Vorzeichen und, wo die Regel verletzt ist, den Grund.
- **Sehnen** (`krm_chords.py`): zufällige Sehnen im Kasten (frei: [−3; 3]², nichtnegativ: [0,1; 3]²), nur dort gerechnet, wo die Funktion in beiden Endpunkten und der Mitte definiert ist; Verletzung relativ zu 1 + größtem Endwert, Schwelle 10⁻⁸.
- **Umformung** (`krm_rewrite.py`): **sieben** kleine Identitäten (`sqrt(Summe von Quadraten) = norm2`, `sqrt(u²) = abs(u)`, `log(Summe von Exponentialfunktionen) = log_sum_exp`, `u·u = square(u)`, `exp(a)·exp(b) = exp(a+b)`, `u²·(1/v) = quad_over_lin(u, v)`, `1/u² = (1/u)²` für u > 0). Eine Umformung gilt erst, wenn beide Fassungen auf 4000 Zufallspunkten bis 10⁻⁹ (relativ) übereinstimmen.
- **Kegelprogramm** (`krm_canon.py`): Epigraph je konvexem Atom (`t ≥ f`), Hypograph je konkavem (`t ≤ f`); Orthant und Lorentz-Kegel; gelöst mit `kegel_kern.py` (wortgleich mit Stück 2, 4 und 5; die Version mit dem Rückmeldefeld `far_from_path`). `exp`, `log` und `log_sum_exp` bekommen ein Urteil, aber kein Kegelprogramm (Exponentialkegel: Stück 7).
- **Absicherung des Kerns:** Startwert `t₀ = ν / max(1, |cᵀx₀|)` und eine unabhängige Prüfung der Zentriertheit am Ende (halbes Dekrement-Quadrat zwischen 0 und 10⁻³ bei `t = ν/Lücke`). Ohne sie meldet der Kern in einem Fall eine Lückenschranke, die nicht gilt (siehe Befunde); sein Feld `far_from_path` zeigt diesen Fall an, die Prüfung am Ende ergänzt es um die Fälle, die das Feld nicht trifft.

## Methodik

Vier Experimente mit live berechnetem Verdikt: (1) der Prüfer am eingegebenen Ausdruck: Baum mit Krümmung, Vorzeichen und verletzter Regel, bestätigte Umformungen; (2) der Katalog der Täuscher: 21 Ausdrücke, was Prüfer, Sehnen und Umformung sagen; (3) Beweisen gegen Widerlegen: Trefferquote der Sehnen, Zufallsbäume gegen Prüfer und Sehnen, Zeit des Prüfers; (4) vom Beweis zum Kegelprogramm: Zeilen, Kegel, Lösung mit Gegenprobe, Zufallsbäume als Kegelprogramm, der Ausreißer im Kern.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung auf Linux, steht die gemessene Zahl trotzdem hier); Stichproben laufen über feste Seeds. Zufallsbäume haben Tiefe 3 über 15 Operationen (bei freien Vorzeichen 12, ohne `log`, `sqrt`, `inv_pos`) und zwei Variablen: ihre Anteile sagen nichts über Modelle aus der Praxis.

| Frage | Befund |
|---|---|
| Standardfall `sqrt(square(x) + 1)` auf [−3; 3] | Prüfer **unbekannt** (Grund am Knoten `sqrt`: konkav und wachsend verlangt ein konkaves Argument, es ist konvex); 4000 gültige Sehnen ohne Verletzung, Urteil **offen**; die Umformung `norm2(x, 1)` ist konvex und stimmt mit dem Original überein (Abweichung 0,0); das Kegelprogramm der Umformung hat den Wert 1,000 |
| Katalog (21 Ausdrücke) | Der Prüfer beweist **6** (5 konvex, 1 konkav), schweigt bei **15**. Von diesen sind 5 bei den Sehnen widerlegt (zu Recht keine Aussage), bei 10 finden die Sehnen nichts; bei **8** dieser 10 rettet eine Umformung den Beweis ohne die Funktion zu ändern (Identität, numerisch auf 4000 Punkten bestätigt; größte Abweichung unter 10⁻⁹). Es bleiben `x·log x` (Entropie-Atom fehlt) und `1/(x·y)`. Widersprüche zwischen Prüfer und Sehnen: **0** |
| Das Vorzeichen entscheidet | `max(x, y)²` ist für freie x, y nicht konvex (Sehne von (1, −5) nach (−5, 1): Endwerte 1 und 1, Mitte (−2, −2) bei 4) und für x, y ≥ 0 **bewiesen konvex**. `√(x·y)` ist konkav, aber „unbekannt“ (kein Atom für das geometrische Mittel); `log(x + y)` ist bewiesen konkav |
| Zufallsbäume gegen Prüfer und Sehnen (je Vorzeichenfall rund 1500, davon 71 bzw. 10 wegen Definitionsbereich ausgeschlossen) | **0 Widersprüche.** Mit x, y ≥ 0: bewiesen konvex oder affin 382, bewiesen konkav 122, „unbekannt“ 925 (davon als nicht konvex widerlegt 599, nicht widerlegt 326). Frei: 461, 95, 934 (665, 269). Bewiesen sind **54 %** (x, y ≥ 0) und **63 %** (frei) der „bewiesen oder nicht widerlegt konvexen“ |
| Trefferquote der Sehnen (150 Zufallsbäume, die der Prüfer nicht beweist und die bei 20000 Sehnen widerlegt sind) | 50 Sehnen widerlegen 91 %, 200: 97 %, 1000: 98 %, 4000: 100 %. Die **kleine Beule** `x² + 0,8·exp(−2000(x−2)²)` (nichtkonvex nur für \|x−2\| < 0,016, zweite Ableitung von Hand): 50 Sehnen finden sie in 13 von 40 Läufen, 200 in 31, ab 1000 in allen 40 (Zählungen mit Streuung von rund 3 Läufen; andere Seeds geben andere Werte, grob ein Drittel bei 50 und drei Viertel bei 200 Sehnen) |
| Zeit des Prüfers | linear: 4 bis 6 µs je Knoten bei 41, 201 und 601 Knoten (die Zeiten hängen vom Rechner ab und werden nicht getestet) |
| Kegelprogramm aus Zufallsbäumen (120 bewiesen konvexe Bäume über `square, abs, maximum, minimum, norm2, sqrt, inv_pos, quad_over_lin`, x, y ∈ [0,1; 3]) | **109 gelöst**, **11 (9,2 %) ohne streng zulässigen Start** im Kastenmittelpunkt (Argument von `sqrt`, `inv_pos`, `quad_over_lin` dort ≤ 0), 0 nicht zentriert. Alle 109 stimmen mit der Direktauswertung am Punkt und dem Gitter (Wert nie über dem Gitterminimum) überein. Im Median 1 Hilfsvariable (höchstens 6), 6 Zeilen (höchstens 20), 1 Lorentz-Kegel, 32 Newton-Schritte |
| Der Ausreißer im Kern: `square(quad_over_lin(square(x), minimum(x, y)))` auf [0,1; 3]² | Ohne Absicherung meldet der Kern den Wert **rund 190** (190,386 auf dem Entwicklungsrechner, 190,368 auf einem GitHub-Runner) mit Lückenschranke unter 10⁻⁹ und `converged = True`; das Optimum ist **1,0·10⁻⁶**. Das halbe Dekrement-Quadrat am Endpunkt ist negativ (−4,8·10¹⁴ auf dem Entwicklungsrechner): die Hesse-Matrix wurde numerisch singulär, das Dekrement negativ, und `dec2/2 ≤ tol` galt als zentriert. Der Kern meldet dabei **`far_from_path = True`** (eine Zentrierung endete nach 100 Schritten weit vom Pfad, ein Test mit der Handinstanz hält das fest). Mit der Absicherung stimmt der Wert (1,0·10⁻⁶, Dekrement 6,6·10⁻⁸, kein Flag) |

## Befunde und Korrekturen gegenüber dem Plan

- **Sieben statt acht Umformungsregeln.** Der Plan sprach von acht kleinen Regeln; gebaut sind sieben (die achte, `sqrt(u²+1)`, ist ein Sonderfall von `sqrt(Summe von Quadraten)`).
- **Atome:** elf im Prüfer, davon acht mit Kegelübersetzung und drei mit Exponentialkegel.
- **Der Prüfer stimmte erst nach vier Korrekturen mit cvxpy überein:** `maximum` ist nichtnegativ, sobald **ein** Argument es ist; `minimum` nur, wenn **alle**; `log_sum_exp`: der Prüfer verlangt, dass **alle** Argumente es sind (hinreichend, nicht scharf, denn schon ein nichtnegatives Argument genügte mathematisch); die Vektornorm hat **ein** Vorzeichen für den ganzen Vektor (pro Argument getrennt wäre großzügiger, aber nicht cvxpy). Die Vorzeichenregeln sind der schwierige Teil, nicht die Verkettung. Heute: 700 feste Bäume je Vorzeichenfall, 0 Abweichungen (`tests/test_oracle_kruemmungsregeln.py`).
- **Ein Fehler des ersten Baus:** die Prüfung speicherte Ergebnisse unter `id(Knoten)`; Streamlits Zwischenspeicher kopiert die Objekte (pickle), und die Kopien haben andere Adressen. Der Fehler zeigte sich erst im zweiten Lauf der App und ist jetzt durch einen Test abgedeckt (`test_a_report_survives_copying_with_pickle`).
- **Die Beule ist schmaler als geschätzt:** nichtkonvex nur für |x−2| < 0,016 (nicht 0,02); die Texte nennen den gemessenen Wert.
- **Ein Optimum auf einer ganzen Fläche** (`max(x, 2y) + |x|`, null auf x ≤ 0, 2y ≤ x) beendet den Kern früher: die gemeldete Lücke liegt bei 10⁻⁶ statt 10⁻⁹ (`converged = False`), und sie gilt. Die App zeigt die erreichte Lücke.
- **Sehr verschieden große Koeffizienten** (etwa `1e8*x`) können den Kern in der ersten Runde scheitern lassen (etwa `1e8*x + abs(y)`; `100000000*abs(x) + square(y)` löst sich dagegen): die Übersetzung meldet „Rechengrenze“ oder „nicht zentriert“ statt eines Werts.
- **Quellen geprüft:** Grant, Boyd und Ye (Buchkapitel, S. 155–210) und Agrawal, Verschueren, Diamond und Boyd (Journal of Control and Decision 5 (1), 2018, S. 42–60) sind an der Quelle bestätigt (Suche nach Fundstellen am 2026-11-01); die Regeln habe ich nicht aus den Arbeiten abgeschrieben, sondern gegen cvxpy 1.9.3 abgeglichen.
- **Port:** 9303 (Netstat und `launch.json` gelesen, frei).

## Ehrliche Grenzen

- **Der Prüfer beweist nur, was seine Regeln zulassen.** Konvexe Funktionen außerhalb der Regeln bleiben „unbekannt“; die sieben Umformungen sind kein vollständiges Verfahren (`x·log x` und `1/(x·y)` bleiben offen).
- **Die Sehnenprobe widerlegt, beweist nicht.** Ein Kasten, endlich viele Sehnen; „nicht widerlegt“ ist kein Beweis. Gerechnet wird nur dort, wo die Funktion an allen drei Punkten definiert ist; Sehnen, bei denen nur die Mitte außerhalb des Definitionsbereichs liegt, fallen weg.
- **Die Anteile aus den Zufallsbäumen** (54 %, 63 %, 91 % …) gelten für diese Bäume, nicht für Modelle aus der Praxis.
- **Bewiesen konvex heißt nicht lösbar:** ohne streng zulässigen Start im Kastenmittelpunkt gibt es keine Rechnung (9,2 % der Zufallsbäume); Phase I ist Stück 6.
- **`converged = True` und `ok = True` heißen im Kern nicht „zentriert“.** Auch erschöpfte Schritte, ein negatives Dekrement-Quadrat und der Rundungsstopp gelten dort als ok; die Schranke `ν/t` gilt streng nur für einen zentrierten Punkt, und bei großem t ist das Dekrement durch Rundung nicht verlässlich bestimmbar. Der Kern meldet nur die gefährlichste Klasse (`far_from_path`); eine strengere Fassung würde veröffentlichte Läufe der Stücke 2 und 4 abbrechen lassen, die richtige Werte liefern (Plan: `PLAN-KERN-HAERTUNG.md`). Diese Demo prüft deshalb zusätzlich die Zentriertheit am Endpunkt; ein Test hält das Verhalten des Kerns ohne diese Absicherung fest.
- **Cvxpy-Aussagen** (gleiche Urteile, `DCPError`) gelten für Version 1.9.3 (Stand 2026-11-01); sie sind nur als Gegenprobe im Test, nicht in der App.
- **Zeiten** hängen vom Rechner ab und werden nicht getestet.

## Verifikation

- **Kern** (`tests/test_kern.py`, wortgleich aus Stück 5): Barrieren gegen finite Differenzen, Mini-Programme, singuläre Hesse-Matrix.
- **Parser** (`tests/test_expr.py`): Werte von Hand, Fehlermeldungen, Einschleusungen (`__import__`, Attributzugriff, Python-Namen), Längen-, Tiefen- und Knotengrenzen, Rundlauf des Textes, Auswertung außerhalb des Definitionsbereichs, sehr tiefe Bäume ohne Rekursion.
- **Prüfer** (`tests/test_dcp.py`): jede Regel an einem Mini-Baum von Hand (rund 60 Fälle: je Atom wachsend/fallend, Vorzeichen, affine Argumente), Vorzeichenregeln, Gründe, ein Durchlauf je Knoten, Pickle-Festigkeit.
- **Sehnen, Umformung, Katalog, Kegelprogramm** (`tests/test_chords.py`, `test_rewrite.py`, `test_canon.py`): Handsehne `(1, −5) – (−5, 1)`; jede Regel an ihrem Muster; Epigraph-Formeln als Eigenschaft (4000 Zufallspunkte, unabhängig vom Baumcode); zwölf Minima von Hand; Kernausfall (`LinAlgError`) wird zur Meldung.
- **Orakel** (`tests/test_oracle_kruemmungsregeln.py`): Urteile gegen cvxpy auf 700 festen Zufallsbäumen je Vorzeichenfall (vollständige Übereinstimmung), Optimalwert von 60 Kegelprogrammen gegen Clarabel, zweite Differenzen auf einem feinen Gitter für alle bewiesenen Ausdrücke einer Variablen, cvxpy beweist jede bestätigte Umformung und das Original nicht.
- **Randfälle** (`tests/test_edge_cases.py`), **Verdikte** (`tests/test_verdicts.py`), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler, jedes Preset, Vorzeichenschalter, 8 Eingabefehler, alle 21 Katalogtexte, PDF für jeden Zustand), Presets, Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe.
- **Umfang:** 296 Tests ohne cvxpy (das Orakelmodul mit 5 weiteren Tests überspringt sich als Ganzes), rund 45 Sekunden im generischen venv (gemessen 2026-11-03, Windows).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne sie überspringen sich die Orakel-Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2, 4 und 5 |
| `krm_expr.py` | Ausdrucksbaum, Parser, Auswertung, Textausgabe |
| `krm_dcp.py` | Der Krümmungsprüfer (Atome, Verkettung, Vorzeichen, Gründe) |
| `krm_chords.py` | Sehnenprobe und dreiwertiges Urteil |
| `krm_rewrite.py` | Sieben Umformungen mit gerechneter Bestätigung |
| `krm_catalog.py` | Der Katalog der 21 Täuscher |
| `krm_canon.py` | Übersetzung ins Kegelprogramm, Lösen, Absicherung des Kerns |
| `krm_constants.py` | Kästen, feste Stichproben, Farben, Presets |
| `krm_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `krm_layout.py` | Fester Kopf (Stylesheet) |
| `krm_pdf_export.py` | PDF-Export (fpdf2) |
| `krm_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `krm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Parser, Prüfer, Sehnen, Umformung, Kegelprogramm, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Exponential- und Potenzkegel:** Stück 7; `exp`, `log` und `log_sum_exp` haben hier ein Urteil, aber kein Kegelprogramm.
- **Phase I** (Start ohne streng zulässigen Punkt) und der Primal-Dual-Schritt: Stück 6.
- **Weitere Atome** (geometrisches Mittel, Entropie, Normen mit Matrizen) und Matrixvariablen: nicht gebaut.
- **Vollständige Umformung** (Suche über alle Identitäten): nur sieben feste Regeln.
- **Strenges Härten des Kerns** (`center` meldet nach `max_iter` oder bei negativem Dekrement nicht mehr `ok = True`): gemessen und verworfen, weil es Stück 2 und 4 bricht (siehe `PLAN-KERN-HAERTUNG.md`); umgesetzt sind Docstrings und das Feld `far_from_path`.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
