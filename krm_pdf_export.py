"""PDF-Export der aktuellen Analyse (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import krm_dcp as D
from krm_evaluation import Analysis, num, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "∘": "o", "ν": "nu", "λ": "lambda", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "√": "sqrt", "‖": "|", "…": "...", "∈": "in", "ᵀ": "^T", "₀": "0", "₁": "1", "∞": "inf"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis, verdict_text: str) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Krümmungsregeln: beweist der Prüfer Konvexität?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Ausdruck: {a.text}\nVariablen: {s.sign}; Kasten [{num(a.box[0])}; {num(a.box[1])}]; {s.chords} Sehnen, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _s("Urteil"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(verdict_text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    head, widths = ["Knoten", "Krümmung", "Vorzeichen", "Regel"], [58, 26, 24, 82]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for depth, label, curv, sign, reason in a.annotation:
        for v, w in zip(["  " * depth + label, curv, sign, (reason or "")[:60]], widths):
            pdf.cell(w, 5, _s(v), border=1)
        pdf.ln()
    if a.rewrites:
        r = a.rewrites[0]
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _s(f"Bestätigte Umformung: {r.text} ist {r.curvature} (Abweichung höchstens {sci(r.deviation)})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Prüfer (DCP) beweist Konvexität oder schweigt; eine verletzte Sehne widerlegt sie, eine unverletzte beweist nichts. Quelle: kruemmungsregeln-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
