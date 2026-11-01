"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import krm_chords as CH
import krm_constants as C
import krm_dcp as D
import krm_expr as X


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ----------------------------------------------------------------------------- Experiment 1: Funktion und Sehne

def typical_chord(root: X.Node, lo: float, hi: float, seed: int) -> CH.Chord | None:
    """Eine weit auseinanderliegende, auswertbare Sehne zum Zeigen, wenn keine verletzt ist."""
    rng = np.random.default_rng(seed + 991)
    best = None
    for _ in range(60):
        a = {k: float(rng.uniform(lo, hi)) for k in X.VARIABLES}
        b = {k: float(rng.uniform(lo, hi)) for k in X.VARIABLES}
        lam = 0.5
        m = {k: lam * a[k] + (1 - lam) * b[k] for k in X.VARIABLES}
        vals = [float(X.evaluate(root, {k: np.array([p[k]]) for k in X.VARIABLES})[0]) for p in (a, b, m)]
        if not all(np.isfinite(vals)):
            continue
        dist = abs(a["x"] - b["x"]) + abs(a["y"] - b["y"])
        if best is None or dist > best[0]:
            avg = lam * vals[0] + (1 - lam) * vals[1]
            best = (dist, CH.Chord((a["x"], a["y"]), (b["x"], b["y"]), lam, vals[0], vals[1], vals[2], avg, vals[2] - avg))
    return None if best is None else best[1]


def build_function(tree: X.Node, box: tuple, chord: CH.Chord | None, violated: bool, solved_point: dict | None = None):
    """Links die Funktion (Kurve bei einer Variablen, Höhenlinien bei zwei) mit der Sehne, rechts der Verlauf der Funktion entlang der Sehne gegen die gerade Verbindung der Endwerte."""
    lo, hi = box
    names = tree.variables()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], subplot_titles=("Die Funktion im Kasten", "Entlang der Sehne"))
    color = C.COLOR_BAD if violated else C.COLOR_LINE
    if len(names) == 1:
        v = names[0]
        xs = np.linspace(lo, hi, 400)
        vals = {k: (xs if k == v else np.zeros_like(xs)) for k in X.VARIABLES}
        ys = X.evaluate(tree, vals)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_CONVEX, width=2), name="f", hovertemplate=f"{v} = %{{x:.3f}}<br>f = %{{y:.4g}}<extra></extra>"), row=1, col=1)
        if chord is not None:
            i = 0 if v == "x" else 1
            fig.add_trace(go.Scatter(x=[chord.a[i], chord.b[i]], y=[chord.f_a, chord.f_b], mode="lines+markers", line=dict(color=color, width=2), name="Sehne"), row=1, col=1)
            xm = chord.lam * chord.a[i] + (1 - chord.lam) * chord.b[i]
            fig.add_trace(go.Scatter(x=[xm], y=[chord.f_mid], mode="markers", marker=dict(color=color, size=9, symbol="diamond"), name="f(Mitte)", showlegend=False), row=1, col=1)
        if solved_point and v in solved_point:
            fig.add_trace(go.Scatter(x=[solved_point[v]], y=[float(X.evaluate(tree, {k: np.array([solved_point.get(k, 0.0)]) for k in X.VARIABLES})[0])], mode="markers",
                                     marker=dict(color="black", size=9, symbol="x"), name="Optimum (Kegelprogramm)"), row=1, col=1)
        fig.update_xaxes(title_text=v, row=1, col=1)
    else:
        g = np.linspace(lo, hi, 70)
        gx, gy = np.meshgrid(g, g)
        z = X.evaluate(tree, {"x": gx, "y": gy})
        fig.add_trace(go.Contour(x=g, y=g, z=z, colorscale="Viridis", showscale=False, contours=dict(showlabels=False), line=dict(width=1), name="f", hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>f = %{z:.4g}<extra></extra>"), row=1, col=1)
        if chord is not None:
            fig.add_trace(go.Scatter(x=[chord.a[0], chord.b[0]], y=[chord.a[1], chord.b[1]], mode="lines+markers", line=dict(color=color, width=3), name="Sehne"), row=1, col=1)
        if solved_point:
            fig.add_trace(go.Scatter(x=[solved_point.get("x", 0.0)], y=[solved_point.get("y", 0.0)], mode="markers", marker=dict(color="white", line=dict(color="black", width=2), size=10, symbol="x"),
                                     name="Optimum (Kegelprogramm)"), row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
    if chord is not None:
        lam = np.linspace(0.0, 1.0, 120)
        pts = {k: lam * chord.a[i] + (1 - lam) * chord.b[i] for i, k in enumerate(X.VARIABLES)}
        along = X.evaluate(tree, pts)
        line = lam * chord.f_a + (1 - lam) * chord.f_b
        # lam = 1 ist Endpunkt a, lam = 0 Endpunkt b; die Achse läuft von a nach b
        s = 1.0 - lam
        order = np.argsort(s)
        fig.add_trace(go.Scatter(x=s[order], y=along[order], mode="lines", line=dict(color=C.COLOR_CONVEX, width=2), name="f entlang der Sehne"), row=1, col=2)
        fig.add_trace(go.Scatter(x=s[order], y=line[order], mode="lines", line=dict(color=color, width=2, dash="dash"), name="gerade Verbindung der Endwerte"), row=1, col=2)
        fig.update_xaxes(title_text="Weg von a (0) nach b (1)", row=1, col=2)
    _base(fig, height=430)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(l=10, r=10, t=50, b=80))        # unter dem Bild: oben stehen die Teiltitel
    return fig


# ----------------------------------------------------------------------------- Experiment 2: Katalog

def build_catalog(rows: list):
    """Je Eintrag drei Zellen: Urteil des Prüfers, Urteil der Sehnen, Urteil nach der Umformung."""
    names, z, text = [], [], []
    for r in rows:
        j, cur = r["judgement"], r["curvature"]
        proven = cur != D.UNK
        a = (2 if proven else 0, f"bewiesen {cur}" if proven else "unbekannt")
        if j.status == CH.REFUTED:
            b = (0, "widerlegt" + (" (weder noch)" if j.neither else ""))
        elif j.status == CH.OPEN:
            b = (1, "nicht widerlegt")
        else:
            b = (2, "nicht widerlegt")
        if proven:
            c = (2, "–")
        elif r["rewrite"] is not None:
            c = (2, r["rewrite"].text)
        else:
            c = (0 if j.status == CH.REFUTED else 1, "keine Umformung" if j.status != CH.REFUTED else "– (nicht konvex)")
        names.append(r["entry"].name)
        z.append([a[0], b[0], c[0]])
        text.append([a[1], b[1], c[1]])
    fig = go.Figure(go.Heatmap(z=z, x=["Prüfer", "Sehnen", "nach Umformung"], y=names, text=text, texttemplate="%{text}", textfont=dict(size=11), showscale=False,
                               colorscale=[[0, "rgba(179,74,74,0.55)"], [0.5, "rgba(176,138,46,0.55)"], [1, "rgba(47,107,101,0.6)"]], zmin=0, zmax=2, xgap=2, ygap=2, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _base(fig, height=60 + 30 * len(rows))


# ----------------------------------------------------------------------------- Experiment 3: Sehnenzahl und Zufallsbäume

def build_power(p: dict):
    steps = list(p["steps"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[100.0 * p["corpus"][n] / p["trees"] for n in steps], mode="lines+markers", name=f"{p['trees']} Zufallsbäume (nicht konvex)", line=dict(color=C.COLOR_LINE)))
    fig.add_trace(go.Scatter(x=steps, y=[100.0 * p["bump"][n] / p["bump_runs"] for n in steps], mode="lines+markers", name="kleine Beule (nichtkonvex nur für |x−2| < 0,016)", line=dict(color=C.COLOR_BAD)))
    fig.update_xaxes(type="log", title_text="Zahl der Sehnen")
    fig.update_yaxes(title_text="widerlegt in % der Fälle bzw. Läufe", range=[0, 105])
    return _base(fig)


def build_corpus(rows: list):
    fig = go.Figure()
    cats = [("proven_convex", "bewiesen konvex", C.COLOR_CONVEX), ("proven_concave", "bewiesen konkav", "#7aa6a1"),
            ("unknown_open_convex", "unbekannt, konvex nicht widerlegt", C.COLOR_UNKNOWN), ("unknown_open_concave", "unbekannt, nur konvex widerlegt", "#d4b873"),
            ("unknown_neither", "unbekannt, weder noch (widerlegt)", C.COLOR_BAD)]
    for key, name, col in cats:
        fig.add_trace(go.Bar(x=[r["sign"] for r in rows], y=[100.0 * r[key] / r["trees"] for r in rows], name=name, marker_color=col))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="% der Zufallsbäume", range=[0, 100])
    return _base(fig)


def build_timing(rows: list):
    fig = go.Figure(go.Scatter(x=[r["nodes"] for r in rows], y=[r["seconds"] * 1e3 for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LINE)))
    fig.update_xaxes(title_text="Knoten im Baum")
    fig.update_yaxes(title_text="Zeit des Prüfers (ms)", rangemode="tozero")
    return _base(fig, height=300)


# ----------------------------------------------------------------------------- Experiment 4: Kegelprogramme

def build_canon_corpus(c: dict):
    labels = ["gelöst", "kein Start", "nicht zentriert"]
    vals = [c["solved"], c["no_start"], c["not_centered"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[C.COLOR_CONVEX, C.COLOR_UNKNOWN, C.COLOR_BAD], text=vals, textposition="outside"))
    fig.update_yaxes(title_text="bewiesen konvexe Zufallsbäume", range=[0, max(vals) * 1.15 + 1])
    return _base(fig, height=300)
