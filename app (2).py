import io
import math
import re
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from scipy.optimize import brentq
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak, KeepTogether
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hispanic Heritage App 2.0", page_icon="📈", layout="wide")

# ----------------------------- Constants and helpers -----------------------------
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]
CONTEXTS = [
    "Business", "Engineering", "Medical", "Sports science", "Biology",
    "Bioengineering", "Economic forecasting", "Weather", "Custom"
]
WB_INDICATORS = {
    "Population, total": "SP.POP.TOTL",
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
    "Life expectancy at birth, total (years)": "SP.DYN.LE00.IN",
    "CO₂ emissions (metric tons per capita)": "EN.ATM.CO2E.PC",
    "School enrollment, secondary (% gross)": "SE.SEC.ENRR",
    "Unemployment, total (% of total labor force)": "SL.UEM.TOTL.ZS",
}
WB_COUNTRIES = {
    "United States": "USA", "Ecuador": "ECU", "Mexico": "MEX",
    "Canada": "CAN", "Brazil": "BRA", "United Kingdom": "GBR",
    "Spain": "ESP", "Japan": "JPN", "India": "IND", "World": "WLD",
}
SUGGESTIONS = {
    "Business": ("1200 + 85*x - 2*x**2", "Months since launch", "Monthly revenue", "USD/month"),
    "Engineering": ("2.5*x**2 + 4*x", "Time", "Distance", "m"),
    "Medical": ("100*exp(-0.18*x)", "Hours after dose", "Concentration", "mg/L"),
    "Sports science": ("-4.9*x**2 + 12*x + 1", "Time", "Height", "m"),
    "Biology": ("50*exp(0.12*x)", "Days", "Population", "cells"),
    "Bioengineering": ("0.5*x**3 - 3*x**2 + 4*x + 8", "Time", "Device response", "units"),
    "Economic forecasting": ("2*x**2 - 15*x + 100", "Years after baseline", "Index", "index points"),
    "Weather": ("12 + 8*sin(2*pi*x/24)", "Hour of day", "Temperature", "°C"),
    "Custom": ("x**2 - 4*x + 3", "Input", "Output", "units"),
}
X = sp.Symbol("x", real=True)
SAFE_LOCALS = {
    "x": X, "t": X, "pi": sp.pi, "e": sp.E, "E": sp.E,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
}

def parse_function(source):
    """Parse a familiar math expression without evaluating arbitrary Python."""
    source = str(source).strip().replace("^", "**").replace("−", "-").replace("×", "*").replace("÷", "/")
    if not source:
        raise ValueError("Enter a function, for example x**2 - 4*x + 3.")
    if len(source) > 250:
        raise ValueError("The function is too long. Try a shorter expression.")
    if re.search(r"[^0-9A-Za-z_+\-*/().,\s]", source):
        raise ValueError("Use numbers, x, parentheses, + - * / **, and supported functions such as sin, cos, exp, log, and sqrt.")
    try:
        expr = sp.sympify(source, locals=SAFE_LOCALS, evaluate=True)
    except Exception as exc:
        raise ValueError("I couldn't read that expression. Check parentheses and operators, such as x**2 rather than x^2 (both are accepted).") from exc
    if not expr.free_symbols.issubset({X}):
        raise ValueError("Use x as the only variable in the function.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The function contains an undefined or infinite constant.")
    return expr

def safe_numeric(expr, x):
    try:
        value = complex(sp.N(expr.subs(X, float(x))))
        if abs(value.imag) > 1e-7 or not np.isfinite(value.real):
            return np.nan
        return float(value.real)
    except Exception:
        return np.nan

def sample_function(expr, xs):
    return np.array([safe_numeric(expr, v) for v in xs], dtype=float)

def world_bank_fetch(country_code, indicator_code, start_year, end_year):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 2000, "date": f"{start_year}:{end_year}"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError("The World Bank API returned no usable data for this selection.")
    rows = [{"year": int(r["date"]), "value": r.get("value")} for r in payload[1] if r.get("value") is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No non-missing observations were returned. Try a different country, indicator, or year range.")
    return df.sort_values("year").drop_duplicates("year").reset_index(drop=True)

def fit_polynomial(df, degree, base_year):
    data = df.dropna(subset=["year", "value"]).copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna().drop_duplicates("year").sort_values("year")
    if len(data) < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct non-missing years; only {len(data)} are available.")
    t = data["year"].to_numpy(dtype=float) - float(base_year)
    y = data["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residual)))
    data["t_centered"] = t
    data["fitted"] = pred
    data["residual"] = residual
    return {"data": data, "coeff": coeff, "base_year": int(base_year), "degree": int(degree),
            "r2": r2, "mae": mae, "mean": float(np.mean(y)), "n": int(len(data)),
            "min_year": int(data.year.min()), "max_year": int(data.year.max()),
            "residuals": residual}

def poly_equation(coeff, base_year, unit="value"):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if abs(c) < 1e-14:
            continue
        if power == 0:
            terms.append(f"{c:.10g}")
        elif power == 1:
            terms.append(f"({c:.10g})t")
        else:
            terms.append(f"({c:.10g})t^{power}")
    return "y = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = year − {base_year}"

def eval_fit(model, years):
    t = np.asarray(years, dtype=float) - model["base_year"]
    return np.polyval(model["coeff"], t)

def chronological_holdout(df, degree, base_year):
    """Use final 20% as a chronological holdout if training set remains estimable."""
    data = df.dropna(subset=["year", "value"]).drop_duplicates("year").sort_values("year")
    n = len(data)
    if n < max(10, degree + 3):
        return None
    test_n = max(2, int(np.ceil(0.2 * n)))
    train, test = data.iloc[:-test_n], data.iloc[-test_n:]
    if len(train) < degree + 1:
        return None
    coeff = np.polyfit(train["year"].to_numpy(dtype=float) - base_year, train["value"].to_numpy(dtype=float), degree)
    pred = np.polyval(coeff, test["year"].to_numpy(dtype=float) - base_year)
    mae = float(np.mean(np.abs(test["value"].to_numpy(dtype=float) - pred)))
    return {"mae": mae, "n_test": len(test), "first_test": int(test.year.min()), "last_test": int(test.year.max())}

def fmt(v, digits=6):
    if v is None:
        return "—"
    try:
        if not np.isfinite(float(v)):
            return "not defined"
        return f"{float(v):,.{digits}g}"
    except Exception:
        return str(v)

def numeric_roots(expr, lo, hi, samples=1200):
    """Find real roots numerically with sign changes and near-zero sampled values."""
    fn = sp.lambdify(X, expr, modules=["numpy"])
    xs = np.linspace(lo, hi, samples)
    roots = []
    try:
        ys = np.asarray(fn(xs), dtype=float)
        if ys.ndim == 0:
            return []
        ys[~np.isfinite(ys)] = np.nan
        scale = max(1.0, np.nanmax(np.abs(ys))) if np.isfinite(ys).any() else 1.0
        for i in range(len(xs)-1):
            a, b, ya, yb = xs[i], xs[i+1], ys[i], ys[i+1]
            if not np.isfinite(ya) or not np.isfinite(yb):
                continue
            if abs(ya) < 1e-7 * scale:
                roots.append(float(a))
            if ya * yb < 0:
                try:
                    roots.append(float(brentq(lambda z: float(np.real(fn(z))), a, b)))
                except Exception:
                    pass
        if np.isfinite(ys[-1]) and abs(ys[-1]) < 1e-7 * scale:
            roots.append(float(xs[-1]))
    except Exception:
        return []
    unique = []
    for r in sorted(roots):
        if lo - 1e-7 <= r <= hi + 1e-7 and (not unique or abs(r - unique[-1]) > 1e-3):
            unique.append(r)
    return unique

def classify_calculus(expr, lo, hi):
    d1 = sp.diff(expr, X)
    d2 = sp.diff(d1, X)
    critical = numeric_roots(d1, lo, hi)
    inflection_candidates = numeric_roots(d2, lo, hi)
    def val(e, z):
        return safe_numeric(e, z)
    critical_rows = []
    for c in critical:
        eps = max(1e-4, (hi-lo)*1e-5)
        left = val(d1, c-eps) if c-eps >= lo else np.nan
        right = val(d1, c+eps) if c+eps <= hi else np.nan
        if np.isfinite(left) and np.isfinite(right) and left > 0 and right < 0:
            kind = "local maximum (first-derivative test)"
        elif np.isfinite(left) and np.isfinite(right) and left < 0 and right > 0:
            kind = "local minimum (first-derivative test)"
        elif np.isfinite(left) and np.isfinite(right):
            kind = "stationary point; no extremum confirmed"
        else:
            kind = "critical candidate at interval boundary"
        critical_rows.append({"x": c, "y": val(expr, c), "classification": kind})
    infl_rows = []
    for c in inflection_candidates:
        eps = max(1e-4, (hi-lo)*1e-5)
        left = val(d2, c-eps) if c-eps >= lo else np.nan
        right = val(d2, c+eps) if c+eps <= hi else np.nan
        changes = np.isfinite(left) and np.isfinite(right) and left * right < 0
        infl_rows.append({"x": c, "y": val(expr, c), "confirmed": bool(changes),
                          "note": "concavity changes" if changes else "candidate only; no concavity change confirmed"})
    # Closed-interval absolute extrema, with endpoints included; skip undefined values.
    candidates = [(float(lo), val(expr, lo), "left endpoint"), (float(hi), val(expr, hi), "right endpoint")]
    candidates += [(r["x"], r["y"], r["classification"]) for r in critical_rows]
    candidates = [(x, y, label) for x, y, label in candidates if np.isfinite(y)]
    abs_max = max(candidates, key=lambda z: z[1]) if candidates else None
    abs_min = min(candidates, key=lambda z: z[1]) if candidates else None
    return d1, d2, critical_rows, infl_rows, abs_min, abs_max

def context_interpretation(context, xname, yname, xunit, yunit, xval, yval, slope, second):
    context_line = {
        "Business": "Read the output as a business measure; a positive first derivative means growth and a negative one means decline.",
        "Engineering": "Read the output as an engineering measurement; the derivatives describe its rate and changing rate.",
        "Medical": "This is a simplified educational model, not medical advice. Concentration trends depend on the chosen function and units.",
        "Sports science": "Read the output as a movement or performance measure; the slope describes how quickly it changes.",
        "Biology": "Read the output as a biological quantity; a positive slope indicates increase and a negative slope indicates decrease.",
        "Bioengineering": "Read the output as a device or biological-system response; check assumptions before applying it to real systems.",
        "Economic forecasting": "This is a mathematical scenario, not a verified forecast. Historical and future economic behavior may differ.",
        "Weather": "This is a simplified mathematical weather indicator, not an official forecast.",
        "Custom": "Interpret the values using the meaning and units you entered."
    }.get(context, "Interpret the values using the selected context.")
    slope_unit = f"{yunit}/{xunit}" if xunit and yunit else "output units per input unit"
    second_unit = f"{yunit}/{xunit}²" if xunit and yunit else "output units per input unit squared"
    return (context_line,
            f"At {xname} = {fmt(xval)}, {yname} ≈ {fmt(yval)} {yunit}.",
            f"First derivative ≈ {fmt(slope)} {slope_unit}: the output is changing by about this amount per one {xunit}.",
            f"Second derivative ≈ {fmt(second)} {second_unit}: this describes whether the rate of change is increasing or decreasing.")

def make_plotly_model(expr, lo, hi, current_x, interval_a, interval_b, show_f=True, show_d1=True, show_d2=True,
                      observed=None, fitted_model=None, batch_df=None, sample_step=1, xname="x", yname="y",
                      xunit="", yunit="", boundary=None):
    fig = go.Figure()
    xs = np.linspace(lo, hi, 900)
    ys = sample_function(expr, xs)
    # Break lines at undefined values, rather than connecting across discontinuities.
    ys[~np.isfinite(ys)] = np.nan
    if show_f:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Function f(x)", line=dict(width=3)))
    d1, d2 = sp.diff(expr, X), sp.diff(expr, X, 2)
    if show_d1:
        yy = sample_function(d1, xs); yy[~np.isfinite(yy)] = np.nan
        fig.add_trace(go.Scatter(x=xs, y=yy, mode="lines", name="f′(x)", line=dict(dash="dash")))
    if show_d2:
        yy = sample_function(d2, xs); yy[~np.isfinite(yy)] = np.nan
        fig.add_trace(go.Scatter(x=xs, y=yy, mode="lines", name="f″(x)", line=dict(dash="dot")))
    if np.isfinite(safe_numeric(expr, current_x)):
        cy = safe_numeric(expr, current_x)
        fig.add_trace(go.Scatter(x=[current_x], y=[cy], mode="markers", name="Current point", marker=dict(size=11, symbol="circle")))
        slope = safe_numeric(d1, current_x)
        if np.isfinite(slope):
            tx = np.array([lo, hi])
            ty = cy + slope*(tx-current_x)
            # Show tangent only near the current point so its scale remains readable.
            span = (hi-lo)*0.18
            tx = np.array([max(lo,current_x-span), min(hi,current_x+span)])
            ty = cy + slope*(tx-current_x)
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", name="Tangent line", line=dict(dash="dashdot", width=2)))
    ya, yb = safe_numeric(expr, interval_a), safe_numeric(expr, interval_b)
    if np.isfinite(ya) and np.isfinite(yb) and abs(interval_b-interval_a) > 1e-12:
        slope = (yb-ya)/(interval_b-interval_a)
        sx = np.array([interval_a, interval_b])
        sy = ya + slope*(sx-interval_a)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers", name="Secant line", line=dict(width=2, dash="longdash")))
    if observed is not None and not observed.empty:
        fig.add_trace(go.Scatter(x=observed["year"], y=observed["value"], mode="markers", name="Observed data", marker=dict(size=7, symbol="diamond")))
    if fitted_model is not None:
        fm = fitted_model
        dense_years = np.linspace(fm["min_year"], fm["max_year"], 500)
        fit_vals = eval_fit(fm, dense_years)
        fig.add_trace(go.Scatter(x=dense_years, y=fit_vals, mode="lines", name="Historical polynomial fit", line=dict(width=3)))
        fig.add_vline(x=fm["min_year"], line_dash="dot", annotation_text="Historical boundary")
        fig.add_vline(x=fm["max_year"], line_dash="dot", annotation_text="Historical boundary")
        # Separate dashed extrapolation on either side of historical coverage.
        if lo < fm["min_year"]:
            ext = np.linspace(lo, fm["min_year"], 100)
            fig.add_trace(go.Scatter(x=ext, y=eval_fit(fm, ext), mode="lines", name="Extrapolation", line=dict(dash="dash", width=2)))
        if hi > fm["max_year"]:
            ext = np.linspace(fm["max_year"], hi, 100)
            fig.add_trace(go.Scatter(x=ext, y=eval_fit(fm, ext), mode="lines", name="Extrapolation", line=dict(dash="dash", width=2)))
        if sample_step > 1:
            marker_years = np.arange(fm["min_year"], fm["max_year"]+1, sample_step)
            fig.add_trace(go.Scatter(x=marker_years, y=eval_fit(fm, marker_years), mode="markers", name=f"Model samples (step {sample_step})", marker=dict(size=6, symbol="circle-open")))
    if batch_df is not None and not batch_df.empty:
        fig.add_trace(go.Scatter(x=batch_df["year"], y=batch_df["estimate"], mode="markers", name="Batch estimates", marker=dict(size=10, symbol="x")))
    fig.update_layout(
        height=560, template="plotly_white", hovermode="x unified",
        xaxis_title=f"{xname}" + (f" ({xunit})" if xunit else ""),
        yaxis_title=f"{yname}" + (f" ({yunit})" if yunit else ""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def excel_bytes(summary, working, calculated):
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    for row in summary:
        ws.append(row)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for title, df in [("Working data", working), ("Calculated years", calculated)]:
        sh = wb.create_sheet(title)
        if df is None or df.empty:
            sh.append(["No results"])
            continue
        for row in dataframe_to_rows(df, index=False, header=True):
            sh.append(row)
        for cell in sh[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
        sh.freeze_panes = "A2"
        sh.auto_filter.ref = sh.dimensions
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()

def pdf_report(title, metadata, equation, metrics, findings, data_preview, calculated, plot_png):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch,
                            topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    story.append(Paragraph("Metadata", styles["Heading2"]))
    story.append(Paragraph("<br/>".join([f"<b>{k}:</b> {str(v)}" for k, v in metadata.items()]), styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("Equation and metrics", styles["Heading2"]),
              Paragraph(equation.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), styles["BodyText"]),
              Paragraph("<br/>".join(metrics), styles["BodyText"]), Spacer(1, 8)]
    if plot_png:
        story.append(Paragraph("Visualization", styles["Heading2"]))
        story.append(RLImage(io.BytesIO(plot_png), width=9.5*inch, height=4.1*inch))
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    story.append(Paragraph("<br/>".join(findings).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), styles["BodyText"]))
    for heading, df in [("Data preview", data_preview), ("Calculated results", calculated)]:
        story.append(Paragraph(heading, styles["Heading2"]))
        if df is not None and not df.empty:
            small = df.head(30).copy().astype(str)
            table_data = [list(small.columns)] + small.values.tolist()
            tbl = Table(table_data, repeatRows=1, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F6FA")]),
            ]))
            story.append(tbl)
            if len(df) > 30:
                story.append(Paragraph("Only the first 30 rows are shown in this PDF preview; the CSV and Excel exports contain the complete result set.", styles["Italic"]))
        else:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
    doc.build(story)
    return buf.getvalue()

# ----------------------------- UI -----------------------------
st.title("📈 Hispanic Heritage App 2.0")
st.caption("Interactive mathematical modeling, historical data, and calculus exploration. Educational use only.")

with st.sidebar:
    st.header("Classroom settings")
    course = st.selectbox("Course mode", COURSES, index=2)
    context = st.selectbox("Context", CONTEXTS)
    suggested, sug_x, sug_y, sug_yunit = SUGGESTIONS[context]
    context_description = st.text_area("Context description", value=f"Illustrative {context.lower()} model; not a validated real-world prediction.", height=75)
    xname = st.text_input("Independent variable name", value=sug_x)
    yname = st.text_input("Dependent variable name", value=sug_y)
    xunit = st.text_input("Independent variable units", value="hours" if context == "Medical" else ("years" if "Economic" in context else "units"))
    yunit = st.text_input("Dependent variable units", value=sug_yunit)
    st.divider()
    st.subheader("Function source")
    function_source = st.radio("Choose function source", [
        "Suggested function", "Build from parameters", "Enter my own function",
        "Best-fit function from historical data"
    ], index=0)

# Session state setup
defaults = {
    "current_x": 2.0, "interval_a": 0.0, "interval_b": 5.0,
    "play": False, "last_model_key": None, "historical_df": pd.DataFrame(),
    "fitted_model": None, "last_fetch_key": None, "calculated_results": None,
    "calc_signature": None, "working_data": pd.DataFrame(), "actual_single": None,
    "data_status": "No historical data loaded."
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Historical data panel
with st.expander("Historical data and polynomial curve of best fit", expanded=(function_source == "Best-fit function from historical data")):
    left, right = st.columns([1, 1])
    with left:
        country_name = st.selectbox("Country / economy", list(WB_COUNTRIES.keys()), index=1, key="wb_country")
        indicator_name = st.selectbox("Indicator", list(WB_INDICATORS.keys()), index=0, key="wb_indicator")
        start_year = st.number_input("Start year", min_value=1960, max_value=2025, value=1990, step=1)
        end_year = st.number_input("End year", min_value=1960, max_value=2025, value=2023, step=1)
        degree = st.slider("Polynomial degree", min_value=1, max_value=6, value=2)
        base_year = st.number_input("Centered-time base year", min_value=1800, max_value=2100, value=2000, step=1)
        sample_step = st.slider("Model marker / sample-table increment (years)", 1, 10, 1)
        fetch_clicked = st.button("Fetch World Bank data and fit model", type="primary")
    with right:
        st.markdown("**Data source:** World Bank Indicators API")
        st.write("The model uses centered time `t = year − base year`. At least degree + 1 distinct years are required.")
        st.write("Fit metrics below are **in-sample**. A separate chronological holdout score is shown only when enough observations exist.")
        st.warning("A polynomial can behave implausibly outside the observed years. Extrapolations are flagged, not clipped.")
    if fetch_clicked:
        if start_year > end_year:
            st.error("Start year must be less than or equal to end year.")
        else:
            fetch_key = (WB_COUNTRIES[country_name], WB_INDICATORS[indicator_name], int(start_year), int(end_year), int(degree), int(base_year))
            try:
                with st.spinner("Fetching historical observations…"):
                    fetched = world_bank_fetch(fetch_key[0], fetch_key[1], fetch_key[2], fetch_key[3])
                model = fit_polynomial(fetched, degree, base_year)
                st.session_state.historical_df = fetched
                st.session_state.fitted_model = model
                st.session_state.last_fetch_key = fetch_key
                st.session_state.working_data = fetched.copy()
                st.session_state.calculated_results = None
                st.session_state.calc_signature = None
                st.session_state.data_status = f"Loaded {len(fetched)} observations from the World Bank API."
                st.success(st.session_state.data_status)
            except requests.RequestException as exc:
                st.error(f"Could not reach the World Bank API: {exc}. Check your internet connection and the app logs, then try again.")
            except Exception as exc:
                st.error(f"Could not load or fit the data: {exc}")
    hist_df = st.session_state.historical_df
    fitted_model = st.session_state.fitted_model
    if not hist_df.empty and fitted_model is not None:
        # User-editable data table; editing is explicit and refits the same selected model.
        st.markdown("**Working data (editable).** Change year/value cells, then click Apply edited data.")
        edited = st.data_editor(hist_df, num_rows="dynamic", use_container_width=True, key="historical_editor")
        if st.button("Apply edited data and refit"):
            try:
                edited_clean = edited.copy()
                edited_clean["year"] = pd.to_numeric(edited_clean["year"], errors="coerce")
                edited_clean["value"] = pd.to_numeric(edited_clean["value"], errors="coerce")
                edited_clean = edited_clean.dropna(subset=["year", "value"])
                if not np.all(np.equal(edited_clean["year"], np.floor(edited_clean["year"]))):
                    raise ValueError("Years must be whole numbers.")
                if edited_clean["year"].duplicated().any():
                    raise ValueError("Years must be unique; remove duplicate years before fitting.")
                if len(edited_clean) < int(degree) + 1:
                    raise ValueError(f"Need at least {int(degree)+1} distinct years for degree {degree}.")
                edited_clean["year"] = edited_clean["year"].astype(int)
                new_model = fit_polynomial(edited_clean, degree, base_year)
                st.session_state.historical_df = edited_clean[["year", "value"]].sort_values("year").reset_index(drop=True)
                st.session_state.fitted_model = new_model
                st.session_state.working_data = st.session_state.historical_df.copy()
                st.session_state.calculated_results = None
                st.session_state.calc_signature = None
                st.session_state.data_status = "Edited data applied; model refit."
                st.success(st.session_state.data_status)
            except Exception as exc:
                st.error(f"Edited data was not applied: {exc}")
        if st.button("Reset to downloaded data"):
            try:
                if st.session_state.last_fetch_key:
                    cc, ii, sy, ey, dd, bb = st.session_state.last_fetch_key
                    fresh = world_bank_fetch(cc, ii, sy, ey)
                    st.session_state.historical_df = fresh
                    st.session_state.working_data = fresh.copy()
                    st.session_state.fitted_model = fit_polynomial(fresh, dd, bb)
                    st.session_state.calculated_results = None
                    st.session_state.calc_signature = None
                    st.session_state.data_status = "Reset to downloaded data."
                    st.success(st.session_state.data_status)
            except Exception as exc:
                st.error(f"Could not reset data from the API: {exc}")
        m = st.session_state.fitted_model
        st.markdown("**Polynomial model summary**")
        st.code(poly_equation(m["coeff"], m["base_year"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (in-sample)", fmt(m["r2"], 7))
        c2.metric("MAE (in-sample)", fmt(m["mae"]), help="Mean absolute residual in indicator units.")
        c3.metric("Observation count", str(m["n"]))
        c4.metric("Mean observed value", fmt(m["mean"]))
        st.write(f"Base year: **{m['base_year']}** · Degree: **{m['degree']}** · Observed range: **{m['min_year']}–{m['max_year']}**")
        st.caption("Residual = observed value − fitted value at each observed year. In-sample metrics describe fit to the same data used to estimate the coefficients; they are not a guarantee of future accuracy.")
        with st.expander("Residuals and fitted values at observed years"):
            st.dataframe(m["data"][["year", "value", "fitted", "residual"]].rename(columns={
                "value":"Actual observation", "fitted":"Fitted value at observed year", "residual":"Residual"
            }), use_container_width=True)
        holdout = chronological_holdout(st.session_state.historical_df, m["degree"], m["base_year"])
        if holdout:
            st.info(f"Chronological holdout MAE: {fmt(holdout['mae'])} indicator units on {holdout['n_test']} later observations ({holdout['first_test']}–{holdout['last_test']}). This is a separate holdout score, not an in-sample metric.")
        else:
            st.caption("Chronological holdout score not shown: there are not enough observations for a useful holdout while retaining enough training points.")
        if country_name != "World":
            st.markdown("**Compare another country using the same indicator**")
            compare_name = st.selectbox("Comparison country", [n for n in WB_COUNTRIES if n != country_name], index=0)
            if st.button("Load country comparison"):
                try:
                    compare_df = world_bank_fetch(WB_COUNTRIES[compare_name], WB_INDICATORS[indicator_name], int(start_year), int(end_year))
                    primary = st.session_state.historical_df.rename(columns={"value": country_name})
                    other = compare_df.rename(columns={"value": compare_name})
                    merged = primary.merge(other, on="year", how="outer").sort_values("year")
                    st.session_state["comparison_df"] = merged
                    st.session_state["comparison_label"] = f"{country_name} vs {compare_name} — {indicator_name}"
                except Exception as exc:
                    st.error(f"Comparison could not be loaded: {exc}")
            if "comparison_df" in st.session_state:
                comp = st.session_state["comparison_df"]
                st.caption("Gaps mean the source has no reported value for that year, or country coverage differs. Do not fill missing values silently.")
                fig_comp = go.Figure()
                for name in [country_name, st.session_state.get("comparison_label","").split(" vs ")[-1].split(" — ")[0]]:
                    if name in comp.columns:
                        fig_comp.add_trace(go.Scatter(x=comp["year"], y=comp[name], mode="lines+markers", name=name))
                fig_comp.update_layout(title=st.session_state.get("comparison_label","Country comparison"), xaxis_title="Year", yaxis_title=indicator_name, template="plotly_white")
                st.plotly_chart(fig_comp, use_container_width=True)

# Determine active function
expr = None
function_error = None
source_note = ""
if function_source == "Best-fit function from historical data":
    model = st.session_state.fitted_model
    if model is not None:
        # Map centered year variable x -> actual calendar year using the same centered transformation once.
        expr = sum(sp.Float(float(c), 14) * (X - model["base_year"])**(len(model["coeff"])-1-i)
                   for i, c in enumerate(model["coeff"]))
        source_note = f"Polynomial fit to {st.session_state.last_fetch_key[0] if st.session_state.last_fetch_key else 'selected economy'} data. Input x is calendar year; centered time is applied once as (year − {model['base_year']})."
    else:
        st.info("Fetch and fit historical data in the expander above, or choose another function source.")
elif function_source == "Suggested function":
    expr = parse_function(suggested)
    source_note = "Suggested illustrative function; coefficients are examples, not fitted real-world values."
elif function_source == "Build from parameters":
    st.subheader("Build a function from parameters")
    model_kind = st.selectbox("Model family", ["Polynomial degree 2", "Polynomial degree 3", "Exponential", "Logarithmic", "Trigonometric", "Rational"])
    pcols = st.columns(4)
    with pcols[0]: a = st.number_input("Parameter a", value=1.0, step=0.5)
    with pcols[1]: b = st.number_input("Parameter b", value=0.0, step=0.5)
    with pcols[2]: c = st.number_input("Parameter c", value=0.0, step=0.5)
    with pcols[3]: d = st.number_input("Parameter d", value=1.0, step=0.5)
    if model_kind == "Polynomial degree 2":
        source = f"{a}*x**2 + {b}*x + {c}"
    elif model_kind == "Polynomial degree 3":
        source = f"{a}*x**3 + {b}*x**2 + {c}*x + {d}"
    elif model_kind == "Exponential":
        source = f"{a}*exp({b}*x) + {c}"
    elif model_kind == "Logarithmic":
        source = f"{a}*log(x) + {b}"
    elif model_kind == "Trigonometric":
        source = f"{a}*sin({b}*x) + {c}"
    else:
        source = f"({a}*x + {b})/(x + {d}) + {c}"
    st.code(f"f(x) = {source}")
    expr = parse_function(source)
    source_note = f"Generated from parameters in the {model_kind.lower()} family."
else:
    own_function = st.text_input("Enter function f(x)", value="x**2 - 4*x + 3", help="Examples: x^2 - 4*x + 3, 100*exp(-0.18*x), 12 + 8*sin(2*pi*x/24), (x+1)/(x-2)")
    try:
        expr = parse_function(own_function)
        source_note = "User-entered function."
    except ValueError as exc:
        function_error = str(exc)

# Main analysis controls
st.divider()
st.header("Function analysis and simulation")
if function_error:
    st.error(function_error)
if expr is not None:
    st.latex(sp.latex(sp.Eq(sp.Function("f")(X), expr)))
    st.caption(source_note)
    c1, c2, c3 = st.columns(3)
    with c1:
        interval_min = st.number_input("Analysis interval minimum", value=-5.0, step=1.0, key="analysis_min")
    with c2:
        interval_max = st.number_input("Analysis interval maximum", value=10.0, step=1.0, key="analysis_max")
    with c3:
        initial_x = st.number_input("Initial current input", value=float(st.session_state.current_x), step=0.25, key="initial_x")
    if interval_min >= interval_max:
        st.error("Analysis interval minimum must be smaller than its maximum.")
        analysis_ok = False
    else:
        analysis_ok = True
    if analysis_ok:
        lo, hi = float(interval_min), float(interval_max)
        # Clamp widget defaults to current analysis interval without changing the mathematical result.
        if "point_slider" not in st.session_state:
            st.session_state.point_slider = min(max(initial_x, lo), hi)
        if "a_slider" not in st.session_state:
            st.session_state.a_slider = lo
        if "b_slider" not in st.session_state:
            st.session_state.b_slider = hi
        st.session_state.point_slider = min(max(float(st.session_state.point_slider), lo), hi)
        st.session_state.a_slider = min(max(float(st.session_state.a_slider), lo), hi)
        st.session_state.b_slider = min(max(float(st.session_state.b_slider), lo), hi)
        sliders = st.columns(3)
        with sliders[0]:
            current_x = st.slider(f"Current {xname}", min_value=float(lo), max_value=float(hi), value=float(st.session_state.point_slider), key="point_slider")
        with sliders[1]:
            interval_a = st.slider("Average-rate interval start", min_value=float(lo), max_value=float(hi), value=float(st.session_state.a_slider), key="a_slider")
        with sliders[2]:
            interval_b = st.slider("Average-rate interval end", min_value=float(lo), max_value=float(hi), value=float(st.session_state.b_slider), key="b_slider")
        st.session_state.current_x, st.session_state.interval_a, st.session_state.interval_b = current_x, interval_a, interval_b
        d1, d2 = sp.diff(expr, X), sp.diff(expr, X, 2)
        yval, slope, second = safe_numeric(expr, current_x), safe_numeric(d1, current_x), safe_numeric(d2, current_x)
        ya, yb = safe_numeric(expr, interval_a), safe_numeric(expr, interval_b)
        avg_rate = (yb-ya)/(interval_b-interval_a) if np.isfinite(ya) and np.isfinite(yb) and abs(interval_b-interval_a)>1e-12 else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{yname} at current input", fmt(yval), yunit)
        c2.metric("Instantaneous rate f′(x)", fmt(slope), f"{yunit}/{xunit}" if xunit and yunit else "")
        c3.metric("Second derivative f″(x)", fmt(second), f"{yunit}/{xunit}²" if xunit and yunit else "")
        c4.metric("Average rate of change", fmt(avg_rate), f"{yunit}/{xunit}" if xunit and yunit else "")
        if abs(interval_b-interval_a) <= 1e-12:
            st.warning("The average rate of change needs two different interval endpoints.")
        elif not np.isfinite(ya) or not np.isfinite(yb):
            st.warning("The average rate of change is undefined because an endpoint is outside the real-valued domain or at a discontinuity.")
        st.markdown("**Derivative formulas**")
        st.latex(r"f'(x) = " + sp.latex(d1))
        st.latex(r"f''(x) = " + sp.latex(d2))
        interp = context_interpretation(context, xname, yname, xunit, yunit, current_x, yval, slope, second)
        st.info(interp[0] + "\n\n" + interp[1] + "\n\n" + interp[2] + "\n\n" + interp[3])
        show_cols = st.columns(3)
        with show_cols[0]: show_f = st.checkbox("Show f(x)", True)
        with show_cols[1]: show_d1 = st.checkbox("Show f′(x)", course == "Calculus")
        with show_cols[2]: show_d2 = st.checkbox("Show f″(x)", course == "Calculus")
        # Visual context and synchronized animation
        visual_options = {
            "Walking person with prosthetic leg": "prosthetic",
            "Moving object / projectile": "projectile",
            "Tank / concentration indicator": "tank",
            "Population / cell colony": "cells",
            "Thermometer / weather indicator": "weather",
            "Business / economic indicator": "business",
        }
        visual = st.selectbox("Contextual visual simulation", list(visual_options.keys()))
        play_cols = st.columns([1,1,1,2])
        with play_cols[0]:
            if st.button("▶ Play", disabled=st.session_state.play):
                st.session_state.play = True
                st.rerun()
        with play_cols[1]:
            if st.button("⏸ Pause", disabled=not st.session_state.play):
                st.session_state.play = False
                st.rerun()
        with play_cols[2]:
            if st.button("↺ Reset"):
                st.session_state.play = False
                st.session_state.point_slider = lo
                st.session_state.current_x = lo
                st.rerun()
        with play_cols[3]:
            playback_speed = st.select_slider("Playback speed", options=[0.25,0.5,1.0,1.5,2.0], value=1.0)
        if st.session_state.play:
            st.caption("Playback is a step-based simulation: the graph point and values advance on reruns. Use Pause to stop. If your deployment does not rerun automatically, move the slider or press Play again.")
            st.session_state.point_slider = lo if current_x >= hi else min(hi, current_x + (hi-lo)/80*playback_speed)
            st.session_state.current_x = st.session_state.point_slider
            time.sleep(0.08)
            st.rerun()
        fig = make_plotly_model(expr, lo, hi, current_x, interval_a, interval_b, show_f, show_d1, show_d2,
                                observed=st.session_state.historical_df if function_source == "Best-fit function from historical data" and not st.session_state.historical_df.empty else None,
                                fitted_model=st.session_state.fitted_model if function_source == "Best-fit function from historical data" else None,
                                sample_step=sample_step if function_source == "Best-fit function from historical data" else 1,
                                xname=xname, yname=yname, xunit=xunit, yunit=yunit)
        st.plotly_chart(fig, use_container_width=True)
        # Lightweight SVG simulation that responds to current point and normalized output.
        normalized = 0.5 if not np.isfinite(yval) else float(np.clip((yval - np.nanmin(sample_function(expr, np.linspace(lo,hi,300)))) /
                            (np.nanmax(sample_function(expr, np.linspace(lo,hi,300))) - np.nanmin(sample_function(expr, np.linspace(lo,hi,300))) + 1e-12), 0, 1))
        st.markdown("**Simulation indicator**")
        st.caption(f"Visual scaling: output is normalized to 0–100% across the selected interval for display only. The graph and numeric values retain the original units. Current normalized level: {normalized*100:.1f}%.")
        if visual_options[visual] == "prosthetic":
            st.markdown(f"🚶 **Walking / prosthetic-leg concept** — step phase {((current_x-lo)/(hi-lo)*100):.0f}%; response indicator {normalized*100:.1f}%. This is a conceptual visual, not a medical gait assessment.")
        elif visual_options[visual] == "projectile":
            st.markdown(f"🚀 **Moving object concept** — input position {current_x:.3g} {xunit}; modeled output {fmt(yval)} {yunit}.")
        elif visual_options[visual] == "tank":
            st.markdown(f"🧪 **Tank / concentration concept** — fill indicator {normalized*100:.1f}%; modeled value {fmt(yval)} {yunit}.")
        elif visual_options[visual] == "cells":
            st.markdown(f"🦠 **Population / cell colony concept** — relative population indicator {normalized*100:.1f}%; modeled value {fmt(yval)} {yunit}.")
        elif visual_options[visual] == "weather":
            st.markdown(f"🌡️ **Weather indicator** — relative display level {normalized*100:.1f}%; modeled value {fmt(yval)} {yunit}. This is not an official forecast.")
        else:
            st.markdown(f"📊 **Business / economic indicator** — relative display level {normalized*100:.1f}%; modeled value {fmt(yval)} {yunit}.")
        # Course level analysis
        if course in ["Honors Algebra 2", "Precalculus"]:
            st.subheader(f"{course}: graph and function behavior")
            st.write(f"**Domain:** determined by the expression and real-valued restrictions. The analysis interval here is [{lo:g}, {hi:g}].")
            ys_all = sample_function(expr, np.linspace(lo,hi,900))
            finite_y = ys_all[np.isfinite(ys_all)]
            if finite_y.size:
                st.write(f"**Approximate range on the selected interval:** [{fmt(np.min(finite_y))}, {fmt(np.max(finite_y))}] {yunit}. This sampled range may miss narrow extrema or discontinuities.")
            st.write(f"**End behavior:** based on the function family and dominant terms. For a polynomial, the leading term controls the behavior as x → ±∞; for rational, exponential, logarithmic, and trigonometric functions, domain and asymptotes/periodicity may matter.")
            if np.isfinite(slope):
                st.write("**Trend at the current point:** increasing locally if f′(x) > 0, decreasing locally if f′(x) < 0, and momentarily flat if f′(x) = 0.")
            if np.isfinite(avg_rate):
                st.write(f"**Average rate:** over [{interval_a:g}, {interval_b:g}], output changes by about {fmt(avg_rate)} {yunit}/{xunit} per input unit.")
            if course == "Precalculus":
                st.write(f"**Degree / coefficient effects:** for a polynomial, the degree affects possible turning points and end behavior; the leading coefficient affects end behavior, and other coefficients shift/reshape the graph. Turning-point candidates are examined numerically below. Concavity is interpreted from f″(x) and changing average rates.")
        if course == "Calculus":
            st.subheader("Calculus: derivative-based analysis")
            try:
                d1n, d2n, crit, infl, abs_min, abs_max = classify_calculus(expr, lo, hi)
                st.write("**Critical points and local behavior**")
                if crit:
                    st.dataframe(pd.DataFrame(crit).rename(columns={"x":xname, "y":yname}), use_container_width=True)
                else:
                    st.write("No critical points were numerically detected on the selected interval. This does not prove none exist.")
                st.write("**Increasing / decreasing intervals (numerical sign sampling)**")
                edges = [lo] + [r["x"] for r in crit if lo < r["x"] < hi] + [hi]
                trends = []
                for aa, bb in zip(edges[:-1], edges[1:]):
                    mid = (aa+bb)/2
                    s = safe_numeric(d1, mid)
                    trends.append(f"({aa:.4g}, {bb:.4g}): {'increasing' if s>1e-8 else 'decreasing' if s<-1e-8 else 'derivative approximately zero/uncertain'}")
                st.write("; ".join(trends) if trends else "No intervals could be classified.")
                st.write("**Absolute extrema on the closed interval** (endpoints included when defined)")
                st.write(f"Minimum: {fmt(abs_min[1])} at x ≈ {fmt(abs_min[0])} ({abs_min[2]})" if abs_min else "Minimum not determined from finite candidate values.")
                st.write(f"Maximum: {fmt(abs_max[1])} at x ≈ {fmt(abs_max[0])} ({abs_max[2]})" if abs_max else "Maximum not determined from finite candidate values.")
                st.write("**Inflection-point candidates** (only confirmed when sampled concavity changes)")
                if infl:
                    st.dataframe(pd.DataFrame(infl).rename(columns={"x":xname, "y":yname}), use_container_width=True)
                else:
                    st.write("No second-derivative zeros were numerically detected in the selected interval.")
                st.write("**Domain restrictions / discontinuities**")
                denom = sp.denom(sp.together(expr))
                if denom != 1:
                    st.write(f"Denominator: {sp.latex(denom)}. Values where it is zero are excluded from the domain.")
                if expr.has(sp.log):
                    st.write("Logarithms require their arguments to be positive for real-valued outputs.")
                if expr.has(sp.sqrt):
                    st.write("Square roots require nonnegative radicands for real-valued outputs.")
                if expr.has(sp.tan):
                    st.write("Tangent is undefined where cos(x) = 0; the plotted curve is broken at non-finite samples.")
                if denom == 1 and not expr.has(sp.log, sp.sqrt, sp.tan):
                    st.write("No explicit rational denominator, logarithm, square root, or tangent restriction was detected. Check the expression for other domain limits.")
                st.caption("Critical points and inflection candidates are numerical approximations over the selected interval. A zero of f′ is not automatically an extremum; a zero of f″ is not automatically an inflection point. Endpoint and near-singularity behavior can require closer inspection.")
            except Exception as exc:
                st.warning(f"Some numerical calculus analysis could not be completed: {exc}")
        # ---------------- Single-year and batch estimates ----------------
        if function_source == "Best-fit function from historical data" and st.session_state.fitted_model is not None:
            st.divider()
            st.header("Calculate historical-model value")
            fm = st.session_state.fitted_model
            actual_years = set(st.session_state.historical_df["year"].astype(int).tolist())
            single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(min(max(round(current_x),1950),2100)), step=1, key="single_year")
            if st.button("Calculate Value", type="primary"):
                estimate = float(eval_fit(fm, [single_year])[0])
                row = {"year": int(single_year), "estimate": estimate,
                       "units": indicator_name,
                       "classification": "within observed fitted range" if fm["min_year"] <= single_year <= fm["max_year"] else "outside observed fitted range",
                       "at_observed_year": bool(single_year in actual_years)}
                st.session_state.actual_single = row
                st.session_state.calc_signature = (st.session_state.last_fetch_key, fm["degree"], fm["base_year"], tuple(fm["coeff"]), tuple(st.session_state.historical_df["year"]), tuple(st.session_state.historical_df["value"]))
                st.session_state.calculated_results = pd.DataFrame([row])
            if st.session_state.actual_single:
                row = st.session_state.actual_single
                st.write(f"**Year:** {row['year']} · **Estimate:** {fmt(row['estimate'])} ({row['units']}) · **Classification:** {row['classification']}.")
                if row["at_observed_year"]:
                    actual_row = st.session_state.historical_df.loc[st.session_state.historical_df["year"] == row["year"]]
                    st.write(f"This year has an observed data point. The fitted estimate ({fmt(row['estimate'])}) is distinct from the actual observation ({fmt(actual_row.iloc[0]['value'])}).")
                else:
                    st.write("No actual observation is available for this exact year in the current working data.")
                if row["classification"].startswith("outside"):
                    st.warning("This is a polynomial extrapolation. Its value may be implausible; it has not been clipped. Residuals and MAE describe historical fit, not prediction certainty.")
            st.subheader("Batch calculations")
            batch_mode = st.radio("Batch input type", ["Year range", "Comma-separated custom years"], horizontal=True)
            requested = []
            invalid_messages = []
            if batch_mode == "Year range":
                bcols = st.columns(3)
                with bcols[0]: batch_start = st.number_input("Range start", min_value=1950, max_value=2100, value=2000, step=1)
                with bcols[1]: batch_end = st.number_input("Range end", min_value=1950, max_value=2100, value=2020, step=1)
                with bcols[2]: batch_step = st.number_input("Range step (1–10)", min_value=1, max_value=10, value=5, step=1)
                if batch_start > batch_end:
                    invalid_messages.append("Range start must be less than or equal to range end.")
                else:
                    requested = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
            else:
                custom_years = st.text_input("Custom years", value="1950, 2000, 2020, 2050, 2100")
                for part in custom_years.split(","):
                    token = part.strip()
                    if not token:
                        continue
                    try:
                        val = int(token)
                        if not 1950 <= val <= 2100:
                            invalid_messages.append(f"{token}: year must be from 1950 through 2100.")
                        else:
                            requested.append(val)
                    except ValueError:
                        invalid_messages.append(f"{token!r}: not a whole-number year.")
            requested = list(dict.fromkeys(requested))
            if invalid_messages:
                for msg in invalid_messages: st.error(msg)
            if requested:
                st.caption("Valid requested years: " + ", ".join(map(str, requested)))
            if st.button("Calculate batch results"):
                if not requested or invalid_messages:
                    st.error("Fix the batch input errors before calculating.")
                else:
                    estimates = eval_fit(fm, requested)
                    result_rows = []
                    for yr, est in zip(requested, estimates):
                        actual_match = st.session_state.historical_df.loc[st.session_state.historical_df["year"] == yr, "value"]
                        result_rows.append({"year": int(yr), "estimate": float(est), "units": indicator_name,
                            "classification": "within observed fitted range" if fm["min_year"] <= yr <= fm["max_year"] else "outside observed fitted range",
                            "at_observed_year": bool(len(actual_match)),
                            "actual_observation": float(actual_match.iloc[0]) if len(actual_match) else np.nan})
                    st.session_state.calculated_results = pd.DataFrame(result_rows)
                    st.session_state.actual_single = None
                    st.session_state.calc_signature = (st.session_state.last_fetch_key, fm["degree"], fm["base_year"], tuple(fm["coeff"]), tuple(st.session_state.historical_df["year"]), tuple(st.session_state.historical_df["value"]))
            results = st.session_state.calculated_results
            current_signature = (st.session_state.last_fetch_key, fm["degree"], fm["base_year"], tuple(fm["coeff"]), tuple(st.session_state.historical_df["year"]), tuple(st.session_state.historical_df["value"]))
            if results is not None and st.session_state.calc_signature != current_signature:
                st.session_state.calculated_results = None
                results = None
                st.warning("Calculated results were cleared because the underlying model or data changed. Recalculate to refresh them.")
            if results is not None:
                # Single-year results use the same columns as batch for export and chart consistency.
                display_results = results.copy()
                if "actual_observation" not in display_results:
                    display_results["actual_observation"] = np.nan
                st.dataframe(display_results, use_container_width=True)
                st.caption("Baseline uncertainty explanation: residuals show how far the fitted curve differs from observed data; MAE summarizes the typical absolute in-sample error. Neither is a validated 95% prediction interval. The polynomial assumes one global polynomial shape and can be unreliable outside the historical range.")
            # Visual plot for historical fit and requested results
            fm = st.session_state.fitted_model
            chart_lo = min(float(fm["min_year"]), float(single_year if "single_year" in locals() else fm["min_year"]))
            chart_hi = max(float(fm["max_year"]), float(single_year if "single_year" in locals() else fm["max_year"]))
            if results is not None and not results.empty:
                chart_lo = min(chart_lo, float(results["year"].min()))
                chart_hi = max(chart_hi, float(results["year"].max()))
            chart_lo, chart_hi = max(1950, chart_lo), min(2100, chart_hi)
            chart_expr = sum(sp.Float(float(c), 14) * (X - fm["base_year"])**(len(fm["coeff"])-1-i) for i,c in enumerate(fm["coeff"]))
            hist_plot = make_plotly_model(chart_expr, chart_lo, chart_hi, float(single_year), float(fm["min_year"]), float(fm["max_year"]),
                show_f=True, show_d1=False, show_d2=False,
                observed=st.session_state.historical_df, fitted_model=fm,
                batch_df=results, sample_step=sample_step, xname="Year", yname=indicator_name, xunit="year", yunit=indicator_name)
            st.plotly_chart(hist_plot, use_container_width=True)
            st.subheader("Exports")
            # Prepare coherent export datasets
            export_working = fm["data"][["year", "value", "fitted", "residual"]].copy()
            export_results = results.copy() if results is not None else pd.DataFrame(columns=["year","estimate","units","classification","at_observed_year","actual_observation"])
            if st.session_state.actual_single and (results is None or len(results) == 0):
                sr = st.session_state.actual_single
                export_results = pd.DataFrame([{"year":sr["year"],"estimate":sr["estimate"],"units":sr["units"],"classification":sr["classification"],"at_observed_year":sr["at_observed_year"],"actual_observation":np.nan}])
            metadata = {
                "Country": country_name, "Indicator": indicator_name, "API indicator": WB_INDICATORS[indicator_name],
                "Source": "World Bank Indicators API", "Degree": fm["degree"], "Base year": fm["base_year"],
                "Historical range": f"{fm['min_year']}–{fm['max_year']}", "Observations": fm["n"],
                "Data status": st.session_state.data_status, "Generated": datetime.now().isoformat(timespec="seconds")
            }
            summary_rows = [["Field", "Value"]] + [[k, str(v)] for k,v in metadata.items()] + [
                ["Equation", poly_equation(fm["coeff"], fm["base_year"])],
                ["R-squared (in-sample)", str(fm["r2"])], ["MAE (in-sample)", str(fm["mae"])],
                ["Mean observed value", str(fm["mean"])],
            ]
            st.download_button("Download CSV", data=csv_bytes(export_results), file_name="calculated_year_results.csv", mime="text/csv")
            st.download_button("Download Excel (.xlsx)", data=excel_bytes(summary_rows, export_working, export_results), file_name="hispanic_heritage_model.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            # Build a static report chart with matplotlib for portable PDF export.
            fig_mpl, ax = plt.subplots(figsize=(10, 4.2))
            ax.scatter(st.session_state.historical_df["year"], st.session_state.historical_df["value"], label="Observed points")
            dense = np.linspace(fm["min_year"], fm["max_year"], 300)
            ax.plot(dense, eval_fit(fm, dense), label="Fitted curve")
            ax.axvline(fm["min_year"], linestyle=":", label="Historical boundary")
            ax.axvline(fm["max_year"], linestyle=":")
            if chart_lo < fm["min_year"]:
                xx = np.linspace(chart_lo, fm["min_year"], 100); ax.plot(xx, eval_fit(fm, xx), linestyle="--", label="Extrapolation")
            if chart_hi > fm["max_year"]:
                xx = np.linspace(fm["max_year"], chart_hi, 100); ax.plot(xx, eval_fit(fm, xx), linestyle="--")
            if results is not None and not results.empty:
                ax.scatter(results["year"], results["estimate"], marker="x", s=65, label="Calculated estimates")
            ax.set_xlabel("Year"); ax.set_ylabel(indicator_name); ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=.25)
            fig_mpl.tight_layout()
            img = io.BytesIO(); fig_mpl.savefig(img, format="png", dpi=150); plt.close(fig_mpl)
            pdf_data = pdf_report("Hispanic Heritage App 2.0 — Model Report", metadata,
                poly_equation(fm["coeff"], fm["base_year"]),
                [f"R-squared (in-sample): {fm['r2']}", f"MAE (in-sample): {fm['mae']}",
                 f"Mean observed value: {fm['mean']}", f"Observation count: {fm['n']}"],
                [f"Data status: {st.session_state.data_status}",
                 "Residuals and MAE describe historical fit; they are not 95% prediction intervals.",
                 "Polynomial extrapolation can be implausible and is not clipped.",
                 "The model does not establish historical causes or guarantee future outcomes."],
                export_working.head(30), export_results, img.getvalue())
            st.download_button("Download PDF report", data=pdf_data, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
            st.checkbox("Printer mode: simplify report panel", key="printer_mode")
            if st.session_state.get("printer_mode"):
                st.markdown("### Printable model report")
                st.write(f"**{country_name} — {indicator_name}** | Degree {fm['degree']} | Base year {fm['base_year']} | Range {fm['min_year']}–{fm['max_year']}")
                st.write(poly_equation(fm["coeff"], fm["base_year"]))
                st.write(f"In-sample R²: {fmt(fm['r2'])} · In-sample MAE: {fmt(fm['mae'])} · n = {fm['n']}")
                st.write("Limitations: residual error is not a prediction interval; extrapolations may be implausible.")
else:
    st.info("Choose a valid suggested or parameter-built function, or correct the custom expression to continue.")

st.divider()
with st.expander("How the app works, assumptions, and limitations"):
    st.markdown("""
**Functions and data flow**
1. A context suggests a function, or the student builds/enters one. The expression is parsed with SymPy using a restricted list of mathematical names.
2. SymPy computes symbolic first and second derivatives. Plotly displays the function, derivatives, selected point, tangent, and secant.
3. For historical modeling, the app requests indicator observations from the World Bank API, checks distinct years, centers time as `t = year − base year`, fits a polynomial, and calculates residuals and in-sample metrics.
4. Single-year and batch estimates use the same stored coefficients and the same centered-time transformation. The app clears stored estimates when the model/data signature changes.
5. CSV contains calculated results; Excel contains summary, working data, and calculated-year results; PDF contains metadata, equation, metrics, chart, limitations, data preview, and calculated results.

**Mathematical assumptions**
- A polynomial fit minimizes squared residuals on the selected observations. Degree is limited to 1–6 in this app.
- R² and MAE are in-sample metrics. The chronological holdout MAE is separate and is available only when enough observations remain for both training and testing.
- A single polynomial may overfit and may become unrealistic outside the observed range. Predictions are not clipped.
- The numerical calculus scanner samples the selected interval and detects roots using sign changes/near-zero values. It may miss repeated roots, tightly spaced roots, singularities, or features between samples. Reported roots and extrema may be approximations.
- No 95% prediction intervals are supplied. Residuals/MAE communicate historical fit error, not full predictive uncertainty.
- Context descriptions are educational interpretations, not validated medical, weather, economic, engineering, or biological advice. No historical cause is inferred from correlation or a curve fit.
- World Bank API availability, country coverage, definitions, and missing years vary by indicator. Missing values are not silently imputed.
""")

st.caption("For class defense: explain what each variable and unit means, why centered time is used, what residuals measure, why holdout MAE differs from in-sample MAE, and why extrapolation needs caution.")
