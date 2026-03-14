"""
Interactive Tunnel of Sorrow visualization dashboard.

Overlay projected expected-move bands on SPX price charts to visually validate
and iterate on formula coefficients.

Usage: streamlit run app.py
"""

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from formulas import (
    PRESETS,
    breach_rate_by_vix_bucket,
    compute_bounds_series,
    compute_move_pct,
    evaluate_containment,
)

DB_PATH = "market_data.db"
DATA_DB_PATH = "data/market_data.db"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def load_daily_data():
    """Load merged VIX+SPX daily data from SQLite."""
    # Try project root first, then data/ subdirectory
    import os

    db = DB_PATH if os.path.exists(DB_PATH) else DATA_DB_PATH
    conn = sqlite3.connect(db)
    vix = pd.read_sql("SELECT * FROM vix", conn, parse_dates=["date"])
    spx = pd.read_sql("SELECT * FROM spx", conn, parse_dates=["date"])
    conn.close()
    df = pd.merge(vix, spx, on="date", suffixes=("_vix", "_spx"))
    df["spx_up_move_pct"] = (df["high_spx"] - df["open_spx"]) / df["open_spx"] * 100
    df["spx_down_move_pct"] = (df["open_spx"] - df["low_spx"]) / df["open_spx"] * 100
    df["spx_max_abs_pct"] = df[["spx_up_move_pct", "spx_down_move_pct"]].max(axis=1)
    return df


@st.cache_data(ttl=3600)
def load_intraday_data():
    """Load SPX 5-min intraday data from SQLite."""
    import os

    db = DB_PATH if os.path.exists(DB_PATH) else DATA_DB_PATH
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql("SELECT * FROM spx_intraday", conn, parse_dates=["datetime"])
    except Exception:
        conn.close()
        return None
    conn.close()
    if df.empty:
        return None
    df["date"] = df["datetime"].dt.date
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar_controls():
    """Render sidebar controls and return current formula parameters."""
    st.sidebar.header("Formula Controls")

    preset_name = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Coefficients")

    a = st.sidebar.number_input(
        "Constant (a)", value=preset["a"], format="%.4f", step=0.01
    )
    b = st.sidebar.number_input(
        "Linear coeff (b)", value=preset["b"], format="%.4f", step=0.001
    )
    c = st.sidebar.number_input(
        "Quadratic coeff (c)", value=preset["c"], format="%.6f", step=0.0001
    )
    floor = st.sidebar.number_input(
        "VIX floor", value=preset["floor"], format="%.1f", step=0.5
    )
    offset = st.sidebar.number_input(
        "Constant offset", value=preset["offset"], format="%.4f", step=0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Display")

    chart_mode = st.sidebar.radio("Chart Mode", ["Daily Backtest", "Intraday (60d)"])
    show_breaches = st.sidebar.checkbox("Show breach markers", value=True)

    return {
        "preset_name": preset_name,
        "a": a,
        "b": b,
        "c": c,
        "floor": floor,
        "offset": offset,
        "chart_mode": chart_mode,
        "show_breaches": show_breaches,
    }


# ---------------------------------------------------------------------------
# Statistics bar
# ---------------------------------------------------------------------------


def render_stats(stats):
    st.markdown(
        f"**Containment:** {stats['containment_rate']:.1f}% · "
        f"**Breaches:** {stats['breach_count']:,} / {stats['total_days']:,} days · "
        f"**Mean Overshoot:** {stats['mean_breach_overshoot']:.3f}%"
    )


# ---------------------------------------------------------------------------
# Breach breakdown chart
# ---------------------------------------------------------------------------


def _breach_color(rate):
    """Return a CSS color for a breach rate, green→yellow→red gradient."""
    if rate <= 15:
        return "#2ecc71"
    elif rate <= 30:
        # Interpolate green → orange
        t = (rate - 15) / 15
        r = int(46 + (243 - 46) * t)
        g = int(204 + (156 - 204) * t)
        b = int(113 + (18 - 113) * t)
        return f"rgb({r},{g},{b})"
    else:
        # Interpolate orange → red
        t = min((rate - 30) / 20, 1.0)
        r = int(243 + (231 - 243) * t)
        g = int(156 + (76 - 156) * t)
        b = int(18 + (60 - 18) * t)
        return f"rgb({r},{g},{b})"


def render_breach_breakdown(df, params):
    bucket_df = breach_rate_by_vix_bucket(
        df, params["a"], params["b"], params["c"], params["floor"], params["offset"]
    )
    bucket_df = bucket_df[bucket_df["total"] > 0]

    cells = []
    for _, row in bucket_df.iterrows():
        bg = _breach_color(row["breach_rate"])
        cells.append(
            f'<td style="background:{bg};color:#fff;text-align:center;padding:4px 6px;font-size:12px;font-weight:600">'
            f'{row["breach_rate"]:.0f}%</td>'
        )
    headers = "".join(
        f'<th style="text-align:center;padding:2px 6px;font-size:11px;font-weight:400;opacity:0.7">'
        f'{row["bucket"]}</th>'
        for _, row in bucket_df.iterrows()
    )
    cells_html = "".join(cells)
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:24px">'
        f'<tr>{headers}</tr><tr>{cells_html}</tr></table>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Daily backtest chart
# ---------------------------------------------------------------------------


def render_daily_chart(df, params, stats, chart_window):
    bounded = compute_bounds_series(
        df, params["a"], params["b"], params["c"], params["floor"], params["offset"]
    )

    # Filter to the selected chart window
    visible = bounded[
        (bounded["date"] >= pd.Timestamp(chart_window[0]))
        & (bounded["date"] <= pd.Timestamp(chart_window[1]))
    ]
    if visible.empty:
        st.warning("No data in selected chart window.")
        return

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=visible["date"],
            open=visible["open_spx"],
            high=visible["high_spx"],
            low=visible["low_spx"],
            close=visible["close_spx"],
            name="SPX",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Upper bound
    fig.add_trace(
        go.Scatter(
            x=visible["date"],
            y=visible["upper_bound"],
            mode="lines",
            name="Upper Bound",
            line=dict(color="red", width=1, dash="dash"),
        )
    )

    # Lower bound
    fig.add_trace(
        go.Scatter(
            x=visible["date"],
            y=visible["lower_bound"],
            mode="lines",
            name="Lower Bound",
            line=dict(color="green", width=1, dash="dash"),
        )
    )

    # Breach markers
    if params["show_breaches"] and not stats["breach_df"].empty:
        breach = stats["breach_df"]
        # Filter breaches to visible window
        breach = breach[
            (breach["date"] >= pd.Timestamp(chart_window[0]))
            & (breach["date"] <= pd.Timestamp(chart_window[1]))
        ]
        breach_high = breach[breach["high_spx"] > breach["upper_bound"]]
        breach_low = breach[breach["low_spx"] < breach["lower_bound"]]

        if not breach_high.empty:
            fig.add_trace(
                go.Scatter(
                    x=breach_high["date"],
                    y=breach_high["high_spx"],
                    mode="markers",
                    name="Breach (high)",
                    marker=dict(color="red", size=6, symbol="triangle-up"),
                )
            )
        if not breach_low.empty:
            fig.add_trace(
                go.Scatter(
                    x=breach_low["date"],
                    y=breach_low["low_spx"],
                    mode="markers",
                    name="Breach (low)",
                    marker=dict(color="blue", size=6, symbol="triangle-down"),
                )
            )

    # TOS-style header with latest day's values
    latest = visible.iloc[-1]
    move_pct = latest["move_pct"]
    header_text = (
        f"SPX EM: {move_pct / 100:.4f} | "
        f"Top: {latest['upper_bound']:,.2f} | "
        f"Bottom: {latest['lower_bound']:,.2f}"
    )

    fig.update_layout(
        title=header_text,
        xaxis_title="Date",
        yaxis_title="SPX",
        height=600,
        margin=dict(t=40, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Intraday chart
# ---------------------------------------------------------------------------


def render_intraday_chart(intraday_df, daily_df, params):
    if intraday_df is None or intraday_df.empty:
        st.warning(
            "No intraday data available. Run `cd data && python fetch_data.py` to fetch."
        )
        return

    # Get VIX opens for the intraday dates
    daily_sub = daily_df[["date", "open_vix", "open_spx"]].copy()
    daily_sub["date_key"] = daily_sub["date"].dt.date
    intraday_df = intraday_df.copy()
    intraday_df["date_key"] = intraday_df["date"]

    merged = intraday_df.merge(
        daily_sub[["date_key", "open_vix", "open_spx"]],
        on="date_key",
        how="inner",
        suffixes=("", "_daily"),
    )

    if merged.empty:
        st.warning("No overlapping dates between intraday and daily data.")
        return

    # Compute bounds per day using daily VIX open and SPX open
    move_pct = compute_move_pct(
        merged["open_vix"].values,
        params["a"],
        params["b"],
        params["c"],
        params["floor"],
        params["offset"],
    )
    move_decimal = move_pct / 100.0
    merged["upper_bound"] = merged["open_spx"] * (1 + move_decimal)
    merged["lower_bound"] = merged["open_spx"] * (1 - move_decimal)

    fig = go.Figure()

    # Candlestick for 5-min bars
    fig.add_trace(
        go.Candlestick(
            x=merged["datetime"],
            open=merged["open"],
            high=merged["high"],
            low=merged["low"],
            close=merged["close"],
            name="SPX 5m",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Add per-day bound lines as horizontal segments
    unique_dates = sorted(merged["date_key"].unique())
    for d in unique_dates:
        day_data = merged[merged["date_key"] == d]
        if day_data.empty:
            continue
        upper = day_data["upper_bound"].iloc[0]
        lower = day_data["lower_bound"].iloc[0]
        x0 = day_data["datetime"].iloc[0]
        x1 = day_data["datetime"].iloc[-1]

        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=upper, y1=upper,
            line=dict(color="red", width=1.5, dash="dash"),
        )
        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=lower, y1=lower,
            line=dict(color="green", width=1.5, dash="dash"),
        )

    # TOS-style header with latest day's values
    latest_date = unique_dates[-1]
    latest_day = merged[merged["date_key"] == latest_date]
    upper_val = latest_day["upper_bound"].iloc[0]
    lower_val = latest_day["lower_bound"].iloc[0]
    move_val = move_pct[merged["date_key"] == latest_date].mean()

    header_text = (
        f"SPX EM: {move_val / 100:.4f} | "
        f"Top: {upper_val:,.2f} | "
        f"Bottom: {lower_val:,.2f}"
    )

    # Hide overnight/weekend gaps
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # weekends
            dict(bounds=[20, 4], pattern="hour"),  # overnight (8pm - 4am)
        ]
    )

    fig.update_layout(
        title=header_text,
        xaxis_title="Time",
        yaxis_title="SPX",
        height=600,
        margin=dict(t=40, b=40, l=60, r=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Formula curve (VIX vs max-move scatter)
# ---------------------------------------------------------------------------


def render_formula_curve(df, params):
    fig = go.Figure()

    # Scatter of actual data
    fig.add_trace(
        go.Scatter(
            x=df["open_vix"],
            y=df["spx_max_abs_pct"],
            mode="markers",
            name="Actual",
            marker=dict(color="steelblue", size=3, opacity=0.15),
        )
    )

    # Formula curve
    vix_range = np.linspace(9, 85, 200)
    move_range = compute_move_pct(
        vix_range, params["a"], params["b"], params["c"], params["floor"], params["offset"]
    )

    fig.add_trace(
        go.Scatter(
            x=vix_range,
            y=move_range,
            mode="lines",
            name=f"Formula ({params['preset_name']})",
            line=dict(color="#e74c3c", width=2.5),
        )
    )

    # Add reference presets as lighter lines
    for name, preset in PRESETS.items():
        if name == params["preset_name"]:
            continue
        ref_move = compute_move_pct(
            vix_range, preset["a"], preset["b"], preset["c"], preset["floor"], preset["offset"]
        )
        fig.add_trace(
            go.Scatter(
                x=vix_range,
                y=ref_move,
                mode="lines",
                name=name,
                line=dict(width=1, dash="dot"),
                opacity=0.4,
            )
        )

    formula_str = (
        f"move% = {params['a']:.4f} + {params['b']:.4f}*(VIX-{params['floor']:.1f})"
    )
    if params["c"] != 0:
        formula_str += f" + {params['c']:.6f}*(VIX-{params['floor']:.1f})²"
    if params["offset"] != 0:
        formula_str += f" + {params['offset']:.4f}"

    fig.update_layout(
        title=f"VIX Open → Max Intraday SPX Move | {formula_str}",
        xaxis_title="VIX Open",
        yaxis_title="Max Abs SPX % Move from Open",
        xaxis_range=[8, 85],
        yaxis_range=[0, 12],
        height=500,
        margin=dict(t=50, b=40, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Tunnel of Sorrow",
        page_icon="📉",
        layout="wide",
    )

    st.markdown("#### Tunnel of Sorrow — Expected Move Bands")
    st.markdown(
        '<div style="font-size:12px;opacity:0.8;line-height:1.6;margin-bottom:8px">'
        "<b>Formula:</b> move% = a + b × (VIX − floor) + c × (VIX − floor)² + offset &nbsp;·&nbsp; "
        "Bounds: SPX_open × (1 ± move%/100)"
        '<table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:11px">'
        "<tr>"
        '<th style="text-align:left;padding:2px 8px;border-bottom:1px solid #555">Parameter</th>'
        '<th style="text-align:left;padding:2px 8px;border-bottom:1px solid #555">What it controls</th>'
        '<th style="text-align:left;padding:2px 8px;border-bottom:1px solid #555">Example effect (SPX ~6800, VIX ~20)</th>'
        "</tr><tr>"
        '<td style="padding:2px 8px"><b>Constant (a)</b></td>'
        '<td style="padding:2px 8px">Baseline move % regardless of VIX level</td>'
        '<td style="padding:2px 8px">+0.1 → bands widen ~6.8 pts each side</td>'
        "</tr><tr>"
        '<td style="padding:2px 8px"><b>Linear coeff (b)</b></td>'
        '<td style="padding:2px 8px">Move % added per VIX point above the floor</td>'
        '<td style="padding:2px 8px">+0.01 → ~1 pt more per VIX point above floor</td>'
        "</tr><tr>"
        '<td style="padding:2px 8px"><b>Quadratic coeff (c)</b></td>'
        '<td style="padding:2px 8px">Extra widening at high VIX (vol-of-vol curve)</td>'
        '<td style="padding:2px 8px">Mostly felt at VIX 30+; widens tail bands</td>'
        "</tr><tr>"
        '<td style="padding:2px 8px"><b>VIX floor</b></td>'
        '<td style="padding:2px 8px">VIX level where expected move ≈ 0%</td>'
        '<td style="padding:2px 8px">Raising 5→6 narrows all bands slightly</td>'
        "</tr><tr>"
        '<td style="padding:2px 8px"><b>Constant offset</b></td>'
        '<td style="padding:2px 8px">Flat addition to move % (uniform band shift)</td>'
        '<td style="padding:2px 8px">+0.01 ≈ 0.7 SPX pts; +1.0 ≈ 68 pts wider</td>'
        "</tr></table></div>",
        unsafe_allow_html=True,
    )

    params = sidebar_controls()

    # Date range filter in sidebar
    daily_df = load_daily_data()
    min_date = daily_df["date"].min().to_pydatetime().date()
    max_date = daily_df["date"].max().to_pydatetime().date()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Date Range (Daily Backtest)")
    date_range = st.sidebar.date_input(
        "Date range",
        value=(datetime(2004, 1, 1).date(), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Apply date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = daily_df[
            (daily_df["date"].dt.date >= start_date)
            & (daily_df["date"].dt.date <= end_date)
        ]
    else:
        filtered_df = daily_df

    if filtered_df.empty:
        st.error("No data in selected date range.")
        return

    # Compute containment stats
    containment = evaluate_containment(
        filtered_df,
        params["a"],
        params["b"],
        params["c"],
        params["floor"],
        params["offset"],
    )

    # 1. Statistics bar
    render_stats(containment)

    # 2. Breach breakdown
    render_breach_breakdown(filtered_df, params)

    # 3. SPX chart (daily or intraday)
    if params["chart_mode"] == "Daily Backtest":
        # Quick-select buttons for chart window
        btn_cols = st.columns(6)
        periods = {
            "1M": timedelta(days=30),
            "3M": timedelta(days=90),
            "6M": timedelta(days=180),
            "1Y": timedelta(days=365),
            "5Y": timedelta(days=5 * 365),
            "All": None,
        }
        for i, (label, delta) in enumerate(periods.items()):
            if btn_cols[i].button(label, use_container_width=True):
                if delta is None:
                    st.session_state["chart_start"] = start_date
                else:
                    st.session_state["chart_start"] = max(
                        start_date, end_date - delta
                    )
                st.session_state["chart_end"] = end_date

        # Default to last 60 trading days
        chart_start = st.session_state.get("chart_start", max(start_date, end_date - timedelta(days=90)))
        chart_end = st.session_state.get("chart_end", end_date)

        render_daily_chart(filtered_df, params, containment, (chart_start, chart_end))
    else:
        intraday_df = load_intraday_data()
        render_intraday_chart(intraday_df, daily_df, params)

    # 4. Formula curve
    render_formula_curve(filtered_df, params)


if __name__ == "__main__":
    main()
