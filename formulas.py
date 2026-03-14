"""
Formula engine for Tunnel of Sorrow expected-move bands.

Provides preset formula configurations and vectorized computation of
upper/lower SPX bounds from VIX open levels.
"""

import numpy as np
import pandas as pd


# Each preset: (display_name, a, b, c, floor, offset)
# Formula: move% = a + b*(VIX - floor) + c*(VIX - floor)^2 + offset
PRESETS = {
    "Quad 75th (2004+)": {
        "a": 0.1953,
        "b": 0.0659,
        "c": 0.000440,
        "floor": 5.0,
        "offset": 0.0,
    },
    "Quad 90th (2004+)": {
        "a": 0.2947,
        "b": 0.0874,
        "c": 0.000531,
        "floor": 5.0,
        "offset": 0.0,
    },
    "Quad 95th (2004+)": {
        "a": 0.4042,
        "b": 0.0970,
        "c": 0.000806,
        "floor": 5.0,
        "offset": 0.0,
    },
    "Linear Mean (2004+)": {
        "a": 0.0,
        "b": 0.0700,
        "c": 0.0,
        "floor": 5.0,
        "offset": 0.0,
    },
    "Linear 75th (2004+)": {
        "a": 0.0,
        "b": 0.0900,
        "c": 0.0,
        "floor": 5.0,
        "offset": 0.0,
    },
    "TOS Original": {
        "a": 0.37,
        "b": 0.0609,
        "c": 0.0,
        "floor": 8.0,
        "offset": 0.0,
    },
}


def compute_move_pct(vix_open, a, b, c, floor, offset):
    """Compute expected move % from VIX open (vectorized).

    move% = a + b*(VIX - floor) + c*(VIX - floor)^2 + offset
    Returns array of move percentages (e.g. 1.28 means 1.28%).
    """
    v = np.maximum(vix_open - floor, 0.0)
    return a + b * v + c * v**2 + offset


def compute_bounds_series(df, a, b, c, floor, offset):
    """Add upper_bound and lower_bound columns to a dataframe.

    Expects columns: open_vix, open_spx.
    Returns a copy with added columns: move_pct, upper_bound, lower_bound.
    """
    out = df.copy()
    move_pct = compute_move_pct(out["open_vix"].values, a, b, c, floor, offset)
    move_decimal = move_pct / 100.0
    out["move_pct"] = move_pct
    out["upper_bound"] = out["open_spx"] * (1 + move_decimal)
    out["lower_bound"] = out["open_spx"] * (1 - move_decimal)
    return out


def evaluate_containment(df, a, b, c, floor, offset):
    """Evaluate how well a formula contains actual SPX moves.

    Expects columns: open_vix, open_spx, high_spx, low_spx, open_spx.
    Returns dict with: containment_rate, breach_count, total_days,
    mean_breach_overshoot, breach_df (DataFrame of breach days).
    """
    bounded = compute_bounds_series(df, a, b, c, floor, offset)

    # A breach occurs when high exceeds upper bound OR low goes below lower bound
    breach_high = bounded["high_spx"] > bounded["upper_bound"]
    breach_low = bounded["low_spx"] < bounded["lower_bound"]
    breached = breach_high | breach_low

    breach_df = bounded[breached].copy()

    # Compute overshoot: how far beyond the bound (as % of open)
    high_overshoot = np.maximum(
        (bounded["high_spx"] - bounded["upper_bound"]) / bounded["open_spx"] * 100, 0
    )
    low_overshoot = np.maximum(
        (bounded["lower_bound"] - bounded["low_spx"]) / bounded["open_spx"] * 100, 0
    )
    breach_df = breach_df.copy()
    breach_df["overshoot_pct"] = np.maximum(
        high_overshoot[breached].values, low_overshoot[breached].values
    )

    total = len(bounded)
    breach_count = int(breached.sum())
    containment_rate = (total - breach_count) / total * 100 if total > 0 else 0.0
    mean_overshoot = (
        float(breach_df["overshoot_pct"].mean()) if breach_count > 0 else 0.0
    )

    return {
        "containment_rate": containment_rate,
        "breach_count": breach_count,
        "total_days": total,
        "mean_breach_overshoot": mean_overshoot,
        "breach_df": breach_df,
    }


def breach_rate_by_vix_bucket(df, a, b, c, floor, offset):
    """Compute breach rate by VIX bucket for the breakdown chart.

    Returns DataFrame with columns: bucket, total, breaches, breach_rate.
    """
    bounded = compute_bounds_series(df, a, b, c, floor, offset)
    breach_high = bounded["high_spx"] > bounded["upper_bound"]
    breach_low = bounded["low_spx"] < bounded["lower_bound"]
    bounded["breached"] = breach_high | breach_low

    bins = list(range(10, 55, 5)) + [100]
    labels = [f"{b}-{b+5}" for b in range(10, 50, 5)] + ["50+"]
    bounded["bucket"] = pd.cut(
        bounded["open_vix"], bins=bins, labels=labels, right=False
    )

    grouped = bounded.groupby("bucket", observed=False).agg(
        total=("breached", "count"),
        breaches=("breached", "sum"),
    )
    grouped["breach_rate"] = (grouped["breaches"] / grouped["total"] * 100).fillna(0)
    return grouped.reset_index()
