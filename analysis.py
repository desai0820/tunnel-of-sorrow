"""
Analyze the relationship between VIX opening level and maximum intraday SPX move.

Core question: Given the VIX open on a given day, what is the expected maximum
absolute change in SPX (from open) for that session?

Max absolute SPX move = max(|SPX_high - SPX_open|, |SPX_open - SPX_low|) / SPX_open * 100
i.e. the largest % swing from the opening print in either direction during the session.

The predictor is zero-based: (VIX_open - floor), where floor is the x-intercept of the
OLS fit, derived per subset.

Two model families:
  - Linear: slope * (VIX - floor)  [simple, interpretable]
  - Quadratic quantile: a + b*(VIX-floor) + c*(VIX-floor)^2  [conservative at high vol,
    accounts for vol-of-vol by curving upward at elevated VIX levels]
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec

DB_PATH = "market_data.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    vix = pd.read_sql("SELECT * FROM vix", conn, parse_dates=["date"])
    spx = pd.read_sql("SELECT * FROM spx", conn, parse_dates=["date"])
    conn.close()

    df = pd.merge(vix, spx, on="date", suffixes=("_vix", "_spx"))

    # Max absolute % move from SPX open (intraday)
    df["spx_up_move_pct"] = (df["high_spx"] - df["open_spx"]) / df["open_spx"] * 100
    df["spx_down_move_pct"] = (df["open_spx"] - df["low_spx"]) / df["open_spx"] * 100
    df["spx_max_abs_pct"] = df[["spx_up_move_pct", "spx_down_move_pct"]].max(axis=1)

    # Close-to-close daily return for context
    df["spx_daily_return_pct"] = df["close_spx"].pct_change() * 100

    return df


def compute_vix_floor(df):
    """Find the VIX shift that zeros the OLS intercept (x-intercept of raw fit)."""
    clean = df.dropna(subset=["open_vix", "spx_max_abs_pct"])
    X = sm.add_constant(clean["open_vix"])
    m = sm.OLS(clean["spx_max_abs_pct"], X).fit()
    floor = -m.params.iloc[0] / m.params.iloc[1]
    return round(floor, 1)


def fit_models(df, vix_floor):
    """
    Fit linear OLS + linear quantile + quadratic quantile models.
    Returns ols_model, linear quantile dict, quadratic quantile dict.
    """
    clean = df.dropna(subset=["open_vix", "spx_max_abs_pct"])
    vs = clean["open_vix"] - vix_floor
    y = clean["spx_max_abs_pct"]

    # Linear OLS (mean)
    X_lin = sm.add_constant(vs)
    ols_model = sm.OLS(y, X_lin).fit()

    print("=" * 70)
    print(f"LINEAR MODEL: (VIX Open - {vix_floor}) -> Max Abs SPX % Move")
    print("=" * 70)
    slope = ols_model.params.iloc[1]
    print(f"\n  OLS mean: ~= {slope:.4f} * (VIX - {vix_floor})")
    print(f"  R² = {ols_model.rsquared:.4f}, n = {len(clean)}")

    # Linear quantile regressions
    lin_qmodels = {}
    print("\n  Linear quantile regressions:")
    for q in [0.50, 0.75, 0.90, 0.95]:
        qr = sm.QuantReg(y, X_lin).fit(q=q)
        lin_qmodels[q] = qr
        print(f"    {int(q*100):>2}th: {qr.params.iloc[0]:.4f} + {qr.params.iloc[1]:.4f} * (VIX - {vix_floor})")

    # Quadratic quantile regressions (accounts for vol-of-vol)
    X_quad = np.column_stack([np.ones_like(vs), vs, vs**2])
    quad_qmodels = {}
    print(f"\n  Quadratic quantile regressions (vol-of-vol adjustment):")
    for q in [0.50, 0.75, 0.90, 0.95]:
        qr = sm.QuantReg(y, X_quad).fit(q=q)
        quad_qmodels[q] = qr
        contain = np.mean(y <= qr.predict(X_quad)) * 100
        v_label = f"V-{vix_floor}"
        print(f"    {int(q*100):>2}th: {qr.params[0]:.4f} + {qr.params[1]:.4f}*({v_label}) "
              f"+ {qr.params[2]:.6f}*({v_label})²  [contains {contain:.0f}%]")

    return ols_model, lin_qmodels, quad_qmodels


def structural_break_analysis(df, vix_floor):
    """Test whether the VIX->SPX relationship has changed over time."""
    print("\n" + "=" * 70)
    print("STRUCTURAL BREAK ANALYSIS")
    print("=" * 70)

    eras = {
        "1990-1999": (df["date"] < "2000-01-01"),
        "2000-2006": (df["date"] >= "2000-01-01") & (df["date"] < "2007-01-01"),
        "2007-2009 (GFC)": (df["date"] >= "2007-01-01") & (df["date"] < "2010-01-01"),
        "2010-2019": (df["date"] >= "2010-01-01") & (df["date"] < "2020-01-01"),
        "2020-present": (df["date"] >= "2020-01-01"),
    }

    for era_name, mask in eras.items():
        sub = df[mask].dropna(subset=["open_vix", "spx_max_abs_pct"])
        if len(sub) < 30:
            continue
        X = sm.add_constant(sub["open_vix"] - vix_floor)
        model = sm.OLS(sub["spx_max_abs_pct"], X).fit()
        print(f"  {era_name:25s} (n={len(sub):>4}): slope={model.params.iloc[1]:.4f}, R²={model.rsquared:.4f}")


def build_lookup_table(df, ols_model, lin_qmodels, quad_qmodels, vix_floor):
    """Build lookup table with both linear and quadratic projections."""
    bins = list(range(10, 55, 5)) + [100]
    labels = [f"{b}-{b+5}" for b in range(10, 50, 5)] + ["50+"]
    df_tmp = df.copy()
    df_tmp["bucket"] = pd.cut(df_tmp["open_vix"], bins=bins, labels=labels, right=False)

    total_days = len(df_tmp.dropna(subset=["bucket"]))
    bucket_stats = df_tmp.groupby("bucket", observed=False).agg(
        days=("spx_max_abs_pct", "count"),
    )
    bucket_stats["pct_days"] = (bucket_stats["days"] / total_days * 100).round(1)

    rows = []
    for label, lo in zip(labels, bins[:-1]):
        vs = lo - vix_floor
        x_lin = np.array([1, vs])
        x_quad = np.array([1, vs, vs**2])
        stats_row = bucket_stats.loc[label]
        rows.append({
            "VIX Bucket": label,
            "Days": int(stats_row["days"]),
            "% of Days": stats_row["pct_days"],
            "Mean (lin)": ols_model.predict(x_lin)[0],
            "75th (lin)": lin_qmodels[0.75].predict(x_lin)[0],
            "75th (quad)": quad_qmodels[0.75].predict(x_quad)[0],
            "90th (quad)": quad_qmodels[0.90].predict(x_quad)[0],
            "95th (quad)": quad_qmodels[0.95].predict(x_quad)[0],
        })
    return pd.DataFrame(rows)


def plot_analysis(df, ols_model, lin_qmodels, quad_qmodels, vix_floor,
                  title_suffix="", filename="vix_spx_analysis.png"):
    """Generate a multi-panel figure with quadratic quantile bands."""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1, 0.7, 1])

    vix_range = np.linspace(9, 85, 200)
    vs_range = vix_range - vix_floor

    # --- Panel 1: Scatter + regression lines (linear + quadratic quantiles) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df["open_vix"], df["spx_max_abs_pct"], alpha=0.08, s=3, c="steelblue")

    # Linear OLS mean
    X_lin = sm.add_constant(vs_range)
    ax1.plot(vix_range, ols_model.predict(X_lin), "r-", lw=2, label="OLS mean (linear)")

    # Quadratic quantile bands
    X_quad = np.column_stack([np.ones_like(vs_range), vs_range, vs_range**2])
    qcolors = {0.50: "#2ecc71", 0.75: "#f39c12", 0.90: "#e74c3c", 0.95: "#8e44ad"}
    for q, color in qcolors.items():
        pred = quad_qmodels[q].predict(X_quad)
        ax1.plot(vix_range, pred, "--", color=color, lw=1.5,
                 label=f"{int(q*100)}th pctile (quad)")

    ax1.set_xlabel("VIX Open", fontsize=11)
    ax1.set_ylabel("Max Abs SPX % Move from Open", fontsize=11)
    ax1.set_title(f"VIX Open -> Max Intraday SPX Move{title_suffix}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_xlim(8, 85)
    ax1.set_ylim(0, 12)

    slope = ols_model.params.iloc[1]
    r2 = ols_model.rsquared
    q75 = quad_qmodels[0.75]
    ax1.text(0.97, 0.05,
             f"Linear: {slope:.4f} * (VIX - {vix_floor}), R²={r2:.3f}\n"
             f"Quad 75th: {q75.params[0]:.3f} + {q75.params[1]:.4f}*(V-{vix_floor}) "
             f"+ {q75.params[2]:.5f}*(V-{vix_floor})²",
             transform=ax1.transAxes, fontsize=7.5, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Panel 2: Lookup table ---
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")
    lookup = build_lookup_table(df, ols_model, lin_qmodels, quad_qmodels, vix_floor)
    table_data = []
    for _, row in lookup.iterrows():
        table_data.append([
            row["VIX Bucket"],
            f"{int(row['Days']):,}",
            f"{row['% of Days']:.1f}%",
            f"{row['Mean (lin)']:.2f}%",
            f"{row['75th (lin)']:.2f}%",
            f"{row['75th (quad)']:.2f}%",
            f"{row['90th (quad)']:.2f}%",
            f"{row['95th (quad)']:.2f}%",
        ])
    col_labels = ["VIX", "Days", "% Days", "Mean", "75th lin", "75th quad", "90th quad", "95th quad"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    cmap_table = plt.cm.YlOrRd(np.linspace(0.05, 0.45, len(table_data)))
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(cmap_table[i])
    ax_table.set_title(
        f"Projected Max Abs SPX % Move from Open\n(VIX floor = {vix_floor}, quad models account for vol-of-vol)",
        fontsize=11, fontweight="bold", pad=20,
    )

    # --- Panel 3: VIX time series ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.fill_between(df["date"], 0, df["open_vix"], alpha=0.4, color="steelblue")
    ax3.plot(df["date"], df["open_vix"], lw=0.4, color="navy")
    ax3.set_ylabel("VIX Open", fontsize=11)
    date_lo = df["date"].min().strftime("%Y")
    date_hi = df["date"].max().strftime("%Y")
    ax3.set_title(f"VIX Open Level ({date_lo} - {date_hi}){title_suffix}", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 90)
    for level, label, color in [(15, "15", "#2ecc71"), (20, "20", "#f39c12"),
                                 (30, "30", "#e74c3c"), (40, "40", "#8e44ad")]:
        ax3.axhline(y=level, color=color, ls=":", lw=1, alpha=0.7)
        ax3.text(df["date"].iloc[-1], level + 1, label, fontsize=8, color=color, va="bottom")

    # --- Panel 4: Rolling regression slope ---
    ax4 = fig.add_subplot(gs[2, 0])
    window = 504
    rolling_slopes = []
    rolling_dates = []
    for i in range(window, len(df)):
        chunk = df.iloc[i - window:i]
        clean = chunk.dropna(subset=["open_vix", "spx_max_abs_pct"])
        if len(clean) < 50:
            continue
        s, _, _, _, _ = stats.linregress(clean["open_vix"] - vix_floor, clean["spx_max_abs_pct"])
        rolling_slopes.append(s)
        rolling_dates.append(df.iloc[i]["date"])
    ax4.plot(rolling_dates, rolling_slopes, lw=1, color="navy")
    ax4.axhline(y=ols_model.params.iloc[1], color="red", ls="--", lw=1, label="Full-sample slope")
    ax4.set_ylabel("Regression Slope", fontsize=11)
    ax4.set_title(f"Rolling {window}-day Slope Stability", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)

    # --- Panel 5: Scatter by decade ---
    ax5 = fig.add_subplot(gs[2, 1])
    eras = {
        "1990s": (df["date"] < "2000-01-01", "#e74c3c"),
        "2000s": ((df["date"] >= "2000-01-01") & (df["date"] < "2010-01-01"), "#3498db"),
        "2010s": ((df["date"] >= "2010-01-01") & (df["date"] < "2020-01-01"), "#2ecc71"),
        "2020s": (df["date"] >= "2020-01-01", "#9b59b6"),
    }
    for era_name, (mask, color) in eras.items():
        sub = df[mask].dropna(subset=["open_vix", "spx_max_abs_pct"])
        ax5.scatter(sub["open_vix"], sub["spx_max_abs_pct"], alpha=0.12, s=5,
                    color=color, label=era_name)
    ax5.plot(vix_range, ols_model.predict(X_lin), "k-", lw=2, label="Linear OLS")
    ax5.plot(vix_range, quad_qmodels[0.75].predict(X_quad), "k--", lw=2, label="Quad 75th")
    ax5.set_xlabel("VIX Open", fontsize=11)
    ax5.set_ylabel("Max Abs SPX % Move", fontsize=11)
    ax5.set_title("Scatter by Decade", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.set_xlim(8, 85)
    ax5.set_ylim(0, 12)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {filename}")
    plt.close()


def run_subset(df, label, start_date, filename):
    """Run full analysis on a date-filtered subset."""
    sub = df[df["date"] >= start_date].copy()
    n = len(sub)
    date_lo = sub["date"].min().strftime("%Y-%m-%d")
    date_hi = sub["date"].max().strftime("%Y-%m-%d")

    print(f"\n{'#' * 70}")
    print(f"# {label}: {date_lo} to {date_hi} (n={n})")
    print(f"{'#' * 70}")

    vix_floor = compute_vix_floor(sub)
    ols_model, lin_qmodels, quad_qmodels = fit_models(sub, vix_floor)
    structural_break_analysis(sub, vix_floor)

    print(f"\n--- Lookup Table: {label} ---")
    lookup = build_lookup_table(sub, ols_model, lin_qmodels, quad_qmodels, vix_floor)
    print(lookup.to_string(index=False, float_format="%.3f"))

    plot_analysis(sub, ols_model, lin_qmodels, quad_qmodels, vix_floor,
                  title_suffix=f" ({label})", filename=filename)

    return vix_floor, ols_model, lin_qmodels, quad_qmodels


def main():
    df = load_data()

    subsets = [
        ("Full Sample (1990+)", "1990-01-01", "vix_spx_analysis.png"),
        ("2000+", "2000-01-01", "vix_spx_analysis_2000.png"),
        ("Post Dot-Com (2004+)", "2004-01-01", "vix_spx_analysis_2004.png"),
    ]

    for label, start_date, filename in subsets:
        run_subset(df, label, start_date, filename)


if __name__ == "__main__":
    main()
