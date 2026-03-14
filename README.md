# Tunnel of Sorrow

Quantitative analysis of the relationship between VIX opening level and maximum intraday SPX move. Given the VIX at the open, how far could SPX move during the session?

Produces expected-move band formulas suitable for day trading — a data-driven replacement for the classic ThinkOrSwim "Tunnel of Sorrow" script.

## Key Result

Quadratic 75th percentile model (2004–2026, 5,583 trading days):

```
move% = 0.1953 + 0.0659 × (VIX - 5) + 0.000440 × (VIX - 5)²
```

Contains 75% of observed daily max moves. For price levels:

```
move  = 0.001953 + 0.000659 × (VIX - 5) + 0.00000440 × (VIX - 5)²
top   = SPX_open × (1 + move)
bottom = SPX_open × (1 - move)
```

Quick reference (SPX ~5700):

| VIX | Expected Move (75th) | Band |
|-----|---------------------|------|
| 12 | 0.54% | 5669 – 5731 |
| 15 | 0.90% | 5649 – 5751 |
| 20 | 1.28% | 5627 – 5773 |
| 25 | 1.69% | 5604 – 5796 |
| 30 | 2.12% | 5579 – 5821 |
| 40 | 3.04% | 5527 – 5873 |
| 50 | 4.05% | 5469 – 5931 |

Mental math: **move% ≈ 0.07 × (VIX - 5)**. Every 1 point of VIX above 5 adds ~7 bps. Add 20–30% buffer above VIX 30 for vol-of-vol.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Fetch VIX and SPX daily OHLC + intraday data into SQLite
cd data && python fetch_data.py && cd ..

# Run analysis (prints model stats, generates charts)
python analysis.py

# Launch interactive visualization dashboard
streamlit run app.py
```

## Interactive Dashboard

The Streamlit dashboard (`app.py`) lets you visually validate and iterate on expected-move formulas against real SPX price action.

```bash
streamlit run app.py
```

### Sidebar Controls

- **Preset dropdown** — Select from built-in formulas: Quad 75th/90th/95th, Linear Mean/75th, TOS Original. Selecting a preset auto-fills all coefficient fields.
- **Coefficient inputs** — Manually tune the constant (a), linear coeff (b), quadratic coeff (c), VIX floor, and constant offset. The formula is: `move% = a + b*(VIX - floor) + c*(VIX - floor)² + offset`
- **Date range** — Filter the daily backtest to any date window (default: 2004–present).
- **Chart mode** — Toggle between:
  - **Daily Backtest**: Full history candlestick chart with bound overlay. Good for backtesting containment across thousands of days.
  - **Intraday (60d)**: 5-minute candles for the last ~60 trading days with per-day bound lines. Mimics the TOS chart experience.
- **Show breach markers** — Toggle red/blue triangle markers on days where price exceeded the projected bounds.

### Dashboard Panels (top to bottom)

1. **Statistics bar** — Containment rate, breach count, total days, and mean breach overshoot for the current formula + date range.
2. **Breach breakdown** — Bar chart showing breach rate by VIX bucket (10–15, 15–20, ..., 50+). Color-coded green/yellow/red to show where the formula leaks.
3. **SPX chart** — Plotly candlestick with red dashed upper bound, green dashed lower bound, and breach markers. Header shows today's expected move and band levels TOS-style. Daily mode includes a range slider (default view: last 60 days).
4. **Formula curve** — VIX vs max-move scatter plot with the current formula overlaid in red, plus all other presets as faint dotted reference lines.

### Typical Workflow

1. Start with the **Quad 75th** preset — verify ~75% containment.
2. Switch to **Daily Backtest** mode, scroll through history to see where breaches cluster.
3. Check the **breach breakdown** chart — if a specific VIX range leaks too much, adjust coefficients.
4. Tweak the quadratic coeff (c) to widen bands at high VIX, or adjust the offset for a uniform shift.
5. Switch to **Intraday (60d)** mode to see how your formula looks on recent 5-minute price action.
6. Compare against other presets using the formula curve panel.

## How It Works

1. **Data**: Daily OHLC for VIX (^VIX) and SPX (^GSPC) from 1990–present via yfinance, stored in SQLite.
2. **Metric**: Max absolute % move from SPX open = `max(|high - open|, |open - low|) / open × 100`.
3. **VIX floor**: The OLS x-intercept (~5.0) is subtracted from VIX to zero-base the predictor — at VIX = 5, expected move ≈ 0%.
4. **Models**: Linear OLS (mean) and quadratic quantile regressions (50th/75th/90th/95th). The quadratic term captures vol-of-vol: residual scatter fans out ~6× from low VIX to crisis VIX.
5. **Validation**: Structural break analysis by era confirms the slope is stable post-2000 (~0.070 ± 0.002). Analysis runs on three subsets (1990+, 2000+, 2004+).

## Files

| File | Description |
|------|-------------|
| `data/fetch_data.py` | Fetches VIX/SPX daily + intraday data from Yahoo Finance into SQLite |
| `data/market_data.db` | SQLite database with `vix`, `spx`, and `spx_intraday` tables |
| `analysis.py` | Regression analysis with linear + quadratic quantile models |
| `formulas.py` | Formula engine with presets and vectorized bound computation |
| `app.py` | Interactive Streamlit dashboard for visual formula validation |
| `findings.txt` | Detailed statistical results and model comparisons |
| `conclusions.txt` | Final recommended formulas and quick-reference table |
| `visualizations/` | Generated multi-panel analysis charts (PNG) |