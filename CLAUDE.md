# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"Tunnel of Sorrow" is a quantitative analysis project that models the relationship between VIX opening level and maximum intraday SPX move. The goal is to produce formulas for expected-move bands usable in day trading (replacing an older ThinkOrSwim script).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch/refresh VIX and SPX data from Yahoo Finance into SQLite
cd data && python fetch_data.py && cd ..

# Run the full analysis (fits models, prints stats, generates charts)
python analysis.py

# Launch interactive visualization dashboard
streamlit run app.py
```

Note: `analysis.py` must be run from the project root (it reads `market_data.db` from cwd). `fetch_data.py` writes `market_data.db` into its own directory (`data/`), while `analysis.py` expects it at `./market_data.db` — the DB path is hardcoded in both files. `app.py` checks both paths.

## Architecture

- **`data/fetch_data.py`** — Downloads VIX (^VIX) and SPX (^GSPC) daily OHLC + 5-min intraday from yfinance, stores in `data/market_data.db` (SQLite with `vix`, `spx`, and `spx_intraday` tables).
- **`analysis.py`** — Core analysis script. Loads data, computes a "VIX floor" (x-intercept of OLS fit, ~5.0), fits linear OLS + quantile regressions (50th/75th/90th/95th) and quadratic quantile regressions, runs structural break analysis by era, builds lookup tables, and generates multi-panel PNG charts into `visualizations/`.
- **`formulas.py`** — Formula engine with preset configurations and vectorized bound computation. Used by `app.py`.
- **`app.py`** — Interactive Streamlit dashboard. Overlays expected-move bands on SPX charts (daily backtest + intraday modes), with adjustable formula coefficients and containment statistics.
- **`findings.txt`** — Detailed statistical results and model comparisons.
- **`conclusions.txt`** — Final recommended formulas and quick-reference table.

## Interactive Dashboard (`app.py`)

`streamlit run app.py` launches a Plotly-based dashboard for visual formula validation. Sidebar has: preset formula dropdown, editable coefficients (a, b, c, floor, offset), date range filter, chart mode toggle (Daily Backtest vs Intraday 60d), and breach marker toggle. Main area shows: containment stats bar, breach-rate-by-VIX-bucket bar chart, SPX candlestick with bound overlay + breach markers, and a VIX-vs-move scatter with formula curve. Formula presets live in `formulas.py:PRESETS`. Intraday mode requires the `spx_intraday` table (populated by `fetch_data.py`).

## Key Concepts

- **Max absolute SPX move**: `max(|high - open|, |open - low|) / open * 100` — largest % swing from open in either direction.
- **VIX floor**: The OLS x-intercept (~5.0), subtracted from VIX to zero-base the predictor.
- **Two model families**: Linear (simple, for mental math) and quadratic quantile (accounts for vol-of-vol fan-out at high VIX).
- **Primary model**: Quadratic 75th percentile on 2004+ data — `move% = 0.1953 + 0.0659*(VIX-5) + 0.000440*(VIX-5)²`.
- Analysis runs on three date subsets (1990+, 2000+, 2004+) to test stability.
