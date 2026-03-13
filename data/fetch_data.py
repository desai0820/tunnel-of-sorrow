"""
Fetch historical VIX and SPX data and store in a SQLite database.

Data sources (via yfinance):
  - ^VIX: CBOE Volatility Index (open, high, low, close)
  - ^GSPC: S&P 500 Index (open, close, high, low)
"""

import sys
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text

DB_PATH = "market_data.db"
START_DATE = "1990-01-01"


def fetch_vix():
    """Fetch VIX historical data: date, open, high, low, close."""
    print("Fetching VIX data...")
    vix = yf.download("^VIX", start=START_DATE, auto_adjust=True, progress=False)
    vix = vix[["Open", "High", "Low", "Close"]].copy()
    vix.columns = ["open", "high", "low", "close"]
    vix.index.name = "date"
    vix = vix.reset_index()
    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    print(f"  VIX: {len(vix)} rows ({vix['date'].min()} to {vix['date'].max()})")
    return vix


def fetch_spx():
    """Fetch SPX historical data: date, open, close, high, low."""
    print("Fetching SPX data...")
    spx = yf.download("^GSPC", start=START_DATE, auto_adjust=True, progress=False)
    spx = spx[["Open", "High", "Low", "Close"]].copy()
    spx.columns = ["open", "high", "low", "close"]
    spx.index.name = "date"
    spx = spx.reset_index()
    spx["date"] = pd.to_datetime(spx["date"]).dt.date
    print(f"  SPX: {len(spx)} rows ({spx['date'].min()} to {spx['date'].max()})")
    return spx


def create_tables(engine):
    """Create tables if they don't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vix (
                date DATE PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS spx (
                date DATE PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL
            )
        """))


def load_to_db(engine, vix_df, spx_df):
    """Write dataframes to SQLite, replacing any existing data."""
    vix_df.to_sql("vix", engine, if_exists="replace", index=False)
    spx_df.to_sql("spx", engine, if_exists="replace", index=False)
    print(f"\nLoaded data into {DB_PATH}")


def verify(engine):
    """Print summary stats to confirm data integrity."""
    with engine.connect() as conn:
        for table in ["vix", "spx"]:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            min_date = conn.execute(text(f"SELECT MIN(date) FROM {table}")).scalar()
            max_date = conn.execute(text(f"SELECT MAX(date) FROM {table}")).scalar()
            nulls = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE open IS NULL OR close IS NULL")
            ).scalar()
            print(f"  {table.upper()}: {row_count} rows, {min_date} to {max_date}, {nulls} nulls")


def main():
    engine = create_engine(f"sqlite:///{DB_PATH}")

    vix_df = fetch_vix()
    spx_df = fetch_spx()

    create_tables(engine)
    load_to_db(engine, vix_df, spx_df)

    print("\nVerification:")
    verify(engine)
    print("\nDone.")


if __name__ == "__main__":
    main()
