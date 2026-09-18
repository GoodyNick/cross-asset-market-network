from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Define the six macro anchor assets
ASSETS = {
    "SPY": "SP500_Equity",
    "USO": "Crude_Oil",
    "GLD": "Gold",
    "UUP": "US_Dollar_Index",
    "IEF": "US_10Y_Treasury",
    "BTC-USD": "Bitcoin",
}

def fetch_and_sync_macro_data(start_date="2021-01-01", end_date="2026-01-01"):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(ASSETS.keys())
    
    print(f"Fetching historical daily bars for: {tickers}...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    
    # Extract adjusted Close prices and rename columns
    prices = raw_data["Close"][tickers].rename(columns=ASSETS)
    
    # Align calendars: Drop non-trading days where traditional markets are closed
    # This filters 24/7 crypto data to active market trading days
    aligned_prices = prices.dropna().copy()
    
    # Compute daily log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(aligned_prices / aligned_prices.shift(1)).dropna()
    
    # Save datasets in Parquet for strict data typing and compression
    aligned_prices.to_parquet(RAW_DATA_DIR / "macro_prices_raw.parquet")
    log_returns.to_parquet(RAW_DATA_DIR / "macro_log_returns.parquet")
    
    print(f"\nSuccess! Captured {len(aligned_prices)} synchronized trading days.")
    print(f"Date range: {aligned_prices.index.min().date()} to {aligned_prices.index.max().date()}")
    print("\nLog Returns Summary Snapshot:")
    print(log_returns.tail())

if __name__ == "__main__":
    fetch_and_sync_macro_data()