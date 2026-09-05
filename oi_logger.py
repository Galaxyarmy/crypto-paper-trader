"""
oi_logger.py — Runs every 15 min (same cron as paper_trader.py) and just
LOGS. No signal, no trading, nothing that touches state.json.

Why this exists: Binance's free openInterestHist endpoint only serves
about 5 days of history (confirmed empirically — see NOTES.md and the
backtest diagnostic output that found 500/2400 candles covered). That's
not enough to backtest or trust the OI-delta filter on. This script's
only job is to start accumulating a genuine, growing history of our own
so that in a few weeks there's real multi-week OI data to re-tune
OI_DELTA_MIN_PCT against — data that wasn't used to pick the threshold in
the first place, unlike the current one.

Appends one row per symbol per run to oi_log.csv. Safe to run as often as
you like — it only ever appends, never reads its own output, so there's
no state to corrupt. If it fails, it fails silently-ish (prints a
warning) and never touches trading state.
"""

import csv
import os
from datetime import datetime, timezone

import requests

import config as cfg

LOG_FILE = "oi_log.csv"


def fetch_point(symbol: str) -> dict | None:
    """One current snapshot: open interest + mark price + funding rate,
    in a single premiumIndex call plus one openInterest call (premiumIndex
    doesn't carry OI)."""
    try:
        oi_resp = requests.get(cfg.OPEN_INTEREST_ENDPOINT, params={"symbol": symbol}, timeout=10)
        oi_resp.raise_for_status()
        oi_data = oi_resp.json()

        mark_resp = requests.get(cfg.MARK_PRICE_ENDPOINT, params={"symbol": symbol}, timeout=10)
        mark_resp.raise_for_status()
        mark_data = mark_resp.json()

        return {
            "open_interest": float(oi_data["openInterest"]),
            "mark_price": float(mark_data["markPrice"]),
            "funding_rate": float(mark_data["lastFundingRate"]),
        }
    except Exception as e:
        print(f"[WARN] oi_logger fetch failed for {symbol}: {e}")
        return None


def log_row(row: list):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "open_interest", "mark_price", "funding_rate"])
        writer.writerow(row)


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    for symbol in cfg.COINS:
        point = fetch_point(symbol)
        if point is None:
            continue
        log_row([timestamp, symbol, point["open_interest"], point["mark_price"], point["funding_rate"]])
        print(f"[LOGGED] {symbol} OI={point['open_interest']:,.1f} "
              f"price={point['mark_price']:,.2f} funding={point['funding_rate']:.4%}")


if __name__ == "__main__":
    main()
