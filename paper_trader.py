"""
Crypto Paper Trading Agent — single-run version for GitHub Actions.

Each run:
1. Loads wallet state from state.json (or creates fresh state if missing)
2. Fetches latest candles for BTC/USDT and ETH/USDT from Binance public API
3. Computes EMA9/EMA21 crossover + RSI signal
4. Simulates BUY/SELL if signal fires
5. Appends the decision to trade_log.csv
6. Saves updated wallet state back to state.json

GitHub Actions is expected to call this script on a schedule (e.g. every
15 minutes) and commit state.json + trade_log.csv back to the repo after
each run, so state persists across runs.

Requirements: pip install requests pandas
"""

import json
import csv
import os
from datetime import datetime, timezone

import requests
import pandas as pd

# ------------------------- CONFIG -------------------------

COINS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "15m"
STARTING_CASH_PER_COIN = 5000.0
FEE_PCT = 0.001

EMA_SHORT = 9
EMA_LONG = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

STATE_FILE = "state.json"
LOG_FILE = "trade_log.csv"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# ------------------------------------------------------------


def load_state() -> dict:
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        symbol: {"cash": STARTING_CASH_PER_COIN, "coin_qty": 0.0, "num_trades": 0}
        for symbol in COINS
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_candles(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def generate_signal(df: pd.DataFrame) -> str:
    prev = df.iloc[-2]
    latest = df.iloc[-1]

    crossed_up = prev["ema_short"] <= prev["ema_long"] and latest["ema_short"] > latest["ema_long"]
    crossed_down = prev["ema_short"] >= prev["ema_long"] and latest["ema_short"] < latest["ema_long"]

    if crossed_up and latest["rsi"] < RSI_OVERBOUGHT:
        return "BUY"
    if crossed_down or latest["rsi"] > RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


def log_row(row: list):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "price", "qty",
                              "cash", "coin_qty", "portfolio_value"])
        writer.writerow(row)


def main():
    state = load_state()
    timestamp = datetime.now(timezone.utc).isoformat()

    for symbol in COINS:
        try:
            df = fetch_candles(symbol, INTERVAL)
            df = compute_indicators(df)
            signal = generate_signal(df)
            price = float(df.iloc[-1]["close"])

            wallet = state[symbol]

            if signal == "BUY" and wallet["cash"] > 0:
                fee = wallet["cash"] * FEE_PCT
                usable_cash = wallet["cash"] - fee
                wallet["coin_qty"] += usable_cash / price
                wallet["cash"] = 0.0
                wallet["num_trades"] += 1

            elif signal == "SELL" and wallet["coin_qty"] > 0:
                proceeds = wallet["coin_qty"] * price
                fee = proceeds * FEE_PCT
                wallet["cash"] += proceeds - fee
                wallet["coin_qty"] = 0.0
                wallet["num_trades"] += 1

            value = wallet["cash"] + wallet["coin_qty"] * price
            log_row([timestamp, symbol, signal, price, wallet["coin_qty"],
                     wallet["cash"], wallet["coin_qty"], value])

            print(f"[{timestamp}] {symbol}: price=${price:.2f} signal={signal} "
                  f"portfolio=${value:.2f}")

        except Exception as e:
            print(f"[{timestamp}] Error processing {symbol}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
