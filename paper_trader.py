"""
Crypto Paper Trading Agent v2 — single-run version for GitHub Actions.

New in v2:
- Stop-loss: auto-exit a position if price drops STOP_LOSS_PCT below entry
- Position sizing: only deploy POSITION_SIZE_PCT of available cash per trade
  (rest stays in reserve, reducing risk of one bad trade wiping the account)
- Fee-aware entry filter: skip BUY signals unless the recent move is big
  enough to plausibly cover round-trip fees
- Daily loss limit: if a coin's value drops more than MAX_DAILY_LOSS_PCT
  from its start-of-day value, pause trading that coin for the rest of the day

Each run:
1. Loads wallet state from state.json (or creates fresh state if missing)
2. Fetches latest candles for BTC/USDT and ETH/USDT
3. Computes EMA9/EMA21 crossover + RSI signal
4. Applies risk filters, then simulates BUY/SELL if signal fires
5. Appends the decision to trade_log.csv
6. Saves updated wallet state back to state.json

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
INTERVAL = "5m"
STARTING_CASH_PER_COIN = 5000.0
FEE_PCT = 0.001

EMA_SHORT = 9
EMA_LONG = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

# --- risk management ---
STOP_LOSS_PCT = 0.02          # exit if price falls 2% below entry
POSITION_SIZE_PCT = 0.5       # only deploy 50% of available cash per buy
MIN_MOVE_PCT = 0.004          # skip buy unless recent move >= 0.4% (covers ~2x round-trip fee)
MAX_DAILY_LOSS_PCT = 0.03     # stop trading a coin for the day after 3% daily loss

STATE_FILE = "state.json"
LOG_FILE = "trade_log.csv"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

# ------------------------------------------------------------


def load_state() -> dict:
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return default_state()


def default_state() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    return {
        symbol: {
            "cash": STARTING_CASH_PER_COIN,
            "coin_qty": 0.0,
            "num_trades": 0,
            "entry_price": None,
            "day": today,
            "day_start_value": STARTING_CASH_PER_COIN,
            "day_halted": False,
        }
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


def recent_move_pct(df: pd.DataFrame, lookback: int = 5) -> float:
    """Rough measure of recent momentum, used as a fee-aware filter."""
    recent = df["close"].iloc[-lookback:]
    return abs(recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]


def reset_day_if_needed(wallet: dict, price: float):
    today = datetime.now(timezone.utc).date().isoformat()
    if wallet.get("day") != today:
        wallet["day"] = today
        wallet["day_start_value"] = wallet["cash"] + wallet["coin_qty"] * price
        wallet["day_halted"] = False


def log_row(row: list):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "reason", "price", "qty",
                              "cash", "coin_qty", "portfolio_value"])
        writer.writerow(row)


def main():
    state = load_state()
    # migrate old-format state (from v1) if needed
    for symbol in COINS:
        if symbol not in state:
            state[symbol] = default_state()[symbol]
        for key, default_val in default_state()[symbol].items():
            state[symbol].setdefault(key, default_val)

    timestamp = datetime.now(timezone.utc).isoformat()

    for symbol in COINS:
        try:
            df = fetch_candles(symbol, INTERVAL)
            df = compute_indicators(df)
            signal = generate_signal(df)
            price = float(df.iloc[-1]["close"])
            move = recent_move_pct(df)

            wallet = state[symbol]
            reset_day_if_needed(wallet, price)

            reason = signal

            if wallet["day_halted"]:
                reason = "DAILY_LOSS_HALT"

            # --- stop-loss check first (applies even if halted) ---
            elif wallet["coin_qty"] > 0 and wallet["entry_price"]:
                loss_pct = (wallet["entry_price"] - price) / wallet["entry_price"]
                if loss_pct >= STOP_LOSS_PCT:
                    proceeds = wallet["coin_qty"] * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] = 0.0
                    wallet["entry_price"] = None
                    wallet["num_trades"] += 1
                    reason = "STOP_LOSS"

                elif signal == "SELL":
                    proceeds = wallet["coin_qty"] * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] = 0.0
                    wallet["entry_price"] = None
                    wallet["num_trades"] += 1
                    reason = "SELL"

            elif signal == "BUY" and wallet["cash"] > 0:
                if move < MIN_MOVE_PCT:
                    reason = "SKIP_SMALL_MOVE"
                else:
                    spend = wallet["cash"] * POSITION_SIZE_PCT
                    fee = spend * FEE_PCT
                    usable = spend - fee
                    wallet["coin_qty"] += usable / price
                    wallet["cash"] -= spend
                    wallet["entry_price"] = price
                    wallet["num_trades"] += 1
                    reason = "BUY"

            value = wallet["cash"] + wallet["coin_qty"] * price

            # check daily loss limit for next run
            day_loss_pct = (wallet["day_start_value"] - value) / wallet["day_start_value"]
            if day_loss_pct >= MAX_DAILY_LOSS_PCT:
                wallet["day_halted"] = True

            log_row([timestamp, symbol, signal, reason, price, wallet["coin_qty"],
                     wallet["cash"], wallet["coin_qty"], value])

            print(f"[{timestamp}] {symbol}: price=${price:.2f} signal={signal} "
                  f"reason={reason} portfolio=${value:.2f}")

        except Exception as e:
            print(f"[{timestamp}] Error processing {symbol}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
