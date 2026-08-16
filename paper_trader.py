"""
Crypto Paper Trading Agent v3.1 — Hardened Single-Run Edition (Fixed)
Requirements: pip install requests pandas
"""

import json
import csv
import os
import time
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

STOP_LOSS_PCT = 0.02
POSITION_SIZE_PCT = 0.5
MIN_MOVE_PCT = 0.004
MAX_DAILY_LOSS_PCT = 0.03

STATE_FILE = "state.json"
LOG_FILE = "trade_log_v3_1.csv"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"

SCHEMA_VERSION = 2

# ------------------------------------------------------------


def default_state() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "coins": {
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
    }


def load_state() -> dict:
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {}

    if "coins" not in state:
        migrated_coins = {}
        has_v3_data = False
        for symbol in COINS:
            if symbol in state:
                migrated_coins[symbol] = state[symbol]
                has_v3_data = True
            else:
                migrated_coins[symbol] = default_state()["coins"][symbol]

        new_state = default_state()
        new_state["coins"] = migrated_coins
        state = new_state

        if has_v3_data:
            print(f"[INFO] Migrated v3 flat state to v3.1 wrapped format. Data preserved.")
    else:
        defaults = default_state()
        for symbol in COINS:
            if symbol not in state["coins"]:
                state["coins"][symbol] = defaults["coins"][symbol]
            for key, default_val in defaults["coins"][symbol].items():
                state["coins"][symbol].setdefault(key, default_val)

    state["schema_version"] = SCHEMA_VERSION
    return state


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_closed_candles(symbol: str, interval: str, limit: int = 101, retries: int = 3) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_sec = 2 ** attempt
                print(f"[WARN] {symbol} API attempt {attempt} failed, retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
            else:
                raise last_err

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    return df.iloc[:-1].reset_index(drop=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
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
    recent = df.iloc[-lookback:]
    avg_range = (recent["high"] - recent["low"]).mean()
    latest_close = df.iloc[-1]["close"]
    return avg_range / latest_close


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
            writer.writerow([
                "timestamp", "symbol", "action", "signal", "reason", "price",
                "trade_qty", "cash", "coin_qty", "portfolio_value",
                "ema_short", "ema_long", "rsi"
            ])
        writer.writerow(row)


def main():
    state = load_state()
    timestamp = datetime.now(timezone.utc).isoformat()

    for symbol in COINS:
        try:
            df = fetch_closed_candles(symbol, INTERVAL)
            df = compute_indicators(df)
            signal = generate_signal(df)
            price = float(df.iloc[-1]["close"])
            move = recent_move_pct(df)

            wallet = state["coins"][symbol]
            reset_day_if_needed(wallet, price)

            if wallet["coin_qty"] > 0 and wallet.get("entry_price") is None:
                print(f"[WARN] {symbol}: coin_qty={wallet['coin_qty']} but entry_price missing. "
                      f"Auto-setting entry_price to current price ${price:.2f}")
                wallet["entry_price"] = price

            reason = signal
            action = "HOLD"
            trade_qty = 0.0

            if wallet["coin_qty"] > 0 and wallet["entry_price"] is not None:
                loss_pct = (wallet["entry_price"] - price) / wallet["entry_price"]
                if loss_pct >= STOP_LOSS_PCT:
                    trade_qty = wallet["coin_qty"]
                    proceeds = trade_qty * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] = 0.0
                    wallet["entry_price"] = None
                    wallet["num_trades"] += 1
                    action = "SELL"
                    reason = "STOP_LOSS"

                elif signal == "SELL":
                    trade_qty = wallet["coin_qty"]
                    proceeds = trade_qty * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] = 0.0
                    wallet["entry_price"] = None
                    wallet["num_trades"] += 1
                    action = "SELL"
                    reason = "SELL"

                elif wallet["day_halted"]:
                    action = "HOLD"
                    reason = "DAILY_LOSS_HALT"

            elif wallet["day_halted"]:
                action = "HOLD"
                reason = "DAILY_LOSS_HALT"

            elif signal == "BUY" and wallet["cash"] > 0:
                hypothetical_value = wallet["cash"] + wallet["coin_qty"] * price
                hypothetical_loss = (wallet["day_start_value"] - hypothetical_value) / wallet["day_start_value"]
                if hypothetical_loss >= MAX_DAILY_LOSS_PCT * 0.9:
                    action = "HOLD"
                    reason = "SKIP_NEAR_DAILY_LIMIT"
                elif move < MIN_MOVE_PCT:
                    action = "HOLD"
                    reason = "SKIP_SMALL_MOVE"
                else:
                    spend = wallet["cash"] * POSITION_SIZE_PCT
                    fee = spend * FEE_PCT
                    usable = spend - fee
                    trade_qty = usable / price
                    wallet["coin_qty"] += trade_qty
                    wallet["cash"] -= spend
                    wallet["entry_price"] = price
                    wallet["num_trades"] += 1
                    action = "BUY"
                    reason = "BUY"

            value = wallet["cash"] + wallet["coin_qty"] * price
            day_loss_pct = (wallet["day_start_value"] - value) / wallet["day_start_value"]
            if day_loss_pct >= MAX_DAILY_LOSS_PCT:
                wallet["day_halted"] = True

            ema_s = float(df.iloc[-1]["ema_short"])
            ema_l = float(df.iloc[-1]["ema_long"])
            rsi_v = float(df.iloc[-1]["rsi"])

            log_row([
                timestamp, symbol, action, signal, reason, price, trade_qty,
                wallet["cash"], wallet["coin_qty"], value,
                round(ema_s, 2), round(ema_l, 2), round(rsi_v, 2)
            ])

            print(f"[{timestamp}] {symbol}: price=${price:.2f} signal={signal} "
                  f"action={action} reason={reason} portfolio=${value:.2f}")

        except Exception as e:
            print(f"[{timestamp}] ERROR processing {symbol}: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
