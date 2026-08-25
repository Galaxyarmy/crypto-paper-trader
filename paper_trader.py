"""
Crypto Paper Trading Agent v6.1 — Robust Production Edition

Improvements vs v6:
1. Centralized HTTP Session with Automatic Retries (Exponential Backoff for 429/5xx errors).
2. Fail-safe Wrappers on ALL API endpoints (fetch_klines, order_book, funding_rate, fear_greed).
3. ThreadPool exception handling to prevent worker crashes during parallel calls.
4. Preserved all core strategies (ADX, 1H Filter, OB Imbalance, Trailing Stops, Win/Loss fixes).

Requirements: pip install requests pandas
"""

import json
import csv
import os
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ------------------------- CONFIG -------------------------

COINS = ["BTCUSDT", "ETHUSDT"]
PRIMARY_INTERVAL = "15m"
TREND_INTERVAL = "1h"

STARTING_CASH_PER_COIN = 5000.0
FEE_PCT = 0.001

EMA_SHORT = 9
EMA_LONG = 21
TREND_EMA = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70

STOP_LOSS_PCT = 0.02
TRAILING_STOP_PCT = 0.015
PARTIAL_PROFIT_PCT = 0.015
POSITION_SIZE_PCT = 0.5
MIN_MOVE_PCT = 0.0025
MAX_DAILY_LOSS_PCT = 0.03
MIN_RISK_REWARD = 1.5

ADX_MIN = 20.0
OB_IMBALANCE_THRESHOLD = 0.10
FUNDING_EXTREME = 0.0005
FEAR_GREED_FEAR = 25
FEAR_GREED_GREED = 75

STATE_FILE = "state.json"
LOG_FILE = "trade_log_v6.csv"

BINANCE_SPOT = "https://data-api.binance.vision/api/v3"
BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

SCHEMA_VERSION = 6

# -------------------- HTTP SESSION SETUP --------------------

def get_robust_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

HTTP = get_robust_session()

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
                "win_trades": 0,
                "loss_trades": 0,
                "entry_price": None,
                "partial_sold": False,
                "partial_qty": 0.0,
                "day": today,
                "day_start_value": STARTING_CASH_PER_COIN,
                "day_halted": False,
                "peak_value": STARTING_CASH_PER_COIN,
                "max_drawdown": 0.0,
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
        migrated = {}
        has_old = False
        for sym in COINS:
            if sym in state:
                migrated[sym] = state[sym]
                has_old = True
            else:
                migrated[sym] = default_state()["coins"][sym]
        new_state = default_state()
        new_state["coins"] = migrated
        state = new_state
        if has_old:
            print("[INFO] Migrated old state -> v6.")
    else:
        defaults = default_state()
        for sym in COINS:
            if sym not in state["coins"]:
                state["coins"][sym] = defaults["coins"][sym]
            for k, v in defaults["coins"][sym].items():
                state["coins"][sym].setdefault(k, v)

    state["schema_version"] = SCHEMA_VERSION
    return state


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# -------------------- DATA FETCHERS (WITH RETRIES) --------------------

def fetch_klines(symbol: str, interval: str, limit: int = 150) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = HTTP.get(f"{BINANCE_SPOT}/klines", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.iloc[:-1].reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] Kline fetch completely failed for {symbol} ({interval}): {e}")
        return pd.DataFrame()


def fetch_order_book(symbol: str, limit: int = 100) -> dict:
    try:
        params = {"symbol": symbol, "limit": limit}
        resp = HTTP.get(f"{BINANCE_SPOT}/depth", params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        bid_vol = sum(float(b[1]) for b in data["bids"])
        ask_vol = sum(float(a[1]) for a in data["asks"])
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0
        return {"imbalance": imbalance, "bid_vol": bid_vol, "ask_vol": ask_vol}
    except Exception as e:
        print(f"[WARN] OrderBook fetch failed for {symbol}: {e}")
        return {"imbalance": 0.0, "bid_vol": 0.0, "ask_vol": 0.0}


def fetch_funding_rate(symbol: str) -> float:
    try:
        params = {"symbol": symbol, "limit": 1}
        resp = HTTP.get(f"{BINANCE_FUTURES}/fundingRate", params=params, timeout=8)
        resp.raise_for_status()
        return float(resp.json()[0]["fundingRate"])
    except Exception as e:
        print(f"[WARN] Funding rate fetch failed for {symbol}: {e}")
        return 0.0


def fetch_fear_greed() -> dict:
    try:
        resp = HTTP.get(FEAR_GREED_URL, timeout=8)
        resp.raise_for_status()
        d = resp.json()["data"][0]
        return {"value": int(d["value"]), "class": d["value_classification"]}
    except Exception as e:
        print(f"[WARN] Fear & Greed fetch failed: {e}")
        return {"value": 50, "class": "Neutral"}


# -------------------- INDICATORS --------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=TREND_EMA, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0
    atr = df["atr"]
    plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, 1e-9)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)) * 100
    df["adx"] = dx.rolling(14).mean()

    return df


def generate_signal(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "HOLD"
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
    if len(df) < lookback:
        return 0.0
    recent = df.iloc[-lookback:]
    avg_range = (recent["high"] - recent["low"]).mean()
    return avg_range / df.iloc[-1]["close"]


# -------------------- FILTER ENGINE --------------------

def apply_filters(symbol: str, base_signal: str,
                  primary_df: pd.DataFrame, trend_df: pd.DataFrame,
                  funding: float, fg: dict, ob: dict) -> tuple:
    reasons = []
    size_mult = 1.0

    latest = primary_df.iloc[-1]
    price = latest["close"]
    adx = latest["adx"]
    atr = latest["atr"]

    if pd.isna(adx) or adx < ADX_MIN:
        reasons.append(f"ADX_WEAK({adx:.1f})" if not pd.isna(adx) else "ADX_WEAK(nan)")
        if base_signal == "BUY":
            return "HOLD", "FILTER:" + ",".join(reasons), 0
    else:
        reasons.append(f"ADX_OK({adx:.1f})")

    if len(trend_df) > TREND_EMA:
        trend_price = float(trend_df.iloc[-1]["close"])
        trend_ema = float(trend_df.iloc[-1]["ema_trend"])
        if base_signal == "BUY" and trend_price < trend_ema:
            reasons.append(f"1H_BELOW_EMA50({trend_price:.0f}<{trend_ema:.0f})")
            return "HOLD", "FILTER:" + ",".join(reasons), 0
        if base_signal == "BUY" and trend_price > trend_ema:
            reasons.append("1H_ABOVE_EMA50")
            size_mult = min(size_mult + 0.25, 1.5)

    ob_imb = ob.get("imbalance", 0)
    if abs(ob_imb) > OB_IMBALANCE_THRESHOLD:
        if base_signal == "BUY" and ob_imb < -OB_IMBALANCE_THRESHOLD:
            reasons.append(f"OB_SELL_PRESSURE({ob_imb:+.2f})")
            size_mult = max(size_mult - 0.25, 0.5)
        elif base_signal == "BUY" and ob_imb > OB_IMBALANCE_THRESHOLD:
            reasons.append(f"OB_BUY_PRESSURE({ob_imb:+.2f})")
            size_mult = min(size_mult + 0.25, 1.5)

    if funding > FUNDING_EXTREME:
        reasons.append(f"FUNDING_HIGH({funding:.4%})")
        if base_signal == "BUY":
            return "HOLD", "FILTER:" + ",".join(reasons), 0
    elif funding < -FUNDING_EXTREME:
        reasons.append(f"FUNDING_NEG({funding:.4%})")
        if base_signal == "BUY":
            size_mult = min(size_mult + 0.25, 1.5)

    if fg["value"] <= FEAR_GREED_FEAR:
        reasons.append(f"EXTREME_FEAR({fg['value']})")
        if base_signal == "BUY":
            size_mult = min(size_mult + 0.25, 1.5)
    elif fg["value"] >= FEAR_GREED_GREED:
        reasons.append(f"EXTREME_GREED({fg['value']})")
        if base_signal == "BUY":
            return "HOLD", "FILTER:" + ",".join(reasons), 0

    if base_signal == "BUY" and not pd.isna(atr):
        stop_dist = max(atr * 2, price * STOP_LOSS_PCT)
        target_dist = stop_dist * MIN_RISK_REWARD
        resistance = primary_df["high"].rolling(20).max().iloc[-1]
        upside_to_res = resistance - price
        if upside_to_res < target_dist:
            reasons.append(f"RR_POOR(upside={upside_to_res:.0f}<target={target_dist:.0f})")
            size_mult = max(size_mult - 0.25, 0.5)
        else:
            reasons.append(f"RR_OK(1:{upside_to_res/stop_dist:.1f})")

    return base_signal, "FILTER_OK:" + ",".join(reasons), size_mult


# -------------------- POSITION & EXIT LOGIC --------------------

def calculate_position(wallet: dict, price: float, atr: float, size_mult: float) -> tuple:
    portfolio = wallet["cash"] + wallet["coin_qty"] * price
    if portfolio <= 0 or pd.isna(atr) or atr <= 0:
        return 0.0, 0.0
    stop_dist = max(atr * 2, price * STOP_LOSS_PCT)
    stop_pct = stop_dist / price
    if stop_pct <= 0:
        return 0.0, 0.0
    max_risk = portfolio * 0.02
    pos_dollars = max_risk / stop_pct
    max_spend = wallet["cash"] * POSITION_SIZE_PCT * size_mult
    spend = min(pos_dollars, max_spend)
    if spend <= 0 or spend > wallet["cash"]:
        return 0.0, 0.0
    fee = spend * FEE_PCT
    usable = spend - fee
    qty = usable / price
    return qty, spend


def check_exits(wallet: dict, price: float, atr: float, final_signal: str) -> tuple:
    if wallet["coin_qty"] <= 0 or wallet.get("entry_price") is None:
        return "HOLD", "", 0.0

    entry = wallet["entry_price"]
    current_qty = wallet["coin_qty"]

    stop_dist = max(atr * 2, entry * STOP_LOSS_PCT) if not pd.isna(atr) else entry * STOP_LOSS_PCT
    hard_stop = entry - stop_dist
    if price <= hard_stop:
        return "SELL", f"STOP_LOSS|hard@{hard_stop:.0f}", current_qty

    if not wallet.get("partial_sold", False):
        partial_target = entry * (1 + PARTIAL_PROFIT_PCT)
        if price >= partial_target:
            partial_qty = current_qty * 0.5
            return "PARTIAL", f"PARTIAL_PROFIT|+{PARTIAL_PROFIT_PCT:.1%}", partial_qty

    if price > entry * 1.03:
        trail_stop = price * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return "SELL", f"TRAILING_STOP|trail@{trail_stop:.0f}", current_qty

    if final_signal == "SELL":
        return "SELL", "SELL_SIGNAL", current_qty

    return "HOLD", "", 0.0


# -------------------- STATE & LOG --------------------

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
                "ema_short", "ema_long", "rsi", "adx", "atr",
                "funding", "fg", "ob_imbalance", "size_mult"
            ])
        writer.writerow(row)


# -------------------- MAIN --------------------

def main():
    start = time.time()
    state = load_state()
    timestamp = datetime.now(timezone.utc).isoformat()

    fg = fetch_fear_greed()
    print(f"[GLOBAL] Fear & Greed: {fg['value']} ({fg['class']})")

    for symbol in COINS:
        try:
            print(f"\n[ANALYZING] {symbol}")

            df_15m = fetch_klines(symbol, PRIMARY_INTERVAL, 150)
            df_1h = fetch_klines(symbol, TREND_INTERVAL, 150)

            if df_15m.empty or df_1h.empty:
                print(f"[SKIP] Skipping {symbol} due to missing kline data.")
                continue

            df_15m = compute_indicators(df_15m)
            df_1h = compute_indicators(df_1h)

            base_signal = generate_signal(df_15m)
            price = float(df_15m.iloc[-1]["close"])
            atr = float(df_15m.iloc[-1]["atr"])
            adx = float(df_15m.iloc[-1]["adx"])
            move = recent_move_pct(df_15m)

            with ThreadPoolExecutor(max_workers=2) as ex:
                f_funding = ex.submit(fetch_funding_rate, symbol)
                f_ob = ex.submit(fetch_order_book, symbol)

                try:
                    funding = f_funding.result(timeout=10)
                except Exception as e:
                    print(f"[WARN] Funding execution error: {e}")
                    funding = 0.0

                try:
                    ob = f_ob.result(timeout=10)
                except Exception as e:
                    print(f"[WARN] OrderBook execution error: {e}")
                    ob = {"imbalance": 0.0, "bid_vol": 0.0, "ask_vol": 0.0}

            print(f"  Price=${price:,.2f} | ADX={adx:.1f} | ATR=${atr:,.2f} | "
                  f"Funding={funding:.4%} | OB={ob['imbalance']:+.2f}")

            final_signal, filter_reason, size_mult = apply_filters(
                symbol, base_signal, df_15m, df_1h, funding, fg, ob
            )
            print(f"  Base={base_signal} | Final={final_signal} | Size={size_mult:.0%} | {filter_reason}")

            wallet = state["coins"][symbol]
            reset_day_if_needed(wallet, price)

            if wallet["coin_qty"] > 0 and wallet.get("entry_price") is None:
                wallet["entry_price"] = price
                wallet["partial_sold"] = False
                wallet["partial_qty"] = 0.0

            action = "HOLD"
            reason = filter_reason
            trade_qty = 0.0

            # --- EXIT LOGIC ---
            if wallet["coin_qty"] > 0:
                exit_action, exit_reason, exit_qty = check_exits(wallet, price, atr, final_signal)
                if exit_action == "SELL":
                    entry_price_at_exit = wallet.get("entry_price")

                    proceeds = exit_qty * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] -= exit_qty
                    if wallet["coin_qty"] <= 1e-9:
                        wallet["coin_qty"] = 0.0
                        wallet["entry_price"] = None
                        wallet["partial_sold"] = False
                        wallet["partial_qty"] = 0.0
                    wallet["num_trades"] += 1
                    trade_qty = exit_qty

                    if entry_price_at_exit is not None and price > entry_price_at_exit:
                        wallet["win_trades"] = wallet.get("win_trades", 0) + 1
                    else:
                        wallet["loss_trades"] = wallet.get("loss_trades", 0) + 1
                    action = "SELL"
                    reason = exit_reason

                elif exit_action == "PARTIAL":
                    proceeds = exit_qty * price
                    fee = proceeds * FEE_PCT
                    wallet["cash"] += proceeds - fee
                    wallet["coin_qty"] -= exit_qty
                    wallet["partial_sold"] = True
                    wallet["partial_qty"] = exit_qty
                    wallet["num_trades"] += 1
                    trade_qty = exit_qty
                    action = "PARTIAL"
                    reason = exit_reason

                elif wallet["day_halted"]:
                    action = "HOLD"
                    reason = "DAILY_LOSS_HALT"

            # --- ENTRY LOGIC ---
            elif wallet["day_halted"]:
                action = "HOLD"
                reason = "DAILY_LOSS_HALT"

            elif final_signal == "BUY" and wallet["cash"] > 0:
                if move < MIN_MOVE_PCT:
                    action = "HOLD"
                    reason = "SKIP_SMALL_MOVE"
                elif size_mult <= 0:
                    action = "HOLD"
                    reason = filter_reason
                else:
                    qty, spend = calculate_position(wallet, price, atr, size_mult)
                    if qty <= 0:
                        action = "HOLD"
                        reason = "RISK_SIZING_FAIL"
                    else:
                        wallet["coin_qty"] += qty
                        wallet["cash"] -= spend
                        wallet["entry_price"] = price
                        wallet["partial_sold"] = False
                        wallet["partial_qty"] = 0.0
                        wallet["num_trades"] += 1
                        trade_qty = qty
                        action = "BUY"
                        reason = f"BUY|{filter_reason}|size={size_mult:.0%}"

            value = wallet["cash"] + wallet["coin_qty"] * price
            if value > wallet.get("peak_value", value):
                wallet["peak_value"] = value
            dd = (wallet.get("peak_value", value) - value) / wallet.get("peak_value", value)
            if dd > wallet.get("max_drawdown", 0):
                wallet["max_drawdown"] = dd

            day_loss = (wallet["day_start_value"] - value) / wallet["day_start_value
