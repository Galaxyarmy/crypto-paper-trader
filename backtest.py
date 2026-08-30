"""
Backtest — Crypto Paper Trading Agent v6.2
(With proper Peak-Price based Trailing Stop)

Changes vs previous:
- Fixed broken trailing stop
- Added peak_price tracking
- Partial sell ke baad trailing continue hota hai
- Same logic as paper_trader.py v6.2

Outputs:
  - backtest_trades.csv
  - backtest_summary.md
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ------------------------- CONFIG (mirrors paper_trader v6.2) -------------------------

COINS = ["BTCUSDT", "ETHUSDT"]
PRIMARY_INTERVAL = "15m"
TREND_INTERVAL = "1h"

BACKTEST_DAYS = 90
BACKTEST_OFFSET_DAYS = 0   # NEW: kitne din pehle se test start ho

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
EMA_SEP_MIN_PCT = 0.0004
FUNDING_EXTREME = 0.0005
FEAR_GREED_FEAR = 25
FEAR_GREED_GREED = 75

BINANCE_SPOT = "https://data-api.binance.vision/api/v3"
BINANCE_FUTURES = "https://fapi.binance.com/fapi/v1"
FEAR_GREED_URL = "https://api.alternative.me/fng/"

RESULTS_CSV = "backtest_trades.csv"
SUMMARY_MD = "backtest_summary.md"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

# ------------------------- DATA FETCHERS -------------------------

def fetch_full_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur,
                  "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(f"{BINANCE_SPOT}/klines", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Kline page fetch failed for {symbol} at {cur}: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=KLINE_COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
    return df


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(f"{BINANCE_FUTURES}/fundingRate", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Funding history fetch failed for {symbol} at {cur}: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    if not all_rows:
        return pd.DataFrame(columns=["time", "funding"])

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["time", "funding"]].sort_values("time").reset_index(drop=True)


def fetch_fear_greed_history() -> pd.DataFrame:
    try:
        resp = requests.get(FEAR_GREED_URL, params={"limit": 0, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
    except Exception as e:
        print(f"[WARN] Fear & Greed history fetch failed: {e}")
        return pd.DataFrame(columns=["date", "fg"])

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.date
    df["fg"] = df["value"].astype(int)
    return df[["date", "fg"]].sort_values("date").reset_index(drop=True)


# ------------------------- INDICATORS -------------------------

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


def generate_signal(prev_row, latest_row) -> str:
    crossed_up = prev_row["ema_short"] <= prev_row["ema_long"] and latest_row["ema_short"] > latest_row["ema_long"]
    crossed_down = prev_row["ema_short"] >= prev_row["ema_long"] and latest_row["ema_short"] < latest_row["ema_long"]

    if crossed_up and latest_row["rsi"] < RSI_OVERBOUGHT:
        return "BUY"
    if crossed_down:
        return "SHORT"
    if latest_row["rsi"] > RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"

# ------------------------- FILTER ENGINE -------------------------

def apply_filters(base_signal, latest, trend_price, trend_ema, funding, fg_value, recent_move):
    size_mult = 1.0
    adx = latest["adx"]
    price = latest["close"]
    atr = latest["atr"]

    is_entry_signal = base_signal in ("BUY", "SHORT")
    breakout_ok = (base_signal != "BUY" or price >= latest.get("res20", price) * 0.998) and (base_signal != "SHORT" or price <= latest.get("sup20", price) * 1.002)
  
    if pd.isna(adx) or adx < ADX_MIN:
        if is_entry_signal:
            return "HOLD", 0.0
    else:
        ema_sep = abs(latest["ema_short"] - latest["ema_long"]) / price
        if ema_sep < EMA_SEP_MIN_PCT and is_entry_signal:
            return "HOLD", 0.0

    if not pd.isna(trend_ema):
        if base_signal == "BUY" and trend_price < trend_ema:
            return "HOLD", 0.0
        if base_signal == "BUY" and trend_price > trend_ema:
            size_mult = min(size_mult + 0.25, 1.5)

        if base_signal == "SHORT" and trend_price > trend_ema:
            return "HOLD", 0.0
        if base_signal == "SHORT" and trend_price < trend_ema:
            size_mult = min(size_mult + 0.25, 1.5)

    if base_signal == "SHORT":
        return (base_signal if breakout_ok else "HOLD"), size_mult

    if funding > FUNDING_EXTREME and base_signal == "BUY":
        return "HOLD", 0.0
    elif funding < -FUNDING_EXTREME and base_signal == "BUY":
        size_mult = min(size_mult + 0.25, 1.5)

    if fg_value <= FEAR_GREED_FEAR and base_signal == "BUY":
        size_mult = min(size_mult + 0.25, 1.5)
    elif fg_value >= FEAR_GREED_GREED and base_signal == "BUY":
        return "HOLD", 0.0

    return (base_signal if breakout_ok else "HOLD"), size_mult

def calculate_position(wallet, price, atr, size_mult):
    portfolio = wallet["cash"] - wallet["coin_qty"] * price if wallet.get("position_type") == "SHORT" else wallet["cash"] + wallet["coin_qty"] * price
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
    qty = (spend - fee) / price
    return qty, spend


def check_exits(wallet, price, atr, final_signal):
    """Proper Peak/Trough-Price based Trailing Stop. Handles LONG and SHORT."""
    if wallet["coin_qty"] <= 0 or wallet.get("entry_price") is None:
        return "HOLD", "", 0.0

    entry = wallet["entry_price"]
    qty = wallet["coin_qty"]
    pos_type = wallet.get("position_type", "LONG")

    if pos_type == "LONG":
        stop_dist = max(atr * 2, entry * STOP_LOSS_PCT) if not pd.isna(atr) else entry * STOP_LOSS_PCT
        hard_stop = entry - stop_dist
        if price <= hard_stop:
            return "SELL", "STOP_LOSS", qty

        if wallet.get("peak_price") is None or price > wallet["peak_price"]:
            wallet["peak_price"] = price

        if not wallet.get("partial_sold", False):
            if price >= entry * (1 + PARTIAL_PROFIT_PCT):
                return "PARTIAL", "PARTIAL_PROFIT", qty * 0.5

        if wallet["peak_price"] is not None and wallet["peak_price"] >= entry * (1 + PARTIAL_PROFIT_PCT):
            trail_stop = wallet["peak_price"] * (1 - TRAILING_STOP_PCT)
            if price <= trail_stop:
                return "SELL", "TRAILING_STOP", qty

        if final_signal in ("SELL", "SHORT"):
            return "SELL", "SELL_SIGNAL", qty

        return "HOLD", "", 0.0

    elif pos_type == "SHORT":
        stop_dist = max(atr * 2, entry * STOP_LOSS_PCT) if not pd.isna(atr) else entry * STOP_LOSS_PCT
        hard_stop = entry + stop_dist
        if price >= hard_stop:
            return "COVER", "STOP_LOSS", qty

        if wallet.get("trough_price") is None or price < wallet["trough_price"]:
            wallet["trough_price"] = price

        if not wallet.get("partial_sold", False):
            if price <= entry * (1 - PARTIAL_PROFIT_PCT):
                return "PARTIAL", "PARTIAL_PROFIT", qty * 0.5

        if wallet["trough_price"] is not None and wallet["trough_price"] <= entry * (1 - PARTIAL_PROFIT_PCT):
            trail_stop = wallet["trough_price"] * (1 + TRAILING_STOP_PCT)
            if price >= trail_stop:
                return "COVER", "TRAILING_STOP", qty

        if final_signal == "BUY":
            return "COVER", "COVER_SIGNAL", qty

        return "HOLD", "", 0.0

    return "HOLD", "", 0.0

# ------------------------- BACKTEST ENGINE -------------------------

def backtest_symbol(symbol, df_15m, df_1h, funding_df, fg_df):
    df_15m = compute_indicators(df_15m.copy())
    df_1h = compute_indicators(df_1h.copy())
    df_1h_trend = df_1h[["open_time", "close", "ema_trend"]].rename(
        columns={"close": "trend_close", "ema_trend": "trend_ema"})

    merged = pd.merge_asof(
        df_15m.sort_values("open_time"),
        df_1h_trend.sort_values("open_time"),
        on="open_time", direction="backward"
    )
    if not funding_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("open_time"),
            funding_df.rename(columns={"time": "open_time"}).sort_values("open_time"),
            on="open_time", direction="backward"
        )
        merged["funding"] = merged["funding"].fillna(0.0)
    else:
        merged["funding"] = 0.0

    merged["date"] = merged["open_time"].dt.date
    if not fg_df.empty:
        merged = merged.merge(fg_df, on="date", how="left")
        merged["fg"] = merged["fg"].ffill().fillna(50)
    else:
        merged["fg"] = 50

    merged["res20"] = merged["high"].rolling(20).max()
    merged["sup20"] = merged["low"].rolling(20).min()
    merged["move5"] = (merged["high"] - merged["low"]).rolling(5).mean() / merged["close"]

    wallet = {
        "cash": STARTING_CASH_PER_COIN,
        "coin_qty": 0.0,
        "entry_price": None,
        "peak_price": None,
        "trough_price": None,
        "position_type": None,
        "partial_sold": False,
        "num_trades": 0,
        "win_trades": 0,
        "loss_trades": 0,
        "day": None,
        "day_start_value": STARTING_CASH_PER_COIN,
        "day_halted": False,
        "peak_value": STARTING_CASH_PER_COIN,
        "max_drawdown": 0.0,
    }

    trades = []
    equity_curve = []
    start_idx = max(TREND_EMA, 25)

    for i in range(start_idx, len(merged)):
        row = merged.iloc[i]
        prev = merged.iloc[i - 1]
        ts = row["open_time"]
        price = row["close"]
        atr = row["atr"]

        day = ts.date().isoformat()
        if wallet["day"] != day:
            wallet["day"] = day
            wallet["day_start_value"] = wallet["cash"] + wallet["coin_qty"] * price
            wallet["day_halted"] = False

        base_signal = generate_signal(prev, row)
        final_signal, size_mult = apply_filters(
            base_signal, row, row.get("trend_close", np.nan), row.get("trend_ema", np.nan),
            row.get("funding", 0.0), row.get("fg", 50), row.get("move5", 0.0)
        )

        action, reason, qty = "HOLD", "", 0.0

        # EXIT
        if wallet["coin_qty"] > 0:
            exit_action, exit_reason, exit_qty = check_exits(wallet, price, atr, final_signal)

            if exit_action in ("SELL", "COVER", "PARTIAL"):
                proceeds = exit_qty * price
                fee = proceeds * FEE_PCT
                entry_price = wallet["entry_price"]
                was_short = wallet.get("position_type") == "SHORT"

                if exit_action == "COVER" or (exit_action == "PARTIAL" and wallet.get("position_type") == "SHORT"):
                    wallet["cash"] -= proceeds + fee
                else:
                    wallet["cash"] += proceeds - fee

                wallet["coin_qty"] -= exit_qty
                wallet["num_trades"] += 1

                if exit_action in ("SELL", "COVER"):
                    if wallet["coin_qty"] <= 1e-9:
                        wallet["coin_qty"] = 0.0
                        wallet["entry_price"] = None
                        wallet["peak_price"] = None
                        wallet["trough_price"] = None
                        wallet["position_type"] = None
                        wallet["partial_sold"] = False

                    if entry_price is not None:
                        won = (price < entry_price) if was_short else (price > entry_price)
                        if won:
                            wallet["win_trades"] += 1
                        else:
                            wallet["loss_trades"] += 1

                elif exit_action == "PARTIAL":
                    wallet["partial_sold"] = True

                action = exit_action
                reason = exit_reason
                qty = exit_qty

                trades.append({
                    "timestamp": ts, "symbol": symbol, "action": action, "reason": reason,
                    "price": price, "qty": qty, "cash": wallet["cash"],
                    "coin_qty": wallet["coin_qty"],
                    "value": wallet["cash"] + wallet["coin_qty"] * price
                })

        # ENTRY
        elif not wallet["day_halted"] and wallet["cash"] > 0 and row.get("move5", 0) >= MIN_MOVE_PCT and size_mult > 0:
            if final_signal == "BUY":
                buy_qty, spend = calculate_position(wallet, price, atr, size_mult)
                if buy_qty > 0:
                    wallet["coin_qty"] += buy_qty
                    wallet["cash"] -= spend
                    wallet["entry_price"] = price
                    wallet["peak_price"] = price
                    wallet["position_type"] = "LONG"
                    wallet["partial_sold"] = False
                    wallet["num_trades"] += 1
                    action, reason, qty = "BUY", "BUY", buy_qty
                    trades.append({
                        "timestamp": ts, "symbol": symbol, "action": action, "reason": reason,
                        "price": price, "qty": qty, "cash": wallet["cash"],
                        "coin_qty": wallet["coin_qty"],
                        "value": wallet["cash"] + wallet["coin_qty"] * price
                    })

            elif final_signal == "SHORT":
                short_qty, spend = calculate_position(wallet, price, atr, size_mult)
                if short_qty > 0:
                    wallet["coin_qty"] += short_qty
                    wallet["cash"] += spend
                    wallet["entry_price"] = price
                    wallet["trough_price"] = price
                    wallet["position_type"] = "SHORT"
                    wallet["partial_sold"] = False
                    wallet["num_trades"] += 1
                    action, reason, qty = "SHORT", "SHORT", short_qty
                    trades.append({
                        "timestamp": ts, "symbol": symbol, "action": action, "reason": reason,
                        "price": price, "qty": qty, "cash": wallet["cash"],
                        "coin_qty": wallet["coin_qty"],
                        "value": wallet["cash"] + wallet["coin_qty"] * price
                    })

        # Equity & Drawdown
        value = wallet["cash"] - wallet["coin_qty"] * price if wallet.get("position_type") == "SHORT" else wallet["cash"] + wallet["coin_qty"] * price
        if value > wallet["peak_value"]:
            wallet["peak_value"] = value
        dd = (wallet["peak_value"] - value) / wallet["peak_value"] if wallet["peak_value"] > 0 else 0
        if dd > wallet["max_drawdown"]:
            wallet["max_drawdown"] = dd

        day_loss = (wallet["day_start_value"] - value) / wallet["day_start_value"] if wallet["day_start_value"] > 0 else 0
        if day_loss >= MAX_DAILY_LOSS_PCT:
            wallet["day_halted"] = True

        equity_curve.append({"timestamp": ts, "value": value})

    final_value = wallet["cash"] - wallet["coin_qty"] * merged.iloc[-1]["close"] if wallet.get("position_type") == "SHORT" else wallet["cash"] + wallet["coin_qty"] * merged.iloc[-1]["close"]
    total_trades = wallet["num_trades"]
    wins = wallet["win_trades"]
    losses = wallet["loss_trades"]
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    return {
        "symbol": symbol,
        "start_value": STARTING_CASH_PER_COIN,
        "end_value": final_value,
        "return_pct": (final_value - STARTING_CASH_PER_COIN) / STARTING_CASH_PER_COIN * 100,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": wallet["max_drawdown"] * 100,
        "trades": trades,
        "equity": equity_curve
    }


def main():
    print("Starting Backtest v6.2 (Peak Trailing Stop)...")
    end = datetime.now(timezone.utc) - timedelta(days=BACKTEST_OFFSET_DAYS)
    start = end - timedelta(days=BACKTEST_DAYS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    fg_df = fetch_fear_greed_history()
    print(f"Fear & Greed history loaded: {len(fg_df)} days")

    all_results = []
    all_trades = []

    for symbol in COINS:
        print(f"\nProcessing {symbol}...")
        df_15m = fetch_full_klines(symbol, PRIMARY_INTERVAL, start_ms, end_ms)
        df_1h = fetch_full_klines(symbol, TREND_INTERVAL, start_ms, end_ms)
        funding_df = fetch_funding_history(symbol, start_ms, end_ms)

        if df_15m.empty or df_1h.empty:
            print(f"Skipping {symbol} - no data")
            continue

        result = backtest_symbol(symbol, df_15m, df_1h, funding_df, fg_df)
        all_results.append(result)
        all_trades.extend(result["trades"])

        print(f"  Return: {result['return_pct']:.2f}% | WinRate: {result['win_rate']:.1f}% | "
              f"MaxDD: {result['max_drawdown']:.2f}% | Trades: {result['total_trades']}")

    # Save trades
    if all_trades:
        pd.DataFrame(all_trades).to_csv(RESULTS_CSV, index=False)
        print(f"\nTrades saved to {RESULTS_CSV}")

    # Summary
    with open(SUMMARY_MD, "w") as f:
        f.write("# Backtest Summary — v6.2 (Peak Trailing Stop)\n\n")
        f.write(f"Period: {start.date()} to {end.date()} ({BACKTEST_DAYS} days)\n\n")

        for r in all_results:
            f.write(f"## {r['symbol']}\n\n")
            f.write(f"- Start: ${r['start_value']:,.2f} → End: ${r['end_value']:,.2f} ({r['return_pct']:+.2f}%)\n")
            f.write(f"- Total trades: {r['total_trades']}\n")
            f.write(f"- Win rate: {r['win_rate']:.1f}% ({r['wins']}W / {r['losses']}L)\n")
            f.write(f"- Max drawdown: {r['max_drawdown']:.2f}%\n\n")

    print(f"Summary saved to {SUMMARY_MD}")
    print("\nBacktest completed.")


if __name__ == "__main__":
    main()
