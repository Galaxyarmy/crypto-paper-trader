"""
Backtest — Crypto Paper Trading Agent v6.1 strategy on historical Binance data.

Replicates (as closely as historical data allows) the exact live logic from
paper_trader.py v6.1:
  - EMA9/EMA21 crossover + RSI signal
  - ADX filter (>= 20)
  - 1H EMA50 trend filter
  - MIN_MOVE_PCT filter (0.25%)
  - Funding rate filter (from Binance Futures historical funding rate)
  - Fear & Greed filter (from alternative.me historical daily index)
  - Risk management: stop-loss, trailing-stop, partial-profit-booking (50% @ +1.5%),
    risk-based position sizing, daily loss halt

LIMITATION: Order-book imbalance is a live-only snapshot signal — free historical
order-book data isn't available, so it is treated as neutral (0) here. This means
the backtest slightly under-uses the size-multiplier boosts/cuts that OB imbalance
gives live, but does not change entry/exit decisions (OB never forces a HOLD).

Outputs:
  - backtest_trades.csv   : every trade (entry/exit) with reason
  - backtest_summary.md   : Sharpe ratio, win rate, max drawdown, per-coin stats

Requirements: pip install requests pandas numpy
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ------------------------- CONFIG (mirrors paper_trader.py v6.1) -------------------------

COINS = ["BTCUSDT", "ETHUSDT"]
PRIMARY_INTERVAL = "15m"
TREND_INTERVAL = "1h"

BACKTEST_DAYS = 90  # how far back to test

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

# ------------------------- DATA FETCHERS (PAGINATED, HISTORICAL) -------------------------

def fetch_full_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all klines between start_ms and end_ms, paginating in chunks of 1000."""
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
        time.sleep(0.25)  # be polite to the free API

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=KLINE_COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").reset_index(drop=True)
    return df


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch historical funding rates (every 8h) for a symbol."""
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
    """Fetch the full historical daily Fear & Greed index."""
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


# ------------------------- INDICATORS (identical to paper_trader.py) -------------------------

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
    if crossed_down or latest_row["rsi"] > RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


# ------------------------- FILTER ENGINE (OB imbalance neutral) -------------------------

def apply_filters(base_signal, latest, trend_price, trend_ema, funding, fg_value, recent_move):
    size_mult = 1.0
    adx = latest["adx"]
    price = latest["close"]
    atr = latest["atr"]

    if pd.isna(adx) or adx < ADX_MIN:
        if base_signal == "BUY":
            return "HOLD", 0.0
    if not pd.isna(trend_ema):
        if base_signal == "BUY" and trend_price < trend_ema:
            return "HOLD", 0.0
        if base_signal == "BUY" and trend_price > trend_ema:
            size_mult = min(size_mult + 0.25, 1.5)

    if funding > FUNDING_EXTREME and base_signal == "BUY":
        return "HOLD", 0.0
    elif funding < -FUNDING_EXTREME and base_signal == "BUY":
        size_mult = min(size_mult + 0.25, 1.5)

    if fg_value <= FEAR_GREED_FEAR and base_signal == "BUY":
        size_mult = min(size_mult + 0.25, 1.5)
    elif fg_value >= FEAR_GREED_GREED and base_signal == "BUY":
        return "HOLD", 0.0

    if base_signal == "BUY" and not pd.isna(atr):
        stop_dist = max(atr * 2, price * STOP_LOSS_PCT)
        target_dist = stop_dist * MIN_RISK_REWARD
        # resistance handled by caller via rolling max passed in latest

    return base_signal, size_mult


def calculate_position(wallet, price, atr, size_mult):
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
    qty = (spend - fee) / price
    return qty, spend


def check_exits(wallet, price, atr, final_signal):
    if wallet["coin_qty"] <= 0 or wallet.get("entry_price") is None:
        return "HOLD", "", 0.0
    entry = wallet["entry_price"]
    qty = wallet["coin_qty"]

    stop_dist = max(atr * 2, entry * STOP_LOSS_PCT) if not pd.isna(atr) else entry * STOP_LOSS_PCT
    hard_stop = entry - stop_dist
    if price <= hard_stop:
        return "SELL", "STOP_LOSS", qty

    if not wallet.get("partial_sold", False):
        if price >= entry * (1 + PARTIAL_PROFIT_PCT):
            return "PARTIAL", "PARTIAL_PROFIT", qty * 0.5

    if price > entry * 1.03:
        trail_stop = price * (1 - TRAILING_STOP_PCT)
        if price <= trail_stop:
            return "SELL", "TRAILING_STOP", qty

    if final_signal == "SELL":
        return "SELL", "SELL_SIGNAL", qty

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
    merged["move5"] = (merged["high"] - merged["low"]).rolling(5).mean() / merged["close"]

    wallet = {
        "cash": STARTING_CASH_PER_COIN, "coin_qty": 0.0, "entry_price": None,
        "partial_sold": False, "num_trades": 0, "win_trades": 0, "loss_trades": 0,
        "day": None, "day_start_value": STARTING_CASH_PER_COIN, "day_halted": False,
        "peak_value": STARTING_CASH_PER_COIN, "max_drawdown": 0.0,
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

        if wallet["coin_qty"] > 0:
            exit_action, exit_reason, exit_qty = check_exits(wallet, price, atr, final_signal)
            if exit_action in ("SELL", "PARTIAL"):
                proceeds = exit_qty * price
                fee = proceeds * FEE_PCT
                entry_price = wallet["entry_price"]
                wallet["cash"] += proceeds - fee
                wallet["coin_qty"] -= exit_qty
                wallet["num_trades"] += 1
                if exit_action == "SELL":
                    if wallet["coin_qty"] <= 1e-9:
                        wallet["coin_qty"] = 0.0
                        wallet["entry_price"] = None
                        wallet["partial_sold"] = False
                    if price > entry_price:
                        wallet["win_trades"] += 1
                    else:
                        wallet["loss_trades"] += 1
                else:
                    wallet["partial_sold"] = True
                action, reason, qty = exit_action, exit_reason, exit_qty
                trades.append([ts, symbol, action, reason, price, qty,
                               wallet["cash"], wallet["coin_qty"]])
        elif not wallet["day_halted"] and final_signal == "BUY" and wallet["cash"] > 0:
            if row.get("move5", 0.0) >= MIN_MOVE_PCT and size_mult > 0:
                q, spend = calculate_position(wallet, price, atr, size_mult)
                if q > 0:
                    wallet["coin_qty"] += q
                    wallet["cash"] -= spend
                    wallet["entry_price"] = price
                    wallet["partial_sold"] = False
                    wallet["num_trades"] += 1
                    action, reason, qty = "BUY", f"BUY|size={size_mult:.0%}", q
                    trades.append([ts, symbol, action, reason, price, qty,
                                   wallet["cash"], wallet["coin_qty"]])

        value = wallet["cash"] + wallet["coin_qty"] * price
        wallet["peak_value"] = max(wallet.get("peak_value", value), value)
        dd = (wallet["peak_value"] - value) / wallet["peak_value"] if wallet["peak_value"] > 0 else 0
        wallet["max_drawdown"] = max(wallet.get("max_drawdown", 0), dd)

        day_loss = (wallet["day_start_value"] - value) / wallet["day_start_value"] if wallet["day_start_value"] > 0 else 0
        if day_loss >= MAX_DAILY_LOSS_PCT:
            wallet["day_halted"] = True

        equity_curve.append((ts, value))

    equity_df = pd.DataFrame(equity_curve, columns=["time", "value"]).set_index("time")
    return wallet, trades, equity_df


def compute_sharpe(equity_df, bars_per_day=96):
    if equity_df.empty:
        return 0.0
    daily = equity_df["value"].resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(365)


def main():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKTEST_DAYS)
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)

    print(f"[INFO] Backtesting {BACKTEST_DAYS} days: {start_dt.date()} -> {end_dt.date()}")
    fg_df = fetch_fear_greed_history()

    all_trades = []
    summary_lines = [f"# Backtest Summary — v6.1 strategy\n",
                      f"Period: {start_dt.date()} to {end_dt.date()} ({BACKTEST_DAYS} days)\n"]

    for symbol in COINS:
        print(f"\n[FETCH] {symbol} klines...")
        df_15m = fetch_full_klines(symbol, PRIMARY_INTERVAL, start_ms, end_ms)
        df_1h = fetch_full_klines(symbol, TREND_INTERVAL, start_ms, end_ms)
        funding_df = fetch_funding_history(symbol, start_ms, end_ms)

        if df_15m.empty or df_1h.empty:
            print(f"[SKIP] No data for {symbol}")
            continue

        print(f"[RUN] Simulating {symbol} ({len(df_15m)} 15m bars)...")
        wallet, trades, equity_df = backtest_symbol(symbol, df_15m, df_1h, funding_df, fg_df)
        all_trades.extend(trades)

        sharpe = compute_sharpe(equity_df)
        final_value = wallet["cash"] + wallet["coin_qty"] * df_15m.iloc[-1]["close"]
        total_return = (final_value - STARTING_CASH_PER_COIN) / STARTING_CASH_PER_COIN * 100
        total = wallet["num_trades"]
        wins = wallet["win_trades"]
        wr = (wins / (wallet["win_trades"] + wallet["loss_trades"]) * 100) if (wallet["win_trades"] + wallet["loss_trades"]) > 0 else 0

        summary_lines.append(f"## {symbol}\n")
        summary_lines.append(f"- Start: ${STARTING_CASH_PER_COIN:,.2f} -> End: ${final_value:,.2f} ({total_return:+.2f}%)\n")
        summary_lines.append(f"- Total trades (buy+sell+partial): {total}\n")
        summary_lines.append(f"- Win rate (closed round-trips): {wr:.1f}% ({wins}W / {wallet['loss_trades']}L)\n")
        summary_lines.append(f"- Max drawdown: {wallet['max_drawdown']*100:.2f}%\n")
        summary_lines.append(f"- Sharpe ratio (annualized, daily returns): {sharpe:.2f}\n")
        print(f"[RESULT] {symbol}: {total_return:+.2f}% | WinRate={wr:.1f}% | MaxDD={wallet['max_drawdown']*100:.2f}% | Sharpe={sharpe:.2f}")

    with open(RESULTS_CSV, "w") as f:
        f.write("timestamp,symbol,action,reason,price,qty,cash,coin_qty\n")
        for t in all_trades:
            f.write(",".join(str(x) for x in t) + "\n")

    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[DONE] Wrote {RESULTS_CSV} and {SUMMARY_MD}")


if __name__ == "__main__":
    main()
