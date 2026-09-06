"""
Crypto Paper Trading Agent v7.0 — SMC / Isolated-Margin Futures Rewrite

Runs once per invocation (designed for a GitHub Actions cron job every
15 minutes — no persistent connection, so no WebSocket; see config.py's
top-of-file note for why). All signal and accounting logic lives in
engine.py and is shared byte-for-byte with backtest.py.

Changes vs v6.2:
- Removed EMA9/21, RSI, Fear & Greed, Order Book Imbalance entirely.
- New entry signal: 24h liquidity sweep + Open Interest delta (engine.py).
- Moved from Binance spot klines to Binance USDT-M Futures klines/OI/funding.
- Accounting rebuilt around isolated-margin collateral + side-aware PnL —
  fixes the short-equity-sign bug and the short win/loss-counting bug
  that Codex's audit found in v6.2's live script.
- Exits now walk the just-closed candle's high/low path (not close-only),
  and a liquidation check runs before the stop check.
- state.json schema changed completely (position_side/qty/margin_locked
  instead of coin_qty/cash-only). Old v6.2 state is NOT migrated — it was
  built on accounting that this rewrite intentionally replaces, and it's
  paper money, so this starts every symbol fresh at $5,000 rather than
  attempting a lossy conversion. You'll see a one-time log line about this.

Requirements: pip install requests pandas
"""

import json
import os
import csv
import time
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

import config as cfg
import engine as eng


# -------------------- HTTP SESSION --------------------

def get_robust_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=4, backoff_factor=1,
                     status_forcelist=[429, 500, 502, 503, 504],
                     raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP = get_robust_session()


# -------------------- STATE --------------------

def default_state() -> dict:
    return {
        "schema_version": cfg.SCHEMA_VERSION,
        "coins": {symbol: eng.default_wallet() for symbol in cfg.COINS},
    }


def load_state() -> dict:
    if not os.path.isfile(cfg.STATE_FILE):
        return default_state()
    with open(cfg.STATE_FILE) as f:
        try:
            state = json.load(f)
        except Exception:
            print("[WARN] state.json unreadable — starting fresh.")
            return default_state()

    if state.get("schema_version") != cfg.SCHEMA_VERSION:
        print(f"[INFO] state.json is schema {state.get('schema_version')}, "
              f"this build is {cfg.SCHEMA_VERSION}. Old accounting model is "
              f"incompatible with the new isolated-margin engine — resetting "
              f"all symbols to fresh ${cfg.STARTING_CASH_PER_COIN:,.0f} paper "
              f"balances rather than converting stale numbers. (No real money "
              f"involved — this is a paper simulator.)")
        return default_state()

    # Fill in any symbols missing from an otherwise-current state file.
    for symbol in cfg.COINS:
        state["coins"].setdefault(symbol, eng.default_wallet())
    return state


def save_state(state: dict):
    with open(cfg.STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# -------------------- DATA FETCHERS (Binance USDT-M Futures) --------------------

def fetch_klines(symbol: str, limit: int = 150) -> pd.DataFrame:
    """Fetches recent futures klines, drops the still-forming last candle."""
    params = {"symbol": symbol, "interval": cfg.PRIMARY_INTERVAL, "limit": limit}
    try:
        resp = HTTP.get(cfg.KLINES_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=cfg.KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.iloc[:-1].reset_index(drop=True)   # drop unfinished candle
    except Exception as e:
        print(f"[ERROR] Kline fetch failed for {symbol}: {e}")
        return pd.DataFrame()


def fetch_oi_hist(symbol: str, limit: int = 12) -> pd.DataFrame:
    """Last `limit` open-interest points at the primary interval. Binance's
    free tier only serves ~30 days of this — fine for live (we only need
    OI_DELTA_LOOKBACK_CANDLES+buffer points), but see BACKTEST_DAYS in
    config.py for how backtest.py stays inside that window."""
    params = {"symbol": symbol, "period": cfg.PRIMARY_INTERVAL, "limit": limit}
    try:
        resp = HTTP.get(cfg.OPEN_INTEREST_HIST_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame(columns=["time", "open_interest"])
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        return df[["time", "open_interest"]].sort_values("time").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] OI history fetch failed for {symbol}: {e}")
        return pd.DataFrame(columns=["time", "open_interest"])


def fetch_funding_rate(symbol: str) -> float:
    try:
        resp = HTTP.get(cfg.MARK_PRICE_ENDPOINT, params={"symbol": symbol}, timeout=8)
        resp.raise_for_status()
        return float(resp.json()["lastFundingRate"])
    except Exception as e:
        print(f"[WARN] Funding rate fetch failed for {symbol}: {e}")
        return 0.0


# -------------------- LOGGING --------------------

def log_row(row: list):
    file_exists = os.path.isfile(cfg.LOG_FILE)
    with open(cfg.LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "action", "reason", "signal",
                "price", "fill_price", "qty", "fee", "realized_pnl",
                "cash", "margin_locked", "equity", "position_side",
                "position_id", "funding_paid",
            ])
        writer.writerow(row)


# -------------------- MAIN --------------------

def main():
    start = time.time()
    state = load_state()
    timestamp = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date().isoformat()

    for symbol in cfg.COINS:
        try:
            print(f"\n[ANALYZING] {symbol}")
            df = fetch_klines(symbol, limit=cfg.SWEEP_LOOKBACK_CANDLES + 20)
            if df.empty or len(df) < cfg.SWEEP_LOOKBACK_CANDLES + 2:
                print(f"[SKIP] Not enough kline data for {symbol}.")
                continue

            oi_df = fetch_oi_hist(symbol, limit=cfg.OI_DELTA_LOOKBACK_CANDLES + 6)
            df = eng.align_oi(df, oi_df)
            df = eng.add_atr(df, cfg.ATR_PERIOD)
            df = eng.add_sweep_signal(df)
            df = eng.add_oi_delta(df)

            latest = df.iloc[-1]
            price = float(latest["close"])
            candle_high = float(latest["high"])
            candle_low = float(latest["low"])
            atr = float(latest["atr"]) if pd.notna(latest["atr"]) else None
            funding = fetch_funding_rate(symbol)

            final_signal = eng.generate_signal(latest)
            print(f"  Price={price:,.2f} | sweep={latest.get('sweep_signal')} | "
                  f"oi_chg={latest.get('oi_pct_change')} | signal={final_signal} | "
                  f"funding={funding:.4%}")

            wallet = state["coins"][symbol]
            eng.reset_day_if_needed(wallet, price, today)
            now_ts = pd.Timestamp(timestamp)
            funding_paid = eng.apply_funding_if_due(wallet, funding, now_ts)

            action, reason, qty, fill_price = "HOLD", "", 0.0, 0.0
            fee = 0.0
            realized_pnl = 0.0

            if wallet["position_side"] is not None:
                action, reason, qty, fill_price = eng.check_exits(
                    wallet, candle_high, candle_low, price, final_signal)
                if action != "HOLD" and qty > 0:
                    realized_pnl, fee = eng.close_position(wallet, fill_price, qty)
                    print(f"  >> {action} {reason} qty={qty:.6f} @ {fill_price:.2f} "
                          f"pnl={realized_pnl:+.2f}")
            elif wallet["day_halted"]:
                reason = "DAILY_LOSS_HALT"
            elif final_signal in ("LONG", "SHORT") and atr is not None:
                stop_dist = max(atr * cfg.ATR_STOP_MULT, price * cfg.STOP_LOSS_PCT)
                entry_qty, margin, entry_fee = eng.size_position(wallet, price, stop_dist, price)
                if entry_qty > 0:
                    eng.open_position(wallet, final_signal, price, entry_qty, margin,
                                       entry_fee, stop_dist, now_ts=now_ts)
                    action, reason, qty, fill_price, fee = "OPEN", final_signal, entry_qty, price, entry_fee
                    print(f"  >> OPEN {final_signal} qty={entry_qty:.6f} @ {price:.2f} "
                          f"stop={wallet['stop_price']:.2f} liq={wallet['liquidation_price']:.2f}")
                else:
                    reason = "SIZING_FAILED"

            eq = eng.update_drawdown_and_halt(wallet, price)
            log_row([
                timestamp, symbol, action, reason, final_signal, price, fill_price,
                qty, round(fee, 4), round(realized_pnl, 4), round(wallet["cash"], 2),
                round(wallet["margin_locked"], 2), round(eq, 2),
                wallet["position_side"], wallet.get("current_position_id"),
                round(funding_paid, 4),
            ])
            print(f"  Equity=${eq:,.2f} | position={wallet['position_side']} | "
                  f"realized_total=${wallet.get('realized_pnl_total', 0):,.2f}")

        except Exception as e:
            print(f"[ERROR] Critical failure processing {symbol}: {e}")

    save_state(state)
    print(f"\n[DONE] Completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
