"""
Backtest — Crypto Paper Trading Agent v7.0 (SMC / Isolated-Margin Futures)

Shares engine.py's signal generation and accounting with paper_trader.py
byte-for-byte — this was Codex's #1 priority fix (live and backtest had
silently diverged before). The only things this file owns are: historical
data fetching, and the bar-by-bar simulation loop.

Look-ahead / realism fixes vs the old v6.2 backtest (Codex findings #2-#6):
- Single timeframe (15m) — no 1h-trend merge, so that whole look-ahead
  class doesn't exist anymore.
- Sweep reference levels use .shift(1) before the rolling window
  (engine.add_sweep_signal), so a candle's own high/low can never leak
  into the level it's compared against.
- A signal confirmed on candle i is only ACTED ON at candle i+1's open —
  never filled at the same candle's own close.
- Exits are OHLC-path aware: a stop/target touched intrabar is caught
  even if the candle's close ends up back on the "safe" side. Same-candle
  ambiguity (both stop and target touched) resolves pessimistically —
  the stop wins.
- Funding is charged only at 8h boundaries, not every bar.

Known limitation (documented, not hidden): Binance's free openInterestHist
endpoint only serves ~30 days of history. BACKTEST_DAYS is set to 25 in
config.py specifically to stay inside that window. If you widen it past
~30 days, the OI series will come back empty for the older portion and
every signal in that stretch will fail the OI-confirmation filter (fails
safe to HOLD, per engine.generate_signal) — the backtest won't crash, but
it will effectively be flat-and-untested for whatever's outside the OI
window. The console output prints how many candles actually got OI
coverage so this isn't silently swallowed.
"""

import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

import config as cfg
import engine as eng


# -------------------- HISTORICAL FETCHERS --------------------

def fetch_klines_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": cfg.PRIMARY_INTERVAL,
                  "startTime": cur, "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(cfg.KLINES_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] Kline page fetch failed for {symbol} at {cur}: {e}")
            break
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cfg.KLINE_COLS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.drop_duplicates(subset="open_time").reset_index(drop=True)


def fetch_oi_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated fetch, but Binance silently caps this at ~30 days back
    regardless of what start_ms asks for — see module docstring."""
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "period": cfg.PRIMARY_INTERVAL,
                  "startTime": cur, "endTime": end_ms, "limit": 500}
        try:
            resp = requests.get(cfg.OPEN_INTEREST_HIST_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] OI history page fetch failed for {symbol} at {cur}: {e}")
            break
        if not data:
            break
        rows.extend(data)
        cur = data[-1]["timestamp"] + 1
        if len(data) < 500:
            break
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame(columns=["time", "open_interest"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    return df[["time", "open_interest"]].sort_values("time").reset_index(drop=True)


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(cfg.FUNDING_RATE_HIST_ENDPOINT, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Funding history fetch failed for {symbol} at {cur}: {e}")
            break
        if not data:
            break
        rows.extend(data)
        cur = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame(columns=["time", "funding"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding"] = df["fundingRate"].astype(float)
    return df[["time", "funding"]].sort_values("time").reset_index(drop=True)


# -------------------- SIMULATION --------------------

def run_backtest(symbol: str, df: pd.DataFrame, oi_df: pd.DataFrame,
                  funding_df: pd.DataFrame) -> dict:
    df = eng.align_oi(df, oi_df)
    df = eng.add_atr(df, cfg.ATR_PERIOD)
    df = eng.add_sweep_signal(df)
    df = eng.add_oi_delta(df)

    df["signal"] = df.apply(eng.generate_signal, axis=1)
    raw_sweeps = df["sweep_signal"].notna().sum()
    sweep_in_oi_window = ((df["sweep_signal"].notna()) & (df["open_interest"].notna())).sum()
    print(f"  Sweeps inside OI-covered window: {sweep_in_oi_window}")
    oi_moves = df.loc[df["sweep_signal"].notna(), "oi_pct_change"]
    print(f"  OI %% change on those sweep candles: {oi_moves.dropna().round(4).tolist()}")
    print(f"  Raw sweep candles (pre-OI-filter): {raw_sweeps}")
    oi_covered = df["open_interest"].notna().sum()
    print(f"  OI coverage: {oi_covered}/{len(df)} candles "
          f"({df['open_time'].iloc[0].date() if oi_covered else 'n/a'} onward)")

    if not funding_df.empty:
        df = pd.merge_asof(df.sort_values("open_time"),
                            funding_df.sort_values("time"),
                            left_on="open_time", right_on="time", direction="backward")
        df["funding"] = df["funding"].fillna(0.0)
    else:
        df["funding"] = 0.0

    wallet = eng.default_wallet()
    trades = []
    equity_curve = []
    warmup = cfg.SWEEP_LOOKBACK_CANDLES + 5
    pending_entry = None   # signal confirmed on the PRIOR candle, filled at THIS candle's open

    for i in range(warmup, len(df)):
        candle = df.iloc[i]
        ts = candle["open_time"]
        day = ts.date().isoformat()
        eng.reset_day_if_needed(wallet, candle["open"], day)

        funding_paid = eng.apply_funding_if_due(wallet, candle.get("funding", 0.0), ts)

        # ---- 1. Execute a deferred entry at THIS candle's open ----
        if pending_entry is not None and wallet["position_side"] is None and not wallet["day_halted"]:
            side = pending_entry
            slip = 1 + cfg.BACKTEST_SLIPPAGE_PCT if side == "LONG" else 1 - cfg.BACKTEST_SLIPPAGE_PCT
            fill_price = candle["open"] * slip
            atr = candle["atr"]
            if pd.notna(atr):
                stop_dist = max(atr * cfg.ATR_STOP_MULT, fill_price * cfg.STOP_LOSS_PCT)
                qty, margin, fee = eng.size_position(wallet, fill_price, stop_dist, fill_price)
                if qty > 0:
                    eng.open_position(wallet, side, fill_price, qty, margin, fee, stop_dist, now_ts=ts)
                    trades.append({"timestamp": ts, "symbol": symbol, "action": "OPEN",
                                    "side": side, "price": fill_price, "qty": qty, "fee": fee,
                                    "pnl": 0.0, "position_id": wallet["current_position_id"]})
        pending_entry = None

        # ---- 2. Exits, using THIS candle's own OHLC path ----
        if wallet["position_side"] is not None:
            action, reason, qty, fill_price = eng.check_exits(
                wallet, candle["high"], candle["low"], candle["close"], candle["signal"])
            if action != "HOLD" and qty > 0:
                pos_id = wallet["current_position_id"]
                pnl, fee = eng.close_position(wallet, fill_price, qty)
                trades.append({"timestamp": ts, "symbol": symbol, "action": action,
                                "reason": reason, "price": fill_price, "qty": qty, "fee": fee,
                                "pnl": pnl, "position_id": pos_id})

        # ---- 3. Register a fresh signal to be filled NEXT candle ----
        if wallet["position_side"] is None and not wallet["day_halted"]:
            if candle["signal"] in ("LONG", "SHORT"):
                pending_entry = candle["signal"]

        eq = eng.update_drawdown_and_halt(wallet, candle["close"])
        equity_curve.append({"timestamp": ts, "equity": eq})

    final_price = df.iloc[-1]["close"]
    final_equity = eng.equity(wallet, final_price)
    wins, losses = wallet["win_trades"], wallet["loss_trades"]
    total_closed = wins + losses

    return {
        "symbol": symbol,
        "start_value": cfg.STARTING_CASH_PER_COIN,
        "end_value": final_equity,
        "return_pct": (final_equity - cfg.STARTING_CASH_PER_COIN) / cfg.STARTING_CASH_PER_COIN * 100,
        "positions_opened": wallet["num_positions_opened"],
        "wins": wins, "losses": losses,
        "win_rate": (wins / total_closed * 100) if total_closed > 0 else 0.0,
        "realized_pnl_total": wallet["realized_pnl_total"],
        "max_drawdown_pct": wallet["max_drawdown"] * 100,
        "oi_coverage_candles": int(oi_covered),
        "total_candles": len(df),
        "trades": trades,
        "equity_curve": equity_curve,
    }


# -------------------- MAIN --------------------

def main():
    print(f"Starting backtest v{cfg.SCHEMA_VERSION} ({cfg.BACKTEST_DAYS} days, "
          f"liquidity-sweep + OI-delta, isolated-margin futures)...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.BACKTEST_DAYS)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    all_results = []
    all_trades = []

    for symbol in cfg.COINS:
        print(f"\nFetching {symbol}...")
        df = fetch_klines_history(symbol, start_ms, end_ms)
        oi_df = fetch_oi_history(symbol, start_ms, end_ms)
        funding_df = fetch_funding_history(symbol, start_ms, end_ms)

        if df.empty or len(df) < cfg.SWEEP_LOOKBACK_CANDLES + 10:
            print(f"  Skipping {symbol} — not enough kline data returned.")
            continue

        result = run_backtest(symbol, df, oi_df, funding_df)
        all_results.append(result)
        all_trades.extend(result["trades"])
        print(f"  Return: {result['return_pct']:+.2f}% | WinRate: {result['win_rate']:.1f}% "
              f"({result['wins']}W/{result['losses']}L) | MaxDD: {result['max_drawdown_pct']:.2f}% | "
              f"Positions opened: {result['positions_opened']}")

    if all_trades:
        pd.DataFrame(all_trades).to_csv(cfg.BACKTEST_TRADES_CSV, index=False)
        print(f"\nTrades saved to {cfg.BACKTEST_TRADES_CSV}")

    with open(cfg.BACKTEST_SUMMARY_MD, "w") as f:
        f.write(f"# Backtest Summary — v{cfg.SCHEMA_VERSION} "
                f"(liquidity sweep + OI delta, isolated-margin futures)\n\n")
        f.write(f"Period: {start.date()} to {end.date()} ({cfg.BACKTEST_DAYS} days)\n\n")
        f.write("Note: entries fill at the next candle's open after the signal candle "
                "closes (not the signal candle's own close). Stops/targets/liquidation "
                "are checked against each candle's actual high/low path, not close-only. "
                "Funding is charged at 8h boundaries only.\n\n")
        for r in all_results:
            f.write(f"## {r['symbol']}\n\n")
            f.write(f"- Start: ${r['start_value']:,.2f} → End: ${r['end_value']:,.2f} "
                    f"({r['return_pct']:+.2f}%)\n")
            f.write(f"- Positions opened: {r['positions_opened']} | "
                    f"Closed round-trips: {r['wins'] + r['losses']} "
                    f"({r['wins']}W / {r['losses']}L, {r['win_rate']:.1f}% win rate)\n")
            f.write(f"- Realized PnL total: ${r['realized_pnl_total']:,.2f}\n")
            f.write(f"- Max drawdown: {r['max_drawdown_pct']:.2f}%\n")
            f.write(f"- OI data coverage: {r['oi_coverage_candles']}/{r['total_candles']} candles "
                    f"— signals outside this window fail the OI filter safe-to-HOLD "
                    f"(see module docstring in backtest.py)\n\n")
    print(f"Summary saved to {cfg.BACKTEST_SUMMARY_MD}")
    print("\nBacktest completed.")


if __name__ == "__main__":
    main()
