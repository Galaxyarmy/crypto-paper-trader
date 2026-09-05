"""
engine.py — Shared logic for the futures paper trader.

Both paper_trader.py (live) and backtest.py (simulation) import
everything from here. This is deliberate: Codex's audit's #1 and #4
findings were that live and backtest silently drifted apart (different
config, different filter logic). If it can drift, it will drift again —
so there is now exactly one copy of the signal logic and exactly one
copy of the accounting logic, and each file just wires it up to a
different data source (live REST calls vs historical candles).

Sections:
  1. Indicators / signal   (liquidity sweep + OI delta — Prompt 2)
  2. Isolated-margin wallet accounting (Prompt 1)
  3. Position sizing
  4. Exit logic (OHLC-aware, not close-only — Codex finding #6)

NOTE ON SCOPE: this drops RSI, EMA9/21, EMA50 trend, Fear&Greed, and
Order-Book-Imbalance entirely, per Prompt 1 + Prompt 2 ("replace the
old indicator-based signal generation"). ATR is kept, but only
internally, only to size the stop-loss distance — it is not a signal
or a filter. If you want the 1H trend filter back as an extra gate on
top of the sweep+OI signal, that's a small addition to
generate_signal() below, not a redesign.
"""

import pandas as pd
import numpy as np

import config as cfg


# =====================================================================
# 1. SIGNAL: liquidity sweep (prior 24h high/low wick) + OI delta
# =====================================================================

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR for stop-loss distance sizing ONLY — not used as a signal."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def add_sweep_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Liquidity sweep detection.

    prior_high / prior_low use .shift(1) before the rolling window, so
    the current candle's own high/low can never leak into the level it
    is being compared against (Codex finding #4 — the old backtest's
    res20/sup20 included the current bar).

    A SHORT setup: this candle's high pokes above the prior-24h high by
    at least SWEEP_WICK_MIN_PCT, then closes back below that high
    (stop-hunt + rejection). A LONG setup is the mirror image at the
    prior-24h low.
    """
    lookback = cfg.SWEEP_LOOKBACK_CANDLES
    wick_min = cfg.SWEEP_WICK_MIN_PCT

    df["prior_high"] = df["high"].shift(1).rolling(lookback).max()
    df["prior_low"] = df["low"].shift(1).rolling(lookback).min()

    swept_high = df["high"] > df["prior_high"] * (1 + wick_min)
    swept_low = df["low"] < df["prior_low"] * (1 - wick_min)
    reclaimed_high = df["close"] < df["prior_high"]
    reclaimed_low = df["close"] > df["prior_low"]

    df["sweep_signal"] = None
    df.loc[swept_high & reclaimed_high, "sweep_signal"] = "SHORT"
    df.loc[swept_low & reclaimed_low, "sweep_signal"] = "LONG"
    return df


def add_oi_delta(df: pd.DataFrame, oi_col: str = "open_interest") -> pd.DataFrame:
    """
    % change in open interest vs OI_DELTA_LOOKBACK_CANDLES ago.
    Requires df[oi_col] to already be populated (live: repeated current
    snapshots forward-filled; backtest: merged historical OI series).
    """
    df["oi_pct_change"] = df[oi_col].pct_change(periods=cfg.OI_DELTA_LOOKBACK_CANDLES)
    return df


def align_oi(df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Time-aligns an open-interest series onto the candle dataframe.
    Used by BOTH live and backtest so the join logic can't drift apart
    the way the old backtest's 1h-trend merge did (Codex finding #2).

    df needs an 'open_time' column (tz-aware UTC). oi_df needs 'time'
    and 'open_interest' columns. direction='backward' means each candle
    only ever sees an OI reading whose timestamp is <= its own open_time
    — never a future OI value.
    """
    if oi_df is None or oi_df.empty:
        df["open_interest"] = float("nan")
        return df
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        oi_df[["time", "open_interest"]].sort_values("time"),
        left_on="open_time", right_on="time", direction="backward",
    )
    return merged.drop(columns=["time"], errors="ignore")


def generate_signal(row) -> str:
    """
    Combines the sweep with OI confirmation. No OI data for this bar ->
    HOLD (fail safe, not fail open) rather than trading on price alone.
    """
    sweep = row.get("sweep_signal")
    if sweep not in ("LONG", "SHORT"):
        return "HOLD"
    oi_chg = row.get("oi_pct_change")
    if oi_chg is None or (isinstance(oi_chg, float) and np.isnan(oi_chg)):
        return "HOLD"
    if oi_chg >= cfg.OI_DELTA_MIN_PCT:
        return sweep
    return "HOLD"


# =====================================================================
# 2. ISOLATED-MARGIN WALLET ACCOUNTING
# =====================================================================

def default_wallet() -> dict:
    return {
        "cash": cfg.STARTING_CASH_PER_COIN,   # free USDT, NOT locked in margin
        "position_side": None,                # "LONG" / "SHORT" / None
        "qty": 0.0,
        "entry_price": None,
        "margin_locked": 0.0,
        "stop_price": None,
        "liquidation_price": None,
        "peak_price": None,                   # LONG trailing reference
        "trough_price": None,                 # SHORT trailing reference
        "partial_sold": False,
        "current_position_id": None,          # ties OPEN/PARTIAL/CLOSE log rows to one round-trip
        "position_pnl_so_far": 0.0,           # partial + final realized PnL for the OPEN position
        "next_position_id": 1,
        "num_positions_opened": 0,            # counts opens only — NOT entries+exits+partials mixed together
        "realized_pnl_total": 0.0,
        "win_trades": 0,
        "loss_trades": 0,
        "day": None,
        "day_start_value": cfg.STARTING_CASH_PER_COIN,
        "day_halted": False,
        "peak_value": cfg.STARTING_CASH_PER_COIN,
        "max_drawdown": 0.0,
        "last_funding_ts": None,   # ISO string of the last 8h funding boundary already charged
    }


def apply_funding_if_due(wallet: dict, funding_rate: float, now_ts) -> float:
    """
    Binance perpetual funding is charged only at 00:00/08:00/16:00 UTC —
    NOT continuously. This checks whether a boundary has been crossed
    since the last time this wallet was charged, and if so, applies it
    exactly once. Call this every run (live: every 15 min: backtest:
    every bar) — it no-ops on every call that isn't a fresh boundary.

    Returns the dollar amount charged (positive = wallet paid, negative
    = wallet received). LONG pays positive funding; SHORT is the mirror.
    """
    if wallet["position_side"] is None:
        return 0.0

    boundary_hour = (now_ts.hour // cfg.FUNDING_INTERVAL_HOURS) * cfg.FUNDING_INTERVAL_HOURS
    boundary = now_ts.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
    last = wallet.get("last_funding_ts")
    if last is not None:
        last_dt = pd.Timestamp(last)
        if boundary <= last_dt:
            return 0.0

    wallet["last_funding_ts"] = str(boundary)
    notional = wallet["qty"] * wallet["entry_price"]
    payment = notional * funding_rate
    if wallet["position_side"] == "LONG":
        wallet["cash"] -= payment
    else:
        wallet["cash"] += payment
    return payment if wallet["position_side"] == "LONG" else -payment


def unrealized_pnl(wallet: dict, mark_price: float) -> float:
    """Side-agnostic by construction — this is what actually fixes the
    'short equity uses the long formula' bug class Codex flagged,
    instead of patching a sign in one spot and missing another."""
    if wallet["position_side"] is None or wallet["qty"] <= 0:
        return 0.0
    if wallet["position_side"] == "LONG":
        return (mark_price - wallet["entry_price"]) * wallet["qty"]
    else:  # SHORT
        return (wallet["entry_price"] - mark_price) * wallet["qty"]


def equity(wallet: dict, mark_price: float) -> float:
    return wallet["cash"] + wallet["margin_locked"] + unrealized_pnl(wallet, mark_price)


def liquidation_price(side: str, entry_price: float, leverage: float, mmr: float) -> float:
    """Simplified isolated-margin liquidation price (ignores funding
    accrual and Binance's tiered MMR brackets — see NOTES.md)."""
    if side == "LONG":
        return entry_price * (1 - 1 / leverage + mmr)
    else:
        return entry_price * (1 + 1 / leverage - mmr)


# =====================================================================
# 3. POSITION SIZING
# =====================================================================

def size_position(wallet: dict, price: float, stop_distance: float, mark_price: float):
    """
    Risk RISK_PER_TRADE_PCT of account equity if the stop is hit, capped
    by MAX_MARGIN_PCT_OF_CASH of free cash. Returns (qty, margin, fee) —
    any of which being 0 means "don't take this trade".
    """
    if stop_distance <= 0 or price <= 0:
        return 0.0, 0.0, 0.0

    acct_equity = equity(wallet, mark_price)
    risk_dollars = acct_equity * cfg.RISK_PER_TRADE_PCT
    qty = risk_dollars / stop_distance

    notional = qty * price
    margin_needed = notional / cfg.LEVERAGE
    max_margin = wallet["cash"] * cfg.MAX_MARGIN_PCT_OF_CASH
    margin_used = min(margin_needed, max_margin)
    if margin_used <= 0:
        return 0.0, 0.0, 0.0

    qty = margin_used * cfg.LEVERAGE / price
    fee = qty * price * cfg.TAKER_FEE_PCT

    if margin_used + fee > wallet["cash"]:
        return 0.0, 0.0, 0.0
    return qty, margin_used, fee


def stop_price_for(side: str, entry_price: float, stop_distance: float) -> float:
    return entry_price - stop_distance if side == "LONG" else entry_price + stop_distance


def open_position(wallet: dict, side: str, price: float, qty: float, margin: float,
                   fee: float, stop_distance: float, now_ts=None):
    """stop_distance is a price-delta (e.g. from ATR), computed by the
    caller at entry time — engine.py doesn't recompute it later so a
    stop never silently moves after entry.

    now_ts (if given) seeds last_funding_ts to the CURRENT funding
    boundary, so a position opened mid-window doesn't get charged for
    the boundary that already passed before it existed — funding only
    starts accruing from the next boundary onward, matching how a real
    exchange bills it."""
    wallet["cash"] -= (margin + fee)
    if now_ts is not None:
        boundary_hour = (now_ts.hour // cfg.FUNDING_INTERVAL_HOURS) * cfg.FUNDING_INTERVAL_HOURS
        wallet["last_funding_ts"] = str(now_ts.replace(hour=boundary_hour, minute=0, second=0, microsecond=0))
    wallet["position_side"] = side
    wallet["qty"] = qty
    wallet["entry_price"] = price
    wallet["margin_locked"] = margin
    wallet["stop_price"] = stop_price_for(side, price, stop_distance)
    wallet["liquidation_price"] = liquidation_price(side, price, cfg.LEVERAGE, cfg.MAINTENANCE_MARGIN_RATE)
    wallet["peak_price"] = price if side == "LONG" else None
    wallet["trough_price"] = price if side == "SHORT" else None
    wallet["partial_sold"] = False
    wallet["current_position_id"] = wallet["next_position_id"]
    wallet["next_position_id"] += 1
    wallet["num_positions_opened"] += 1
    wallet["position_pnl_so_far"] = 0.0   # accumulates across partial + final closes of THIS position


def close_position(wallet: dict, price: float, qty_to_close: float, fee_pct: float = cfg.TAKER_FEE_PCT):
    """
    Closes (fully or partially) at `price`. Returns realized_pnl for
    this chunk. Releases margin/PnL proportionally on a partial close so
    cash and margin_locked always stay consistent with the remaining
    position size.
    """
    side = wallet["position_side"]
    frac = qty_to_close / wallet["qty"] if wallet["qty"] > 0 else 0.0
    pnl_chunk = unrealized_pnl(wallet, price) * frac
    margin_chunk = wallet["margin_locked"] * frac
    fee = qty_to_close * price * fee_pct

    wallet["cash"] += margin_chunk + pnl_chunk - fee
    wallet["margin_locked"] -= margin_chunk
    wallet["qty"] -= qty_to_close
    wallet["realized_pnl_total"] += pnl_chunk
    wallet["position_pnl_so_far"] = wallet.get("position_pnl_so_far", 0.0) + pnl_chunk

    if wallet["qty"] <= 1e-9:
        # Win/loss is judged on the WHOLE position's realized PnL (partial +
        # final leg combined), not just the last fill — a position that took
        # partial profit and then stopped out on the remainder for a smaller
        # loss is still a net win overall.
        if wallet["position_pnl_so_far"] > 0:
            wallet["win_trades"] += 1
        else:
            wallet["loss_trades"] += 1
        wallet["position_side"] = None
        wallet["qty"] = 0.0
        wallet["entry_price"] = None
        wallet["margin_locked"] = 0.0
        wallet["stop_price"] = None
        wallet["liquidation_price"] = None
        wallet["peak_price"] = None
        wallet["trough_price"] = None
        wallet["partial_sold"] = False
        wallet["current_position_id"] = None
        wallet["position_pnl_so_far"] = 0.0

    return pnl_chunk, fee


# =====================================================================
# 4. EXIT LOGIC — OHLC-aware, not close-only (Codex finding #6)
# =====================================================================

def check_exits(wallet: dict, candle_high: float, candle_low: float, candle_close: float,
                 final_signal: str):
    """
    Walks the just-completed candle's actual high/low path instead of
    only its close, so a stop or target that was touched intrabar isn't
    silently ignored. SAME_CANDLE_AMBIGUITY="pessimistic": if one
    candle's range would trigger both the stop and a profit target,
    the stop wins.

    Returns (action, reason, qty_to_close, fill_price) where action is one
    of "HOLD" / "LIQUIDATED" / "STOP" / "PARTIAL" / "TRAIL" / "SIGNAL".
    fill_price is the actual trigger level the order would fill near
    (stop/target/liquidation price) — not the candle close — so a stop
    that fires mid-candle doesn't get marked-to-close-price by mistake.
    A SIGNAL exit has no trigger level, so it fills at candle_close.
    """
    if wallet["position_side"] is None or wallet["qty"] <= 0:
        return "HOLD", "", 0.0, 0.0

    side = wallet["position_side"]
    entry = wallet["entry_price"]
    qty = wallet["qty"]
    stop_price = wallet["stop_price"]
    liq_price = wallet["liquidation_price"]

    if side == "LONG":
        # 1. Liquidation (worst case, exchange-forced)
        if liq_price is not None and candle_low <= liq_price:
            return "LIQUIDATED", f"LIQUIDATED@{liq_price:.2f}", qty, liq_price
        # 2. Hard stop — pessimistic: check before any target this candle
        if stop_price is not None and candle_low <= stop_price:
            return "STOP", f"STOP_LOSS@{stop_price:.2f}", qty, stop_price
        # 3. Partial profit
        if not wallet["partial_sold"]:
            target = entry * (1 + cfg.PARTIAL_PROFIT_PCT)
            if candle_high >= target:
                return "PARTIAL", f"PARTIAL_PROFIT@{target:.2f}", qty * 0.5, target
        # 4. Trailing stop from peak
        wallet["peak_price"] = max(wallet["peak_price"] or entry, candle_high)
        if wallet["peak_price"] >= entry * (1 + cfg.PARTIAL_PROFIT_PCT):
            trail = wallet["peak_price"] * (1 - cfg.TRAILING_STOP_PCT)
            if candle_low <= trail:
                return "TRAIL", f"TRAILING_STOP peak={wallet['peak_price']:.2f}", qty, trail
        # 5. Signal flip
        if final_signal == "SHORT":
            return "SIGNAL", "SIGNAL_FLIP", qty, candle_close

    else:  # SHORT
        if liq_price is not None and candle_high >= liq_price:
            return "LIQUIDATED", f"LIQUIDATED@{liq_price:.2f}", qty, liq_price
        if stop_price is not None and candle_high >= stop_price:
            return "STOP", f"STOP_LOSS@{stop_price:.2f}", qty, stop_price
        if not wallet["partial_sold"]:
            target = entry * (1 - cfg.PARTIAL_PROFIT_PCT)
            if candle_low <= target:
                return "PARTIAL", f"PARTIAL_PROFIT@{target:.2f}", qty * 0.5, target
        wallet["trough_price"] = min(wallet["trough_price"] or entry, candle_low)
        if wallet["trough_price"] <= entry * (1 - cfg.PARTIAL_PROFIT_PCT):
            trail = wallet["trough_price"] * (1 + cfg.TRAILING_STOP_PCT)
            if candle_high >= trail:
                return "TRAIL", f"TRAILING_STOP trough={wallet['trough_price']:.2f}", qty, trail
        if final_signal == "LONG":
            return "SIGNAL", "SIGNAL_FLIP", qty, candle_close

    return "HOLD", "", 0.0, 0.0


def reset_day_if_needed(wallet: dict, mark_price: float, today: str):
    if wallet.get("day") != today:
        wallet["day"] = today
        wallet["day_start_value"] = equity(wallet, mark_price)
        wallet["day_halted"] = False


def update_drawdown_and_halt(wallet: dict, mark_price: float):
    val = equity(wallet, mark_price)
    if val > wallet.get("peak_value", val):
        wallet["peak_value"] = val
    peak = wallet.get("peak_value", val)
    dd = (peak - val) / peak if peak > 0 else 0.0
    if dd > wallet.get("max_drawdown", 0.0):
        wallet["max_drawdown"] = dd
    day_start = wallet.get("day_start_value", val)
    day_loss = (day_start - val) / day_start if day_start > 0 else 0.0
    if day_loss >= cfg.MAX_DAILY_LOSS_PCT:
        wallet["day_halted"] = True
    return val
