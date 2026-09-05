"""
config.py — Single shared config for paper_trader.py and backtest.py.

Codex audit finding #1 was that live and backtest used different
EMA_SEP_MIN_PCT values, so the backtest never actually represented the
live bot. Fix: both files import every parameter from here. Never
hardcode a strategy/risk constant directly in paper_trader.py or
backtest.py again — add it here instead.
"""

# ---- Symbols & timeframe ----
COINS = ["BTCUSDT", "ETHUSDT"]
PRIMARY_INTERVAL = "15m"
CANDLES_PER_DAY = 96                 # 24h / 15m
SWEEP_LOOKBACK_CANDLES = CANDLES_PER_DAY   # prior-24h high/low window for liquidity sweeps

# ---- Account / isolated-margin futures ----
STARTING_CASH_PER_COIN = 5000.0      # simulated USDT wallet, per symbol, isolated per position
LEVERAGE = 3                         # isolated margin leverage. ASSUMPTION — change this to whatever you actually intend to run.
MAINTENANCE_MARGIN_RATE = 0.005      # flat approximation. Real Binance MMR is tiered by notional size — see NOTES.md.
TAKER_FEE_PCT = 0.0005               # Binance USDT-M futures taker fee tier (0.05%). Update if your fee tier differs.

# ---- Risk management (unchanged intent from v6.2, same values) ----
STOP_LOSS_PCT = 0.02
TRAILING_STOP_PCT = 0.015
PARTIAL_PROFIT_PCT = 0.015
RISK_PER_TRADE_PCT = 0.02            # fraction of account equity risked per trade -> sizes the position
MAX_MARGIN_PCT_OF_CASH = 0.5         # hard cap: never lock more than this fraction of free cash as margin on one entry
MAX_DAILY_LOSS_PCT = 0.03

# ---- Volatility (kept — ATR sizes the stop distance; it's not a signal or a trend filter) ----
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0                  # stop distance = max(ATR * this, price * STOP_LOSS_PCT)

# ---- Signal: liquidity sweep + OI delta (replaces EMA/RSI/Fear&Greed/OrderBookImbalance) ----
SWEEP_WICK_MIN_PCT = 0.0015          # wick must clear the prior 24h level by at least this % to count as a real sweep
OI_DELTA_LOOKBACK_CANDLES = 4        # compare current OI to OI this many 15m candles ago (4 = 1h)
OI_DELTA_MIN_PCT = 0.003              # OI must have moved at least this % in the confirming direction

# ---- Execution realism (backtest only — live fills at whatever the market gives it) ----
BACKTEST_SLIPPAGE_PCT = 0.0005
FUNDING_INTERVAL_HOURS = 8           # Binance perpetual funding cadence (00:00 / 08:00 / 16:00 UTC)
SAME_CANDLE_AMBIGUITY = "pessimistic"   # documents the rule below — checking stop before target/trail
                                         # in engine.check_exits() IS the implementation; changing this
                                         # string alone does nothing, you'd need to reorder that function.
BACKTEST_DAYS = 25                   # kept under Binance's ~30-day openInterestHist retention (see NOTES.md) so OI-delta is testable on the whole window

# ---- Files ----
STATE_FILE = "state.json"
LOG_FILE = "trade_log_v7.csv"
BACKTEST_TRADES_CSV = "backtest_trades_v7.csv"
BACKTEST_SUMMARY_MD = "backtest_summary_v7.md"
SCHEMA_VERSION = 7.0

# ---- Binance USDT-M Futures endpoints (Prompt 1: move off spot onto futures) ----
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
MARK_PRICE_ENDPOINT = f"{BINANCE_FUTURES_BASE}/fapi/v1/premiumIndex"
OPEN_INTEREST_ENDPOINT = f"{BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
OPEN_INTEREST_HIST_ENDPOINT = f"{BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
FUNDING_RATE_HIST_ENDPOINT = f"{BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
