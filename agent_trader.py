"""
Agent Mode Stock Day Trader — Alpaca (Paper or Live)

Capabilities
- Scans a watchlist every N minutes during U.S. market hours
- Computes EMA9/EMA21, VWAP, ATR, Relative Volume
- Generates entry signals (EMA crossover + VWAP filter + rel vol)
- Sizes positions by fixed account risk % using ATR-based stop distance
- Places bracket orders (stop-loss + take-profit) via Alpaca API
- Enforces guardrails: max daily loss, max trades/day, max concurrent positions
- End-of-day auto-flatten + CSV journaling

Requirements
- Python 3.10+
- pip install: requests, pandas, numpy, python-dotenv
- Alpaca account + API keys (paper or live)

Environment Variables (recommended via .env or Agent Mode secrets)
- APCA_API_KEY_ID
- APCA_API_SECRET_KEY
- APCA_PAPER=true|false (default true)
- APCA_DATA_FEED=iex|sip (default iex)

Usage
- Configure WATCHLIST and risk settings below
- Run inside ChatGPT Agent Mode or locally: `python agent_trader.py`
"""

from __future__ import annotations
import os
import time
import csv
import math
import json
import signal
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

# --------------------------- Config ---------------------------
US_EAST = ZoneInfo("America/New_York")

WATCHLIST = [
    "AAPL", "NVDA", "MSFT", "META", "AMD", "TSLA", "AMZN", "NFLX",
    # add tickers you actively trade
]

TIMEFRAME = "5Min"          # bar timeframe for scanning
SCAN_INTERVAL_SEC = 60       # scan frequency
BARS_LOOKBACK = 200          # bars fetched each scan (must be >= max indicator window)

# Risk/safeguards
RISK_PER_TRADE_PCT = 0.4
DAILY_MAX_LOSS_PCT = 2.0     # stop for the day if equity drawdown exceeds this %
MAX_TRADES_PER_DAY = 6
MAX_CONCURRENT_POSITIONS = 2
TAKE_PROFIT_R_MULTIPLE = 1.3
MIN_STOP_PCT = 0.004         # minimum stop 0.4% if ATR-based calc is too tight
END_OF_DAY_FLATTEN_MINUTE = 15*60 + 55  # 15:55 ET

JOURNAL_CSV = "trade_journal.csv"

# Strategy thresholds
REL_VOL_MIN = 1.5

# --------------------------- Helpers ---------------------------

# Batch helper
from itertools import islice

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

# S&P 500 fetchers (DataHub symbols dump -> Wikipedia fallback)
import pandas as _pd

def fetch_sp500_tickers() -> list[str]:
    urls = [
        "https://datahub.io/core/s-and-p-500-companies/r/constituents_symbols.txt",
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=12)
            r.raise_for_status()
            if u.endswith(".txt"):
                syms = [s.strip().upper() for s in r.text.splitlines() if s.strip()]
                if len(syms) > 400:
                    return syms
            else:
                tables = _pd.read_html(r.text)
                for df in tables:
                    if "Symbol" in df.columns:
                        syms = [str(s).split(":")[-1].strip().upper() for s in df["Symbol"].tolist()]
                        syms = [s for s in syms if s and (s.replace(".","").isalpha())]
                        if len(syms) > 400:
                            return syms
        except Exception:
            continue
    return ["AAPL","MSFT","NVDA","META","AMZN","TSLA","GOOGL","GOOG","AMD"]

# Alpaca snapshots for after-hours movers
def get_snapshots(symbols: list[str]) -> dict:
    out = {}
    for chunk in chunked(symbols, 50):
        try:
            params = {"symbols": ",".join(chunk)}
            r = requests.get(f"{DATA_BASE}/v2/stocks/snapshots", headers=ALPACA_HEADERS, params=params, timeout=12)
            r.raise_for_status()
            js = r.json() or {}
            out.update(js.get("snapshots", {}))
        except Exception:
            pass
    return out

def build_daily_priority(symbols: list[str]) -> list[str]:
    snaps = get_snapshots(symbols)
    ranked = []
    for sym, s in snaps.items():
        try:
            prev = s.get("prevDailyBar") or {}
            last = s.get("minuteBar") or s.get("dailyBar") or {}
            prev_c = float(prev.get("c", 0) or 0)
            last_c = float(last.get("c", 0) or 0)
            vol = float(last.get("v", 0) or 0)
            if prev_c > 0 and last_c > 0:
                chg = 100.0 * (last_c - prev_c) / prev_c
                ranked.append((sym, abs(chg), chg, vol))
        except Exception:
            continue
    ranked.sort(key=lambda x: (x[1], x[3]), reverse=True)
    return [sym for sym, *_ in ranked[:PRIORITY_COUNT]]

# Global current watchlist (priority first)
CURRENT_WATCHLIST: list[str] = []

def refresh_watchlist_daily():
    global CURRENT_WATCHLIST
    spx = fetch_sp500_tickers()
    try:
        priority = build_daily_priority(spx)
    except Exception:
        priority = []
    base_core = ["NVDA","TSLA","AAPL","MSFT","AMZN","META"]
    merged, seen = [], set()
    for sym in (priority + base_core + spx):
        if sym not in seen:
            seen.add(sym)
            merged.append(sym)
    CURRENT_WATCHLIST = merged
    print(f"[INFO] Watchlist built: {len(CURRENT_WATCHLIST)} symbols; priority sample: {priority[:10]}")
    return {"priority": priority, "count": len(CURRENT_WATCHLIST)}


def env_bool(var: str, default: bool) -> bool:
    v = os.getenv(var)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}

APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
APCA_PAPER = env_bool("APCA_PAPER", True)
APCA_DATA_FEED = os.getenv("APCA_DATA_FEED", "iex").lower()

if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
    print("[WARN] Alpaca API keys are not set. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

TRADING_BASE = "https://paper-api.alpaca.markets" if APCA_PAPER else "https://api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": APCA_API_KEY_ID,
    "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
}

# --------------------------- Data & Indicators ---------------------------

def get_clock() -> Dict:
    r = requests.get(f"{TRADING_BASE}/v2/clock", headers=ALPACA_HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def get_account() -> Dict:
    r = requests.get(f"{TRADING_BASE}/v2/account", headers=ALPACA_HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def list_positions() -> List[Dict]:
    r = requests.get(f"{TRADING_BASE}/v2/positions", headers=ALPACA_HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def list_orders(status="open") -> List[Dict]:
    r = requests.get(f"{TRADING_BASE}/v2/orders", headers=ALPACA_HEADERS, params={"status": status}, timeout=10)
    r.raise_for_status()
    return r.json()


def cancel_all_orders():
    requests.delete(f"{TRADING_BASE}/v2/orders", headers=ALPACA_HEADERS, timeout=10)


def close_all_positions():
    requests.delete(f"{TRADING_BASE}/v2/positions", headers=ALPACA_HEADERS, timeout=10)


def get_bars(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    # Alpaca Market Data v2
    params = {
        "timeframe": timeframe,
        "limit": limit,
        "feed": APCA_DATA_FEED,
        # optionally add start/end for precise ranges
    }
    r = requests.get(f"{DATA_BASE}/v2/stocks/{symbol}/bars", headers=ALPACA_HEADERS, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    bars = js.get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    # normalize
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(US_EAST)
    df.rename(columns={"t":"time", "o":"open", "h":"high", "l":"low", "c":"close", "v":"volume"}, inplace=True)
    return df[["time","open","high","low","close","volume"]]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close).abs()
    tr3 = (df["low"] - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    cumvol = df["volume"].cumsum()
    return pv / cumvol


def relative_volume(df: pd.DataFrame, window: int = 20) -> float:
    # current bar volume vs trailing average
    if len(df) < window + 1:
        return 1.0
    current = float(df.iloc[-1]["volume"])
    avg = float(df.iloc[-(window+1):-1]["volume"].mean())
    return current / max(avg, 1.0)

# --------------------------- Strategy ---------------------------
@dataclass
class Signal:
    symbol: str
    side: str  # "buy" or "sell"
    entry: float
    stop: float
    take_profit: float
    r_multiple: float


def generate_signal(df: pd.DataFrame) -> Optional[Signal]:
    if df.empty or len(df) < 50:
        return None

    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["vwap"] = vwap(df)
    df["atr14"] = atr(df, 14)

    rv = relative_volume(df, 20)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish crossover (EMA9 crossed above EMA21) + price above VWAP + rel vol filter
    crossed_up = (prev["ema9"] <= prev["ema21"]) and (last["ema9"] > last["ema21"]) and (last["close"] > last["vwap"]) and (rv >= REL_VOL_MIN)

    # Bearish: cross down (for optional shorts — disabled by default for stocks)
    crossed_down = (prev["ema9"] >= prev["ema21"]) and (last["ema9"] < last["ema21"]) and (last["close"] < last["vwap"]) and (rv >= REL_VOL_MIN)

    if crossed_up:
        entry = float(last["close"])
        stop_dist = max(float(last["atr14"]) * 0.8, entry * MIN_STOP_PCT)  # ATR-based floor
        stop = entry - stop_dist
        take_profit = entry + TAKE_PROFIT_R_MULTIPLE * stop_dist
        return Signal("", "buy", entry, stop, take_profit, TAKE_PROFIT_R_MULTIPLE)

    # if you want to enable shorting, you can return a sell signal here
    return None

# --------------------------- Broker ---------------------------

def place_bracket_order(symbol: str, qty: int, side: str, entry_price: float, stop_price: float, take_profit_price: float, tif: str = "day") -> Dict:
    # Use a LIMIT entry at current price for determinism; you may switch to market
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "limit",
        "limit_price": round(entry_price, 2),
        "time_in_force": tif,
        "order_class": "bracket",
        "take_profit": {"limit_price": round(take_profit_price, 2)},
        "stop_loss": {"stop_price": round(stop_price, 2)}
    }
    r = requests.post(f"{TRADING_BASE}/v2/orders", headers=ALPACA_HEADERS, json=order, timeout=10)
    if r.status_code >= 400:
        try:
            print("[ERROR] order rejected:", r.json())
        except Exception:
            print("[ERROR] order rejected, status:", r.status_code, r.text)
        r.raise_for_status()
    js = r.json()
    return js

# --------------------------- Risk & Sizing ---------------------------

def calc_position_size(account_equity: float, entry: float, stop: float, risk_pct: float) -> int:
    risk_amount = account_equity * (risk_pct / 100.0)
    per_share_risk = max(entry - stop, 0.01)
    qty = int(risk_amount // per_share_risk)
    return max(qty, 0)

# --------------------------- Journal ---------------------------

def ensure_journal():
    if not os.path.exists(JOURNAL_CSV):
        with open(JOURNAL_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_et","symbol","side","entry","stop","take_profit","qty","order_id","reason","r_multiple"
            ])


def log_trade(symbol: str, side: str, entry: float, stop: float, tp: float, qty: int, order_id: str, reason: str, r_mult: float):
    ensure_journal()
    with open(JOURNAL_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(US_EAST).strftime("%Y-%m-%d %H:%M:%S"), symbol, side, round(entry,4), round(stop,4), round(tp,4), qty, order_id, reason, r_mult
        ])

# --------------------------- Agent Loop ---------------------------
class KillSwitch(Exception):
    pass


def within_regular_hours(now_et: datetime) -> bool:
    # 9:30 - 16:00 ET
    start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now_et <= end


def minutes_since_midnight(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def agent_day_loop():
    print("[INFO] Starting Agent Mode day loop (stocks)")
    acct = get_account()
    start_equity = float(acct.get("equity", acct.get("last_equity", 0)))
    print(f"[INFO] Starting equity: {start_equity}")

    trade_count = 0

    try:
        while True:
            now_et = datetime.now(US_EAST)

            # End-of-day safety flatten
            if minutes_since_midnight(now_et) >= END_OF_DAY_FLATTEN_MINUTE and within_regular_hours(now_et):
                print("[INFO] EOD flattening positions and cancelling orders...")
                cancel_all_orders()
                close_all_positions()
                break

            # Trade only during regular hours
            if not within_regular_hours(now_et):
                time.sleep(10)
                continue

            # Check daily loss limit
            acct = get_account()
            equity = float(acct.get("equity", start_equity))
            drawdown_pct = 100.0 * (start_equity - equity) / max(start_equity, 1.0)
            if drawdown_pct >= DAILY_MAX_LOSS_PCT:
                print(f"[WARN] Daily loss limit hit ({drawdown_pct:.2f}%). Stopping for today.")
                break

            # Respect max concurrent positions
            open_positions = list_positions()
            if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            # Respect max trades/day
            if trade_count >= MAX_TRADES_PER_DAY:
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            # Rebuild around scheduled time and scan watchlist
            now_tuple = (now_et.hour, now_et.minute)
            if now_tuple == REBUILD_TIME_ET:
                try:
                    info = refresh_watchlist_daily()
                    print(f"[INFO] Watchlist rebuilt: {info['count']} symbols; priority sample: {info['priority'][:10]}")
                except Exception as e:
                    print(f"[WARN] watchlist rebuild failed: {e}")

            # Scan current watchlist (priority first)
            wl = CURRENT_WATCHLIST if CURRENT_WATCHLIST else WATCHLIST
            for batch in chunked(wl, BATCH_SIZE):
                for symbol in batch:
                    try:
                        df = get_bars(symbol, TIMEFRAME, BARS_LOOKBACK)
                        if df.empty:
                            continue
                        sig = generate_signal(df)
                        if not sig:
                            continue

                        # Prepare order
                        sig.symbol = symbol
                        qty = calc_position_size(float(acct.get("equity", start_equity)), sig.entry, sig.stop, RISK_PER_TRADE_PCT)
                        if qty <= 0:
                            continue

                        # Place order
                        print(f"[TRADE] {symbol} {sig.side} qty={qty} entry={sig.entry:.2f} stop={sig.stop:.2f} tp={sig.take_profit:.2f}")
                        order = place_bracket_order(symbol, qty, sig.side, sig.entry, sig.stop, sig.take_profit)
                        order_id = order.get("id", "")
                        log_trade(symbol, sig.side, sig.entry, sig.stop, sig.take_profit, qty, order_id, "EMA9>EMA21 + VWAP + RelVol", sig.r_multiple)
                        trade_count += 1
                    except requests.HTTPError as e:
                        print(f"[HTTP] {symbol}: {e}")
                    except Exception as e:
                        print(f"[ERR] {symbol}: {e}")

                        # Re-check caps after a trade
                        if trade_count >= MAX_TRADES_PER_DAY:
                            print("[INFO] Max trades reached; pausing further entries today.")
                        break
                        open_positions = list_positions()
                        if len(open_positions) >= MAX_CONCURRENT_POSITIONS:
                            print("[INFO] Max concurrent positions reached; pausing entries.")
                        break

                    except requests.HTTPError as e:
                        print(f"[HTTP] {symbol}: {e}")
                    except Exception as e:
                        print(f"[ERR] {symbol}: {e}")

            time.sleep(SCAN_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user. Flattening...")
        cancel_all_orders()
        close_all_positions()


if __name__ == "__main__":
    agent_day_loop()