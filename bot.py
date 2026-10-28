
import os
import asyncio
import logging
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# =====================================================
# RALPHFX INTELLIGENCE - SINGLE FILE VERSION
# =====================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "8000"))

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD"
]

TIMEFRAMES = {
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h"
}

DEFAULT_TIMEFRAME = "H1"
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5

DISCLAIMER = (
    "⚠️ Educational analysis only, not financial advice. "
    "Forex and gold trading involve substantial risk. "
    "Signals are not guaranteed to be profitable."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("RalphFX")


# =====================================================
# RENDER HEALTH SERVER
# =====================================================

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/health"):
            self.send_error(404)
            return

        body = b"RalphFX Intelligence is running"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def start_health_server():
    try:
        server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
        logger.info("Health server listening on port %s", PORT)
        server.serve_forever()
    except Exception:
        logger.exception("Health server failed")


# =====================================================
# MARKET DATA - TWELVE DATA
# =====================================================

class MarketData:
    URL = "https://api.twelvedata.com/time_series"

    def __init__(self):
        self.api_key = TWELVE_DATA_API_KEY
        self.last_success = None
        self.cache = {}
        self.cache_lock = threading.Lock()

    @staticmethod
    def symbol(pair):
        pair = pair.upper().replace("/", "")
        if pair == "XAUUSD":
            return "XAU/USD"
        if len(pair) != 6:
            raise ValueError("Invalid symbol")
        return pair[:3] + "/" + pair[3:]

    async def get_rates(self, pair, interval, outputsize=300):
        return await asyncio.to_thread(
            self._fetch, pair, interval, outputsize
        )

    def _fetch(self, pair, interval, outputsize):
        if not self.api_key:
            logger.error("TWELVE_DATA_API_KEY is missing")
            return None

        key = (pair, interval, outputsize)
        now = datetime.now(timezone.utc).timestamp()

        with self.cache_lock:
            cached = self.cache.get(key)
            if cached and now - cached[0] < 60:
                return cached[1].copy()

        params = {
            "symbol": self.symbol(pair),
            "interval": interval,
            "outputsize": min(max(int(outputsize), 1), 5000),
            "apikey": self.api_key,
            "timezone": "UTC"
        }

        try:
            response = requests.get(
                self.URL, params=params, timeout=25
            )
            response.raise_for_status()
            payload = response.json()

            if "values" not in payload:
                logger.warning(
                    "Twelve Data error: %s",
                    payload.get("message", payload)
                )
                return None

            df = pd.DataFrame(payload["values"])
            if df.empty:
                return None

            df["time"] = pd.to_datetime(
                df["datetime"], utc=True, errors="coerce"
            )

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(
                subset=["time", "open", "high", "low", "close"]
            )
            df = (
                df.sort_values("time")
                .drop_duplicates("time")
                .reset_index(drop=True)
            )

            if df.empty:
                return None

            df = df[["time", "open", "high", "low", "close"]]
            self.last_success = datetime.now(timezone.utc)

            with self.cache_lock:
                self.cache[key] = (now, df.copy())

            return df

        except Exception:
            logger.exception("Market data request failed for %s", pair)
            return None

    @staticmethod
    def is_fresh(df, max_age_minutes=180):
        if df is None or df.empty:
            return False

        last_time = pd.to_datetime(
            df.iloc[-1]["time"], utc=True, errors="coerce"
        )

        if pd.isna(last_time):
            return False

        age = (
            datetime.now(timezone.utc) - last_time.to_pydatetime()
        ).total_seconds() / 60

        return 0 <= age <= max_age_minutes


market = MarketData()


# =====================================================
# TECHNICAL INDICATORS
# =====================================================

def compute_indicators(df):
    data = df.copy()

    data["ema_fast"] = data["close"].ewm(
        span=EMA_FAST, adjust=False
    ).mean()

    data["ema_slow"] = data["close"].ewm(
        span=EMA_SLOW, adjust=False
    ).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    data["rsi"] = 100 - (100 / (1 + rs))

    data.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100
    data.loc[(avg_gain == 0) & (avg_loss > 0), "rsi"] = 0

    previous_close = data["close"].shift(1)

    tr = pd.concat([
        data["high"] - data["low"],
        (data["high"] - previous_close).abs(),
        (data["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    data["atr"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        min_periods=ATR_PERIOD,
        adjust=False
    ).mean()

    return data


def analyze_setup(df):
    if df is None or len(df) < EMA_SLOW + 5:
        return {
            "valid": False,
            "reason": "Not enough candles for reliable EMA200 analysis."
        }

    data = compute_indicators(df).dropna(
        subset=["ema_fast", "ema_slow", "rsi", "atr"]
    )

    if data.empty:
        return {
            "valid": False,
            "reason": "Indicator values are unavailable."
        }

    row = data.iloc[-1]
    close = float(row["close"])
    fast = float(row["ema_fast"])
    slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    atr = float(row["atr"])

    direction = None
    reasons = []

    if fast > slow and close > fast and 50 <= rsi <= 70:
        direction = "BUY"
        reasons = [
            "EMA50 is above EMA200",
            "Price is above EMA50",
            "RSI supports bullish momentum"
        ]

    elif fast < slow and close < fast and 30 <= rsi <= 50:
        direction = "SELL"
        reasons = [
            "EMA50 is below EMA200",
            "Price is below EMA50",
            "RSI supports bearish momentum"
        ]

    if direction is None:
        return {
            "valid": False,
            "reason": "Trend and RSI conditions do not align."
        }

    distance = atr * ATR_STOP_MULTIPLIER

    if not pd.notna(distance) or distance <= 0:
        return {
            "valid": False,
            "reason": "ATR value is invalid."
        }

    if direction == "BUY":
        sl = close - distance
        tp1 = close + distance
        tp2 = close + 2 * distance
    else:
        sl = close + distance
        tp1 = close - distance
        tp2 = close - 2 * distance

    return {
        "valid": True,
        "direction": direction,
        "timestamp": row["time"],
        "entry": close,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "ema_fast": fast,
        "ema_slow": slow,
        "rsi": rsi,
        "atr": atr,
        "reasons": reasons
    }


def calculate_position_size(equity, risk_pct, entry, stop):
    try:
        equity = float(equity)
        risk_pct = float(risk_pct)
        entry = float(entry)
        stop = float(stop)
    except (ValueError, TypeError):
        return {"error": "Enter valid numeric values."}

    if equity <= 0:
        return {"error": "Equity must be positive."}
    if not 0 < risk_pct <= 10:
        return {"error": "Risk must be above 0% and at most 10%."}
    if entry <= 0 or stop <= 0 or entry == stop:
        return {"error": "Entry and stop must be positive and different."}

    risk_amount = equity * risk_pct / 100
    distance = abs(entry - stop)

    # Approximation for USD-quoted forex pairs only.
    lots = risk_amount / (distance * 100000)
    pips = distance * 10000

    return {
        "risk_amount": round(risk_amount, 2),
        "sl_pips": round(pips, 2),
        "lots": round(lots, 2),
        "note": (
            "Approximate estimate for USD-quoted forex pairs. "
            "Not reliable for every pair or gold; verify broker "
            "contract specifications and pip value."
        )
    }


# =====================================================
# TELEGRAM COMMANDS
# =====================================================

async def send(update, text, **kwargs):
    if update.effective_message:
        await update.effective_message.reply_text(text, **kwargs)


def parse_args(context):
    pair = (
        context.args[0].upper().replace("/", "")
        if context.args else ""
    )
    tf = context.args[1].upper() if len(context.args) > 1 else DEFAULT_TIMEFRAME
    return pair, tf


def valid_args(pair, tf):
    return pair in PAIRS and tf in TIMEFRAMES


def precision(pair):
    if pair == "XAUUSD":
        return 2
    if "JPY" in pair:
        return 3
    return 5


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send(
        update,
        "🤖 *RalphFX Intelligence*\n"
        "_Analyze smarter. Trade with a plan._\n\n"
        "*COMMANDS*\n"
        "/analyze EURUSD H1 — Full analysis\n"
        "/pairs — Supported instruments\n"
        "/setups — Scan H1 setups\n"
        "/timeframes — Available timeframes\n"
        "/trend EURUSD H1 — Trend summary\n"
        "/indicators EURUSD H1 — Indicators\n"
        "/risk 1000 1 1.0850 1.0820 — Risk estimate\n"
        "/status — Bot status\n"
        "/help — Help\n\n"
        f"{DISCLAIMER}",
        parse_mode="Markdown"
    )


async def pairs_cmd(update, context):
    await send(
        update,
        "📋 *SUPPORTED PAIRS*\n\n" +
        "\n".join(f"• `{p}`" for p in PAIRS),
        parse_mode="Markdown"
    )


async def timeframes_cmd(update, context):
    await send(
        update,
        "⏱ *TIMEFRAMES*\n\n" +
        "\n".join(f"• `{tf}`" for tf in TIMEFRAMES),
        parse_mode="Markdown"
    )


async def analyze_cmd(update, context):
    pair, tf = parse_args(context)

    if not valid_args(pair, tf):
        await send(update, "Usage: /analyze EURUSD H1\nUse /pairs and /timeframes.")
        return

    msg = await update.effective_message.reply_text("⏳ Analyzing market...")

    try:
        df = await market.get_rates(pair, TIMEFRAMES[tf], 300)

        if df is None or df.empty:
            await msg.edit_text("❌ Data unavailable. Check your Twelve Data API key and usage limits.")
            return

        if not market.is_fresh(df):
            await msg.edit_text("⚠️ Market data is stale or the market may be closed.")
            return

        result = analyze_setup(df)
        timestamp = result.get("timestamp")
        if hasattr(timestamp, "strftime"):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M UTC")

        if not result["valid"]:
            await msg.edit_text(
                f"📊 {pair} {tf}\nNO VALID SETUP\n"
                f"Last candle: {timestamp}\n"
                f"Reason: {result['reason']}\n\n{DISCLAIMER}"
            )
            return

        p = precision(pair)
        fmt = f"{{:.{p}f}}"
        arrow = "🟢" if result["direction"] == "BUY" else "🔴"

        text = (
            f"{arrow} *{result['direction']} SETUP — {pair} {tf}*\n"
            f"Last candle: {timestamp}\n\n"
            "*REFERENCE LEVELS*\n"
            f"Entry: `{fmt.format(result['entry'])}`\n"
            f"Stop Loss: `{fmt.format(result['sl'])}`\n"
            f"Target 1: `{fmt.format(result['tp1'])}`\n"
            f"Target 2: `{fmt.format(result['tp2'])}`\n\n"
            "*INDICATORS*\n"
            f"EMA50: `{fmt.format(result['ema_fast'])}`\n"
            f"EMA200: `{fmt.format(result['ema_slow'])}`\n"
            f"RSI14: `{result['rsi']:.2f}`\n"
            f"ATR14: `{fmt.format(result['atr'])}`\n\n"
            "*REASONS*\n" +
            "\n".join(f"• {r}" for r in result["reasons"]) +
            f"\n\n_{DISCLAIMER}_"
        )

        await msg.edit_text(text, parse_mode="Markdown")

    except Exception:
        logger.exception("Analyze command failed")
        await msg.edit_text("❌ Analysis failed. Check Render logs.")


async def trend_cmd(update, context):
    pair, tf = parse_args(context)

    if not valid_args(pair, tf):
        await send(update, "Usage: /trend EURUSD H1")
        return

    try:
        df = await market.get_rates(pair, TIMEFRAMES[tf], 300)
        if df is None or df.empty:
            await send(update, "❌ Market data unavailable.")
            return

        data = compute_indicators(df).dropna()
        if data.empty:
            await send(update, "❌ Not enough indicator data.")
            return

        row = data.iloc[-1]
        close = float(row["close"])
        fast = float(row["ema_fast"])
        slow = float(row["ema_slow"])

        trend = (
            "🟢 BULLISH" if fast > slow and close > fast
            else "🔴 BEARISH" if fast < slow and close < fast
            else "🟡 MIXED"
        )

        p = precision(pair)
        await send(
            update,
            f"📈 *{pair} {tf} TREND*\n\n"
            f"Direction: {trend}\n"
            f"Close: `{close:.{p}f}`\n"
            f"EMA50: `{fast:.{p}f}`\n"
            f"EMA200: `{slow:.{p}f}`\n"
            f"RSI14: `{row['rsi']:.2f}`",
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("Trend command failed")
        await send(update, "❌ Trend calculation failed.")


async def indicators_cmd(update, context):
    pair, tf = parse_args(context)

    if not valid_args(pair, tf):
        await send(update, "Usage: /indicators EURUSD H1")
        return

    try:
        df = await market.get_rates(pair, TIMEFRAMES[tf], 300)
        if df is None or df.empty:
            await send(update, "❌ Market data unavailable.")
            return

        data = compute_indicators(df).dropna()
        if data.empty:
            await send(update, "❌ Not enough indicator data.")
            return

        row = data.iloc[-1]

        await send(
            update,
            f"📊 *{pair} {tf} INDICATORS*\n\n"
            f"EMA50: `{row['ema_fast']:.5f}`\n"
            f"EMA200: `{row['ema_slow']:.5f}`\n"
            f"RSI14: `{row['rsi']:.2f}`\n"
            f"ATR14: `{row['atr']:.5f}`",
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("Indicators command failed")
        await send(update, "❌ Indicator calculation failed.")


async def risk_cmd(update, context):
    if len(context.args) < 4:
        await send(
            update,
            "🛡 *RISK CALCULATOR*\n"
            "Usage: `/risk equity risk_percent entry stop_loss`\n"
            "Example: `/risk 1000 1 1.0850 1.0820`",
            parse_mode="Markdown"
        )
        return

    result = calculate_position_size(*context.args[:4])

    if "error" in result:
        await send(update, "❌ " + result["error"])
        return

    await send(
        update,
        "🛡 *POSITION SIZE ESTIMATE*\n\n"
        f"Risk amount: `{result['risk_amount']:.2f}`\n"
        f"Stop distance: `{result['sl_pips']:.2f}` pips\n"
        f"Approximate lots: `{result['lots']:.2f}`\n\n"
        f"_{result['note']}_",
        parse_mode="Markdown"
    )


async def status_cmd(update, context):
    try:
        df = await market.get_rates("EURUSD", "1h", 5)
        available = df is not None and not df.empty
        fresh = market.is_fresh(df) if available else False
        last = market.last_success

        await send(
            update,
            "🤖 *RALPHFX STATUS*\n\n"
            "Telegram: ✅ Running\n"
            f"Market data: {'✅ Available' if available else '❌ Unavailable'}\n"
            f"Data freshness: {'✅ Fresh' if fresh else '⚠️ Stale/unavailable'}\n"
            f"Last successful fetch: "
            f"{last.strftime('%Y-%m-%d %H:%M UTC') if last else 'N/A'}",
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("Status command failed")
        await send(update, "⚠️ Status check failed. Check Render logs.")


async def setups_cmd(update, context):
    msg = await update.effective_message.reply_text(
        "🔍 Scanning supported pairs on H1..."
    )
    found = []

    try:
        for pair in PAIRS:
            df = await market.get_rates(pair, "1h", 300)

            if df is None or df.empty or not market.is_fresh(df):
                continue

            result = analyze_setup(df)
            if result["valid"]:
                found.append((pair, result))

        if not found:
            await msg.edit_text("🔍 No valid setups found right now.")
            return

        lines = ["🔍 *VALID H1 REFERENCE SETUPS*\n"]

        for pair, result in found:
            p = precision(pair)
            lines.append(
                f"{'🟢' if result['direction'] == 'BUY' else '🔴'} "
                f"*{pair} — {result['direction']}*\n"
                f"Entry: `{result['entry']:.{p}f}` | "
                f"SL: `{result['sl']:.{p}f}`"
            )

        lines.append(f"\n_{DISCLAIMER}_")

        await msg.edit_text(
            "\n\n".join(lines),
            parse_mode="Markdown"
        )

    except Exception:
        logger.exception("Setup scan failed")
        await msg.edit_text("❌ Setup scan failed. Check Render logs and API limits.")


# =====================================================
# ERROR HANDLING AND STARTUP
# =====================================================

async def error_handler(update, context):
    logger.error(
        "Telegram handler error",
        exc_info=context.error
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN in Render Environment.")

    if not TWELVE_DATA_API_KEY:
        raise SystemExit("Missing TWELVE_DATA_API_KEY in Render Environment.")

    threading.Thread(
        target=start_health_server,
        daemon=True
    ).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    commands = [
        ("start", start),
        ("help", start),
        ("pairs", pairs_cmd),
        ("timeframes", timeframes_cmd),
        ("analyze", analyze_cmd),
        ("trend", trend_cmd),
        ("indicators", indicators_cmd),
        ("risk", risk_cmd),
        ("status", status_cmd),
        ("setups", setups_cmd),
    ]

    for name, handler in commands:
        app.add_handler(CommandHandler(name, handler))

    app.add_error_handler(error_handler)

    logger.info("RalphFX Intelligence starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
