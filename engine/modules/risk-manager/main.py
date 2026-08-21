import os
import talib
from datetime import datetime, timezone
from dotenv import load_dotenv

import db

from shared.schemas.order_spec import OrderSpec

load_dotenv()

DSN = os.getenv("DATABASE_URL")
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M15"

ACCOUNT_EQUITY = float(os.getenv("ACCOUNT_EQUITY", "10000"))
RISK_FRACTION = float(os.getenv("RISK_FRACTION", "0.01"))
PIP_VALUE = float(os.getenv("PIP_VALUE", "10"))
FIXED_PIP_FALLBACK = float(os.getenv("FIXED_PIP_FALLBACK", "0.0020"))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "1000"))
TP_RATIO = float(os.getenv("TP_RATIO", "1.5"))


def process_symbol(conn, symbol: str):
    weighted_signal = db.fetch_latest_signal(conn, symbol)
    if weighted_signal is None or weighted_signal == 0:
        return

    regime = db.fetch_latest_regime(conn, symbol)
    if regime is None:
        return

    bars = db.fetch_recent_bars(conn, symbol, TIMEFRAME, limit=30)
    if len(bars) < 20:
        return

    atr_series = talib.ATR(
        bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), timeperiod=14
    )
    atr = atr_series[-1]
    if atr != atr:
        return

    current_drawdown = db.fetch_current_drawdown(conn)

    result = db.call_position_size(
        conn,
        regime["regime_label"], float(atr), regime["atr_percentile"], regime["confidence"],
        ACCOUNT_EQUITY, RISK_FRACTION, PIP_VALUE, FIXED_PIP_FALLBACK,
        current_drawdown, MAX_DRAWDOWN, TP_RATIO,
    )

    direction = "BUY" if weighted_signal > 0 else "SELL"

    spec = OrderSpec(
        time=datetime.now(timezone.utc),
        symbol=symbol,
        direction=direction,
        position_size=result["position_size"],
        sl_distance=result["sl_distance"],
        tp_distance=result["tp_distance"],
        used_atr=result["used_atr"],
        regime_label=regime["regime_label"],
        confidence=regime["confidence"],
    )
    db.insert_order_spec(conn, spec)


if __name__ == "__main__":
    conn = db.get_conn(DSN)
    listener = db.listen_new_raw_signal(DSN)

    for symbol in SYMBOLS:
        process_symbol(conn, symbol)

    for notify in listener.notifies():
        symbol = notify.payload
        if symbol in SYMBOLS:
            process_symbol(conn, symbol)
