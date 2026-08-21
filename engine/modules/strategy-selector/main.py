import os
from datetime import datetime, timezone
from dotenv import load_dotenv

import db
import allocator

from shared.schemas.raw_signal import RawSignal

load_dotenv()

DSN = os.getenv("DATABASE_URL")
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M15"


def process_symbol(conn, symbol: str):
    regime_probs = db.fetch_latest_regime(conn, symbol)
    if regime_probs is None:
        return

    bars = db.fetch_recent_bars(conn, symbol, TIMEFRAME, limit=200)
    if len(bars) < 30:
        return

    result = allocator.allocate(
        regime_probs,
        bars["high"].to_numpy(),
        bars["low"].to_numpy(),
        bars["close"].to_numpy(),
    )

    signal = RawSignal(
        time=datetime.now(timezone.utc),
        symbol=symbol,
        active_strategy=result["active_strategy"],
        weighted_signal=result["weighted_signal"],
        weight_trend_following=result["strategy_weights"]["TREND_FOLLOWING"],
        weight_mean_reversion=result["strategy_weights"]["MEAN_REVERSION"],
        weight_breakout=result["strategy_weights"]["BREAKOUT"],
    )
    db.insert_raw_signal(conn, signal)


if __name__ == "__main__":
    conn = db.get_conn(DSN)
    listener = db.listen_new_regime_update(DSN)

    for symbol in SYMBOLS:
        process_symbol(conn, symbol)

    for notify in listener.notifies():
        symbol = notify.payload
        if symbol in SYMBOLS:
            process_symbol(conn, symbol)
