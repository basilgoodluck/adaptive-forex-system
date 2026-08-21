import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

import mt5_connector
import db

load_dotenv()

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M15"
DSN = os.getenv("DATABASE_URL")
MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
POLL_SECONDS = 5


def backfill(conn, symbol: str, start: datetime, end: datetime):
    rates = mt5_connector.fetch_historical_bars(symbol, TIMEFRAME, start, end)
    count = 0
    for rate in rates:
        bar = db.rate_to_bar(rate, symbol, TIMEFRAME)
        db.insert_bar(conn, bar)
        count += 1
    print(f"[{symbol}] backfilled {count} bars from {start} to {end}")


def live_loop(conn):
    last_seen = {s: None for s in SYMBOLS}
    while True:
        for symbol in SYMBOLS:
            try:
                rate = mt5_connector.fetch_latest_bar(symbol, TIMEFRAME)
                bar = db.rate_to_bar(rate, symbol, TIMEFRAME)
                if last_seen[symbol] != bar.time:
                    db.insert_bar(conn, bar)
                    last_seen[symbol] = bar.time
            except Exception as e:
                print(f"[{symbol}] live update failed: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    mt5_connector.connect(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    conn = db.get_conn(DSN)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5 * 365)
    for symbol in SYMBOLS:
        try:
            backfill(conn, symbol, start, end)
        except Exception as e:
            print(f"[{symbol}] backfill failed: {e}")

    live_loop(conn)