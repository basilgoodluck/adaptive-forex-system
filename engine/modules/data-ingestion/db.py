import psycopg
from datetime import datetime, timezone

from shared.schemas.bar_data import BarData


def get_conn(dsn: str):
    return psycopg.connect(dsn, autocommit=True)


def insert_bar(conn, bar: BarData):
    conn.execute(
        """
        INSERT INTO bars (time, symbol, timeframe, open, high, low, close, tick_volume, spread)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol, timeframe) DO NOTHING
        """,
        (
            bar.time, bar.symbol, bar.timeframe,
            bar.open, bar.high, bar.low, bar.close,
            bar.tick_volume, bar.spread,
        ),
    )
    conn.execute("SELECT pg_notify('new_bar', %s)", (bar.symbol,))


def rate_to_bar(rate, symbol: str, timeframe: str) -> BarData:
    return BarData(
        time=datetime.fromtimestamp(rate["time"], tz=timezone.utc),
        symbol=symbol,
        timeframe=timeframe,
        open=float(rate["open"]),
        high=float(rate["high"]),
        low=float(rate["low"]),
        close=float(rate["close"]),
        tick_volume=float(rate["tick_volume"]),
        spread=float(rate["spread"]),
    )