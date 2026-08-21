import psycopg
import pandas as pd

from shared.schemas.dc_event import DCEvent
from shared.schemas.regime_update import RegimeUpdate


def get_conn(dsn: str):
    return psycopg.connect(dsn, autocommit=True)


def fetch_recent_bars(conn, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT time, open, high, low, close, tick_volume
        FROM bars
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time DESC
        LIMIT %s
        """,
        (symbol, timeframe, limit),
    ).fetchall()
    rows.reverse()
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "tick_volume"])


def insert_dc_event(conn, event: DCEvent):
    conn.execute(
        """
        INSERT INTO dc_events (
            time, symbol, event_type, price, extreme_price,
            overshoot_ratio, duration_bars, directional_imbalance,
            realized_volatility, adx
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO NOTHING
        """,
        (
            event.time, event.symbol, event.event_type, event.price, event.extreme_price,
            event.overshoot_ratio, event.duration_bars, event.directional_imbalance,
            event.realized_volatility, event.adx,
        ),
    )


def insert_regime_update(conn, update: RegimeUpdate):
    conn.execute(
        """
        INSERT INTO regime_updates (
            time, symbol, regime_label,
            prob_trend_calm, prob_trend_volatile, prob_range_calm,
            prob_range_volatile, prob_breakout, confidence,
            log_likelihood, atr_percentile
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO NOTHING
        """,
        (
            update.time, update.symbol, update.regime_label,
            update.prob_trend_calm, update.prob_trend_volatile, update.prob_range_calm,
            update.prob_range_volatile, update.prob_breakout, update.confidence,
            update.log_likelihood, update.atr_percentile,
        ),
    )
    conn.execute("SELECT pg_notify('new_regime_update', %s)", (update.symbol,))


def listen_new_bar(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("LISTEN new_bar")
    return conn
