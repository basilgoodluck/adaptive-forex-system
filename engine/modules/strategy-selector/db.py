import psycopg
import pandas as pd

from shared.schemas.raw_signal import RawSignal


def get_conn(dsn: str):
    return psycopg.connect(dsn, autocommit=True)


def fetch_recent_bars(conn, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT time, high, low, close
        FROM bars
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time DESC
        LIMIT %s
        """,
        (symbol, timeframe, limit),
    ).fetchall()
    rows.reverse()
    return pd.DataFrame(rows, columns=["time", "high", "low", "close"])


def fetch_latest_regime(conn, symbol: str):
    row = conn.execute(
        """
        SELECT prob_trend_calm, prob_trend_volatile, prob_range_calm, prob_range_volatile, prob_breakout
        FROM regime_updates
        WHERE symbol = %s
        ORDER BY time DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    return {
        "TREND_CALM": row[0],
        "TREND_VOLATILE": row[1],
        "RANGE_CALM": row[2],
        "RANGE_VOLATILE": row[3],
        "BREAKOUT": row[4],
    }


def insert_raw_signal(conn, signal: RawSignal):
    conn.execute(
        """
        INSERT INTO raw_signals (
            time, symbol, active_strategy, weighted_signal,
            weight_trend_following, weight_mean_reversion, weight_breakout
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO NOTHING
        """,
        (
            signal.time, signal.symbol, signal.active_strategy, signal.weighted_signal,
            signal.weight_trend_following, signal.weight_mean_reversion, signal.weight_breakout,
        ),
    )
    conn.execute("SELECT pg_notify('new_raw_signal', %s)", (signal.symbol,))


def listen_new_regime_update(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("LISTEN new_regime_update")
    return conn
