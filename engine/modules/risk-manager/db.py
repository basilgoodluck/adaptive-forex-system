import psycopg
import pandas as pd

from shared.schemas.order_spec import OrderSpec


def get_conn(dsn: str):
    return psycopg.connect(dsn, autocommit=True)


def fetch_latest_signal(conn, symbol: str):
    row = conn.execute(
        """
        SELECT weighted_signal
        FROM raw_signals
        WHERE symbol = %s
        ORDER BY time DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    return row[0] if row else None


def fetch_latest_regime(conn, symbol: str):
    row = conn.execute(
        """
        SELECT regime_label, confidence, atr_percentile
        FROM regime_updates
        WHERE symbol = %s
        ORDER BY time DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    return {"regime_label": row[0], "confidence": row[1], "atr_percentile": row[2]}


def fetch_recent_bars(conn, symbol: str, timeframe: str, limit: int = 30) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT high, low, close
        FROM bars
        WHERE symbol = %s AND timeframe = %s
        ORDER BY time DESC
        LIMIT %s
        """,
        (symbol, timeframe, limit),
    ).fetchall()
    rows.reverse()
    return pd.DataFrame(rows, columns=["high", "low", "close"])


def fetch_current_drawdown(conn) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status = 'CLOSED'"
    ).fetchone()
    total_pnl = row[0]
    return abs(min(total_pnl, 0.0))


def call_position_size(
    conn, regime_label, atr, atr_percentile, confidence,
    account_equity, risk_fraction, pip_value, fixed_pip_fallback,
    current_drawdown, max_drawdown, tp_ratio,
):
    row = conn.execute(
        "SELECT * FROM compute_position_size(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            regime_label, atr, atr_percentile, confidence,
            account_equity, risk_fraction, pip_value, fixed_pip_fallback,
            current_drawdown, max_drawdown, tp_ratio,
        ),
    ).fetchone()
    return {"position_size": row[0], "sl_distance": row[1], "tp_distance": row[2], "used_atr": row[3]}


def insert_order_spec(conn, spec: OrderSpec):
    conn.execute(
        """
        INSERT INTO order_specs (
            time, symbol, direction, position_size, sl_distance, tp_distance,
            used_atr, regime_label, confidence
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO NOTHING
        """,
        (
            spec.time, spec.symbol, spec.direction, spec.position_size,
            spec.sl_distance, spec.tp_distance, spec.used_atr,
            spec.regime_label, spec.confidence,
        ),
    )
    conn.execute("SELECT pg_notify('new_order_spec', %s)", (spec.symbol,))


def listen_new_raw_signal(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("LISTEN new_raw_signal")
    return conn
