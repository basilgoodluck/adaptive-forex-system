import psycopg


def get_conn(dsn: str):
    return psycopg.connect(dsn, autocommit=True)


def fetch_latest_order_spec(conn, symbol: str):
    row = conn.execute(
        """
        SELECT direction, position_size, sl_distance, tp_distance
        FROM order_specs
        WHERE symbol = %s
        ORDER BY time DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    return {"direction": row[0], "position_size": row[1], "sl_distance": row[2], "tp_distance": row[3]}


def insert_trade(conn, open_time, symbol, direction, entry_price, size, sl_price, tp_price):
    conn.execute(
        """
        INSERT INTO trades (open_time, symbol, direction, entry_price, size, sl_price, tp_price, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'OPEN')
        """,
        (open_time, symbol, direction, entry_price, size, sl_price, tp_price),
    )
    conn.execute("SELECT pg_notify('new_trade', %s)", (symbol,))


def listen_new_order_spec(dsn: str):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("LISTEN new_order_spec")
    return conn
