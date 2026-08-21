import os
from datetime import datetime, timezone
from dotenv import load_dotenv

import db
import mt5_broker

load_dotenv()

DSN = os.getenv("DATABASE_URL")
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")


def process_symbol(conn, symbol: str):
    spec = db.fetch_latest_order_spec(conn, symbol)
    if spec is None:
        return

    fill = mt5_broker.place_order(
        symbol, spec["direction"], spec["position_size"], spec["sl_distance"], spec["tp_distance"]
    )

    db.insert_trade(
        conn,
        datetime.now(timezone.utc),
        symbol,
        spec["direction"],
        fill["entry_price"],
        spec["position_size"],
        fill["sl_price"],
        fill["tp_price"],
    )


if __name__ == "__main__":
    mt5_broker.connect(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)
    conn = db.get_conn(DSN)
    listener = db.listen_new_order_spec(DSN)

    for notify in listener.notifies():
        symbol = notify.payload
        if symbol in SYMBOLS:
            process_symbol(conn, symbol)
