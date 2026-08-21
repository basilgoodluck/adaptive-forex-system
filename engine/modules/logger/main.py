import os
import select
import psycopg
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL")

CHANNELS = ["new_bar", "new_regime_update", "new_raw_signal", "new_order_spec", "new_trade"]

if __name__ == "__main__":
    conn = psycopg.connect(DSN, autocommit=True)
    for channel in CHANNELS:
        conn.execute(f"LISTEN {channel}")

    for notify in conn.notifies():
        conn.execute(
            "INSERT INTO system_events (time, event_type, symbol, detail) VALUES (%s, %s, %s, %s)",
            (datetime.now(timezone.utc), notify.channel, notify.payload, None),
        )
