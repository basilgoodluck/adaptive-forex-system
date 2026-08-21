import os
import json
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

import db
import dc_scanner
import features
import train_hmm
import infer

from shared.schemas.dc_event import DCEvent
from shared.schemas.regime_update import RegimeUpdate

load_dotenv()

DSN = os.getenv("DATABASE_URL")
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M15"
THETA = float(os.getenv("DC_THRESHOLD", "0.001"))
FEATURE_WINDOW = 20

model, feat_mean, feat_std = train_hmm.load_model("models/hmm_model.pkl")
with open("models/label_map.json") as f:
    raw_label_map = json.load(f)
    label_map = {int(k): v for k, v in raw_label_map.items()}


def process_symbol(conn, symbol: str):
    bars = db.fetch_recent_bars(conn, symbol, TIMEFRAME, limit=500)
    if len(bars) < FEATURE_WINDOW + 5:
        return

    events = dc_scanner.scan_dc_events(
        bars["close"].tolist(), bars["time"].tolist(), THETA
    )
    if len(events) == 0:
        return

    feature_matrix, kept_events = features.build_feature_matrix(events, bars, THETA)
    if len(feature_matrix) == 0:
        return

    last_event = kept_events[-1]
    dc_event = DCEvent(
        time=last_event["time"],
        symbol=symbol,
        event_type=last_event["type"],
        price=last_event["price"],
        extreme_price=last_event["extreme_price"],
        overshoot_ratio=float(feature_matrix[-1][0]),
        duration_bars=int(feature_matrix[-1][1]),
        directional_imbalance=float(feature_matrix[-1][2]),
        realized_volatility=float(feature_matrix[-1][3]),
        adx=float(feature_matrix[-1][4]),
    )
    db.insert_dc_event(conn, dc_event)

    scaled_matrix = train_hmm.apply_scaler(feature_matrix, feat_mean, feat_std)
    window = scaled_matrix[-FEATURE_WINDOW:] if len(scaled_matrix) >= FEATURE_WINDOW else scaled_matrix
    regime_label, probs, confidence, log_prob = infer.infer_regime(model, label_map, window)

    percentile = infer.atr_percentile(
        bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    )

    update = RegimeUpdate(
        time=datetime.now(timezone.utc),
        symbol=symbol,
        regime_label=regime_label,
        prob_trend_calm=probs.get("TREND_CALM", 0.0),
        prob_trend_volatile=probs.get("TREND_VOLATILE", 0.0),
        prob_range_calm=probs.get("RANGE_CALM", 0.0),
        prob_range_volatile=probs.get("RANGE_VOLATILE", 0.0),
        prob_breakout=probs.get("BREAKOUT", 0.0),
        confidence=confidence,
        log_likelihood=log_prob,
        atr_percentile=percentile,
    )
    db.insert_regime_update(conn, update)


if __name__ == "__main__":
    conn = db.get_conn(DSN)
    listener = db.listen_new_bar(DSN)

    for symbol in SYMBOLS:
        process_symbol(conn, symbol)

    for notify in listener.notifies():
        symbol = notify.payload
        if symbol in SYMBOLS:
            process_symbol(conn, symbol)