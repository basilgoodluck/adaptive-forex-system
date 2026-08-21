import os
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

import db
import dc_scanner
import features
import train_hmm

load_dotenv()

DSN = os.getenv("DATABASE_URL")
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = "M15"
THETA = float(os.getenv("DC_THRESHOLD", "0.001"))
TRAIN_START = datetime(2020, 1, 1)
TRAIN_END = datetime(2022, 12, 31)

if __name__ == "__main__":
    conn = db.get_conn(DSN)

    all_features = []
    for symbol in SYMBOLS:
        bars = db.fetch_recent_bars(conn, symbol, TIMEFRAME, limit=2_000_000)
        bars = bars[(bars["time"] >= TRAIN_START) & (bars["time"] <= TRAIN_END)]

        events = dc_scanner.scan_dc_events(
            bars["close"].tolist(), bars["time"].tolist(), THETA
        )
        feature_matrix, _ = features.build_feature_matrix(events, bars, THETA)
        all_features.append(feature_matrix)

    combined = np.vstack(all_features)

    mean, std = train_hmm.fit_scaler(combined)
    scaled = train_hmm.apply_scaler(combined, mean, std)

    model = train_hmm.train_hmm(scaled, n_states=5)
    label_map = train_hmm.map_states_to_labels(model, scaled)

    train_hmm.save_model(model, mean, std, "models/hmm_model.pkl")

    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f)

    print("Training complete. Label map:", label_map)