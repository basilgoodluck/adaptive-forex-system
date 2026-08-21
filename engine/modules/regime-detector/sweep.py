import os
import json
import itertools
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

TRAIN_START = datetime(2020, 1, 1)
TRAIN_END = datetime(2022, 12, 31)
VAL_START = datetime(2023, 1, 1)
VAL_END = datetime(2023, 12, 31)

THETA_GRID = [0.0005, 0.001, 0.002, 0.005]
WINDOW_GRID = [10, 20, 40]
K_GRID = [5]


def load_period_features(conn, start, end, theta, window):
    all_features = []
    for symbol in SYMBOLS:
        bars = db.fetch_recent_bars(conn, symbol, TIMEFRAME, limit=2_000_000)
        bars = bars[(bars["time"] >= start) & (bars["time"] <= end)]
        if len(bars) < 100:
            continue
        events = dc_scanner.scan_dc_events(bars["close"].tolist(), bars["time"].tolist(), theta)
        if len(events) < window + 5:
            continue
        fm, _ = features.build_feature_matrix(events, bars, theta, window=window)
        if len(fm) == 0:
            continue
        all_features.append(fm)
    if not all_features:
        return None
    return np.vstack(all_features)


def score_candidate(model, val_features):
    log_prob = model.score(val_features)
    hidden_states = model.predict(val_features)
    n_states_used = len(set(hidden_states))
    transitions = np.diff(hidden_states)
    flip_rate = np.mean(transitions != 0)
    return log_prob, n_states_used, flip_rate


if __name__ == "__main__":
    conn = db.get_conn(DSN)
    results = []

    for theta, window, k in itertools.product(THETA_GRID, WINDOW_GRID, K_GRID):
        train_features = load_period_features(conn, TRAIN_START, TRAIN_END, theta, window)
        val_features = load_period_features(conn, VAL_START, VAL_END, theta, window)

        if train_features is None or val_features is None:
            continue

        mean, std = train_hmm.fit_scaler(train_features)
        train_scaled = train_hmm.apply_scaler(train_features, mean, std)
        val_scaled = train_hmm.apply_scaler(val_features, mean, std)

        model = train_hmm.train_hmm(train_scaled, n_states=k)
        log_prob, n_states_used, flip_rate = score_candidate(model, val_scaled)

        results.append({
            "theta": theta,
            "window": window,
            "k": k,
            "val_log_likelihood": log_prob,
            "states_used": n_states_used,
            "flip_rate": flip_rate,
        })
        print(results[-1])

    results.sort(key=lambda r: r["val_log_likelihood"], reverse=True)

    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Best candidate:", results[0])