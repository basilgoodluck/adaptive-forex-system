import numpy as np
import talib


def overshoot_ratio(event, threshold: float) -> float:
    p_dc = event["extreme_price"]
    p_end = event["price"]
    return (p_end - p_dc) / (p_dc * threshold)


def directional_imbalance(events, index, window):
    start = max(0, index - window)
    window_events = events[start:index + 1]
    n_up = sum(1 for e in window_events if e["type"] == "UP")
    n_down = sum(1 for e in window_events if e["type"] == "DOWN")
    total = n_up + n_down
    if total == 0:
        return 0.0
    return (n_up - n_down) / total


def realized_volatility(close_prices: np.ndarray) -> float:
    log_returns = np.diff(np.log(close_prices))
    return float(np.std(log_returns))


def build_feature_matrix(events, bars_df, threshold: float, window: int = 20):
    high = bars_df["high"].to_numpy()
    low = bars_df["low"].to_numpy()
    close = bars_df["close"].to_numpy()
    times = bars_df["time"].to_numpy()

    adx_series = talib.ADX(high, low, close, timeperiod=14)

    bar_indices = []
    for event in events:
        idx = np.searchsorted(times, event["time"])
        bar_indices.append(min(idx, len(close) - 1))

    features = []
    kept_events = []
    for i, event in enumerate(events):
        bar_idx = bar_indices[i]

        if np.isnan(adx_series[bar_idx]):
            continue

        os_ratio = overshoot_ratio(event, threshold)
        r_imbalance = directional_imbalance(events, i, window)
        duration_bars = bar_idx - bar_indices[i - 1] if i > 0 else 0

        vol_start = max(0, bar_idx - window)
        vol = realized_volatility(close[vol_start:bar_idx + 1]) if bar_idx > vol_start else 0.0

        adx_val = float(adx_series[bar_idx])

        features.append([os_ratio, duration_bars, r_imbalance, vol, adx_val])
        kept_events.append(event)

    return np.array(features), kept_events