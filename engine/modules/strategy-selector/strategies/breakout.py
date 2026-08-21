import numpy as np


def generate_signal(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 20) -> int:
    if len(high) < lookback + 1:
        return 0

    channel_high = np.max(high[-lookback - 1:-1])
    channel_low = np.min(low[-lookback - 1:-1])
    price = close[-1]

    if price > channel_high:
        return 1
    elif price < channel_low:
        return -1
    return 0
