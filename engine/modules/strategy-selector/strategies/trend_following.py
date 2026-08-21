import talib
import numpy as np


def generate_signal(close: np.ndarray, fast_period: int = 12, slow_period: int = 26) -> int:
    ema_fast = talib.EMA(close, timeperiod=fast_period)
    ema_slow = talib.EMA(close, timeperiod=slow_period)

    if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
        return 0

    if ema_fast[-1] > ema_slow[-1]:
        return 1
    elif ema_fast[-1] < ema_slow[-1]:
        return -1
    return 0
