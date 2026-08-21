import talib
import numpy as np


def generate_signal(close: np.ndarray, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14) -> int:
    upper, middle, lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
    rsi = talib.RSI(close, timeperiod=rsi_period)

    if np.isnan(upper[-1]) or np.isnan(rsi[-1]):
        return 0

    price = close[-1]

    if price <= lower[-1] and rsi[-1] < 30:
        return 1
    elif price >= upper[-1] and rsi[-1] > 70:
        return -1
    return 0
