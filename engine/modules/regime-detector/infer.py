import numpy as np
import talib

from train_hmm import STATE_LABELS


def infer_regime(model, label_map: dict[int, str], feature_window: np.ndarray):
    hidden_states = model.predict(feature_window)
    current_state = hidden_states[-1]
    regime_label = label_map[current_state]

    log_prob, posteriors = model.score_samples(feature_window)
    probs = posteriors[-1]

    ordered_probs = {label_map[i]: probs[i] for i in range(model.n_components)}
    confidence = float(max(probs))

    return regime_label, ordered_probs, confidence, float(log_prob)


def atr_percentile(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 200) -> float:
    atr_series = talib.ATR(high, low, close, timeperiod=14)
    atr_series = atr_series[~np.isnan(atr_series)]
    if len(atr_series) < 20:
        return 50.0
    recent = atr_series[-lookback:]
    current = recent[-1]
    percentile = (recent < current).sum() / len(recent) * 100
    return float(percentile)
