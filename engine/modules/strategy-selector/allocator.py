import numpy as np

from strategies import trend_following, mean_reversion, breakout

REGIME_STRATEGY_MAP = {
    "TREND_CALM": "TREND_FOLLOWING",
    "TREND_VOLATILE": "TREND_FOLLOWING",
    "RANGE_CALM": "MEAN_REVERSION",
    "RANGE_VOLATILE": "MEAN_REVERSION",
    "BREAKOUT": "BREAKOUT",
}


def compute_signal(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    tf_signal = trend_following.generate_signal(close)
    mr_signal = mean_reversion.generate_signal(close)
    bo_signal = breakout.generate_signal(high, low, close)
    return {
        "TREND_FOLLOWING": tf_signal,
        "MEAN_REVERSION": mr_signal,
        "BREAKOUT": bo_signal,
    }


def allocate(regime_probs: dict, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    signals = compute_signal(high, low, close)

    strategy_weight = {"TREND_FOLLOWING": 0.0, "MEAN_REVERSION": 0.0, "BREAKOUT": 0.0}
    for regime_label, prob in regime_probs.items():
        strategy = REGIME_STRATEGY_MAP[regime_label]
        strategy_weight[strategy] += prob

    weighted_signal = sum(strategy_weight[s] * signals[s] for s in signals)

    active_regime = max(regime_probs, key=regime_probs.get)
    active_strategy = REGIME_STRATEGY_MAP[active_regime]

    return {
        "weighted_signal": weighted_signal,
        "active_strategy": active_strategy,
        "strategy_weights": strategy_weight,
        "raw_signals": signals,
    }
