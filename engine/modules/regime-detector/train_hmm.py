import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM

STATE_LABELS = ["TREND_CALM", "TREND_VOLATILE", "RANGE_CALM", "RANGE_VOLATILE", "BREAKOUT"]


def fit_scaler(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(feature_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (feature_matrix - mean) / std


def train_hmm(feature_matrix: np.ndarray, n_states: int = 5, n_iter: int = 100, n_restarts: int = 10):
    best_model = None
    best_score = -np.inf

    for seed in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=seed,
        )
        model.fit(feature_matrix)
        score = model.score(feature_matrix)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model


def save_model(model, mean: np.ndarray, std: np.ndarray, path: str):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "mean": mean, "std": std}, f)


def load_model(path: str):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["mean"], bundle["std"]


def map_states_to_labels(model, feature_matrix: np.ndarray) -> dict[int, str]:
    """
    Assigns human-readable labels to HMM's arbitrary state indices by ranking
    states on mean directional_imbalance (col 2) and mean realized_volatility (col 3).
    """
    hidden_states = model.predict(feature_matrix)
    state_stats = []
    for state in range(model.n_components):
        mask = hidden_states == state
        if mask.sum() == 0:
            state_stats.append((state, 0.0, 0.0))
            continue
        mean_imbalance = feature_matrix[mask, 2].mean()
        mean_vol = feature_matrix[mask, 3].mean()
        state_stats.append((state, mean_imbalance, mean_vol))

    vol_sorted = sorted(state_stats, key=lambda x: x[2])
    breakout_state = vol_sorted[-1][0]

    remaining = [s for s in state_stats if s[0] != breakout_state]
    remaining_by_trend = sorted(remaining, key=lambda x: abs(x[1] - 1.0), reverse=True)

    label_map = {breakout_state: "BREAKOUT"}
    trend_states = remaining_by_trend[:2]
    range_states = remaining_by_trend[2:]

    trend_states_by_vol = sorted(trend_states, key=lambda x: x[2])
    range_states_by_vol = sorted(range_states, key=lambda x: x[2])

    if len(trend_states_by_vol) == 2:
        label_map[trend_states_by_vol[0][0]] = "TREND_CALM"
        label_map[trend_states_by_vol[1][0]] = "TREND_VOLATILE"
    if len(range_states_by_vol) == 2:
        label_map[range_states_by_vol[0][0]] = "RANGE_CALM"
        label_map[range_states_by_vol[1][0]] = "RANGE_VOLATILE"

    return label_map