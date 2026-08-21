from pydantic import BaseModel
from datetime import datetime
from typing import Literal

RegimeLabel = Literal[
    "TREND_CALM", "TREND_VOLATILE", "RANGE_CALM", "RANGE_VOLATILE", "BREAKOUT"
]


class RegimeUpdate(BaseModel):
    time: datetime
    symbol: str
    regime_label: RegimeLabel
    prob_trend_calm: float
    prob_trend_volatile: float
    prob_range_calm: float
    prob_range_volatile: float
    prob_breakout: float
    confidence: float
    log_likelihood: float | None = None
    atr_percentile: float | None = None
