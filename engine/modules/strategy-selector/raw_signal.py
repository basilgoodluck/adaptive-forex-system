from pydantic import BaseModel
from datetime import datetime
from typing import Literal


class RawSignal(BaseModel):
    time: datetime
    symbol: str
    active_strategy: Literal["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT"]
    weighted_signal: float
    weight_trend_following: float
    weight_mean_reversion: float
    weight_breakout: float
