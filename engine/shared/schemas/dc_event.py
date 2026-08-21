from pydantic import BaseModel
from datetime import datetime
from typing import Literal


class DCEvent(BaseModel):
    time: datetime
    symbol: str
    event_type: Literal["UP", "DOWN"]
    price: float
    extreme_price: float
    overshoot_ratio: float | None = None
    duration_bars: int | None = None
    directional_imbalance: float | None = None
    realized_volatility: float | None = None
    adx: float | None = None
