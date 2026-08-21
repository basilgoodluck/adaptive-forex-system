from pydantic import BaseModel
from datetime import datetime


class BarData(BaseModel):
    time: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    tick_volume: float | None = None
    spread: float | None = None
