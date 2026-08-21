from pydantic import BaseModel
from datetime import datetime
from typing import Literal


class OrderSpec(BaseModel):
    time: datetime
    symbol: str
    direction: Literal["BUY", "SELL"]
    position_size: float
    sl_distance: float
    tp_distance: float
    used_atr: bool
    regime_label: str
    confidence: float
