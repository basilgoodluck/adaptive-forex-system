from datetime import datetime
from typing import Optional
from fastapi import Query
from pydantic import BaseModel


class RangeParams(BaseModel):
    symbol: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 1000


def range_params(
    symbol: Optional[str] = Query(default=None),
    from_: Optional[datetime] = Query(default=None, alias="from"),
    to: Optional[datetime] = Query(default=None),
    limit: int = Query(default=1000, le=5000, gt=0),
) -> RangeParams:
    return RangeParams(symbol=symbol, date_from=from_, date_to=to, limit=limit)
