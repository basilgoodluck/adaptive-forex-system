from fastapi import APIRouter, Depends

from app.db import get_pool
from app.params import RangeParams, range_params

router = APIRouter(prefix="/dc-events", tags=["dc-events"])

BASE_COLUMNS = """
    time, symbol, event_type, price, extreme_price,
    overshoot_ratio, duration_bars, directional_imbalance,
    realized_volatility, adx
"""


@router.get("")
async def list_dc_events(params: RangeParams = Depends(range_params)):
    conditions = []
    args = []

    if params.symbol:
        conditions.append("symbol = %s")
        args.append(params.symbol)
    if params.date_from:
        conditions.append("time >= %s")
        args.append(params.date_from)
    if params.date_to:
        conditions.append("time <= %s")
        args.append(params.date_to)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT {BASE_COLUMNS}
        FROM dc_events
        {where_clause}
        ORDER BY time ASC
        LIMIT %s
    """
    args.append(params.limit)

    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            columns = [desc.name for desc in cur.description]
            rows = await cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
