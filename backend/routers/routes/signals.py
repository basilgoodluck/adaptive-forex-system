from fastapi import APIRouter, Depends

from app.db import get_pool
from app.params import RangeParams, range_params

router = APIRouter(prefix="/signals", tags=["signals"])

BASE_COLUMNS = """
    time, symbol, active_strategy, weighted_signal,
    weight_trend_following, weight_mean_reversion, weight_breakout
"""


@router.get("")
async def list_signals(params: RangeParams = Depends(range_params)):
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
        FROM raw_signals
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
