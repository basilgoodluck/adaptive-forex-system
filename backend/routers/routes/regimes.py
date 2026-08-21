from fastapi import APIRouter, Depends

from app.db import get_pool
from app.params import RangeParams, range_params

router = APIRouter(prefix="/regimes", tags=["regimes"])

BASE_COLUMNS = """
    time, symbol, regime_label,
    prob_trend_calm, prob_trend_volatile, prob_range_calm,
    prob_range_volatile, prob_breakout,
    confidence, log_likelihood, atr_percentile
"""


@router.get("")
async def list_regimes(params: RangeParams = Depends(range_params)):
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
        FROM regime_updates
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


@router.get("/latest")
async def latest_regimes(symbol: str | None = None):
    conditions = []
    args = []
    if symbol:
        conditions.append("symbol = %s")
        args.append(symbol)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT DISTINCT ON (symbol) {BASE_COLUMNS}
        FROM regime_updates
        {where_clause}
        ORDER BY symbol, time DESC
    """

    pool = get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, args)
            columns = [desc.name for desc in cur.description]
            rows = await cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
