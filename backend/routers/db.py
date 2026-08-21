import os
from psycopg_pool import AsyncConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]

pool: AsyncConnectionPool | None = None


async def connect():
    global pool
    pool = AsyncConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=10, open=False)
    await pool.open()


async def disconnect():
    global pool
    if pool is not None:
        await pool.close()
        pool = None


def get_pool() -> AsyncConnectionPool:
    if pool is None:
        raise RuntimeError("DB pool not initialized")
    return pool
