from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db
from app.routes import regimes, dc_events, signals


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    yield
    await db.disconnect()


app = FastAPI(title="regime-backend", lifespan=lifespan)

app.include_router(regimes.router)
app.include_router(dc_events.router)
app.include_router(signals.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
