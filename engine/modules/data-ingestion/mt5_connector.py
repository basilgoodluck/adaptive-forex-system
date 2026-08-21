import MetaTrader5 as mt5
from datetime import datetime

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


def connect(login: int, password: str, server: str):
    if not mt5.initialize(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")


def disconnect():
    mt5.shutdown()


def fetch_historical_bars(symbol: str, timeframe: str, start: datetime, end: datetime):
    tf = TIMEFRAME_MAP[timeframe]
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None:
        raise RuntimeError(f"copy_rates_range failed: {mt5.last_error()}")
    return rates


def fetch_latest_bar(symbol: str, timeframe: str):
    tf = TIMEFRAME_MAP[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    return rates[0]