import MetaTrader5 as mt5


def connect(login: int, password: str, server: str):
    if not mt5.initialize(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")


def get_price(symbol: str, direction: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick failed for {symbol}: {mt5.last_error()}")
    return tick.ask if direction == "BUY" else tick.bid


def place_order(symbol: str, direction: str, volume: float, sl_distance: float, tp_distance: float):
    price = get_price(symbol, direction)
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    if direction == "BUY":
        sl = price - sl_distance
        tp = price + tp_distance
    else:
        sl = price + sl_distance
        tp = price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 100001,
        "comment": "regime-aware-system",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send failed: {result}")

    return {"entry_price": result.price, "sl_price": sl, "tp_price": tp}
