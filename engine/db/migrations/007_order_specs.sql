CREATE TABLE order_specs (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    position_size DOUBLE PRECISION NOT NULL,
    sl_distance DOUBLE PRECISION NOT NULL,
    tp_distance DOUBLE PRECISION NOT NULL,
    used_atr BOOLEAN NOT NULL,
    regime_label TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('order_specs', 'time');

CREATE INDEX idx_order_specs_symbol_time ON order_specs (symbol, time DESC);
