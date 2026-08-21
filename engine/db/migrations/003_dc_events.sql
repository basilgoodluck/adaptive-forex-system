CREATE TABLE dc_events (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('UP', 'DOWN')),
    price DOUBLE PRECISION NOT NULL,
    extreme_price DOUBLE PRECISION NOT NULL,
    overshoot_ratio DOUBLE PRECISION,
    duration_bars INTEGER,
    directional_imbalance DOUBLE PRECISION,
    realized_volatility DOUBLE PRECISION,
    adx DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('dc_events', 'time');

CREATE INDEX idx_dc_events_symbol_time ON dc_events (symbol, time DESC);
