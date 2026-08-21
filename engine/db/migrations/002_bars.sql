CREATE TABLE bars (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    tick_volume DOUBLE PRECISION,
    spread DOUBLE PRECISION,
    PRIMARY KEY (time, symbol, timeframe)
);

SELECT create_hypertable('bars', 'time');

CREATE INDEX idx_bars_symbol_tf_time ON bars (symbol, timeframe, time DESC);
