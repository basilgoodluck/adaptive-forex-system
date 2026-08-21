CREATE TABLE raw_signals (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    active_strategy TEXT NOT NULL CHECK (active_strategy IN ('TREND_FOLLOWING', 'MEAN_REVERSION', 'BREAKOUT')),
    weighted_signal DOUBLE PRECISION NOT NULL,
    weight_trend_following DOUBLE PRECISION NOT NULL,
    weight_mean_reversion DOUBLE PRECISION NOT NULL,
    weight_breakout DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('raw_signals', 'time');

CREATE INDEX idx_raw_signals_symbol_time ON raw_signals (symbol, time DESC);
