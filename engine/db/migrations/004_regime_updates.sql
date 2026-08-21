CREATE TABLE regime_updates (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    regime_label TEXT NOT NULL CHECK (regime_label IN (
        'TREND_CALM', 'TREND_VOLATILE', 'RANGE_CALM', 'RANGE_VOLATILE', 'BREAKOUT'
    )),
    prob_trend_calm DOUBLE PRECISION NOT NULL,
    prob_trend_volatile DOUBLE PRECISION NOT NULL,
    prob_range_calm DOUBLE PRECISION NOT NULL,
    prob_range_volatile DOUBLE PRECISION NOT NULL,
    prob_breakout DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    log_likelihood DOUBLE PRECISION,
    atr_percentile DOUBLE PRECISION,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('regime_updates', 'time');

CREATE INDEX idx_regime_updates_symbol_time ON regime_updates (symbol, time DESC);
