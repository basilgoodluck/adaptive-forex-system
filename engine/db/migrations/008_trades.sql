CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    open_time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    sl_price DOUBLE PRECISION,
    tp_price DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
    close_time TIMESTAMPTZ,
    close_price DOUBLE PRECISION,
    pnl DOUBLE PRECISION
);

CREATE INDEX idx_trades_symbol_open_time ON trades (symbol, open_time DESC);
CREATE INDEX idx_trades_status ON trades (status);
