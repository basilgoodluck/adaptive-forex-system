CREATE TABLE system_events (
    time TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    symbol TEXT,
    detail TEXT
);

SELECT create_hypertable('system_events', 'time');

CREATE INDEX idx_system_events_type_time ON system_events (event_type, time DESC);
