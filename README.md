# Adaptive Forex System

A regime-aware forex trading system built as a set of small, event-driven Python services.
It works out what kind of market each currency pair is in (calm or volatile, trending,
ranging or breaking out) and uses that to pick a strategy and size the risk. Every trade
it makes can be traced back to the regime and signal that caused it.

*Built as a final-year project.*

## Architecture

The services don't call each other. Each one writes a row to PostgreSQL/TimescaleDB and then calls
`pg_notify`. The next service in the chain is `LISTEN`ing on that channel
and picks it up. The database is both the message bus and the audit log.

```
 MT5 terminal
     │
 data-ingestion ──new_bar──▶ regime-detector ──new_regime_update──▶ strategy-selector
                                                                        │
                                                                  new_raw_signal
                                                                        ▼
 execution ◀──new_order_spec── risk-manager (sizing via SQL stored procedure)
     │
  new_trade ─────────▶ logger (records every event into system_events)

 FastAPI backend ──▶ read-only API over regimes, DC events, signals ──▶ Astro dashboard
```

| Service | What it does |
|---|---|
| `data-ingestion` | Backfills history and streams live M15 bars from MetaTrader 5 into a TimescaleDB hypertable |
| `regime-detector` | Finds **directional-change (DC) events**, builds features (overshoot ratio, directional imbalance, realized volatility, ATR percentile) and classifies the regime with a trained **Gaussian Hidden Markov Model** (`hmmlearn`). Outputs a label and the probability of each state |
| `strategy-selector` | Weights trend-following, mean-reversion and breakout strategies by regime probability to produce a raw signal |
| `risk-manager` | Sizes the position with a Postgres stored procedure. The stop distance is a regime-dependent ATR multiple, falling back to a fixed pip stop when volatility is extreme. It also applies an account drawdown limit |
| `execution` | Places the order with stop-loss and take-profit on MT5 and records the trade |
| `logger` | Listens on every channel and writes a unified event log |
| `backend` | FastAPI service exposing regimes, DC events and signals |
| `frontend` | Astro + TypeScript dashboard (falls back to mock data if no API is configured) |

**Pairs:** EURUSD, GBPUSD and USDJPY on the M15 timeframe.

## Model

`regime-detector/train_hmm.py` fits a 5-state Gaussian HMM with multiple random restarts
and keeps the best log-likelihood. `sweep.py` grid-searches the DC threshold (θ), feature window
and number of states on a held-out validation period. The hidden states are mapped to readable labels
(`TREND_CALM`, `TREND_VOLATILE`, `RANGE_CALM`, `RANGE_VOLATILE`, `BREAKOUT`) from their
volatility and directional statistics. The model and label map are saved to
`models/` and mounted into the container.

## Running

```bash
# 1. Database: PostgreSQL with the TimescaleDB extension, then run the migrations
python engine/db/migrate.py

# 2. Analysis services (Docker)
cd engine                            # create .env: DATABASE_URL, DC_THRESHOLD, ACCOUNT_EQUITY,
                                     #   RISK_FRACTION, MAX_DRAWDOWN, PIP_VALUE, TP_RATIO, FIXED_PIP_FALLBACK
docker compose up -d                 # regime-detector, strategy-selector, risk-manager, logger

# 3. Broker-facing services, run on a Windows host with MetaTrader 5 installed
python engine/modules/data-ingestion/main.py
python engine/modules/execution/main.py

# 4. Dashboard
cd frontend && npm install && npm run dev
```

The data-ingestion and execution services run outside Docker because the `MetaTrader5`
Python package only works on Windows, next to a running MT5 terminal.

## Tech

Python 3, psycopg 3, pandas, NumPy, hmmlearn, TA-Lib, Pydantic (shared message schemas),
FastAPI, PostgreSQL + TimescaleDB (hypertables, stored procedures, LISTEN/NOTIFY),
Docker Compose, Astro, TypeScript.
