# Modular Adaptive Forex Trading Framework
## Detailed Implementation Plan (LSTM-Based, Python + MT4/MT5)

> Based on: *A Modular Adaptive Forex Trading Framework Incorporating Market Regime Detection and Strategy Allocation*  
> Author: BASIL Goodluck Chibueze | Dept: Computer Science | Matric: 2022/43802  
> Adaptation: LSTM replaces HMM for regime detection; data sourced from MT4/MT5

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Tech Stack](#3-tech-stack)
4. [Module 1 — Data Ingestion](#4-module-1--data-ingestion)
5. [Module 2 — Directional Change (DC) Algorithm](#5-module-2--directional-change-dc-algorithm)
6. [Module 3 — LSTM Regime Detection](#6-module-3--lstm-regime-detection)
7. [Module 4 — Strategy Selection](#7-module-4--strategy-selection)
8. [Module 5 — Risk Management](#8-module-5--risk-management)
9. [Module 6 — Execution Engine](#9-module-6--execution-engine)
10. [Module 7 — Logging & Transparency](#10-module-7--logging--transparency)
11. [Event Bus (Inter-Module Communication)](#11-event-bus-inter-module-communication)
12. [Data Pipeline & Preprocessing](#12-data-pipeline--preprocessing)
13. [LSTM Model Design (Deep Dive)](#13-lstm-model-design-deep-dive)
14. [Backtesting Framework](#14-backtesting-framework)
15. [Walk-Forward Validation](#15-walk-forward-validation)
16. [Demo Trading Validation (MT5)](#16-demo-trading-validation-mt5)
17. [Project Folder Structure](#17-project-folder-structure)
18. [Implementation Phases & Timeline](#18-implementation-phases--timeline)
19. [Key Formulas & Equations](#19-key-formulas--equations)
20. [Common Pitfalls & How to Avoid Them](#20-common-pitfalls--how-to-avoid-them)

---

## 1. Project Overview

### What Are We Building?
A **fully modular, adaptive** Forex trading system that:
- Pulls real-time and historical price data from MT4/MT5
- Applies the **Directional Change (DC)** algorithm to create event-based features (instead of fixed time-interval features)
- Uses an **LSTM neural network** (replacing HMM) to classify the current market regime into: `TRENDING`, `RANGING`, or `VOLATILE`
- Automatically switches between trading strategies (trend-following, mean-reversion, breakout) based on the detected regime
- Applies dynamic, volatility-aware risk management per trade
- Logs every decision for audit and transparency

### Why LSTM Instead of HMM?
| Feature | HMM (Original) | LSTM (Your Adaptation) |
|---|---|---|
| Model type | Probabilistic graphical model | Deep recurrent neural network |
| Sequence learning | Markov assumption (1-step memory) | Long-range temporal dependencies |
| Non-linearity | Limited | Full (activation functions, gates) |
| Feature handling | Univariate or low-dimensional | Multi-dimensional input vectors |
| Training data needed | Low | Moderate–High |
| Interpretability | Moderate | Lower (use SHAP to compensate) |
| Regime boundary sharpness | Soft probabilities | Softmax probabilities per class |

LSTM is better suited when you have multi-feature input (DC indicators + price stats + volume) and enough historical data (2020–2024 = ~5 years).

---

## 2. Architecture Overview

```
                         ┌──────────────────────────────────┐
                         │          EVENT BUS               │
                         │  (asyncio Queue / pub-sub)       │
                         └────────────┬─────────────────────┘
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        │                             │                              │
        ▼                             ▼                              ▼
┌──────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  DATA        │  ──────▶  │  DIRECTIONAL     │  ──────▶  │  LSTM REGIME     │
│  INGESTION   │           │  CHANGE MODULE   │           │  DETECTOR        │
│  (MT4/MT5)   │           │  (Feature Eng.)  │           │  (Classifier)    │
└──────────────┘           └──────────────────┘           └────────┬─────────┘
                                                                    │
                                                           regime label
                                                                    │
                                                                    ▼
                                                        ┌──────────────────┐
                                                        │  STRATEGY        │
                                                        │  SELECTOR        │
                                                        │  (Rule Engine)   │
                                                        └────────┬─────────┘
                                                                 │
                                                        signal + direction
                                                                 │
                                          ┌──────────────────────┴──────────────────────┐
                                          │                                             │
                                          ▼                                             ▼
                                ┌──────────────────┐                       ┌──────────────────┐
                                │  RISK MANAGEMENT │                       │  LOGGING &       │
                                │  MODULE          │                       │  MONITORING      │
                                │  (Sizing, SL/TP) │                       │  (SQLite + file) │
                                └────────┬─────────┘                       └──────────────────┘
                                         │
                                  sized order
                                         │
                                         ▼
                                ┌──────────────────┐
                                │  EXECUTION       │
                                │  ENGINE          │
                                │  (MT5 API)       │
                                └──────────────────┘
```

Each module communicates **only** via the event bus. No module imports another module's internals directly. This is what makes the system modular.

---

## 3. Tech Stack

### Core Libraries
```
Python                 3.11+
MetaTrader5            5.0.45+       # MT5 Python bridge
pandas                 2.x           # Data manipulation
numpy                  1.26+         # Numerical ops
ta-lib                 0.4.x         # Technical indicators (ATR, ADX, RSI, BB)
torch                  2.x           # PyTorch for LSTM
scikit-learn           1.4+          # Preprocessing, metrics
hmmlearn               0.3+          # Optional: compare LSTM vs HMM baseline
backtrader             1.9.78+       # Backtesting engine
plotly                 5.x           # Interactive charts
matplotlib             3.x           # Static plots
shap                   0.44+         # Explainability
SQLite (built-in)                    # Logging backend
loguru                 0.7+          # Enhanced logging
asyncio (built-in)                   # Event bus
pyyaml                 6.x           # Config files
pytest                 8.x           # Unit testing
```

### Install Command
```bash
pip install MetaTrader5 pandas numpy TA-Lib torch scikit-learn hmmlearn \
            backtrader plotly matplotlib shap loguru pyyaml pytest asyncio
```

> **Note on TA-Lib**: Requires C library. On Windows, download the prebuilt `.whl` from [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/). On Linux: `apt-get install libta-lib-dev`.

---

## 4. Module 1 — Data Ingestion

### Responsibility
- Connect to MT4/MT5 terminal
- Pull historical OHLCV bars (for training/backtesting)
- Stream real-time tick data (for live demo trading)
- Validate, clean, and normalize all data
- Emit cleaned `BarEvent` or `TickEvent` onto the event bus

### MT5 Connection Setup
```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def connect_mt5(login: int, password: str, server: str) -> bool:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    authorized = mt5.login(login, password=password, server=server)
    if not authorized:
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    return True
```

### Pulling Historical Data
```python
def get_historical_bars(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    """
    timeframe options:
        mt5.TIMEFRAME_M1   → 1-minute bars
        mt5.TIMEFRAME_M5   → 5-minute bars
        mt5.TIMEFRAME_H1   → 1-hour bars
        mt5.TIMEFRAME_D1   → Daily bars
    """
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]
```

### Streaming Real-Time Ticks
```python
import asyncio

async def stream_ticks(symbol: str, event_bus, interval: float = 0.5):
    """Polls MT5 for latest tick every `interval` seconds."""
    last_tick_time = None
    while True:
        tick = mt5.symbol_info_tick(symbol)
        if tick and tick.time != last_tick_time:
            last_tick_time = tick.time
            await event_bus.put({
                'type': 'TICK',
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'time': tick.time
            })
        await asyncio.sleep(interval)
```

### Data Cleaning Steps
1. **Remove zero-volume bars** — market closed periods (weekends, holidays)
2. **Remove duplicate timestamps** — keep last occurrence
3. **Forward-fill NaNs** — for missing OHLC values (max 3 consecutive)
4. **Drop rows with NaN after fill** — don't interpolate price data, it introduces look-ahead bias
5. **Normalize timestamps to UTC** — MT5 servers may use broker time zones
6. **Check OHLC validity** — `high >= max(open, close)` and `low <= min(open, close)` must hold

```python
def clean_ohlcv(df: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    df = df[df['volume'] > 0]                        # Step 1
    df = df[~df.index.duplicated(keep='last')]       # Step 2
    df = df.ffill(limit=max_ffill)                   # Step 3
    df = df.dropna()                                 # Step 4
    # Step 6: OHLC sanity check
    valid = (df['high'] >= df[['open','close']].max(axis=1)) & \
            (df['low']  <= df[['open','close']].min(axis=1))
    df = df[valid]
    return df
```

### Data Split Strategy
```
Total data: Jan 2020 → Dec 2024 (5 years)

┌─────────────────┬────────────────┬───────────────┐
│ TRAINING SET    │ VALIDATION SET │ TEST SET      │
│ Jan 2020 –      │ Jan 2023 –     │ Jan 2024 –    │
│ Dec 2022        │ Dec 2023       │ Dec 2024      │
│ (3 years, 60%)  │ (1 year, 20%)  │ (1 year, 20%) │
└─────────────────┴────────────────┴───────────────┘
```

**CRITICAL**: Never expose test set data to model selection or hyperparameter tuning. Use validation set only for that.

---

## 5. Module 2 — Directional Change (DC) Algorithm

### Concept
Instead of sampling price at every bar (time-based), DC samples price only when a **significant reversal** has occurred. This filters noise and captures only meaningful market movements.

### How It Works
1. Start tracking from the first price point
2. Define a threshold `θ` (e.g., 0.004 = 0.4%)
3. Track the current **extreme** (highest high or lowest low since last DC event)
4. When price reverses from the extreme by `θ` or more → record a **DC event** (directional change confirmed)
5. After a DC event, the price continues in the new direction → this continuation is called the **Overshoot (OS) phase**
6. Repeat

```
Price goes UP:
  ────────────────── Peak (extreme)
         /\
        /  \  ← falls by θ → DC DOWN event recorded
       /    \
      /      \
─────/        \──── (overshoot continues down)
```

### DC Feature Engineering
From the DC events, compute the following features for each window:

| Feature | Description | Formula |
|---|---|---|
| `dc_duration` | Time between DC events (in bars) | `t_dc_end - t_dc_start` |
| `dc_magnitude` | Price change during DC event | `abs(P_end - P_start) / P_start` |
| `os_magnitude` | Price change during overshoot | `abs(P_os_end - P_dc_end) / P_dc_end` |
| `dc_count_up` | Number of upward DC events in window | Count |
| `dc_count_down` | Number of downward DC events in window | Count |
| `dc_ratio` | Trend bias | `dc_count_up / (dc_count_up + dc_count_down)` |
| `avg_dc_mag` | Average DC magnitude in window | Rolling mean |
| `avg_os_mag` | Average OS magnitude in window | Rolling mean |

### DC Algorithm Implementation
```python
class DirectionalChange:
    def __init__(self, theta: float = 0.004):
        """
        theta: reversal threshold as a decimal (0.004 = 0.4%)
        """
        self.theta = theta
        self.events = []

    def run(self, prices: list) -> list:
        """
        Returns list of DC event dicts.
        Each event: {
            'type': 'UP' or 'DOWN',
            'start_idx': int,
            'end_idx': int,
            'start_price': float,
            'end_price': float,
            'magnitude': float
        }
        """
        extreme = prices[0]
        extreme_idx = 0
        direction = None  # None until first DC event

        for i, price in enumerate(prices[1:], start=1):
            if direction != 'DOWN':
                # Tracking upward movement
                if price > extreme:
                    extreme = price
                    extreme_idx = i
                elif (extreme - price) / extreme >= self.theta:
                    # DC DOWN event
                    self.events.append({
                        'type': 'DOWN',
                        'start_idx': extreme_idx,
                        'end_idx': i,
                        'start_price': extreme,
                        'end_price': price,
                        'magnitude': (extreme - price) / extreme
                    })
                    extreme = price
                    extreme_idx = i
                    direction = 'DOWN'

            if direction != 'UP':
                # Tracking downward movement
                if price < extreme:
                    extreme = price
                    extreme_idx = i
                elif (price - extreme) / extreme >= self.theta:
                    # DC UP event
                    self.events.append({
                        'type': 'UP',
                        'start_idx': extreme_idx,
                        'end_idx': i,
                        'start_price': extreme,
                        'end_price': price,
                        'magnitude': (price - extreme) / extreme
                    })
                    extreme = price
                    extreme_idx = i
                    direction = 'UP'

        return self.events
```

### Threshold Selection Strategy
- Too small (`θ < 0.1%`): Too many events, sensitive to noise
- Too large (`θ > 1%`): Too few events, misses important structure
- **Recommended starting range**: `θ ∈ {0.002, 0.004, 0.006, 0.008, 0.01}` (0.2%–1%)
- Run the system with multiple `θ` values and pick the one that maximizes regime classification accuracy on the validation set
- Consider using **volatility-adaptive `θ`**: `θ_t = k × ATR_t / P_t` where `k ≈ 1.0–2.0`

---

## 6. Module 3 — LSTM Regime Detection

### What LSTM Brings
LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network that can:
- Remember patterns across many time steps (via cell state `C_t`)
- Forget irrelevant history (via forget gate)
- Learn which features matter for regime transitions (via input/output gates)

This makes it ideal for detecting regime shifts because regime changes are **sequential** and **context-dependent** — the last 50–200 bars matter, not just the current bar.

### LSTM Gates (Reference)
```
Forget gate:   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell state:    C_t = f_t * C_{t-1} + i_t * C̃_t
Output gate:   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state:  h_t = o_t * tanh(C_t)
```
The output `h_t` at the final timestep goes into a Dense layer with Softmax to produce regime probabilities.

### Input Features Vector
For each timestep in the LSTM sequence window, the feature vector `x_t` contains:

```
[
  # Price-based features
  log_return,          # log(close_t / close_{t-1})
  high_low_range,      # (high - low) / close
  close_position,      # (close - low) / (high - low)

  # Technical indicators
  rsi_14,              # RSI (0–100, normalized to 0–1)
  atr_14_norm,         # ATR(14) / close  ← normalized ATR
  adx_14,              # ADX (0–100, normalized to 0–1)
  bb_width,            # (BB_upper - BB_lower) / BB_mid
  macd_hist,           # MACD histogram (normalized)

  # DC-derived features
  dc_ratio,            # up_events / total_events in rolling window
  avg_dc_magnitude,    # average DC event size
  avg_os_magnitude,    # average OS phase size
  dc_duration_mean,    # mean bars between DC events
]
```

Total: ~12 features per timestep.

### Labeling Regimes for Training
Since we don't have ground truth regime labels, we **create them** using rule-based labeling on training data:

```python
def label_regime(df: pd.DataFrame, adx_period=14, atr_period=14) -> pd.Series:
    """
    Rule-based regime labeling for LSTM training targets.
    Labels: 0=RANGING, 1=TRENDING, 2=VOLATILE
    """
    import talib
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=adx_period)
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    atr_pct = atr / df['close']  # normalized ATR
    atr_mean = atr_pct.rolling(50).mean()
    atr_std  = atr_pct.rolling(50).std()

    labels = pd.Series(0, index=df.index)  # default: RANGING

    # TRENDING: strong directional movement
    labels[adx > 25] = 1

    # VOLATILE: abnormally high ATR, regardless of trend
    labels[atr_pct > (atr_mean + 1.5 * atr_std)] = 2

    return labels
```

> **Important**: These labels are your **proxy** for regime truth. The quality of the LSTM depends on how well this labeling captures real market states. Experiment with thresholds.

### LSTM Model Architecture (PyTorch)
```python
import torch
import torch.nn as nn

class RegimeLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 12,     # number of features
        hidden_size: int = 128,   # LSTM hidden units
        num_layers: int = 2,      # stacked LSTM layers
        num_classes: int = 3,     # RANGING, TRENDING, VOLATILE
        dropout: float = 0.3,
        seq_len: int = 60         # lookback window (60 bars)
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]          # take last timestep
        normed = self.batch_norm(last_step)
        logits = self.fc(normed)
        probs = self.softmax(logits)
        return probs
```

### Training Setup
```python
# Hyperparameters
SEQ_LEN      = 60       # 60 bars of history per sample
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 0.001
WEIGHT_DECAY = 1e-5     # L2 regularization

# Loss: handle class imbalance with weights
# If VOLATILE is rare, weight it higher
class_counts = [n_ranging, n_trending, n_volatile]
total = sum(class_counts)
weights = torch.tensor([total / c for c in class_counts])
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
```

### Sequence Dataset Builder
```python
from torch.utils.data import Dataset

class RegimeDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 60):
        self.X = []
        self.y = []
        for i in range(seq_len, len(features)):
            self.X.append(features[i - seq_len:i])  # (seq_len, n_features)
            self.y.append(labels[i])                 # scalar label
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### Training Loop
```python
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

### Real-Time Inference
```python
def predict_regime(model, feature_window: np.ndarray, scaler) -> dict:
    """
    feature_window: numpy array of shape (seq_len, n_features), raw (unscaled)
    Returns: {'regime': str, 'probabilities': dict, 'confidence': float}
    """
    model.eval()
    scaled = scaler.transform(feature_window)  # normalize using training scaler
    tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        probs = model(tensor).squeeze(0).numpy()
    regime_idx = probs.argmax()
    regime_map = {0: 'RANGING', 1: 'TRENDING', 2: 'VOLATILE'}
    return {
        'regime': regime_map[regime_idx],
        'probabilities': {regime_map[i]: float(p) for i, p in enumerate(probs)},
        'confidence': float(probs[regime_idx])
    }
```

### Explainability with SHAP
```python
import shap

def explain_regime_prediction(model, background_data, sample):
    """Use SHAP DeepExplainer to identify which features drove the regime prediction."""
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(sample)
    # shap_values[regime_class] → array of shape (seq_len, n_features)
    return shap_values
```

---

## 7. Module 4 — Strategy Selection

### Responsibility
- Receive the regime label from Module 3
- Select the appropriate strategy class
- Generate trading signals (BUY/SELL/HOLD)
- Pass signal to the Risk Management module

### Strategy → Regime Mapping
```
Regime      │ Strategy          │ Logic
────────────┼───────────────────┼──────────────────────────────────────────
TRENDING    │ Trend-Following   │ Moving average crossover; buy on golden
            │                   │ cross, sell on death cross; confirm w/ ADX
────────────┼───────────────────┼──────────────────────────────────────────
RANGING     │ Mean-Reversion    │ Bollinger Band bounce; buy at lower band,
            │                   │ sell at upper band; RSI confirmation
────────────┼───────────────────┼──────────────────────────────────────────
VOLATILE    │ Breakout          │ Channel/range breakout; enter on momentum;
            │                   │ use wider SL due to noise; or reduce size
```

### Base Strategy Interface
```python
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> dict:
        """
        Returns: {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'confidence': float (0–1),
            'reason': str
        }
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
```

### Trend-Following Strategy
```python
class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, fast_period=20, slow_period=50, adx_threshold=25):
        self.fast = fast_period
        self.slow = slow_period
        self.adx_thresh = adx_threshold

    def generate_signal(self, df):
        import talib
        fast_ma = talib.EMA(df['close'], timeperiod=self.fast)
        slow_ma = talib.EMA(df['close'], timeperiod=self.slow)
        adx     = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        last_fast, prev_fast = fast_ma.iloc[-1], fast_ma.iloc[-2]
        last_slow, prev_slow = slow_ma.iloc[-1], slow_ma.iloc[-2]
        last_adx = adx.iloc[-1]

        if last_adx < self.adx_thresh:
            return {'action': 'HOLD', 'confidence': 0.3, 'reason': 'ADX too weak for trend'}

        if prev_fast < prev_slow and last_fast > last_slow:
            return {'action': 'BUY', 'confidence': min(last_adx / 100, 1.0), 'reason': 'Golden cross'}
        elif prev_fast > prev_slow and last_fast < last_slow:
            return {'action': 'SELL', 'confidence': min(last_adx / 100, 1.0), 'reason': 'Death cross'}
        return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'No crossover'}

    def get_name(self): return "TrendFollowing"
```

### Mean-Reversion Strategy
```python
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, bb_period=20, bb_std=2.0, rsi_period=14):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    def generate_signal(self, df):
        import talib
        upper, mid, lower = talib.BBANDS(df['close'], timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
        rsi = talib.RSI(df['close'], timeperiod=self.rsi_period)

        price = df['close'].iloc[-1]
        rsi_val = rsi.iloc[-1]

        if price <= lower.iloc[-1] and rsi_val < 35:
            return {'action': 'BUY', 'confidence': 0.75, 'reason': 'Price at lower BB, RSI oversold'}
        elif price >= upper.iloc[-1] and rsi_val > 65:
            return {'action': 'SELL', 'confidence': 0.75, 'reason': 'Price at upper BB, RSI overbought'}
        return {'action': 'HOLD', 'confidence': 0.4, 'reason': 'Price within bands'}

    def get_name(self): return "MeanReversion"
```

### Breakout Strategy
```python
class BreakoutStrategy(BaseStrategy):
    def __init__(self, channel_period=20, atr_multiplier=1.5):
        self.period = channel_period
        self.atr_mult = atr_multiplier

    def generate_signal(self, df):
        import talib
        high_channel = df['high'].rolling(self.period).max()
        low_channel  = df['low'].rolling(self.period).min()
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        price = df['close'].iloc[-1]
        prev_high = high_channel.iloc[-2]
        prev_low  = low_channel.iloc[-2]
        atr_val   = atr.iloc[-1]

        if price > prev_high + (atr_val * self.atr_mult * 0.1):
            return {'action': 'BUY', 'confidence': 0.7, 'reason': 'Upside channel breakout'}
        elif price < prev_low - (atr_val * self.atr_mult * 0.1):
            return {'action': 'SELL', 'confidence': 0.7, 'reason': 'Downside channel breakout'}
        return {'action': 'HOLD', 'confidence': 0.3, 'reason': 'Inside channel'}

    def get_name(self): return "Breakout"
```

### Strategy Selector Engine
```python
class StrategySelector:
    def __init__(self):
        self.strategies = {
            'TRENDING': TrendFollowingStrategy(),
            'RANGING':  MeanReversionStrategy(),
            'VOLATILE': BreakoutStrategy()
        }

    def select_and_signal(self, regime: str, df: pd.DataFrame) -> dict:
        strategy = self.strategies.get(regime)
        if strategy is None:
            return {'action': 'HOLD', 'confidence': 0, 'reason': 'Unknown regime'}
        signal = strategy.generate_signal(df)
        signal['strategy_used'] = strategy.get_name()
        signal['regime'] = regime
        return signal
```

---

## 8. Module 5 — Risk Management

### Responsibility
- Calculate position size based on current volatility and account equity
- Set Stop-Loss (SL) and Take-Profit (TP) levels dynamically
- Enforce maximum drawdown limits
- Reduce exposure during volatile regimes
- Cap total portfolio exposure

### Volatility-Based Position Sizing
The formula adjusts lot size so that if price moves against you by 1×ATR, you lose exactly `risk_pct` of your equity:

```
risk_amount  = account_equity × risk_pct
atr_value    = ATR(14) of the current bar (in price terms)
pip_value    = monetary value of 1 pip for the symbol (from MT5)
atr_pips     = atr_value / pip_size
position_size = risk_amount / (atr_pips × pip_value)
```

```python
class RiskManager:
    def __init__(self, base_risk_pct=0.01, max_drawdown=0.15, max_total_exposure=0.06):
        self.base_risk     = base_risk_pct      # 1% per trade by default
        self.max_dd        = max_drawdown        # halt at 15% drawdown
        self.max_exposure  = max_total_exposure  # max 6% equity at risk total

    def _get_regime_risk_multiplier(self, regime: str) -> float:
        """Reduce risk in volatile regimes."""
        return {'TRENDING': 1.0, 'RANGING': 0.8, 'VOLATILE': 0.5}.get(regime, 1.0)

    def calculate_position(self, equity: float, atr: float, pip_value: float,
                           pip_size: float, regime: str) -> dict:
        multiplier   = self._get_regime_risk_multiplier(regime)
        adjusted_risk = equity * self.base_risk * multiplier
        atr_pips     = atr / pip_size
        if atr_pips == 0:
            return {'lots': 0, 'sl_pips': 0, 'tp_pips': 0}

        lots  = adjusted_risk / (atr_pips * pip_value)
        lots  = round(lots, 2)                  # MT5 needs rounded lot sizes
        lots  = max(0.01, min(lots, 10.0))      # hard min/max

        sl_pips = atr_pips * 1.5               # SL = 1.5× ATR
        tp_pips = sl_pips * 2.0                # TP = 2× SL (2:1 R:R ratio)

        return {'lots': lots, 'sl_pips': sl_pips, 'tp_pips': tp_pips,
                'risk_pct': adjusted_risk / equity}

    def check_drawdown_halt(self, peak_equity: float, current_equity: float) -> bool:
        """Returns True if trading should halt."""
        dd = (peak_equity - current_equity) / peak_equity
        return dd >= self.max_dd
```

### Dynamic Risk Table by Regime
| Regime   | Risk Multiplier | Typical Lot Reduction | Rationale |
|---|---|---|---|
| TRENDING | 1.0× (full) | None | Clean directional move, higher confidence |
| RANGING  | 0.8× | 20% smaller | False signals more common |
| VOLATILE | 0.5× | 50% smaller | High unpredictability, protect capital |

---

## 9. Module 6 — Execution Engine

### Responsibility
- Translate signals + position sizes into actual MT5 orders
- Handle order lifecycle: place, monitor, modify, close
- Abstract broker details from strategy logic

### Placing a Trade via MT5
```python
import MetaTrader5 as mt5

class ExecutionEngine:
    def __init__(self, symbol: str, magic: int = 12345):
        self.symbol = symbol
        self.magic  = magic  # unique identifier for this EA

    def place_order(self, action: str, lots: float, sl_pips: float, tp_pips: float) -> dict:
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return {'success': False, 'error': 'Symbol not found'}

        tick     = mt5.symbol_info_tick(self.symbol)
        pip_size = symbol_info.point * 10  # standard pip = 10 points for most pairs

        if action == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price      = tick.ask
            sl         = price - sl_pips * pip_size
            tp         = price + tp_pips * pip_size
        elif action == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
            price      = tick.bid
            sl         = price + sl_pips * pip_size
            tp         = price - tp_pips * pip_size
        else:
            return {'success': False, 'error': 'Invalid action'}

        request = {
            'action':    mt5.TRADE_ACTION_DEAL,
            'symbol':    self.symbol,
            'volume':    lots,
            'type':      order_type,
            'price':     price,
            'sl':        round(sl, symbol_info.digits),
            'tp':        round(tp, symbol_info.digits),
            'deviation': 10,
            'magic':     self.magic,
            'comment':   'AdaptiveFramework',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        return {
            'success': result.retcode == mt5.TRADE_RETCODE_DONE,
            'order_id': result.order,
            'retcode': result.retcode,
            'comment': result.comment
        }

    def close_all_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        for pos in (positions or []):
            close_action = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(self.symbol)
            price = tick.bid if close_action == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': pos.volume,
                'type':   close_action,
                'price':  price,
                'position': pos.ticket,
                'magic':  self.magic,
                'comment': 'Close by Framework'
            }
            mt5.order_send(request)
```

---

## 10. Module 7 — Logging & Transparency

### What to Log
Every decision the system makes must be recorded so it can be reviewed, debugged, and explained:

| Event | Fields to Log |
|---|---|
| Regime Detection | timestamp, symbol, regime, confidence, feature_values |
| Strategy Signal | timestamp, symbol, strategy_name, action, confidence, reason |
| Risk Calculation | timestamp, lots, sl_pips, tp_pips, risk_pct, equity |
| Order Placed | timestamp, order_id, symbol, action, price, lots, sl, tp |
| Order Closed | timestamp, order_id, profit, duration, exit_reason |
| System Error | timestamp, module, error_type, message, stack_trace |

### SQLite Schema
```sql
CREATE TABLE regime_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT,
    regime TEXT,                   -- 'TRENDING', 'RANGING', 'VOLATILE'
    confidence REAL,
    prob_trending REAL,
    prob_ranging REAL,
    prob_volatile REAL,
    features TEXT                  -- JSON-encoded feature dict
);

CREATE TABLE trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    order_id INTEGER,
    symbol TEXT,
    action TEXT,                   -- 'BUY', 'SELL'
    strategy_used TEXT,
    regime_at_entry TEXT,
    lots REAL,
    entry_price REAL,
    sl REAL,
    tp REAL,
    exit_price REAL,
    profit REAL,
    duration_minutes REAL,
    exit_reason TEXT               -- 'SL', 'TP', 'MANUAL', 'REGIME_CHANGE'
);

CREATE TABLE system_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT,                    -- 'INFO', 'WARNING', 'ERROR'
    module TEXT,
    message TEXT
);
```

### Python Logger Setup
```python
from loguru import logger
import sqlite3

logger.add("logs/system_{time}.log", rotation="1 day", retention="30 days", level="INFO")

def log_regime(conn, timestamp, symbol, result: dict):
    conn.execute("""
        INSERT INTO regime_log (timestamp, symbol, regime, confidence,
            prob_trending, prob_ranging, prob_volatile)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, symbol,
        result['regime'], result['confidence'],
        result['probabilities'].get('TRENDING', 0),
        result['probabilities'].get('RANGING', 0),
        result['probabilities'].get('VOLATILE', 0)
    ))
    conn.commit()
    logger.info(f"Regime: {result['regime']} ({result['confidence']:.2%}) for {symbol}")
```

---

## 11. Event Bus (Inter-Module Communication)

### Why an Event Bus?
Without an event bus, modules would call each other directly, creating tight coupling. With an event bus, each module only knows about event types, not other modules.

### asyncio-Based Event Bus
```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Event:
    type: str       # e.g., 'NEW_BAR', 'REGIME_DETECTED', 'SIGNAL_GENERATED', 'ORDER_PLACED'
    payload: Any = field(default=None)

class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event_type: str, handler):
        self._subscribers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event):
        for handler in self._subscribers.get(event.type, []):
            await handler(event)
```

### Event Flow
```
DataIngestion  → publishes 'NEW_BAR'
   ↓
DCModule      subscribes to 'NEW_BAR' → computes DC features → publishes 'DC_FEATURES'
   ↓
LSTMDetector  subscribes to 'DC_FEATURES' → runs inference → publishes 'REGIME_DETECTED'
   ↓
StrategySelector subscribes to 'REGIME_DETECTED' → generates signal → publishes 'SIGNAL_GENERATED'
   ↓
RiskManager   subscribes to 'SIGNAL_GENERATED' → calculates size → publishes 'ORDER_READY'
   ↓
Execution     subscribes to 'ORDER_READY' → places trade → publishes 'ORDER_PLACED'
   ↓
Logger        subscribes to ALL events → writes to DB and log file
```

---

## 12. Data Pipeline & Preprocessing

### Feature Scaling
**CRITICAL**: Scale features using StandardScaler (or MinMaxScaler) fitted on **training data only**. Apply the same scaler to validation and test data.

```python
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # FIT on train
X_val_scaled   = scaler.transform(X_val)          # TRANSFORM only
X_test_scaled  = scaler.transform(X_test)         # TRANSFORM only

# Save scaler for use in real-time prediction
joblib.dump(scaler, 'models/feature_scaler.pkl')
```

### Rolling Feature Window
```python
def build_feature_matrix(df: pd.DataFrame, dc_events: list, window: int = 20) -> pd.DataFrame:
    """
    Builds the full feature matrix combining price-based,
    technical indicator, and DC features.
    """
    import talib
    features = pd.DataFrame(index=df.index)

    # Price features
    features['log_return']      = np.log(df['close'] / df['close'].shift(1))
    features['hl_range']        = (df['high'] - df['low']) / df['close']
    features['close_position']  = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # Technical indicators
    features['rsi']    = talib.RSI(df['close'], 14) / 100
    features['adx']    = talib.ADX(df['high'], df['low'], df['close'], 14) / 100
    atr                = talib.ATR(df['high'], df['low'], df['close'], 14)
    features['atr']    = atr / df['close']
    upper, mid, lower  = talib.BBANDS(df['close'], 20, 2, 2)
    features['bb_w']   = (upper - lower) / (mid + 1e-8)
    _, _, macd_hist    = talib.MACD(df['close'])
    features['macd_h'] = macd_hist / df['close']

    # DC features (map events back to bar index)
    dc_df = _dc_events_to_bar_features(df, dc_events, window)
    features = features.join(dc_df)

    features.dropna(inplace=True)
    return features
```

---

## 13. LSTM Model Design (Deep Dive)

### Hyperparameter Search Space
```
seq_len:       [30, 60, 100, 200]    # bars of history
hidden_size:   [64, 128, 256]
num_layers:    [1, 2, 3]
dropout:       [0.2, 0.3, 0.5]
learning_rate: [0.001, 0.0005, 0.0001]
batch_size:    [32, 64, 128]
```

Use **Bayesian optimization** or **random search** on the validation set to find the best combination.

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

### Evaluation Metrics for Regime Classifier
| Metric | Target |
|---|---|
| Macro F1 Score | > 0.70 |
| Per-class Recall | > 0.65 for each regime |
| Accuracy | > 0.72 |
| Confusion matrix | Low off-diagonal for adjacent regimes |

### Preventing Look-Ahead Bias
- Always use `df['feature'].shift(1)` when referencing "previous bar" values
- Never use future price data when computing labels for past bars
- The DC algorithm is by nature backward-looking (it waits for a reversal), so no bias there
- During backtesting, ensure the LSTM model was trained only on data before the current bar

---

## 14. Backtesting Framework

### Walk-Forward Backtesting with Backtrader
```python
import backtrader as bt

class AdaptiveStrategy(bt.Strategy):
    def __init__(self):
        self.lstm_model = load_model('models/lstm_regime.pt')
        self.scaler     = load_scaler('models/feature_scaler.pkl')
        self.selector   = StrategySelector()
        self.risk_mgr   = RiskManager()

    def next(self):
        if len(self.data) < 200:
            return  # wait for enough bars

        df = self._get_current_df()
        features = build_feature_matrix(df, ...)
        regime_result = predict_regime(self.lstm_model, features.values[-60:], self.scaler)
        signal = self.selector.select_and_signal(regime_result['regime'], df)

        if signal['action'] == 'BUY' and not self.position:
            size = self._calculate_size(regime_result['regime'])
            self.buy(size=size)
        elif signal['action'] == 'SELL' and not self.position:
            size = self._calculate_size(regime_result['regime'])
            self.sell(size=size)
```

### Performance Metrics to Track
```
Total Return         = (final_equity - initial_equity) / initial_equity
Sharpe Ratio         = mean(returns) / std(returns) × sqrt(252)       # annualized
Sortino Ratio        = mean(returns) / std(negative_returns) × sqrt(252)
Maximum Drawdown     = max((peak - trough) / peak)
Win Rate             = n_winning_trades / n_total_trades
Profit Factor        = sum(winning_trades) / abs(sum(losing_trades))
Average Trade Return = mean(trade_returns)
Calmar Ratio         = Annual Return / Max Drawdown
```

### Baseline Comparison
You must compare against:
1. **Static Trend-Following** — always uses EMA crossover regardless of regime
2. **Buy-and-Hold** — hold the base currency for the entire period
3. **Random strategy** — random BUY/SELL to set floor performance

If your adaptive framework doesn't beat all three, something is wrong.

---

## 15. Walk-Forward Validation

### Why Walk-Forward?
Regular backtesting optimizes on the same data it tests → overfitting. Walk-forward testing simulates real trading:

```
Window 1:  Train on [Jan 2020–Jun 2021] → Test on [Jul 2021–Dec 2021]
Window 2:  Train on [Jan 2020–Dec 2021] → Test on [Jan 2022–Jun 2022]
Window 3:  Train on [Jan 2020–Jun 2022] → Test on [Jul 2022–Dec 2022]
Window 4:  Train on [Jan 2020–Dec 2022] → Test on [Jan 2023–Jun 2023]
Window 5:  Train on [Jan 2020–Jun 2023] → Test on [Jul 2023–Dec 2023]
```

Each test set is out-of-sample. Results are averaged across all windows for a realistic picture.

### Sensitivity Analysis
Test system robustness by varying one parameter at a time while holding others constant:

| Parameter | Test Range | Metric |
|---|---|---|
| DC threshold θ | 0.002–0.010 | Regime accuracy |
| LSTM seq_len | 30–200 bars | Val F1 |
| ATR multiplier for SL | 1.0–3.0 | Sharpe ratio |
| Risk per trade | 0.005–0.02 | Max drawdown |
| ADX trend threshold | 20–35 | Win rate |

---

## 16. Demo Trading Validation (MT5)

### Minimum Validation Period
- At least **3 months** of continuous demo trading
- Must cover different market conditions (if possible)
- Log every single regime, signal, and trade

### Demo Trading Loop (Main Async Loop)
```python
import asyncio
import MetaTrader5 as mt5

async def main_trading_loop(symbol, timeframe, model, scaler, event_bus):
    last_bar_time = None
    while True:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        current_bar_time = df['time'].iloc[-2]  # use second-to-last (completed bar)
        if current_bar_time != last_bar_time:
            last_bar_time = current_bar_time
            await event_bus.publish(Event('NEW_BAR', payload={'symbol': symbol, 'df': df}))

        await asyncio.sleep(10)  # check every 10 seconds

asyncio.run(main_trading_loop(...))
```

### Checklist Before Going Live (Demo)
- [ ] All modules unit-tested individually
- [ ] Integration tested end-to-end on historical replay
- [ ] LSTM model validated on 2024 test set (F1 > 0.70)
- [ ] Backtesting Sharpe ratio > 1.0
- [ ] Drawdown never exceeded 15% in backtest
- [ ] Logging confirmed working (entries appear in SQLite)
- [ ] Drawdown halt logic tested
- [ ] Order placement tested with dummy orders on demo account

---

## 17. Project Folder Structure

```
adaptive_forex/
│
├── config/
│   ├── config.yaml              # All parameters (θ, risk %, etc.)
│   └── symbols.yaml             # Currency pair configurations
│
├── data/
│   ├── raw/                     # Raw downloaded MT5 data
│   ├── processed/               # Cleaned, featurized data
│   └── splits/                  # Train/val/test pickles
│
├── models/
│   ├── lstm_regime.pt           # Trained LSTM weights
│   ├── feature_scaler.pkl       # Fitted StandardScaler
│   └── checkpoints/             # Epoch checkpoints during training
│
├── modules/
│   ├── __init__.py
│   ├── data_ingestion.py        # Module 1
│   ├── directional_change.py    # Module 2
│   ├── lstm_detector.py         # Module 3
│   ├── strategy_selector.py     # Module 4
│   ├── risk_manager.py          # Module 5
│   ├── execution_engine.py      # Module 6
│   ├── logger.py                # Module 7
│   └── event_bus.py             # Event bus
│
├── strategies/
│   ├── base_strategy.py
│   ├── trend_following.py
│   ├── mean_reversion.py
│   └── breakout.py
│
├── training/
│   ├── train_lstm.py            # LSTM training script
│   ├── evaluate.py              # Evaluation metrics
│   ├── hyperparameter_search.py # Tuning
│   └── label_generator.py      # Rule-based regime labeling
│
├── backtesting/
│   ├── backtrader_runner.py
│   ├── walk_forward.py
│   └── performance.py
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_dc_analysis.ipynb     # DC feature exploration
│   ├── 03_lstm_training.ipynb   # Model training playground
│   └── 04_backtest_results.ipynb
│
├── logs/
│   ├── system_*.log             # Loguru log files
│   └── trading.db               # SQLite trade/regime database
│
├── tests/
│   ├── test_dc.py
│   ├── test_lstm.py
│   ├── test_risk.py
│   ├── test_execution.py
│   └── test_integration.py
│
├── main.py                      # Entry point for live demo trading
├── requirements.txt
└── README.md
```

---

## 18. Implementation Phases & Timeline

### Phase 1 — Data & Infrastructure (Weeks 1–3)
- [ ] Set up Python project structure
- [ ] Install all dependencies
- [ ] Connect MT5 terminal; pull and save EUR/USD, GBP/USD, USD/JPY historical data (2020–2024)
- [ ] Implement data cleaning pipeline
- [ ] Create train/val/test splits
- [ ] Set up event bus skeleton
- [ ] Set up SQLite logger

### Phase 2 — DC Algorithm & Feature Engineering (Weeks 4–6)
- [ ] Implement DC algorithm
- [ ] Validate DC algorithm visually on chart
- [ ] Extract DC features (duration, magnitude, ratio, etc.)
- [ ] Combine with TA-Lib indicators into feature matrix
- [ ] Generate rule-based regime labels
- [ ] Unit test DC module

### Phase 3 — LSTM Training & Evaluation (Weeks 7–11)
- [ ] Build RegimeDataset and DataLoader
- [ ] Define and test RegimeLSTM model
- [ ] Train on 2020–2022 data, validate on 2023
- [ ] Tune hyperparameters
- [ ] Evaluate: F1, accuracy, confusion matrix
- [ ] Add SHAP explainability
- [ ] Save model and scaler
- [ ] Unit test LSTM module

### Phase 4 — Strategy & Risk Modules (Weeks 12–14)
- [ ] Implement all 3 strategy classes
- [ ] Implement StrategySelector
- [ ] Implement RiskManager with volatility-based sizing
- [ ] Unit test each strategy
- [ ] Integration test: regime → signal → size

### Phase 5 — Backtesting (Weeks 15–16)
- [ ] Wire up Backtrader integration
- [ ] Run walk-forward backtest (2020–2024)
- [ ] Compute performance metrics
- [ ] Compare vs static baseline strategies
- [ ] Sensitivity analysis on key parameters

### Phase 6 — Execution & Demo Trading (Weeks 17–19)
- [ ] Implement ExecutionEngine with MT5 API
- [ ] Test order placement on demo account manually
- [ ] Wire up full async event loop
- [ ] Run continuous demo trading (minimum 3 months, or compressed simulation)
- [ ] Monitor logs; debug any issues
- [ ] Document all results

### Phase 7 — Analysis & Write-Up (Week 20)
- [ ] Compile all performance data
- [ ] Write conclusions and comparisons
- [ ] Prepare visualizations for report
- [ ] Complete project documentation

---

## 19. Key Formulas & Equations

### Sharpe Ratio
```
S = (R_p - R_f) / σ_p × √252

R_p = mean daily return of portfolio
R_f = risk-free rate (approximate as 0 for simplicity)
σ_p = std deviation of daily returns
252 = trading days in a year (Forex ≈ 261, use 252 for conservatism)
```

### Maximum Drawdown
```
MDD = max over t of [ (max_{s≤t} P_s - P_t) / max_{s≤t} P_s ]

P_t = portfolio value at time t
```

### Volatility-Based Position Size
```
Risk Amount  = Account Equity × Risk %
ATR Pips     = ATR(14) / Pip Size
Position Lots = Risk Amount / (ATR Pips × Pip Value)
```

### DC Magnitude
```
For a DOWN DC event: magnitude = (P_peak - P_reversal) / P_peak
For an UP DC event:  magnitude = (P_reversal - P_trough) / P_trough
```

### LSTM Cross-Entropy Loss
```
L = -Σ_c [ y_c × log(ŷ_c) ]

y_c  = true label (one-hot)
ŷ_c  = predicted probability for class c
```

### Kelly Criterion (for position sizing, optional)
```
f* = (p × b - q) / b

f* = fraction of capital to risk
p  = probability of winning trade
q  = 1 - p (probability of losing)
b  = average win / average loss ratio
```

---

## 20. Common Pitfalls & How to Avoid Them

| Pitfall | Consequence | Solution |
|---|---|---|
| Fitting scaler on full dataset | Look-ahead bias | Fit scaler on train split only |
| Using `close` of current bar for signal | Look-ahead | Use previous bar's close or use `shift(1)` |
| Not accounting for spread in backtest | Inflated returns | Add spread to every trade cost |
| Retraining LSTM on test data | Overfitting | Lock test set, never touch until final eval |
| Ignoring slippage | Unrealistic backtest | Add 1–3 pip slippage to all backtest executions |
| DC threshold too small | Noise amplification | Always visualize DC events on a chart before using |
| LSTM hidden size too large | Overfitting | Use dropout + weight decay; monitor val loss |
| Hard-coded SL/TP | Bad risk control in volatility | Always derive SL from ATR, not fixed pips |
| Ignoring class imbalance | LSTM biased to dominant regime | Use weighted loss or oversample minority classes |
| No drawdown halt | Account wipeout | Implement and test the `check_drawdown_halt` function |
| Running on live account | Financial loss | Use demo account only until proven over 3+ months |

---

## Quick Start Command Sequence

```bash
# 1. Clone / create your project
mkdir adaptive_forex && cd adaptive_forex

# 2. Set up virtual environment
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install MetaTrader5 pandas numpy TA-Lib torch scikit-learn backtrader plotly matplotlib shap loguru pyyaml pytest joblib

# 4. Download data from MT5 (run your data_ingestion.py)
python modules/data_ingestion.py --symbols EURUSD GBPUSD USDJPY --start 2020-01-01 --end 2024-12-31

# 5. Train LSTM
python training/train_lstm.py --symbol EURUSD --epochs 100

# 6. Run backtest
python backtesting/backtrader_runner.py --symbol EURUSD --start 2024-01-01 --end 2024-12-31

# 7. Start demo trading loop
python main.py --symbol EURUSD --mode demo
```

---

*This plan covers the complete implementation roadmap for the adaptive Forex framework described in the project document, adapted to use LSTM for regime classification and MT4/MT5 as the data and execution layer. Each module is independently testable. Follow the phases in order and do not skip the walk-forward validation step before moving to demo trading.*