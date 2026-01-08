"""
Project Knowledge Base for RALPH Agent Ensemble

This module contains specialized knowledge about the Polymarket AI Bot project
that agents need for context-aware task execution.

The agents are specifically designed to work on the E:\Polymarket AI Bot project.
"""

# =============================================================================
# PROJECT OVERVIEW
# =============================================================================

PROJECT_OVERVIEW = """
## Polymarket AI Bot - Project Overview

A hybrid trading system combining:
1. **Arbitrage Detection**: Buy YES+NO when sum < $0.995 (risk-free profit)
2. **AI Directional Predictions**: Multi-timeframe ML ensemble for market predictions

### Core Systems
- **Polymarket Trading**: Paper trading on prediction markets
- **Spot Trading**: Live trading on Coinbase (BTC, ETH, SOL, XRP)
- **Dashboard**: Next.js real-time monitoring (port 3000)
- **API Backend**: FastAPI REST + WebSocket (port 8000)
- **Market Scanner**: Priority-queue based (450+ markets/cycle)

### Key Statistics
- Historical Data: 5,364+ BTC daily rows (2010-2025)
- Training Samples: 736+ candlestick samples
- Active Markets: 450+ markets scanned per cycle
- ML Ensemble: 8 models with auto-calibration
"""

# =============================================================================
# API ENDPOINTS & INTEGRATION
# =============================================================================

POLYMARKET_API = """
## Polymarket API Integration

### 1. PNL Subgraph (Goldsky) - Resolution Labels
```
Endpoint: https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn
Type: GraphQL
Purpose: Fetch market resolution outcomes
```

Resolution Mapping:
- `payoutNumerators: [1, 0]` → YES won (label = 1)
- `payoutNumerators: [0, 1]` → NO won (label = 0)

### 2. Gamma API - Market Metadata
```
Base URL: https://gamma-api.polymarket.com
Endpoints:
  - /markets (list with filters)
  - /markets/{id} (single market)
  - /tags (categories)
```

Key Query Parameters:
- `active`, `closed`, `limit`, `offset`
- `order` (e.g., `endDate desc`)
- `end_date_min`, `end_date_max` (ISO format)
- `uma_resolution_status` (`resolved`, `pending`)

### 3. CLOB API - Real-Time Prices
```
Base URL: https://clob.polymarket.com
Endpoints:
  - /markets (with order book)
  - /prices
  - /book/{tokenId}
```

Arbitrage Detection:
```python
if yes_price + no_price < 0.995:
    # Guaranteed 0.5%+ profit
    buy_both_sides_equally()
```
"""

BINANCE_PROXY_INFO = """
## Binance API & Proxy Configuration

### Problem
Binance.com is **blocked in the US**. The bot needs alternative approaches.

### Solutions
1. **Proxy Service**: webshare.io (configured in config.yaml)
2. **Binance.US**: For US users (different API endpoint)
3. **Fallback to Coinbase**: Primary for spot trading

### Proxy Manager Pattern
```python
class ProxyManager:
    def __init__(self, config):
        self.proxies = []
        # Primary + backup proxies

    def rotate(self) -> bool:
        \"\"\"Rotate to next proxy on failure\"\"\"
        self.current_index = (self.current_index + 1) % len(self.proxies)
```

### Local Development Note
- Binance CVD/Whale features disabled locally (blocked API)
- Use Coinbase order book imbalance only
- Full functionality on VPS with proxy
"""

# =============================================================================
# KEY FILES & STRUCTURE
# =============================================================================

KEY_FILES = """
## Key Files to Know

| File | Purpose | Notes |
|------|---------|-------|
| `run_bot.py` | Unified launcher | Main entry point |
| `src/inference.py` | Trading engine | Largest file (519KB) |
| `src/ensemble.py` | ML ensemble | 8 models + calibration |
| `src/features.py` | Feature engineering | MTF features |
| `src/train.py` | Training pipeline | Model training |
| `config.yaml` | Configuration | All settings |
| `README.md` | Documentation | Getting started |
| `API_ENDPOINTS.md` | API docs | All endpoints |

### Directory Structure
```
E:\\Polymarket AI Bot\\
├── config.yaml              # Main configuration
├── run_bot.py               # Entry point
├── src/
│   ├── inference.py         # Trading logic
│   ├── ensemble.py          # ML models
│   ├── features.py          # Feature engineering
│   ├── train.py             # Model training
│   ├── dynamic_scanner/     # Market priority queue
│   ├── live_trading/        # Coinbase integration
│   └── spot_trading/        # Spot trading module
├── models/                  # Trained models
├── data/
│   ├── raw/                 # Historical parquet files
│   └── paper_trading/       # Captured predictions
├── logs/                    # Runtime logs
├── vps/                     # VPS deployment scripts
└── tests/                   # Test suite
```
"""

# =============================================================================
# ML ENSEMBLE DETAILS
# =============================================================================

ML_ENSEMBLE_INFO = """
## ML Ensemble Architecture

### 8-Model Ensemble
| Model | Type | Role |
|-------|------|------|
| XGBoost (x3) | Tree | Diversity via different seeds |
| LightGBM | Tree | Fast gradient boosting |
| CatBoost | Tree | Category handling |
| HistGradientBoosting | Tree | Sklearn native |
| MiniRocket + Ridge | Time-series | Fast feature extraction |
| Logistic Regression | Linear | Baseline/calibration |

### 3-Stage Calibration
1. **Isotonic Regression**: Non-parametric calibration
2. **Static Offset**: Fixed bias correction
3. **Emergency Recalibration**: Runtime bias detection

### Multi-Timeframe Features (46-53 total)
- 1h: Primary features (RSI, MACD, BB, ATR, ADX, EMAs)
- 4h: With `_4h` suffix
- 1d: With `_1d` suffix

### Feature List
```python
['close', 'open', 'high', 'low', 'volume',
 'rsi', 'macd', 'macd_signal', 'macd_hist',
 'bb_upper', 'bb_lower', 'bb_middle',
 'atr', 'adx', 'stoch_k', 'stoch_d',
 'ema_9', 'ema_21', 'ema_50', 'ema_200']
```
"""

# =============================================================================
# TRADING CONCEPTS
# =============================================================================

# =============================================================================
# DATA SOURCES - DETAILED
# =============================================================================

MYSQL_DATABASE = """
## MySQL Database - Candle Storage

### Connection Details
```yaml
database:
  host: "70.180.16.4"
  port: 3306
  user: "cryptouser2"
  password: "#f5EjqW3rS1nK52gQx"
  database: "crypto_data"
```

### Table: crypto_prices (Main Candle Storage)
```sql
CREATE TABLE crypto_prices (
  id INT PRIMARY KEY AUTO_INCREMENT,
  symbol VARCHAR(20),           -- BTC, ETH, SOL, XRP
  timestamp DATETIME,           -- Candle timestamp
  open DECIMAL(20,8),
  high DECIMAL(20,8),
  low DECIMAL(20,8),
  close DECIMAL(20,8),
  volume DECIMAL(30,8),
  -- Pre-computed indicators (optional)
  rsi DECIMAL(10,4),
  macd DECIMAL(20,8),
  macd_signal DECIMAL(20,8),
  ema_12 DECIMAL(20,8),
  ema_26 DECIMAL(20,8)
);
```

### Data Coverage
| Symbol | Rows | Date Range | Timeframes |
|--------|------|------------|------------|
| BTC | 211,000+ | 2010-2025 | 1h, 4h, 1d |
| ETH | 50,000+ | 2017-2025 | 1h, 4h, 1d |
| SOL | 20,000+ | 2020-2025 | 1h, 4h, 1d |
| XRP | 80,000+ | 2013-2025 | 1h, 4h, 1d |

### Query Patterns
```python
# Fetch candles for backtesting
SELECT * FROM crypto_prices
WHERE symbol = 'BTC'
  AND timestamp BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY timestamp ASC;

# Get latest candle
SELECT * FROM crypto_prices
WHERE symbol = 'BTC'
ORDER BY timestamp DESC
LIMIT 1;

# Aggregate to different timeframes
SELECT
  DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as hour,
  FIRST_VALUE(open) as open,
  MAX(high) as high,
  MIN(low) as low,
  LAST_VALUE(close) as close,
  SUM(volume) as volume
FROM crypto_prices
WHERE symbol = 'BTC'
GROUP BY hour;
```

### Data Agent Tasks
- Ingest new candles from exchanges
- Backfill missing historical data
- Compute and store derived indicators
- Monitor data quality and freshness
"""

TAAPI_INTEGRATION = """
## TAAPI.io - Technical Indicator API

### API Overview
- **Website**: https://taapi.io/indicators/
- **Base URL**: https://api.taapi.io
- **Auth**: API key in query parameter or header
- **Rate Limits**: Depends on plan (free: 1 req/15s, paid: higher)

### Available Indicators (100+)

#### Momentum Indicators
- `rsi` - Relative Strength Index
- `stochrsi` - Stochastic RSI
- `willr` - Williams %R
- `mfi` - Money Flow Index
- `cci` - Commodity Channel Index
- `ao` - Awesome Oscillator
- `mom` - Momentum
- `roc` - Rate of Change
- `cmo` - Chande Momentum Oscillator

#### Trend Indicators
- `macd` - Moving Average Convergence Divergence
- `adx` - Average Directional Index
- `dmi` - Directional Movement Index
- `aroon` - Aroon Indicator
- `psar` - Parabolic SAR
- `supertrend` - Super Trend
- `ichimoku` - Ichimoku Cloud
- `vortex` - Vortex Indicator

#### Volatility Indicators
- `atr` - Average True Range
- `bbands` - Bollinger Bands
- `keltner` - Keltner Channel
- `donchian` - Donchian Channel
- `natr` - Normalized ATR

#### Moving Averages
- `sma` - Simple Moving Average
- `ema` - Exponential Moving Average
- `wma` - Weighted Moving Average
- `dema` - Double EMA
- `tema` - Triple EMA
- `kama` - Kaufman Adaptive MA
- `vwma` - Volume Weighted MA

#### Volume Indicators
- `obv` - On Balance Volume
- `vwap` - Volume Weighted Average Price
- `ad` - Accumulation/Distribution
- `cmf` - Chaikin Money Flow
- `fi` - Force Index
- `efi` - Elder Force Index

#### Pattern Recognition
- `cdlengulfing` - Engulfing Pattern
- `cdldoji` - Doji Pattern
- `cdlhammer` - Hammer Pattern
- `cdlmorningstar` - Morning Star
- 60+ candlestick patterns

### API Usage Patterns

#### Single Indicator
```python
import requests

response = requests.get(
    "https://api.taapi.io/rsi",
    params={
        "secret": TAAPI_KEY,
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "interval": "1h"
    }
)
data = response.json()  # {"value": 65.42}
```

#### Bulk Request (Multiple Indicators)
```python
# Get multiple indicators in one request
response = requests.post(
    "https://api.taapi.io/bulk",
    json={
        "secret": TAAPI_KEY,
        "construct": {
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "interval": "1h",
            "indicators": [
                {"indicator": "rsi"},
                {"indicator": "macd"},
                {"indicator": "bbands"},
                {"indicator": "atr"},
                {"indicator": "adx"}
            ]
        }
    }
)
```

### Multi-Timeframe Strategy
```python
# Fetch indicators across timeframes
timeframes = ["1h", "4h", "1d"]
indicators = ["rsi", "macd", "adx", "atr"]

for tf in timeframes:
    for ind in indicators:
        value = fetch_taapi(ind, timeframe=tf)
        features[f"{ind}_{tf}"] = value
```

### Currently Used in Project (46-53 features)
```python
PRIMARY_INDICATORS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_lower', 'bb_middle',
    'atr', 'adx', 'stoch_k', 'stoch_d',
    'ema_9', 'ema_21', 'ema_50', 'ema_200'
]
```
"""

WEBSOCKET_DATA = """
## Real-Time WebSocket Data

### Coinbase WebSocket (Primary for US)
```python
# WebSocket URL
wss://advanced-trade-ws.coinbase.com

# Subscription message
{
    "type": "subscribe",
    "product_ids": ["BTC-USD", "ETH-USD"],
    "channel": "level2",  # Order book
    "jwt": "<your_jwt_token>"
}

# Channels available:
# - level2: Full order book updates
# - ticker: Price ticker updates
# - matches: Trade executions
# - heartbeats: Connection health

# Example order book message
{
    "channel": "l2_data",
    "events": [{
        "type": "update",
        "product_id": "BTC-USD",
        "updates": [
            {"side": "bid", "price": "42000.00", "qty": "1.5"},
            {"side": "ask", "price": "42001.00", "qty": "0.8"}
        ]
    }]
}
```

### Binance WebSocket (Via Proxy for US)
```python
# WebSocket URL (requires proxy in US)
wss://stream.binance.com:9443/ws

# Subscription for multiple streams
{
    "method": "SUBSCRIBE",
    "params": [
        "btcusdt@depth20@100ms",    # Order book
        "btcusdt@aggTrade",          # Trades
        "btcusdt@kline_1m",          # Candles
        "btcusdt@ticker"             # 24h stats
    ],
    "id": 1
}

# Order book stream
{
    "lastUpdateId": 160,
    "bids": [["42000.00", "1.50"]],
    "asks": [["42001.00", "0.80"]]
}

# Aggregate trade stream
{
    "e": "aggTrade",
    "s": "BTCUSDT",
    "p": "42000.50",     # Price
    "q": "0.25",         # Quantity
    "m": false           # Is buyer maker
}
```

### Order Book Imbalance Calculation
```python
async def calculate_imbalance(order_book):
    bids = order_book['bids'][:50]  # Top 50 levels
    asks = order_book['asks'][:50]

    buy_volume = sum(float(b[1]) * float(b[0]) for b in bids)
    sell_volume = sum(float(a[1]) * float(a[0]) for a in asks)

    # Range: -1 (all sell pressure) to +1 (all buy pressure)
    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    return imbalance
```

### CVD (Cumulative Volume Delta) from Binance
```python
async def calculate_cvd(trades, lookback_minutes=5):
    \"\"\"Track buy vs sell pressure over time\"\"\"
    cutoff = datetime.now() - timedelta(minutes=lookback_minutes)

    buy_volume = 0
    sell_volume = 0

    for trade in trades:
        if trade['timestamp'] > cutoff:
            if trade['is_buyer_maker']:
                sell_volume += trade['qty'] * trade['price']
            else:
                buy_volume += trade['qty'] * trade['price']

    # Normalized: -1 to +1
    total = buy_volume + sell_volume
    cvd = (buy_volume - sell_volume) / total if total > 0 else 0
    return cvd
```

### Whale Detection
```python
WHALE_THRESHOLD_USD = 500_000  # $500k+ trades

async def detect_whales(trades, lookback_hours=1):
    \"\"\"Identify large trades that may impact price\"\"\"
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    whale_trades = []

    for trade in trades:
        if trade['timestamp'] > cutoff:
            value_usd = trade['qty'] * trade['price']
            if value_usd >= WHALE_THRESHOLD_USD:
                whale_trades.append({
                    'side': 'buy' if not trade['is_buyer_maker'] else 'sell',
                    'value': value_usd,
                    'price': trade['price'],
                    'timestamp': trade['timestamp']
                })

    return whale_trades
```

### Data Agent Real-Time Tasks
- Subscribe to price feeds for active markets
- Calculate order book imbalance every tick
- Track CVD for momentum signals
- Detect and alert on whale trades
- Feed real-time data to prediction models
"""

EXPERIMENTATION_FRAMEWORK = """
## Experimentation & Parameter Optimization

### Philosophy
The agents should be able to run **hundreds or thousands** of parameter
combinations to find optimal configurations. Every hypothesis should be
tested with data, not assumptions.

### Grid Search Capabilities

#### Basic Grid Search
```python
# Example: Learning rate sweep
param_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, 10]
}
# Total combinations: 5 × 4 × 4 = 80 experiments

# Execute with cross-validation
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_log_loss',
    n_jobs=-1  # Parallel execution
)
```

#### Bayesian Optimization (Efficient Search)
```python
from optuna import create_study

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_est', 50, 2000),
        'max_depth': trial.suggest_int('depth', 2, 15),
        'min_child_weight': trial.suggest_int('mcw', 1, 20)
    }

    model = XGBClassifier(**params)
    # Cross-validate and return score
    return cross_val_score(model, X, y, cv=5).mean()

study = create_study(direction='maximize')
study.optimize(objective, n_trials=500)  # 500 intelligent trials
```

### Multi-Parameter Sweep Example
```python
# Comprehensive strategy tuning
TUNING_GRID = {
    # Entry thresholds
    'edge_threshold': [0.05, 0.07, 0.10, 0.12, 0.15],
    'probability_threshold': [0.52, 0.55, 0.58, 0.60],

    # Position sizing
    'kelly_fraction': [0.1, 0.15, 0.2, 0.25, 0.33, 0.5],
    'max_position_pct': [0.05, 0.10, 0.15, 0.20],

    # Risk management
    'stop_loss_pct': [0.02, 0.03, 0.05, 0.07],
    'take_profit_pct': [0.05, 0.10, 0.15, 0.20],
    'max_hold_hours': [4, 8, 12, 24, 48],

    # Technical indicators
    'rsi_period': [7, 14, 21],
    'macd_fast': [8, 12, 16],
    'macd_slow': [21, 26, 30],
    'bb_period': [10, 20, 30],
    'bb_std': [1.5, 2.0, 2.5, 3.0]
}

# This generates 5×4×6×4×4×4×5×3×3×3×3×4 = 31,104,000 combinations!
# Use Bayesian optimization to sample intelligently
```

### A/B Testing Framework
```python
class ExperimentTracker:
    \"\"\"Track and compare strategy variants\"\"\"

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.variants = {}
        self.results = []

    def register_variant(self, name: str, params: dict):
        \"\"\"Register a strategy variant\"\"\"
        self.variants[name] = {
            'params': params,
            'trades': [],
            'metrics': {}
        }

    def record_trade(self, variant: str, trade: dict):
        \"\"\"Record a trade for a variant\"\"\"
        self.variants[variant]['trades'].append(trade)

    def calculate_metrics(self, variant: str) -> dict:
        \"\"\"Calculate performance metrics for variant\"\"\"
        trades = self.variants[variant]['trades']
        returns = [t['pnl_pct'] for t in trades]

        return {
            'total_return': sum(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'max_drawdown': calculate_max_drawdown(returns),
            'trade_count': len(trades)
        }

    def compare_variants(self) -> pd.DataFrame:
        \"\"\"Compare all variants side by side\"\"\"
        comparison = []
        for name, data in self.variants.items():
            metrics = self.calculate_metrics(name)
            metrics['variant'] = name
            comparison.append(metrics)
        return pd.DataFrame(comparison)
```

### Walk-Forward Optimization
```python
def walk_forward_optimization(
    data: pd.DataFrame,
    param_grid: dict,
    train_size: int = 252,  # 1 year of daily data
    test_size: int = 63     # 3 months out-of-sample
) -> list:
    \"\"\"
    Optimize parameters on rolling windows to avoid overfitting.
    Tests on unseen data to validate robustness.
    \"\"\"
    results = []

    for start in range(0, len(data) - train_size - test_size, test_size):
        # Training window
        train_data = data[start:start + train_size]

        # Test window (out-of-sample)
        test_data = data[start + train_size:start + train_size + test_size]

        # Optimize on training data
        best_params = grid_search(train_data, param_grid)

        # Validate on test data
        test_metrics = backtest(test_data, best_params)

        results.append({
            'window_start': data.index[start],
            'window_end': data.index[start + train_size + test_size],
            'best_params': best_params,
            'in_sample_sharpe': best_params['sharpe'],
            'out_sample_sharpe': test_metrics['sharpe']
        })

    return results
```

### Experiment Commands for Agents
```
# Tuning Agent commands
!sweep learning_rate         # Sweep single parameter
!grid model_params           # Full grid search
!bayesian --trials 500       # Bayesian optimization
!walkforward --windows 8     # Walk-forward validation

# Backtest Agent commands
!experiment <name>           # Start new experiment
!variant <name> <params>     # Add variant to experiment
!compare <exp_name>          # Compare all variants
!significance <var1> <var2>  # Statistical significance test
```

### Statistical Significance Testing
```python
from scipy import stats

def test_significance(returns_a: list, returns_b: list) -> dict:
    \"\"\"Test if strategy B is significantly better than A\"\"\"

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(returns_b, returns_a)

    # Effect size (Cohen's d)
    diff = np.array(returns_b) - np.array(returns_a)
    cohens_d = np.mean(diff) / np.std(diff)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d,
        'interpretation': (
            'Large effect' if abs(cohens_d) > 0.8 else
            'Medium effect' if abs(cohens_d) > 0.5 else
            'Small effect'
        )
    }
```
"""

TRADING_CONCEPTS = """
## Critical Trading Concepts

### 1. Arbitrage (Risk-Free)
- **Trigger**: YES_price + NO_price < $0.995
- **Action**: Buy both YES and NO equally
- **Profit**: ~0.5% guaranteed (before fees)

### 2. Directional Trading (AI-Powered)
- **Entry**: probability > 0.5 + threshold + edge > 7%
- **Edge**: probability - 0.5 (adjusted for sentiment)
- **Sizing**: Tiered Kelly with liquidity caps

### 3. Position Sizing Tiers
| Tier | Edge Range | Kelly Fraction |
|------|------------|----------------|
| A | > 15% | Full Kelly |
| B | 10-15% | 0.5 Kelly |
| C | 7-10% | 0.25 Kelly |

### 4. Risk Thresholds
- Max Drawdown: 25%
- Max Position Size: 10% of portfolio
- Min Sharpe Ratio: 0.8
- Max Leverage: 2.0x
- VaR (95%): 5%
"""

# =============================================================================
# CONFIGURATION PATTERNS
# =============================================================================

CONFIG_PATTERNS = """
## Configuration Patterns (config.yaml)

### Key Sections
1. **API**: TAAPI, Telegram, Polymarket, Coinbase credentials
2. **Database**: MySQL connection (historical data)
3. **MTF**: Multi-timeframe settings (1h, 4h, 1d)
4. **Indicators**: Technical indicator parameters
5. **Position Sizing**: Tiered Kelly with liquidity awareness
6. **Risk Management**: Drawdown halts, exposure limits
7. **Hybrid**: Arbitrage + directional settings
8. **Oracle Fusion**: Multi-source signal weighting
9. **Spot Trading**: Coinbase live trading config

### Oracle Fusion Weights
| Source | Weight | Purpose |
|--------|--------|---------|
| Polymarket Model | 30% | AI predictions |
| Binance Order Book | 35% | Global liquidity |
| Coinbase Order Book | 20% | US market depth |
| Sentiment (F&G) | 15% | Fear & Greed |

### Paper vs Live Mode
```yaml
spot_trading:
  mode: "paper"  # "paper" for testing, "live" for real
  exchanges:
    primary: "coinbase"
    fallback: "binance"
```
"""

# =============================================================================
# DEBUGGING CHECKLIST
# =============================================================================

DEBUGGING_CHECKLIST = """
## Debugging Checklist

### 1. Check Model Calibration
```python
from src.ensemble import EnsembleModel
model = EnsembleModel.load('models/mtf_latest')
print(f"Offset: {model.calibration_offset}")
print(f"Bullish: {(model.predict_proba(X) > 0.5).mean():.1%}")
```

### 2. Check Portfolio State
```python
with open('logs/portfolio_state.json') as f:
    portfolio = json.load(f)
    print(f"Cash: ${portfolio['cash']:.2f}")
```

### 3. Check Risk Halt Status
```python
with open('logs/risk_halt.json') as f:
    risk = json.load(f)
    print(f"Halted: {risk['is_halted']}")
```

### 4. Check Market Scanner
```python
queue.refresh_hotlist()
top_5 = queue.peek_top(5)
```

### Common Issues
1. **Binance blocked**: Use proxy or Coinbase fallback
2. **Model all bullish**: Check calibration_offset
3. **No arbitrage found**: Market efficient (normal)
4. **Position too large**: Liquidity cap at 5%
"""

# =============================================================================
# VPS DEPLOYMENT
# =============================================================================

VPS_INFO = """
## VPS Deployment

### Connection Details
- Host: 46.252.192.140
- User: flashwing
- SSH Key: ~/.ssh/id_rsa
- Project: /home/flashwing/polymarket-ai-bot

### Service Commands
```bash
sudo bash vps/polymarket.sh start    # Start all
sudo bash vps/polymarket.sh stop     # Stop all
sudo bash vps/polymarket.sh restart  # Restart all
sudo bash vps/polymarket.sh status   # Show status
sudo bash vps/polymarket.sh update   # Pull & restart
```

### View Logs
```bash
sudo journalctl -u polymarket-bot -f
sudo journalctl -u polymarket-api -f
```

### Services
- polymarket-bot.service: Main trading bot
- polymarket-api.service: FastAPI backend
- polymarket-dashboard.service: Next.js dashboard
"""

# =============================================================================
# AGENT-SPECIFIC CONTEXT
# =============================================================================

AGENT_CONTEXT = {
    "tuning": """
## Tuning Agent Context

### Your Superpower: Comprehensive Parameter Optimization

You can test **hundreds or thousands** of parameter combinations to find
optimal configurations. Every hypothesis should be validated with data.

### Optimization Methods

#### 1. Grid Search (Exhaustive)
```python
param_grid = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, 10]
}
# 5 × 4 × 4 = 80 combinations
```

#### 2. Bayesian Optimization (Intelligent)
```python
from optuna import create_study
study = create_study(direction='maximize')
study.optimize(objective, n_trials=500)
# 500 trials, learns from previous results
```

#### 3. Walk-Forward Validation
```python
# Optimize on rolling windows to avoid overfitting
for window in rolling_windows(data):
    train, test = split(window)
    best_params = optimize(train)
    validate(test, best_params)  # Out-of-sample test
```

### Parameters to Optimize

#### Model Hyperparameters
| Model | Key Parameters |
|-------|----------------|
| XGBoost | learning_rate, n_estimators, max_depth, min_child_weight |
| LightGBM | learning_rate, num_leaves, feature_fraction |
| CatBoost | learning_rate, depth, l2_leaf_reg |

#### Trading Parameters
```yaml
# Entry/Exit Thresholds
edge_threshold: [0.05, 0.07, 0.10, 0.12, 0.15]
probability_threshold: [0.52, 0.55, 0.58, 0.60]

# Position Sizing
kelly_fraction: [0.1, 0.15, 0.2, 0.25, 0.33, 0.5]
max_position_pct: [0.05, 0.10, 0.15, 0.20]

# Risk Management
stop_loss_pct: [0.02, 0.03, 0.05, 0.07]
take_profit_pct: [0.05, 0.10, 0.15, 0.20]
max_hold_hours: [4, 8, 12, 24, 48]

# Technical Indicators
rsi_period: [7, 14, 21]
macd_fast: [8, 12, 16]
macd_slow: [21, 26, 30]
bb_period: [10, 20, 30]
bb_std: [1.5, 2.0, 2.5, 3.0]
```

### Calibration System (3-Stage)
1. **Isotonic Regression**: Non-parametric calibration
2. **Static Offset**: Fixed bias correction
3. **Emergency Recalibration**: Runtime bias detection

### Key Files
- `config.yaml`: position_sizing section
- `src/ensemble.py`: calibration_offset, model configs
- `src/train.py`: training hyperparameters

### Statistical Validation
Always test significance before accepting improvements:
```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(new_returns, baseline_returns)
if p_value < 0.05:
    # Statistically significant improvement
```

### Tuning Agent Responsibilities
1. **Sweep**: Test parameter ranges systematically
2. **Optimize**: Find best configurations via search
3. **Validate**: Ensure improvements are statistically significant
4. **Document**: Record all experiments and results
5. **Hand off**: Send optimized params to Backtest Agent
""",

    "backtest": """
## Backtest Agent Context

### Your Focus Areas
- Historical data in `data/raw/` (parquet files)
- Backtesting logic in `src/inference.py`
- Performance metrics calculation
- Walk-forward validation

### Key Files
- `src/inference.py`: Trading simulation logic
- `data/raw/{SYMBOL}/`: Historical candles
- `logs/`: Performance logs

### Metrics to Calculate
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown, Average Drawdown
- Win Rate, Profit Factor
- VaR (95%), Expected Shortfall
""",

    "risk": """
## Risk Agent Context

### Your Focus Areas
- Risk thresholds in `config.yaml`
- Drawdown monitoring in `logs/`
- Position limits enforcement
- Risk halt system

### Key Files
- `config.yaml`: risk_management section
- `logs/risk_halt.json`: Current halt status
- `logs/portfolio_state.json`: Positions

### Risk Thresholds
```yaml
risk_management:
  max_drawdown: 0.25
  max_daily_loss: 0.05
  halt_on_loss_streak: 5
  max_position_pct: 0.10
```
""",

    "strategy": """
## Strategy Agent Context (Mission Lead)

### Your Focus Areas
- Trading logic in `src/inference.py`
- Signal generation conditions
- Feature combinations
- Strategy iteration

### Key Files
- `src/inference.py`: Entry/exit logic
- `src/features.py`: Feature engineering
- `config.yaml`: Strategy parameters

### Polymarket-Specific
- Binary outcome markets (YES/NO)
- Resolution timing awareness
- Liquidity constraints
- Arbitrage detection (YES+NO < 0.995)
""",

    "data": """
## Data Agent Context

### Your Superpower: Data Pivoting & Experimentation

You have access to an incredibly rich data ecosystem that enables rapid
experimentation. Your job is to BUILD, TEST, and ITERATE on data-driven
hypotheses.

### Available Data Sources

#### 1. MySQL Database (Historical Candles)
```yaml
host: 70.180.16.4:3306
database: crypto_data
tables: crypto_prices (BTC: 211k rows, ETH: 50k, SOL: 20k, XRP: 80k)
```
- Query any timeframe (1h, 4h, 1d)
- Pre-computed indicators available
- 15+ years of BTC data

#### 2. TAAPI.io (100+ Technical Indicators)
Website: https://taapi.io/indicators/
- Momentum: RSI, Stochastic RSI, Williams %R, MFI, CCI, AO
- Trend: MACD, ADX, DMI, Aroon, Parabolic SAR, Supertrend
- Volatility: ATR, Bollinger Bands, Keltner, Donchian
- Volume: OBV, VWAP, A/D, CMF, Force Index
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA, KAMA, VWMA
- Patterns: 60+ candlestick patterns
- Bulk API for multiple indicators in one call

#### 3. Coinbase WebSocket (Real-Time US Data)
```
wss://advanced-trade-ws.coinbase.com
Channels: level2, ticker, matches, heartbeats
```
- Order book depth (bid/ask imbalance)
- Real-time trade executions
- Price tickers

#### 4. Binance WebSocket (Global Data via Proxy)
```
wss://stream.binance.com:9443/ws (proxy required in US)
Streams: depth, aggTrade, kline, ticker
```
- CVD (Cumulative Volume Delta)
- Whale trade detection ($500k+ trades)
- Global liquidity view

### Experimentation Capabilities

#### Build New Features
```python
# Example: Create a custom momentum indicator
def custom_momentum(df, short=5, long=20):
    short_ma = df['close'].rolling(short).mean()
    long_ma = df['close'].rolling(long).mean()
    return (short_ma - long_ma) / long_ma
```

#### Test Hypotheses
```python
# Example: Does RSI divergence predict reversals?
divergence_signals = detect_rsi_divergence(df)
forward_returns = df['close'].pct_change(periods=24).shift(-24)
correlation = divergence_signals.corr(forward_returns)
# Report findings to Strategy Agent
```

#### A/B Test Data Preprocessing
```python
# Compare different normalization methods
variants = {
    'z_score': (df - df.mean()) / df.std(),
    'min_max': (df - df.min()) / (df.max() - df.min()),
    'robust': (df - df.median()) / (df.quantile(0.75) - df.quantile(0.25))
}
# Test which produces better model predictions
```

### Key Files
- `src/features.py`: Feature extraction logic
- `src/dynamic_scanner/`: Market scanning queue
- `data/raw/`: Historical parquet files
- `config.yaml`: API credentials

### Data Agent Responsibilities
1. **Ingest**: Pull data from all sources
2. **Clean**: Handle missing values, outliers
3. **Engineer**: Create derived features
4. **Validate**: Monitor data quality
5. **Experiment**: Test new indicators and preprocessing
6. **Report**: Share findings with other agents
"""
}


def get_full_context() -> str:
    """Get the complete project knowledge base as a single string."""
    return "\n\n".join([
        PROJECT_OVERVIEW,
        POLYMARKET_API,
        BINANCE_PROXY_INFO,
        KEY_FILES,
        ML_ENSEMBLE_INFO,
        MYSQL_DATABASE,
        TAAPI_INTEGRATION,
        WEBSOCKET_DATA,
        EXPERIMENTATION_FRAMEWORK,
        TRADING_CONCEPTS,
        CONFIG_PATTERNS,
        DEBUGGING_CHECKLIST,
        VPS_INFO
    ])


def get_agent_context(agent_type: str) -> str:
    """Get agent-specific context."""
    return AGENT_CONTEXT.get(agent_type, "")
