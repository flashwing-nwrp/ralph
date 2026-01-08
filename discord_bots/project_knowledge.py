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

### Your Focus Areas in Polymarket AI Bot
- Model hyperparameters in `src/ensemble.py`
- Calibration offsets for prediction bias
- Feature importance analysis
- Position sizing Kelly fractions in `config.yaml`

### Key Files
- `config.yaml`: position_sizing section
- `src/ensemble.py`: calibration_offset, model configs
- `src/train.py`: training hyperparameters

### Important Parameters
```yaml
position_sizing:
  kelly_fraction: 0.25
  max_position_pct: 0.10
  min_edge_threshold: 0.07
```
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

### Your Focus Areas
- API integrations in `src/`
- Data pipelines and preprocessing
- Feature engineering
- Data quality monitoring

### Key Files
- `src/features.py`: Feature extraction
- `src/dynamic_scanner/`: Market scanning
- `data/raw/`: Historical data
- API integration files

### Data Sources
1. Polymarket Gamma API (market metadata)
2. Polymarket CLOB API (real-time prices)
3. Goldsky GraphQL (resolution outcomes)
4. TAAPI.io (technical indicators)
5. Coinbase Advanced Trade (order book)
6. MySQL database (historical data)
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
        TRADING_CONCEPTS,
        CONFIG_PATTERNS,
        DEBUGGING_CHECKLIST,
        VPS_INFO
    ])


def get_agent_context(agent_type: str) -> str:
    """Get agent-specific context."""
    return AGENT_CONTEXT.get(agent_type, "")
