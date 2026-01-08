# RALPH Discord Agent Ensemble

Discord bot system for the RALPH AI trading bot project. Five specialized agents communicate in a private Discord server to collaborate on bot development, tuning, backtesting, and risk management.

## Agents

| Agent | Role | Primary Channel |
|-------|------|-----------------|
| **Tuning Agent** | Parameter optimization, hyperparameter tuning | `#tuning` |
| **Backtest Agent** | Historical simulation, performance metrics | `#backtesting` |
| **Risk Agent** | Safety auditing, risk limits enforcement | `#risk` |
| **Strategy Agent** | Trading logic, feature engineering | `#strategy` |
| **Data Agent** | Data preprocessing, cleaning, feature extraction | `#data` |

## Quick Start

### 1. Create Discord Applications

Create 5 bot applications at https://discord.com/developers/applications

See [SETUP.md](SETUP.md) for exact names/descriptions to paste.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your 5 bot tokens + server ID
```

### 3. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 4. Run

```bash
# All agents concurrently
python run_all.py

# Single agent (for testing)
python run_single.py tuning
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Why multiple bots vs single bot
- [SETUP.md](SETUP.md) - Complete setup guide with Discord Developer Portal instructions
- [COMMUNICATION.md](COMMUNICATION.md) - Channel structure and example conversations

## Commands

All agents respond to:
- `!ping` - Check latency
- `!status` - Agent status
- `!help` - Available commands

Agent-specific commands documented in [COMMUNICATION.md](COMMUNICATION.md).

## Security

- Tokens stored in `.env` (gitignored)
- Minimal Discord intents (message content only)
- Minimal permissions (send messages, read history, threads)
- Private server only

## Requirements

- Python 3.8+
- discord.py 2.3+
- python-dotenv
