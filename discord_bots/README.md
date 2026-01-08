# RALPH Discord Agent Ensemble

**Autonomous multi-agent system** for AI trading bot development. Five specialized agents communicate via Discord and execute tasks using Claude Code CLI - they can work autonomously, handing off tasks to each other without user intervention.

## Key Features

- **Autonomous Execution**: Agents use Claude Code to perform real work on your codebase
- **Automatic Handoffs**: Agents trigger each other based on workflow rules
- **Focused Context**: Each agent has a narrow persona for accurate execution
- **Discord Transparency**: All communication visible in Discord channels
- **Ralph Pattern**: Memory persists via git history and progress files

## Agents

| Agent | Role | Primary Channel |
|-------|------|-----------------|
| **Tuning Agent** | Parameter optimization, hyperparameter tuning | `#tuning` |
| **Backtest Agent** | Historical simulation, performance metrics | `#backtesting` |
| **Risk Agent** | Safety auditing, risk limits enforcement (VETO power) | `#risk` |
| **Strategy Agent** | Trading logic, feature engineering | `#strategy` |
| **Data Agent** | Data preprocessing, cleaning, feature extraction | `#data` |

## Quick Start

### 1. Create Discord Applications

Create 5 bot applications at https://discord.com/developers/applications

See [SETUP.md](SETUP.md) for exact names/descriptions to paste.

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with:
- Your 5 bot tokens
- Discord server ID
- **Project directory** (where Claude Code will work)

```env
RALPH_PROJECT_DIR=E:\Polymarket AI Bot
CLAUDE_CMD=claude
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

### Execution Commands (NEW)

| Command | Description |
|---------|-------------|
| `!do <task>` | Execute a task using Claude Code |
| `!handoff <agent> <task>` | Hand off task to another agent |
| `!tasks` | Show running and completed tasks |

### Common Commands

| Command | Description |
|---------|-------------|
| `!ping` | Check latency |
| `!status` | Agent status |
| `!help` | Available commands |

Agent-specific commands documented in [COMMUNICATION.md](COMMUNICATION.md).

## Autonomous Workflow Example

```
User: !do Implement a momentum reversal strategy for Polymarket

Strategy Agent → Designs strategy, requests features
       ↓ (automatic handoff)
Data Agent → Prepares price_momentum, volume features
       ↓ (automatic handoff)
Backtest Agent → Runs simulation, calculates Sharpe ratio
       ↓ (automatic handoff)
Risk Agent → Audits results, checks against limits
       ↓ (if approved)
Tuning Agent → Optimizes parameters
       ↓ (automatic handoff)
Backtest Agent → Validates optimized version
       ↓
Risk Agent → Final approval

All visible in Discord. User intervention only if needed.
```

## Security

- Tokens stored in `.env` (gitignored)
- Minimal Discord intents (message content only)
- Minimal permissions (send messages, read history, threads)
- Private server only

## Requirements

- Python 3.8+
- discord.py 2.3+
- python-dotenv
