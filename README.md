# RALPH - Autonomous AI Agent Ensemble

![Ralph](ralph.webp)

RALPH (Reinforced Autonomous Learning & Processing Hub) is a Discord-based multi-agent AI system designed to autonomously develop, test, and optimize trading strategies for the Polymarket AI trading bot.

Based on [Geoffrey Huntley's Ralph pattern](https://ghuntley.com/ralph/), extended into a collaborative agent ensemble with Claude Code integration.

[![Ralph Flowchart](ralph-flowchart.png)](https://snarktank.github.io/ralph/)

**[View Interactive Flowchart](https://snarktank.github.io/ralph/)**

## Overview

RALPH consists of 5 specialized AI agents that collaborate through Discord, each with distinct expertise:

| Agent | Role | Personality |
|-------|------|-------------|
| **Strategy Agent** | Mission Lead & Trading Logic | "The Visionary" - Creative, big-picture thinker |
| **Tuning Agent** | Parameter Optimization | "The Perfectionist" - Meticulous, data-driven |
| **Backtest Agent** | Simulation & Validation | "The Skeptic" - Evidence-based, thorough |
| **Risk Agent** | Safety & Compliance | "The Guardian" - Cautious, protective (has VETO power) |
| **Data Agent** | Data Pipeline & Preprocessing | "The Librarian" - Methodical, detail-oriented |

## Key Features

### Mission System
Set high-level goals via Discord. The Strategy Agent breaks them into tasks and delegates to appropriate agents.

```
!mission Improve the momentum strategy Sharpe ratio by 20%
```

### SCRUM Methodology
Built-in agile workflow with sprints, backlog, user stories, and retrospectives.

```
!sprint create Q1 Optimization | Improve core strategy metrics
!story add Implement adaptive learning rate | Auto-tune based on market regime | improvement | 5
!standup
```

### Self-Improvement Proposals
Agents can propose system improvements during their work, queued for operator review.

```
!propose accuracy high Model predictions biased | Add isotonic calibration | Reduce bias by 15%
!proposals
!approve IMP-0001
```

### Production-Ready Operations (P0/P1/P2)

#### P0 Critical Systems
- **Emergency Controls**: Kill switch, circuit breakers, trading halt/resume
- **Real-time Monitoring**: Dashboards, threshold alerts, metric tracking
- **Decision Logging**: Complete audit trail with integrity verification

#### P1 Important Systems
- **Model Lifecycle**: Version registry, deployment, rollback, shadow mode
- **Data Quality**: Freshness, completeness, outlier detection, drift monitoring

#### P2 Operational Systems
- **Task Scheduler**: Cron-like scheduling for automated operations
- **Testing Framework**: Unit, integration, simulation, and regression tests
- **Context Persistence**: Agent memory, sessions, cross-agent context sharing

## Discord Commands

### Core Commands
| Command | Description |
|---------|-------------|
| `!mission <goal>` | Set a new mission for the agent ensemble |
| `!mission_status` | Check current mission progress |
| `!do <task>` | Execute a task using Claude Code |
| `!handoff <agent> <task>` | Hand off a task to another agent |
| `!ask <question>` | Ask the agent a question |

### Emergency Controls (P0)
| Command | Description |
|---------|-------------|
| `!killswitch [reason]` | HALT all trading immediately |
| `!halt [reason]` | Pause trading (less severe) |
| `!resume_trading [notes]` | Resume after halt/kill |
| `!trading_status` | Check trading system status |
| `!circuit_breakers` | View circuit breaker config |

### Monitoring & Alerts (P0)
| Command | Description |
|---------|-------------|
| `!dashboard` | View monitoring dashboard |
| `!alerts [limit]` | Show active alerts |
| `!ack <alert_id>` | Acknowledge an alert |
| `!resolve <alert_id> [notes]` | Resolve an alert |

### Decision Logging (P0/P1)
| Command | Description |
|---------|-------------|
| `!decisions [limit]` | Show recent decision log |
| `!decision <id>` | View decision details |
| `!trading_summary [days]` | Trading decision summary |

### Model Management (P1)
| Command | Description |
|---------|-------------|
| `!models` | Show model registry |
| `!model <id> [version]` | View model details |
| `!deploy_model <id> <version>` | Deploy model to production |
| `!rollback_model <id> <version>` | Rollback to previous version |
| `!training [model_id]` | Show training history |

### Data Quality (P1)
| Command | Description |
|---------|-------------|
| `!data_quality` | View data quality dashboard |
| `!data_source <source>` | Check specific data source |
| `!quality_history [source]` | View quality check history |

### SCRUM Commands
| Command | Description |
|---------|-------------|
| `!story add <title> \| <desc> \| <type> \| <points>` | Create a user story |
| `!backlog` | View product backlog |
| `!sprint create <name> \| <goal>` | Create a new sprint |
| `!sprint start` | Start the sprint |
| `!sprint board` | View sprint board |
| `!sprint end` | End current sprint |
| `!standup` | Daily standup summary |
| `!velocity` | Team velocity report |

### Proposals
| Command | Description |
|---------|-------------|
| `!propose <cat> <priority> <problem> \| <solution> \| <impact>` | Submit improvement proposal |
| `!proposals` | View pending proposals |
| `!approve <id> [notes]` | Approve a proposal |
| `!reject <id> [reason]` | Reject a proposal |
| `!defer <id> [reason]` | Defer for later |

### Scheduler (P2)
| Command | Description |
|---------|-------------|
| `!schedule` | Show scheduled tasks |
| `!task_detail <id>` | View task details |
| `!run_task <id>` | Run task immediately |
| `!pause_task <id>` | Pause a scheduled task |
| `!resume_task <id>` | Resume paused task |

### Testing (P2)
| Command | Description |
|---------|-------------|
| `!tests` | Show test summary |
| `!test <id>` | View test details |
| `!run_test <id>` | Run specific test |
| `!run_tests [type]` | Run all tests |
| `!test_results [run_id]` | View test results |

### Context/Memory (P2)
| Command | Description |
|---------|-------------|
| `!context` | View context store summary |
| `!agent_memory [agent]` | Show agent's memory |
| `!share_context <key> <value>` | Share context with agents |
| `!sessions` | Show active sessions |

### VPS Deployment
| Command | Description |
|---------|-------------|
| `!deploy` | Deploy latest code to VPS |
| `!vps` | Check VPS status |
| `!logs [lines]` | Get VPS logs |
| `!restart` | Restart VPS service |

### Inter-Bot Communication
| Command | Description |
|---------|-------------|
| `!team` | Show registered RALPH bots and status |
| `!msg <agent> <message>` | Send a task to another agent |
| `!ask_agent <agent> <question>` | Ask another agent a question |
| `!alert_team [severity] <message>` | Alert all agents |
| `!delegate <agent> <task>` | Hand off a task to another agent |
| `!pending` | Show pending tasks from other agents |

### Cost Optimization (Orchestration)
| Command | Description |
|---------|-------------|
| `!orch_stats` | View token savings statistics |
| `!orch_provider` | Show orchestration LLM provider |
| `!classify <task>` | Preview how a task would be classified |

## Tiered LLM Architecture (Cost Optimization)

RALPH uses a tiered approach to minimize Claude Code token usage:

```
┌─────────────────────────────────────────────────────────────┐
│                     Incoming Task                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   Orchestration Layer  │  ← GPT-4o-mini / Claude Haiku
          │   (Cheap & Fast)       │     ~$0.15-0.25/1M tokens
          └───────────┬───────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌────────┐     ┌────────────┐    ┌───────────┐
│TRIVIAL │     │  SIMPLE    │    │  COMPLEX  │
│ Local  │     │ Cheap LLM  │    │Claude Code│
│ Free   │     │ ~90% less  │    │ Full power│
└────────┘     └────────────┘    └───────────┘
```

### What Each Tier Handles

| Tier | Handled By | Examples |
|------|-----------|----------|
| **Trivial** | Local patterns (free) | Status checks, pings, ACKs |
| **Simple** | GPT-4o-mini/Haiku | Questions, routing, summaries |
| **Complex** | Claude Code | Code writing, multi-file edits, analysis |

### Token Savings

- **Simple Q&A**: ~90% savings (500 tokens vs 5000)
- **Routing decisions**: ~95% savings (handled locally)
- **Context summarization**: ~60% savings (compress before Claude)

### Configuration

The orchestration layer automatically finds your OpenAI API key from:

1. **Environment variable**: `OPENAI_API_KEY` in `.env`
2. **Polymarket config**: `config.yaml` in your Polymarket AI Bot directory

```env
# Option 1: Set directly in .env
OPENAI_API_KEY=sk-...

# Option 2: Already in Polymarket config.yaml? Just set the path:
POLYMARKET_PROJECT_DIR=/path/to/polymarket-ai-bot
# The key will be loaded from config.yaml automatically

# Option 3: Use Anthropic Haiku instead
# Falls back to ANTHROPIC_API_KEY (already set for Claude Code)

# If no API key is found, falls back to local pattern matching only
```

## Inter-Bot Communication

RALPH agents communicate with each other using Discord @mentions. This enables autonomous collaboration where agents can:
- Delegate tasks to specialists
- Ask questions and get answers
- Send alerts about issues
- Hand off work when their part is complete

### Message Format

Agents use structured message prefixes to indicate intent:

```
@TargetAgent [ACTION] message content
```

**Actions:**
- `[TASK]` - Request the target agent to do something
- `[RESPONSE]` - Reply to a previous task
- `[HANDOFF]` - Transfer work ownership
- `[ALERT]` - Urgent notification
- `[INFO]` - Informational message
- `[QUESTION]` - Ask for input/decision
- `[ACK]` - Acknowledgment

### Example Workflow

1. **Strategy Agent** plans a mission and delegates:
   ```
   @BacktestAgent [TASK] Run backtests on the new momentum parameters
   ```

2. **Backtest Agent** acknowledges and works:
   ```
   [ACK] Received task. Processing...
   ```

3. **Backtest Agent** completes and hands off to Risk:
   ```
   @RiskAgent [HANDOFF] Validate backtests results. Sharpe improved 15%.
   ```

4. **Risk Agent** reviews and alerts if issues:
   ```
   @StrategyAgent @TuningAgent [ALERT] Drawdown exceeds 15% limit
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Discord Server                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Strategy │ │ Tuning  │ │Backtest │ │  Risk   │ │  Data   │   │
│  │  Agent  │ │  Agent  │ │  Agent  │ │  Agent  │ │  Agent  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│       └───────────┴───────────┼───────────┴───────────┘         │
│                               │                                  │
│                    ┌──────────┴──────────┐                      │
│                    │  Agent Coordinator   │                      │
│                    │  (Handoff Queue)     │                      │
│                    └──────────┬──────────┘                      │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    Claude Executor     │
                    │   (Claude Code CLI)    │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────┴───────┐     ┌────────┴────────┐     ┌───────┴───────┐
│ Polymarket AI │     │   MySQL/TAAPI   │     │  Coinbase/    │
│    Bot Code   │     │   Indicators    │     │   Binance WS  │
└───────────────┘     └─────────────────┘     └───────────────┘
```

## Project Knowledge

RALPH includes comprehensive knowledge about the Polymarket AI Bot:

- **MySQL Database**: Candle storage schema, query patterns
- **TAAPI.io Integration**: 100+ technical indicators
- **WebSocket Data**: Real-time feeds from Coinbase/Binance
- **Experimentation Framework**: Grid search, Bayesian optimization, walk-forward analysis

## Setup

### Prerequisites

- Python 3.10+
- Discord Bot tokens (5 bots, one per agent)
- Claude API key (for Claude Code execution)
- Discord server with appropriate channels

### Environment Variables

Create a `.env` file:

```env
# Discord
DISCORD_GUILD_ID=your_guild_id
OWNER_USER_ID=your_user_id

# Agent Bot Tokens
TUNING_BOT_TOKEN=...
BACKTEST_BOT_TOKEN=...
RISK_BOT_TOKEN=...
STRATEGY_BOT_TOKEN=...
DATA_BOT_TOKEN=...

# Claude Code
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-sonnet-4-20250514

# Project
RALPH_PROJECT_DIR=/path/to/polymarket-ai-bot
POLYMARKET_PROJECT_DIR=/path/to/polymarket-ai-bot

# VPS (optional)
VPS_HOST=...
VPS_USER=...
VPS_SSH_KEY_PATH=...
```

### Discord Server Setup

Create these channels:
- `#ralph-team` - Team coordination
- `#tuning` - Tuning Agent primary
- `#backtesting` - Backtest Agent primary
- `#risk` - Risk Agent primary
- `#strategy` - Strategy Agent primary
- `#data` - Data Agent primary
- `#bot-logs` - Bot status messages
- `#error-logs` - Error notifications

### Installation

```bash
# Clone the repository
git clone https://github.com/flashwing-nwrp/ralph.git
cd ralph

# Install dependencies
pip install -r discord_bots/requirements.txt

# Run all agents
python discord_bots/run_all.py
```

## File Structure

```
ralph/
├── discord_bots/
│   ├── base_agent.py          # Base class for all agents
│   ├── claude_executor.py     # Claude Code integration
│   ├── agent_prompts.py       # Agent role definitions
│   ├── project_knowledge.py   # Polymarket AI Bot knowledge
│   │
│   ├── # Agent Implementations
│   ├── tuning_agent.py
│   ├── backtest_agent.py
│   ├── risk_agent.py
│   ├── strategy_agent.py
│   ├── data_agent.py
│   │
│   ├── # Mission & Workflow
│   ├── mission_manager.py     # Mission tracking
│   ├── scrum_manager.py       # SCRUM methodology
│   ├── improvement_proposals.py
│   ├── bot_communication.py   # Inter-bot @mention communication
│   ├── orchestration_layer.py # Tiered LLM cost optimization
│   │
│   ├── # P0 Critical Systems
│   ├── emergency_controls.py  # Kill switch, circuit breakers
│   ├── monitoring_alerts.py   # Dashboards, alerts
│   ├── decision_logger.py     # Audit trail
│   │
│   ├── # P1 Important Systems
│   ├── model_lifecycle.py     # Model versioning
│   ├── data_quality.py        # Data monitoring
│   │
│   ├── # P2 Operational Systems
│   ├── scheduler.py           # Task scheduling
│   ├── testing_framework.py   # Test management
│   ├── context_persistence.py # Agent memory
│   │
│   ├── vps_deployer.py        # VPS deployment
│   ├── autonomous_orchestrator.py
│   └── run_all.py             # Launch all agents
│
├── flowchart/                 # Interactive visualization
├── skills/                    # Amp skills (original Ralph)
└── README.md
```

## Workflow Example

1. **Operator sets mission**:
   ```
   !mission Improve win rate on high-volatility markets by 15%
   ```

2. **Strategy Agent plans**:
   - Breaks down into tasks
   - Creates sprint with user stories
   - Delegates to appropriate agents

3. **Agents collaborate**:
   - Data Agent prepares features
   - Tuning Agent optimizes parameters
   - Backtest Agent validates changes
   - Risk Agent audits safety

4. **Agents propose improvements**:
   ```
   !propose accuracy high Calibration drift detected | Implement online recalibration | Maintain prediction accuracy
   ```

5. **Operator reviews and approves**:
   ```
   !proposals
   !approve IMP-0001
   ```

6. **Deployment**:
   ```
   !deploy
   ```

## Safety Features

- **Kill Switch**: Immediate halt via `!killswitch`
- **Circuit Breakers**: Auto-halt on drawdown (15%), loss streaks (5), volatility spikes (3x)
- **Risk Agent Veto**: Risk Agent can reject any strategy that fails safety audit
- **Complete Audit Trail**: All decisions logged with checksums
- **Model Rollback**: Instant rollback to previous model versions

## Original Ralph Pattern

The original Ralph pattern (for Amp CLI) is still available in the `scripts/` and `skills/` directories. See [AGENTS.md](AGENTS.md) for details on the autonomous iteration loop.

## References

- [Geoffrey Huntley's Ralph article](https://ghuntley.com/ralph/)
- [Amp documentation](https://ampcode.com/manual)
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)

## License

MIT
