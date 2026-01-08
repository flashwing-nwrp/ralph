# Discord Bot Architecture for RALPH Agent Ensemble

## Recommendation: Multiple Bots (One Per Agent)

### Why Multiple Bots Over Single Bot?

| Criteria | Multiple Bots | Single Bot |
|----------|---------------|------------|
| Visual Identity | Each agent has unique name/avatar | All messages from one identity |
| Conversation Feel | Realistic multi-agent discussion | Needs prefix/embeds to distinguish |
| Failure Isolation | One crash doesn't affect others | Single point of failure |
| Rate Limits | Distributed across 5 apps | Shared limits |
| Development | Can update/restart one agent | Must restart entire system |
| Token Management | 5 tokens to manage | 1 token |
| Resource Usage | 5 processes (minimal with asyncio) | 1 process |

**Verdict: Multiple bots is strongly recommended** for your use case because:

1. **Agent Identity Matters** - When Tuning Agent posts, you want to see "Tuning Agent" as the sender, not "RALPH Bot: [Tuning]"
2. **Authentic Collaboration Feel** - The Discord server should look like 5 entities having real conversations
3. **Independent Operation** - Agents can be restarted/updated individually during development
4. **Future Flexibility** - Each agent can evolve with different capabilities

### Alternative: Webhook Hybrid (Not Recommended)

A single bot could use webhooks to post "as" different agents with custom names/avatars. However:
- Webhooks can't read messages or respond to events
- Requires complex orchestration
- Loses bidirectional communication capability

## Architecture Overview

```
discord_bots/
├── .env                    # All 5 bot tokens (gitignored)
├── .env.example            # Template without secrets
├── requirements.txt        # Dependencies
├── base_agent.py           # Shared bot logic (abstract class)
├── agents/
│   ├── __init__.py
│   ├── tuning_agent.py     # Parameter optimization
│   ├── backtest_agent.py   # Simulation & metrics
│   ├── risk_agent.py       # Safety auditing
│   ├── strategy_agent.py   # Logic & features
│   └── data_agent.py       # Preprocessing & denoising
├── run_all.py              # Launches all 5 bots concurrently
├── run_single.py           # Run one specific agent
└── SETUP.md                # Complete setup instructions
```

## Channel Structure (Recommended)

```
RALPH Trading Server
├── #announcements          (read-only, important updates)
├── #ralph-team             (main collaboration channel)
├── AGENT CHANNELS
│   ├── #tuning             (parameter discussions)
│   ├── #backtesting        (simulation results)
│   ├── #risk               (safety audits)
│   ├── #strategy           (logic/feature design)
│   └── #data               (preprocessing pipeline)
├── LOGS
│   ├── #bot-logs           (system messages)
│   └── #error-logs         (exceptions/crashes)
└── VOICE (optional)
    └── #standup            (for future audio features)
```

## Communication Patterns

### 1. Direct Channel Posts
Each agent monitors its dedicated channel and the #ralph-team channel.

### 2. Cross-Agent Mentions
Agents can @mention each other for collaboration:
```
[Strategy Agent in #strategy]:
"@Risk Agent - I'm proposing a 3x leverage strategy. Can you audit this?"

[Risk Agent responds]:
"Analyzing... Max drawdown at 3x would exceed 40% threshold. Recommending 2x max."
```

### 3. Thread-Based Discussions
For complex topics, agents create threads:
```
#ralph-team
└── Thread: "Market Regime Detection v2"
    ├── Data Agent: "Preprocessed 90 days of volatility data"
    ├── Strategy Agent: "Proposed regime classifier attached"
    ├── Backtest Agent: "Running simulation now..."
    └── Backtest Agent: "Results: 67% regime accuracy, 12% alpha improvement"
```

## Security Model

1. **Minimal Intents** - Only `guilds`, `guild_messages`, `message_content`
2. **Minimal Permissions** - Send messages, read history, use threads
3. **Private Server Only** - Bots should only be in your private server
4. **No Token Exposure** - All tokens in `.env`, never committed
5. **Rate Limit Handling** - Built-in backoff and retry logic

## Resource Requirements

- **RAM**: ~50MB per bot (250MB total for all 5)
- **CPU**: Minimal (event-driven, mostly idle)
- **Network**: Standard WebSocket connections
- **Disk**: Minimal (logs can be configured)

Running all 5 bots on a single machine is trivial.
