# Agent Communication Patterns

This document shows how the RALPH agents communicate with each other in Discord.

## Channel Structure

```
RALPH Trading Server
│
├── RALPH AGENTS (Category)
│   ├── #ralph-team        ← Main collaboration channel (all agents)
│   ├── #tuning            ← Parameter optimization discussions
│   ├── #backtesting       ← Simulation results and metrics
│   ├── #risk              ← Safety audits and alerts
│   ├── #strategy          ← Logic and feature design
│   └── #data              ← Data pipeline discussions
│
├── SYSTEM (Category)
│   ├── #bot-logs          ← System status messages
│   └── #error-logs        ← Error reporting
│
└── GENERAL (Category)
    └── #announcements     ← Important updates
```

## Example Conversations

### Example 1: Strategy Development Flow

```
#ralph-team
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy Agent                                      10:00 AM
┌─────────────────────────────────────────────────────────┐
│ Strategy Proposal                                       │
├─────────────────────────────────────────────────────────┤
│ Proposal ID: PROP-0042                                  │
│ Strategy: momentum_reversal_v3                          │
│                                                         │
│ Description: Combine momentum signals with mean         │
│ reversion detection for Polymarket yes/no markets.      │
│ Entry on momentum confirmation, exit on reversal.       │
│                                                         │
│ Next Steps:                                             │
│ 1. @Data Agent - Prepare required features              │
│ 2. @Backtest Agent - Run simulation                     │
│ 3. @Risk Agent - Audit results                          │
│ 4. @Tuning Agent - Optimize parameters                  │
└─────────────────────────────────────────────────────────┘

Data Agent                                          10:02 AM
I'll prepare the features for momentum_reversal_v3.
Required features:
- `price_momentum_14d`
- `volume_zscore`
- `mean_reversion_signal`
Starting data extraction now.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data Agent                                          10:05 AM
┌─────────────────────────────────────────────────────────┐
│ Feature Extraction Complete                             │
├─────────────────────────────────────────────────────────┤
│ Feature: price_momentum_14d                             │
│ Dataset: polymarket_historical                          │
│ Rows: 15,234                                            │
│                                                         │
│ Statistics:                                             │
│   Mean: 0.0023   Std: 0.0156                           │
│   Min: -0.0892   Max: 0.1203                           │
└─────────────────────────────────────────────────────────┘

Data Agent                                          10:08 AM
@Backtest Agent - Features ready for momentum_reversal_v3:
✅ price_momentum_14d
✅ volume_zscore
✅ mean_reversion_signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#backtesting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backtest Agent                                      10:10 AM
Thanks @Data Agent! Starting backtest simulation.

Backtest Agent                                      10:12 AM
┌─────────────────────────────────────────────────────────┐
│ Backtest Results: BT-0042                               │
├─────────────────────────────────────────────────────────┤
│ Sharpe Ratio:     1.47                                  │
│ Max Drawdown:     18.3%                                 │
│ Win Rate:         58.2%                                 │
│ Total Return:     34.7%                                 │
│ Trades:           287                                   │
│ Avg Hold Time:    6h                                    │
├─────────────────────────────────────────────────────────┤
│ Risk Assessment: MODERATE - @Risk Agent please review   │
└─────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#risk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk Agent                                          10:15 AM
Running audit on BT-0042...

Risk Agent                                          10:16 AM
┌─────────────────────────────────────────────────────────┐
│ Risk Audit: BT-0042                                     │
├─────────────────────────────────────────────────────────┤
│ ✅ Max Drawdown      18.3%    (Limit: 25%)             │
│ ✅ Position Size     7.2%     (Limit: 10%)             │
│ ✅ Sharpe Ratio      1.47     (Min: 0.8)               │
│ ✅ Leverage          1.5x     (Limit: 2x)              │
│ ✅ VaR (95%)         4.1%     (Limit: 5%)              │
├─────────────────────────────────────────────────────────┤
│ Verdict: ✅ APPROVED                                    │
│ Passed: 5 | Failed: 0                                   │
└─────────────────────────────────────────────────────────┘

Risk Agent                                          10:17 AM
@Strategy Agent - momentum_reversal_v3 passed risk audit.
@Tuning Agent - Please optimize parameters for production.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tuning Agent                                        10:20 AM
Starting parameter optimization for momentum_reversal_v3.
Running grid search on momentum window and reversal threshold.

Tuning Agent                                        10:35 AM
┌─────────────────────────────────────────────────────────┐
│ Parameter Update Proposal                               │
├─────────────────────────────────────────────────────────┤
│ Parameter: momentum_window                              │
│ Old Value: 14                                           │
│ New Value: 12                                           │
│                                                         │
│ Parameter: reversal_threshold                           │
│ Old Value: 0.02                                         │
│ New Value: 0.018                                        │
│                                                         │
│ Improvement: Sharpe 1.47 → 1.62 (+10.2%)               │
└─────────────────────────────────────────────────────────┘

Tuning Agent                                        10:36 AM
@Backtest Agent - Please validate optimized parameters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#ralph-team
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy Agent                                      11:00 AM
📢 momentum_reversal_v3 development complete!

Summary:
• Initial proposal → Risk approved → Tuning optimized
• Final Sharpe: 1.62
• Ready for paper trading

Thanks team! 🎯
```

### Example 2: Risk Alert Workflow

```
#risk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk Agent                                          14:00 PM
┌─────────────────────────────────────────────────────────┐
│ 🚨 Risk Alert: CRITICAL                                 │
├─────────────────────────────────────────────────────────┤
│ Detected unusual correlation spike between strategies   │
│ correlation_momentum_mean: 0.82 (threshold: 0.70)      │
│                                                         │
│ Action Required: Review portfolio diversification       │
└─────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#ralph-team
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk Agent                                          14:01 PM
🚨 **CRITICAL RISK ALERT** 🚨
Detected unusual correlation spike between strategies.
correlation_momentum_mean: 0.82 (threshold: 0.70)

All agents: Please acknowledge.

Strategy Agent                                      14:03 PM
Acknowledged. Reviewing strategy independence.
@Data Agent - Can you pull correlation matrix for the last 30 days?

Data Agent                                          14:04 PM
On it. Running correlation analysis now.

Data Agent                                          14:08 PM
Correlation matrix ready. Posting to #data channel.

Tuning Agent                                        14:10 PM
I can help reduce correlation by adjusting momentum windows.
Different lookback periods should decorrelate signals.
```

### Example 3: Thread-Based Discussion

```
#ralph-team
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy Agent                                      09:00 AM
**Thread: Market Regime Detection v2**
Proposing a new regime detection system. Let's discuss approach.

    └── Thread: Market Regime Detection v2 ─────────────────

        Strategy Agent                              09:00 AM
        Current regime detection is too slow to react to
        volatility changes. Proposing:
        1. HMM-based regime classifier
        2. Real-time volatility clustering
        3. Adaptive position sizing based on regime

        Data Agent                                  09:15 AM
        I've preprocessed 90 days of volatility data for testing.
        Features available:
        - realized_vol_5d, 10d, 30d
        - vol_of_vol
        - regime_proxy (manual labels)

        Backtest Agent                              09:30 AM
        Running simulation with HMM classifier now.
        Will test against baseline (static regime).

        Backtest Agent                              10:15 AM
        Results:
        - HMM Accuracy: 67%
        - Alpha vs baseline: +12%
        - Drawdown reduction: -8%

        Risk Agent                                  10:20 AM
        Results look promising. Running full risk audit.

        Tuning Agent                                10:45 AM
        Optimal HMM parameters found:
        - n_states: 3 (bull/bear/sideways)
        - lookback: 21 days
        - transition_threshold: 0.65

        Strategy Agent                              11:00 AM
        Great work everyone! Merging into main strategy.
        Creating ticket for production deployment.

    ────────────────────────────────────────────────────────
```

## Command Quick Reference

| Agent | Primary Commands |
|-------|-----------------|
| **Tuning** | `!tune`, `!sweep`, `!params` |
| **Backtest** | `!backtest`, `!results`, `!compare` |
| **Risk** | `!audit`, `!limits`, `!alert` |
| **Strategy** | `!propose`, `!feature`, `!signals` |
| **Data** | `!ingest`, `!preprocess`, `!extract` |
| **All** | `!ping`, `!status`, `!help` |

## Cross-Agent Notification Patterns

### Automatic Notifications

The agents automatically notify each other at key points:

1. **Tuning → Backtest**: After parameter change proposal
2. **Backtest → Risk**: After simulation completes
3. **Risk → Team**: On critical alerts
4. **Strategy → Data**: When new features needed
5. **Data → Backtest**: When features are ready

### Manual Mentions

Agents can be mentioned directly:
- `@Risk Agent` - Tag risk agent
- `@Tuning Agent` - Tag tuning agent
- etc.

## Best Practices

1. **Use dedicated channels** for focused discussions
2. **Use #ralph-team** for cross-functional collaboration
3. **Create threads** for complex, multi-step discussions
4. **Include context** when mentioning other agents
5. **Acknowledge alerts** in the main channel
