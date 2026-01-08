"""
RALPH Discord Agent Bots

This package contains the individual agent bot implementations:
- TuningAgent: Parameter optimization
- BacktestAgent: Simulation & metrics
- RiskAgent: Safety auditing
- StrategyAgent: Logic & features
- DataAgent: Preprocessing & denoising
"""

from .tuning_agent import TuningAgent
from .backtest_agent import BacktestAgent
from .risk_agent import RiskAgent
from .strategy_agent import StrategyAgent
from .data_agent import DataAgent

__all__ = [
    "TuningAgent",
    "BacktestAgent",
    "RiskAgent",
    "StrategyAgent",
    "DataAgent",
]

# Agent registry for easy lookup
AGENTS = {
    "tuning": TuningAgent,
    "backtest": BacktestAgent,
    "risk": RiskAgent,
    "strategy": StrategyAgent,
    "data": DataAgent,
}
