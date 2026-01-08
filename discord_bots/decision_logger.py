"""
Decision Logger & Audit Trail for RALPH Agent Ensemble

Provides comprehensive logging and audit trail for:
- Trading decisions (entries, exits, adjustments)
- Agent actions and handoffs
- System changes and configurations
- Model predictions and confidence
- Compliance and regulatory requirements

P0/P1: Critical for debugging, compliance, and system improvement.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("decision_logger")


class DecisionType(Enum):
    """Types of decisions that are logged."""
    # Trading decisions
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_ADJUST = "position_adjust"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"

    # Agent decisions
    AGENT_HANDOFF = "agent_handoff"
    AGENT_TASK_START = "agent_task_start"
    AGENT_TASK_COMPLETE = "agent_task_complete"
    AGENT_ERROR = "agent_error"

    # Strategy decisions
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_CHANGE = "strategy_change"
    RISK_ASSESSMENT = "risk_assessment"
    PARAMETER_CHANGE = "parameter_change"

    # System decisions
    SYSTEM_CONFIG = "system_config"
    EMERGENCY_ACTION = "emergency_action"
    MODEL_UPDATE = "model_update"
    DATA_PIPELINE = "data_pipeline"

    # Mission/SCRUM decisions
    MISSION_START = "mission_start"
    MISSION_COMPLETE = "mission_complete"
    SPRINT_ACTION = "sprint_action"
    PROPOSAL_ACTION = "proposal_action"


class DecisionOutcome(Enum):
    """Outcome of a decision."""
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DecisionContext:
    """Context information for a decision."""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    model_predictions: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    external_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    """A complete record of a decision."""
    decision_id: str
    decision_type: DecisionType
    agent: str  # Which agent made the decision
    timestamp: str

    # Decision details
    action: str  # What was decided
    rationale: str  # Why this decision was made
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Context at time of decision
    context: Dict[str, Any] = field(default_factory=dict)

    # Outcome tracking
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    outcome_timestamp: str = ""
    outcome_details: Dict[str, Any] = field(default_factory=dict)

    # Audit trail
    parent_decision_id: str = ""  # For decision chains
    child_decision_ids: List[str] = field(default_factory=list)

    # Integrity
    checksum: str = ""

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "agent": self.agent,
            "timestamp": self.timestamp,
            "action": self.action,
            "rationale": self.rationale,
            "parameters": self.parameters,
            "context": self.context,
            "outcome": self.outcome.value,
            "outcome_timestamp": self.outcome_timestamp,
            "outcome_details": self.outcome_details,
            "parent_decision_id": self.parent_decision_id,
            "child_decision_ids": self.child_decision_ids,
            "checksum": self.checksum
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        return cls(
            decision_id=data["decision_id"],
            decision_type=DecisionType(data["decision_type"]),
            agent=data["agent"],
            timestamp=data["timestamp"],
            action=data["action"],
            rationale=data["rationale"],
            parameters=data.get("parameters", {}),
            context=data.get("context", {}),
            outcome=DecisionOutcome(data.get("outcome", "pending")),
            outcome_timestamp=data.get("outcome_timestamp", ""),
            outcome_details=data.get("outcome_details", {}),
            parent_decision_id=data.get("parent_decision_id", ""),
            child_decision_ids=data.get("child_decision_ids", []),
            checksum=data.get("checksum", "")
        )

    def compute_checksum(self) -> str:
        """Compute integrity checksum for the decision."""
        content = f"{self.decision_id}:{self.decision_type.value}:{self.agent}:{self.timestamp}:{self.action}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def format_for_discord(self) -> str:
        """Format decision for Discord display."""
        type_emoji = {
            DecisionType.TRADE_ENTRY: "📈",
            DecisionType.TRADE_EXIT: "📉",
            DecisionType.POSITION_ADJUST: "⚖️",
            DecisionType.ORDER_PLACED: "📝",
            DecisionType.ORDER_FILLED: "✅",
            DecisionType.ORDER_CANCELLED: "❌",
            DecisionType.AGENT_HANDOFF: "🔄",
            DecisionType.AGENT_TASK_START: "▶️",
            DecisionType.AGENT_TASK_COMPLETE: "✔️",
            DecisionType.AGENT_ERROR: "⚠️",
            DecisionType.STRATEGY_SIGNAL: "🎯",
            DecisionType.STRATEGY_CHANGE: "🔧",
            DecisionType.RISK_ASSESSMENT: "🛡️",
            DecisionType.PARAMETER_CHANGE: "⚙️",
            DecisionType.SYSTEM_CONFIG: "🔧",
            DecisionType.EMERGENCY_ACTION: "🚨",
            DecisionType.MODEL_UPDATE: "🤖",
            DecisionType.DATA_PIPELINE: "📊",
            DecisionType.MISSION_START: "🚀",
            DecisionType.MISSION_COMPLETE: "🏁",
            DecisionType.SPRINT_ACTION: "🏃",
            DecisionType.PROPOSAL_ACTION: "💡"
        }

        outcome_emoji = {
            DecisionOutcome.PENDING: "⏳",
            DecisionOutcome.EXECUTED: "✅",
            DecisionOutcome.REJECTED: "❌",
            DecisionOutcome.FAILED: "💥",
            DecisionOutcome.CANCELLED: "🚫"
        }

        emoji = type_emoji.get(self.decision_type, "📌")
        out_emoji = outcome_emoji.get(self.outcome, "❓")

        output = [
            f"{emoji} **{self.decision_id}** | {self.decision_type.value}",
            f"Agent: {self.agent} | Outcome: {out_emoji} {self.outcome.value}",
            f"",
            f"**Action:** {self.action}",
            f"**Rationale:** {self.rationale[:100]}{'...' if len(self.rationale) > 100 else ''}",
            f"**Time:** {self.timestamp}"
        ]

        if self.parameters:
            params = ", ".join([f"{k}={v}" for k, v in list(self.parameters.items())[:3]])
            output.append(f"**Params:** {params}")

        return "\n".join(output)


class DecisionLogger:
    """
    Central decision logging system for RALPH.

    Provides:
    - Complete audit trail of all decisions
    - Decision chain tracking (parent/child relationships)
    - Outcome tracking and analysis
    - Query and search capabilities
    - Integrity verification
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.log_dir = self.project_dir / "decision_logs"
        self.log_dir.mkdir(exist_ok=True)

        self.current_log_file = self.log_dir / f"decisions_{datetime.utcnow().strftime('%Y%m%d')}.json"

        # In-memory recent decisions for quick access
        self.recent_decisions: List[DecisionRecord] = []
        self.recent_limit = 500

        self.decision_counter = 0
        self._lock = asyncio.Lock()

        # Load today's decisions
        self._load_current_log()

    def _load_current_log(self):
        """Load current day's decision log."""
        try:
            if self.current_log_file.exists():
                with open(self.current_log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.recent_decisions = [
                        DecisionRecord.from_dict(d) for d in data.get("decisions", [])
                    ]
                    self.decision_counter = data.get("counter", 0)
                    logger.info(f"Loaded {len(self.recent_decisions)} decisions from log")

        except Exception as e:
            logger.error(f"Failed to load decision log: {e}")

    async def _save_log(self):
        """Save decisions to current log file."""
        async with self._lock:
            try:
                # Check if we need to rotate to a new day
                today_file = self.log_dir / f"decisions_{datetime.utcnow().strftime('%Y%m%d')}.json"
                if today_file != self.current_log_file:
                    self.current_log_file = today_file
                    self.recent_decisions = []

                data = {
                    "counter": self.decision_counter,
                    "date": datetime.utcnow().strftime('%Y-%m-%d'),
                    "decisions": [d.to_dict() for d in self.recent_decisions[-self.recent_limit:]]
                }

                with open(self.current_log_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save decision log: {e}")

    # =========================================================================
    # DECISION LOGGING
    # =========================================================================

    async def log_decision(
        self,
        decision_type: DecisionType,
        agent: str,
        action: str,
        rationale: str,
        parameters: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
        parent_decision_id: str = ""
    ) -> DecisionRecord:
        """
        Log a new decision.

        Args:
            decision_type: Type of decision
            agent: Agent making the decision
            action: What was decided
            rationale: Why this decision was made
            parameters: Decision parameters
            context: Context at time of decision
            parent_decision_id: Parent decision for chains

        Returns:
            The logged decision record
        """
        self.decision_counter += 1
        timestamp = datetime.utcnow().isoformat()

        decision = DecisionRecord(
            decision_id=f"DEC-{timestamp[:10].replace('-', '')}-{self.decision_counter:05d}",
            decision_type=decision_type,
            agent=agent,
            timestamp=timestamp,
            action=action,
            rationale=rationale,
            parameters=parameters or {},
            context=context or {},
            parent_decision_id=parent_decision_id
        )

        # Compute integrity checksum
        decision.checksum = decision.compute_checksum()

        # Update parent's child list if applicable
        if parent_decision_id:
            for d in self.recent_decisions:
                if d.decision_id == parent_decision_id:
                    d.child_decision_ids.append(decision.decision_id)
                    break

        self.recent_decisions.append(decision)
        await self._save_log()

        logger.info(f"Logged decision {decision.decision_id}: {decision_type.value} by {agent}")
        return decision

    async def update_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        details: Dict[str, Any] = None
    ) -> Optional[DecisionRecord]:
        """
        Update the outcome of a decision.

        Args:
            decision_id: ID of the decision to update
            outcome: The outcome
            details: Additional outcome details

        Returns:
            Updated decision record, or None if not found
        """
        for decision in self.recent_decisions:
            if decision.decision_id == decision_id:
                decision.outcome = outcome
                decision.outcome_timestamp = datetime.utcnow().isoformat()
                decision.outcome_details = details or {}
                await self._save_log()
                return decision

        # Check older log files
        decision = await self._find_in_archives(decision_id)
        if decision:
            decision.outcome = outcome
            decision.outcome_timestamp = datetime.utcnow().isoformat()
            decision.outcome_details = details or {}
            # For archived decisions, we'd need to update the archive file
            logger.warning(f"Updated archived decision {decision_id} - archive not modified")
            return decision

        return None

    async def _find_in_archives(self, decision_id: str) -> Optional[DecisionRecord]:
        """Search archived log files for a decision."""
        for log_file in sorted(self.log_dir.glob("decisions_*.json"), reverse=True):
            if log_file == self.current_log_file:
                continue
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for d in data.get("decisions", []):
                        if d["decision_id"] == decision_id:
                            return DecisionRecord.from_dict(d)
            except Exception as e:
                logger.error(f"Error reading archive {log_file}: {e}")
        return None

    # =========================================================================
    # QUERY AND SEARCH
    # =========================================================================

    def get_recent_decisions(self, limit: int = 20) -> List[DecisionRecord]:
        """Get most recent decisions."""
        return list(reversed(self.recent_decisions[-limit:]))

    def get_decisions_by_type(
        self,
        decision_type: DecisionType,
        limit: int = 50
    ) -> List[DecisionRecord]:
        """Get decisions by type."""
        matching = [d for d in self.recent_decisions if d.decision_type == decision_type]
        return list(reversed(matching[-limit:]))

    def get_decisions_by_agent(
        self,
        agent: str,
        limit: int = 50
    ) -> List[DecisionRecord]:
        """Get decisions by agent."""
        matching = [d for d in self.recent_decisions if d.agent == agent]
        return list(reversed(matching[-limit:]))

    def get_decision_chain(self, decision_id: str) -> List[DecisionRecord]:
        """Get the full chain of related decisions."""
        chain = []

        # Find root
        decision = next((d for d in self.recent_decisions if d.decision_id == decision_id), None)
        if not decision:
            return chain

        # Walk up to find root
        current = decision
        while current.parent_decision_id:
            parent = next(
                (d for d in self.recent_decisions if d.decision_id == current.parent_decision_id),
                None
            )
            if parent:
                current = parent
            else:
                break

        # Walk down to collect full chain
        def collect_children(d: DecisionRecord):
            chain.append(d)
            for child_id in d.child_decision_ids:
                child = next(
                    (dd for dd in self.recent_decisions if dd.decision_id == child_id),
                    None
                )
                if child:
                    collect_children(child)

        collect_children(current)
        return chain

    def get_trading_decisions(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100
    ) -> List[DecisionRecord]:
        """Get all trading-related decisions in a time range."""
        trading_types = [
            DecisionType.TRADE_ENTRY, DecisionType.TRADE_EXIT,
            DecisionType.POSITION_ADJUST, DecisionType.ORDER_PLACED,
            DecisionType.ORDER_FILLED, DecisionType.ORDER_CANCELLED
        ]

        matching = [d for d in self.recent_decisions if d.decision_type in trading_types]

        if start_date:
            matching = [d for d in matching if d.timestamp >= start_date]
        if end_date:
            matching = [d for d in matching if d.timestamp <= end_date]

        return list(reversed(matching[-limit:]))

    def get_failed_decisions(self, limit: int = 20) -> List[DecisionRecord]:
        """Get decisions that failed or were rejected."""
        failed = [
            d for d in self.recent_decisions
            if d.outcome in [DecisionOutcome.FAILED, DecisionOutcome.REJECTED]
        ]
        return list(reversed(failed[-limit:]))

    # =========================================================================
    # VERIFICATION AND INTEGRITY
    # =========================================================================

    def verify_integrity(self, decision_id: str) -> bool:
        """Verify the integrity of a decision record."""
        decision = next(
            (d for d in self.recent_decisions if d.decision_id == decision_id),
            None
        )
        if not decision:
            return False

        computed = decision.compute_checksum()
        return computed == decision.checksum

    def verify_all_integrity(self) -> Dict[str, bool]:
        """Verify integrity of all recent decisions."""
        return {d.decision_id: self.verify_integrity(d.decision_id) for d in self.recent_decisions}

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_decision_log_display(self, limit: int = 10) -> str:
        """Get formatted decision log for Discord display."""
        recent = self.get_recent_decisions(limit)

        if not recent:
            return "No decisions logged yet."

        output = ["## 📋 Recent Decisions\n"]

        for decision in recent:
            output.append(decision.format_for_discord())
            output.append("---")

        return "\n".join(output)

    def get_trading_summary(self, days: int = 1) -> str:
        """Get trading decision summary for Discord."""
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        trading = self.get_trading_decisions(start_date=start_date)

        if not trading:
            return f"No trading decisions in the last {days} day(s)."

        # Count by type
        by_type = {}
        for d in trading:
            t = d.decision_type.value
            by_type[t] = by_type.get(t, 0) + 1

        # Count by outcome
        by_outcome = {}
        for d in trading:
            o = d.outcome.value
            by_outcome[o] = by_outcome.get(o, 0) + 1

        output = [f"## Trading Decisions ({days} day{'s' if days > 1 else ''})\n"]
        output.append("**By Type:**")
        for t, count in sorted(by_type.items()):
            output.append(f"  {t}: {count}")

        output.append("\n**By Outcome:**")
        for o, count in sorted(by_outcome.items()):
            output.append(f"  {o}: {count}")

        output.append(f"\n**Total:** {len(trading)} decisions")

        return "\n".join(output)

    def get_agent_activity(self, agent: str, limit: int = 20) -> str:
        """Get agent activity for Discord."""
        decisions = self.get_decisions_by_agent(agent, limit)

        if not decisions:
            return f"No decisions logged for {agent}."

        output = [f"## 🤖 {agent} Agent Activity\n"]

        for decision in decisions[:10]:
            output.append(decision.format_for_discord())
            output.append("---")

        return "\n".join(output)

    def get_decision_details(self, decision_id: str) -> str:
        """Get detailed view of a decision."""
        decision = next(
            (d for d in self.recent_decisions if d.decision_id == decision_id),
            None
        )

        if not decision:
            return f"Decision {decision_id} not found."

        output = [f"## Decision Details: {decision_id}\n"]
        output.append(decision.format_for_discord())

        if decision.parameters:
            output.append("\n**Parameters:**")
            for k, v in decision.parameters.items():
                output.append(f"  {k}: {v}")

        if decision.context:
            output.append("\n**Context:**")
            for k, v in list(decision.context.items())[:5]:
                output.append(f"  {k}: {v}")

        if decision.outcome_details:
            output.append("\n**Outcome Details:**")
            for k, v in decision.outcome_details.items():
                output.append(f"  {k}: {v}")

        # Show chain
        chain = self.get_decision_chain(decision_id)
        if len(chain) > 1:
            output.append(f"\n**Decision Chain:** {len(chain)} related decisions")
            for d in chain:
                marker = "→" if d.decision_id == decision_id else "  "
                output.append(f"  {marker} {d.decision_id}: {d.action[:30]}...")

        # Integrity
        is_valid = self.verify_integrity(decision_id)
        output.append(f"\n**Integrity:** {'✅ Valid' if is_valid else '❌ Invalid'}")

        return "\n".join(output)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def log_trade_entry(
    logger_instance: DecisionLogger,
    agent: str,
    market: str,
    side: str,
    size: float,
    price: float,
    rationale: str,
    context: Dict[str, Any] = None
) -> DecisionRecord:
    """Convenience function for logging trade entries."""
    return await logger_instance.log_decision(
        decision_type=DecisionType.TRADE_ENTRY,
        agent=agent,
        action=f"{side.upper()} {size} @ {price} on {market}",
        rationale=rationale,
        parameters={"market": market, "side": side, "size": size, "price": price},
        context=context or {}
    )


async def log_trade_exit(
    logger_instance: DecisionLogger,
    agent: str,
    market: str,
    size: float,
    entry_price: float,
    exit_price: float,
    pnl: float,
    rationale: str,
    entry_decision_id: str = ""
) -> DecisionRecord:
    """Convenience function for logging trade exits."""
    return await logger_instance.log_decision(
        decision_type=DecisionType.TRADE_EXIT,
        agent=agent,
        action=f"EXIT {size} @ {exit_price} on {market} (PnL: {pnl:+.2f})",
        rationale=rationale,
        parameters={
            "market": market,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl
        },
        parent_decision_id=entry_decision_id
    )


async def log_agent_handoff(
    logger_instance: DecisionLogger,
    from_agent: str,
    to_agent: str,
    task: str,
    context: Dict[str, Any] = None
) -> DecisionRecord:
    """Convenience function for logging agent handoffs."""
    return await logger_instance.log_decision(
        decision_type=DecisionType.AGENT_HANDOFF,
        agent=from_agent,
        action=f"Handoff to {to_agent}: {task}",
        rationale=f"{from_agent} delegating task to {to_agent}",
        parameters={"from": from_agent, "to": to_agent, "task": task},
        context=context or {}
    )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_decision_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """Get or create the decision logger instance."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger()
    return _decision_logger


def set_decision_logger(logger_instance: DecisionLogger):
    """Set the decision logger instance."""
    global _decision_logger
    _decision_logger = logger_instance
