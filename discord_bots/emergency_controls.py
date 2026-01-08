"""
Emergency Controls & Kill Switch System for RALPH Agent Ensemble

Provides critical safety mechanisms for the trading bot:
- Kill switch for immediate trading halt
- Circuit breakers for automatic risk triggers
- Emergency notifications to all channels
- Trading state management

P0 Critical: Must be operational before any live trading.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("emergency_controls")


class TradingState(Enum):
    """Current trading system state."""
    ACTIVE = "active"              # Normal operation
    HALTED = "halted"              # Manual halt, can resume
    KILLED = "killed"              # Emergency kill, requires manual review
    CIRCUIT_BREAKER = "circuit_breaker"  # Auto-triggered, time-based resume
    MAINTENANCE = "maintenance"    # Scheduled maintenance


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    DRAWDOWN = "drawdown"          # Portfolio drawdown exceeded
    LOSS_STREAK = "loss_streak"    # Consecutive losses
    VOLATILITY = "volatility"      # Market volatility spike
    ERROR_RATE = "error_rate"      # System error rate
    API_FAILURE = "api_failure"    # External API failures
    POSITION_LIMIT = "position_limit"  # Max position exceeded


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    breaker_type: CircuitBreakerType
    threshold: float
    cooldown_minutes: int = 30
    auto_reset: bool = True
    description: str = ""


@dataclass
class EmergencyEvent:
    """Record of an emergency event."""
    event_id: str
    event_type: str
    triggered_by: str  # "manual", "circuit_breaker", "api_failure", etc.
    reason: str
    timestamp: str
    previous_state: str
    new_state: str
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    resolved_at: str = ""
    resolution_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "triggered_by": self.triggered_by,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "metrics_snapshot": self.metrics_snapshot,
            "resolved_at": self.resolved_at,
            "resolution_notes": self.resolution_notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmergencyEvent":
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            triggered_by=data["triggered_by"],
            reason=data["reason"],
            timestamp=data["timestamp"],
            previous_state=data["previous_state"],
            new_state=data["new_state"],
            metrics_snapshot=data.get("metrics_snapshot", {}),
            resolved_at=data.get("resolved_at", ""),
            resolution_notes=data.get("resolution_notes", "")
        )


class EmergencyControlSystem:
    """
    Central emergency control system for the RALPH trading bot.

    Provides:
    - Kill switch for immediate trading halt
    - Circuit breakers for automatic risk triggers
    - State persistence across restarts
    - Event logging and notifications
    """

    # Default circuit breaker configurations
    DEFAULT_CIRCUIT_BREAKERS = [
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.DRAWDOWN,
            threshold=0.15,  # 15% drawdown
            cooldown_minutes=60,
            auto_reset=False,
            description="Triggers when portfolio drawdown exceeds 15%"
        ),
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.LOSS_STREAK,
            threshold=5,  # 5 consecutive losses
            cooldown_minutes=30,
            auto_reset=True,
            description="Triggers after 5 consecutive losing trades"
        ),
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=3.0,  # 3x normal volatility
            cooldown_minutes=15,
            auto_reset=True,
            description="Triggers when market volatility exceeds 3x normal"
        ),
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.ERROR_RATE,
            threshold=0.10,  # 10% error rate
            cooldown_minutes=10,
            auto_reset=True,
            description="Triggers when system error rate exceeds 10%"
        ),
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.API_FAILURE,
            threshold=3,  # 3 consecutive API failures
            cooldown_minutes=5,
            auto_reset=True,
            description="Triggers after 3 consecutive API failures"
        ),
        CircuitBreakerConfig(
            breaker_type=CircuitBreakerType.POSITION_LIMIT,
            threshold=0.25,  # 25% of portfolio in single position
            cooldown_minutes=0,  # Requires manual intervention
            auto_reset=False,
            description="Triggers when single position exceeds 25% of portfolio"
        ),
    ]

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.state_file = self.project_dir / "emergency_state.json"
        self.events_file = self.project_dir / "emergency_events.json"

        # Current state
        self.trading_state: TradingState = TradingState.ACTIVE
        self.state_reason: str = ""
        self.state_changed_at: str = ""
        self.circuit_breaker_reset_at: Optional[datetime] = None

        # Circuit breaker configs
        self.circuit_breakers: Dict[CircuitBreakerType, CircuitBreakerConfig] = {
            cb.breaker_type: cb for cb in self.DEFAULT_CIRCUIT_BREAKERS
        }

        # Tracking
        self.events: List[EmergencyEvent] = []
        self.event_counter = 0
        self._lock = asyncio.Lock()

        # Callbacks for notifications
        self._notification_callbacks: List[Callable] = []

        # Metrics for circuit breakers
        self._metrics: Dict[str, Any] = {
            "current_drawdown": 0.0,
            "consecutive_losses": 0,
            "current_volatility": 1.0,
            "error_rate": 0.0,
            "api_failures": 0,
            "max_position_pct": 0.0,
        }

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.trading_state = TradingState(data.get("trading_state", "active"))
                    self.state_reason = data.get("state_reason", "")
                    self.state_changed_at = data.get("state_changed_at", "")
                    self.event_counter = data.get("event_counter", 0)

                    reset_at = data.get("circuit_breaker_reset_at")
                    if reset_at:
                        self.circuit_breaker_reset_at = datetime.fromisoformat(reset_at)

                    logger.info(f"Loaded emergency state: {self.trading_state.value}")

            if self.events_file.exists():
                with open(self.events_file, "r", encoding="utf-8") as f:
                    events_data = json.load(f)
                    self.events = [EmergencyEvent.from_dict(e) for e in events_data]
                    logger.info(f"Loaded {len(self.events)} emergency events")

        except Exception as e:
            logger.error(f"Failed to load emergency state: {e}")

    async def _save_state(self):
        """Save state to file."""
        async with self._lock:
            try:
                state_data = {
                    "trading_state": self.trading_state.value,
                    "state_reason": self.state_reason,
                    "state_changed_at": self.state_changed_at,
                    "event_counter": self.event_counter,
                    "circuit_breaker_reset_at": (
                        self.circuit_breaker_reset_at.isoformat()
                        if self.circuit_breaker_reset_at else None
                    ),
                    "metrics": self._metrics,
                }

                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=2)

                with open(self.events_file, "w", encoding="utf-8") as f:
                    json.dump([e.to_dict() for e in self.events[-100:]], f, indent=2)  # Keep last 100

            except Exception as e:
                logger.error(f"Failed to save emergency state: {e}")

    def register_notification_callback(self, callback: Callable):
        """Register a callback for emergency notifications."""
        self._notification_callbacks.append(callback)

    async def _notify_all(self, message: str, severity: str = "warning"):
        """Send notifications to all registered callbacks."""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message, severity)
                else:
                    callback(message, severity)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    async def _create_event(
        self,
        event_type: str,
        triggered_by: str,
        reason: str,
        previous_state: TradingState,
        new_state: TradingState
    ) -> EmergencyEvent:
        """Create and record an emergency event."""
        self.event_counter += 1
        event = EmergencyEvent(
            event_id=f"EMG-{self.event_counter:05d}",
            event_type=event_type,
            triggered_by=triggered_by,
            reason=reason,
            timestamp=datetime.utcnow().isoformat(),
            previous_state=previous_state.value,
            new_state=new_state.value,
            metrics_snapshot=self._metrics.copy()
        )
        self.events.append(event)
        await self._save_state()
        return event

    # =========================================================================
    # KILL SWITCH OPERATIONS
    # =========================================================================

    async def activate_kill_switch(self, reason: str, triggered_by: str = "operator") -> EmergencyEvent:
        """
        Activate the emergency kill switch.

        This immediately halts ALL trading activity. Requires manual review
        and explicit resume to restart operations.

        Args:
            reason: Why the kill switch was activated
            triggered_by: Who/what triggered it (operator, agent, system)

        Returns:
            The emergency event record
        """
        previous_state = self.trading_state
        self.trading_state = TradingState.KILLED
        self.state_reason = reason
        self.state_changed_at = datetime.utcnow().isoformat()
        self.circuit_breaker_reset_at = None  # No auto-reset for kill switch

        event = await self._create_event(
            event_type="KILL_SWITCH",
            triggered_by=triggered_by,
            reason=reason,
            previous_state=previous_state,
            new_state=TradingState.KILLED
        )

        # Emergency notification
        await self._notify_all(
            f"🚨 **KILL SWITCH ACTIVATED** 🚨\n"
            f"Reason: {reason}\n"
            f"Triggered by: {triggered_by}\n"
            f"Event ID: {event.event_id}\n"
            f"ALL TRADING HALTED. Manual intervention required.",
            severity="critical"
        )

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        return event

    async def halt_trading(self, reason: str, triggered_by: str = "operator") -> EmergencyEvent:
        """
        Halt trading operations (less severe than kill switch).

        Trading is paused but can be resumed without full review.

        Args:
            reason: Why trading was halted
            triggered_by: Who/what triggered it

        Returns:
            The emergency event record
        """
        previous_state = self.trading_state
        self.trading_state = TradingState.HALTED
        self.state_reason = reason
        self.state_changed_at = datetime.utcnow().isoformat()

        event = await self._create_event(
            event_type="TRADING_HALT",
            triggered_by=triggered_by,
            reason=reason,
            previous_state=previous_state,
            new_state=TradingState.HALTED
        )

        await self._notify_all(
            f"⚠️ **TRADING HALTED**\n"
            f"Reason: {reason}\n"
            f"Triggered by: {triggered_by}\n"
            f"Use `!resume_trading` to resume.",
            severity="warning"
        )

        logger.warning(f"Trading halted: {reason}")
        return event

    async def resume_trading(
        self,
        confirmed_by: str,
        notes: str = ""
    ) -> Optional[EmergencyEvent]:
        """
        Resume trading operations.

        For KILLED state, this requires explicit confirmation that issues
        have been reviewed and resolved.

        Args:
            confirmed_by: Who is confirming the resume
            notes: Optional notes about the resolution

        Returns:
            The emergency event record, or None if already active
        """
        if self.trading_state == TradingState.ACTIVE:
            return None

        # For KILLED state, log a warning
        if self.trading_state == TradingState.KILLED:
            logger.warning(f"Resuming from KILLED state - confirmed by {confirmed_by}")

        previous_state = self.trading_state
        self.trading_state = TradingState.ACTIVE
        self.state_reason = ""
        self.state_changed_at = datetime.utcnow().isoformat()
        self.circuit_breaker_reset_at = None

        # Mark the triggering event as resolved
        if self.events:
            last_event = self.events[-1]
            if last_event.new_state in [TradingState.KILLED.value, TradingState.HALTED.value]:
                last_event.resolved_at = datetime.utcnow().isoformat()
                last_event.resolution_notes = notes

        event = await self._create_event(
            event_type="TRADING_RESUMED",
            triggered_by=confirmed_by,
            reason=notes or "Trading resumed",
            previous_state=previous_state,
            new_state=TradingState.ACTIVE
        )

        await self._notify_all(
            f"✅ **TRADING RESUMED**\n"
            f"Confirmed by: {confirmed_by}\n"
            f"Previous state: {previous_state.value}\n"
            f"Notes: {notes or 'None'}",
            severity="info"
        )

        logger.info(f"Trading resumed by {confirmed_by}")
        return event

    # =========================================================================
    # CIRCUIT BREAKER OPERATIONS
    # =========================================================================

    async def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update metrics used for circuit breaker evaluation.

        Args:
            metrics: Dictionary of metric updates
        """
        self._metrics.update(metrics)
        await self._check_circuit_breakers()

    async def _check_circuit_breakers(self):
        """Check all circuit breakers against current metrics."""
        if self.trading_state in [TradingState.KILLED, TradingState.MAINTENANCE]:
            return  # Don't trigger circuit breakers if already in critical state

        for breaker_type, config in self.circuit_breakers.items():
            triggered = False
            current_value = None

            if breaker_type == CircuitBreakerType.DRAWDOWN:
                current_value = self._metrics.get("current_drawdown", 0)
                triggered = current_value >= config.threshold

            elif breaker_type == CircuitBreakerType.LOSS_STREAK:
                current_value = self._metrics.get("consecutive_losses", 0)
                triggered = current_value >= config.threshold

            elif breaker_type == CircuitBreakerType.VOLATILITY:
                current_value = self._metrics.get("current_volatility", 1.0)
                triggered = current_value >= config.threshold

            elif breaker_type == CircuitBreakerType.ERROR_RATE:
                current_value = self._metrics.get("error_rate", 0)
                triggered = current_value >= config.threshold

            elif breaker_type == CircuitBreakerType.API_FAILURE:
                current_value = self._metrics.get("api_failures", 0)
                triggered = current_value >= config.threshold

            elif breaker_type == CircuitBreakerType.POSITION_LIMIT:
                current_value = self._metrics.get("max_position_pct", 0)
                triggered = current_value >= config.threshold

            if triggered:
                await self._trigger_circuit_breaker(breaker_type, config, current_value)

    async def _trigger_circuit_breaker(
        self,
        breaker_type: CircuitBreakerType,
        config: CircuitBreakerConfig,
        current_value: Any
    ):
        """Trigger a specific circuit breaker."""
        previous_state = self.trading_state
        self.trading_state = TradingState.CIRCUIT_BREAKER
        self.state_reason = f"{breaker_type.value}: {current_value} >= {config.threshold}"
        self.state_changed_at = datetime.utcnow().isoformat()

        if config.auto_reset and config.cooldown_minutes > 0:
            self.circuit_breaker_reset_at = datetime.utcnow() + timedelta(minutes=config.cooldown_minutes)

        event = await self._create_event(
            event_type="CIRCUIT_BREAKER",
            triggered_by=f"circuit_breaker:{breaker_type.value}",
            reason=f"{config.description}. Value: {current_value}, Threshold: {config.threshold}",
            previous_state=previous_state,
            new_state=TradingState.CIRCUIT_BREAKER
        )

        reset_msg = ""
        if config.auto_reset:
            reset_msg = f"\nAuto-reset in: {config.cooldown_minutes} minutes"
        else:
            reset_msg = "\nManual intervention required."

        await self._notify_all(
            f"🔴 **CIRCUIT BREAKER TRIGGERED**\n"
            f"Type: {breaker_type.value}\n"
            f"Reason: {config.description}\n"
            f"Current: {current_value} | Threshold: {config.threshold}"
            f"{reset_msg}",
            severity="warning"
        )

        logger.warning(f"Circuit breaker triggered: {breaker_type.value}")

    async def check_auto_reset(self) -> bool:
        """
        Check if circuit breaker should auto-reset.

        Returns:
            True if reset occurred
        """
        if self.trading_state != TradingState.CIRCUIT_BREAKER:
            return False

        if self.circuit_breaker_reset_at and datetime.utcnow() >= self.circuit_breaker_reset_at:
            await self.resume_trading(
                confirmed_by="auto_reset",
                notes="Circuit breaker cooldown expired"
            )
            return True

        return False

    def configure_circuit_breaker(
        self,
        breaker_type: CircuitBreakerType,
        threshold: float = None,
        cooldown_minutes: int = None,
        auto_reset: bool = None
    ):
        """Update circuit breaker configuration."""
        if breaker_type in self.circuit_breakers:
            config = self.circuit_breakers[breaker_type]
            if threshold is not None:
                config.threshold = threshold
            if cooldown_minutes is not None:
                config.cooldown_minutes = cooldown_minutes
            if auto_reset is not None:
                config.auto_reset = auto_reset

    # =========================================================================
    # MAINTENANCE MODE
    # =========================================================================

    async def enter_maintenance(
        self,
        reason: str,
        triggered_by: str = "operator"
    ) -> EmergencyEvent:
        """Enter maintenance mode for scheduled operations."""
        previous_state = self.trading_state
        self.trading_state = TradingState.MAINTENANCE
        self.state_reason = reason
        self.state_changed_at = datetime.utcnow().isoformat()

        event = await self._create_event(
            event_type="MAINTENANCE_START",
            triggered_by=triggered_by,
            reason=reason,
            previous_state=previous_state,
            new_state=TradingState.MAINTENANCE
        )

        await self._notify_all(
            f"🔧 **MAINTENANCE MODE**\n"
            f"Reason: {reason}\n"
            f"Trading paused for maintenance.",
            severity="info"
        )

        return event

    # =========================================================================
    # STATUS AND REPORTING
    # =========================================================================

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return self.trading_state == TradingState.ACTIVE

    def get_status(self) -> Dict[str, Any]:
        """Get current emergency control status."""
        return {
            "trading_state": self.trading_state.value,
            "is_trading_allowed": self.is_trading_allowed(),
            "state_reason": self.state_reason,
            "state_changed_at": self.state_changed_at,
            "circuit_breaker_reset_at": (
                self.circuit_breaker_reset_at.isoformat()
                if self.circuit_breaker_reset_at else None
            ),
            "metrics": self._metrics,
            "recent_events": len([e for e in self.events if not e.resolved_at])
        }

    def get_status_display(self) -> str:
        """Get formatted status for Discord display."""
        state_emoji = {
            TradingState.ACTIVE: "🟢",
            TradingState.HALTED: "🟡",
            TradingState.KILLED: "🔴",
            TradingState.CIRCUIT_BREAKER: "🟠",
            TradingState.MAINTENANCE: "🔧"
        }

        status = self.get_status()
        emoji = state_emoji.get(self.trading_state, "❓")

        output = [
            f"## {emoji} Trading System Status\n",
            f"**State:** {self.trading_state.value.upper()}",
            f"**Trading Allowed:** {'Yes' if status['is_trading_allowed'] else 'No'}",
        ]

        if self.state_reason:
            output.append(f"**Reason:** {self.state_reason}")

        if self.state_changed_at:
            output.append(f"**Since:** {self.state_changed_at}")

        if self.circuit_breaker_reset_at:
            output.append(f"**Auto-Reset At:** {self.circuit_breaker_reset_at.isoformat()}")

        # Metrics summary
        output.append("\n**Current Metrics:**")
        output.append(f"  Drawdown: {status['metrics'].get('current_drawdown', 0):.2%}")
        output.append(f"  Loss Streak: {status['metrics'].get('consecutive_losses', 0)}")
        output.append(f"  Volatility: {status['metrics'].get('current_volatility', 1.0):.2f}x")
        output.append(f"  Error Rate: {status['metrics'].get('error_rate', 0):.2%}")
        output.append(f"  Max Position: {status['metrics'].get('max_position_pct', 0):.2%}")

        # Circuit breaker status
        output.append("\n**Circuit Breakers:**")
        for breaker_type, config in self.circuit_breakers.items():
            output.append(
                f"  {breaker_type.value}: threshold={config.threshold}, "
                f"cooldown={config.cooldown_minutes}m, auto_reset={config.auto_reset}"
            )

        return "\n".join(output)

    def get_event_history(self, limit: int = 10) -> str:
        """Get formatted event history for Discord display."""
        if not self.events:
            return "No emergency events recorded."

        recent = self.events[-limit:]
        recent.reverse()  # Most recent first

        output = ["## Emergency Event History\n"]

        for event in recent:
            resolved = "✅" if event.resolved_at else "⏳"
            output.append(
                f"{resolved} **{event.event_id}** | {event.event_type}\n"
                f"   {event.timestamp}\n"
                f"   {event.previous_state} → {event.new_state}\n"
                f"   {event.reason[:50]}...\n"
            )

        return "\n".join(output)

    def reset_consecutive_losses(self):
        """Reset consecutive loss counter (after a winning trade)."""
        self._metrics["consecutive_losses"] = 0

    def reset_api_failures(self):
        """Reset API failure counter (after successful API call)."""
        self._metrics["api_failures"] = 0


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_emergency_system: Optional[EmergencyControlSystem] = None


def get_emergency_system() -> EmergencyControlSystem:
    """Get or create the emergency control system instance."""
    global _emergency_system
    if _emergency_system is None:
        _emergency_system = EmergencyControlSystem()
    return _emergency_system


def set_emergency_system(system: EmergencyControlSystem):
    """Set the emergency control system instance."""
    global _emergency_system
    _emergency_system = system
