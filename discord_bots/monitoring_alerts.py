"""
Real-time Monitoring & Alerts System for RALPH Agent Ensemble

Provides comprehensive monitoring for:
- System health and performance
- Trading metrics and P&L
- Agent activity and errors
- Resource usage
- External API health

P0 Critical: Essential for production operation.
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

logger = logging.getLogger("monitoring_alerts")


class AlertSeverity(Enum):
    """Alert severity levels."""
    DEBUG = "debug"        # Development/debugging info
    INFO = "info"          # Informational
    WARNING = "warning"    # Attention needed
    ERROR = "error"        # Error occurred
    CRITICAL = "critical"  # Immediate action required


class AlertCategory(Enum):
    """Categories of alerts."""
    TRADING = "trading"           # Trading-related alerts
    SYSTEM = "system"             # System health alerts
    AGENT = "agent"               # Agent activity alerts
    DATA = "data"                 # Data quality alerts
    API = "api"                   # External API alerts
    PERFORMANCE = "performance"   # Performance metrics
    SECURITY = "security"         # Security-related alerts


class MetricType(Enum):
    """Types of metrics tracked."""
    # Trading metrics
    PORTFOLIO_VALUE = "portfolio_value"
    DAILY_PNL = "daily_pnl"
    TOTAL_PNL = "total_pnl"
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    DRAWDOWN = "drawdown"
    OPEN_POSITIONS = "open_positions"
    TRADE_COUNT = "trade_count"

    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_LATENCY = "network_latency"

    # Agent metrics
    AGENT_ACTIVE_COUNT = "agent_active_count"
    AGENT_ERROR_COUNT = "agent_error_count"
    TASK_QUEUE_SIZE = "task_queue_size"

    # API metrics
    API_RESPONSE_TIME = "api_response_time"
    API_ERROR_RATE = "api_error_rate"
    API_CALL_COUNT = "api_call_count"

    # Data metrics
    DATA_FRESHNESS = "data_freshness"
    DATA_QUALITY_SCORE = "data_quality_score"


@dataclass
class AlertThreshold:
    """Threshold configuration for an alert."""
    metric_type: MetricType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # "gt" (greater than), "lt" (less than), "eq" (equal)
    cooldown_minutes: int = 5  # Minimum time between repeated alerts


@dataclass
class Alert:
    """An alert instance."""
    alert_id: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_type: Optional[MetricType] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: str = ""
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: str = ""
    resolved: bool = False
    resolved_at: str = ""
    resolution_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "metric_type": self.metric_type.value if self.metric_type else None,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
            "resolution_notes": self.resolution_notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Alert":
        return cls(
            alert_id=data["alert_id"],
            severity=AlertSeverity(data["severity"]),
            category=AlertCategory(data["category"]),
            title=data["title"],
            message=data["message"],
            metric_type=MetricType(data["metric_type"]) if data.get("metric_type") else None,
            metric_value=data.get("metric_value"),
            threshold=data.get("threshold"),
            timestamp=data.get("timestamp", ""),
            acknowledged=data.get("acknowledged", False),
            acknowledged_by=data.get("acknowledged_by", ""),
            acknowledged_at=data.get("acknowledged_at", ""),
            resolved=data.get("resolved", False),
            resolved_at=data.get("resolved_at", ""),
            resolution_notes=data.get("resolution_notes", "")
        )

    def format_for_discord(self) -> str:
        """Format alert for Discord display."""
        severity_emoji = {
            AlertSeverity.DEBUG: "🔍",
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }

        status = "✅ Resolved" if self.resolved else ("👀 Acknowledged" if self.acknowledged else "⏳ Active")

        output = [
            f"{severity_emoji[self.severity]} **{self.title}**",
            f"ID: `{self.alert_id}` | {self.category.value} | {status}",
            f"",
            f"{self.message}",
        ]

        if self.metric_value is not None:
            output.append(f"**Value:** {self.metric_value} | **Threshold:** {self.threshold}")

        output.append(f"**Time:** {self.timestamp}")

        return "\n".join(output)


@dataclass
class MetricSnapshot:
    """A snapshot of a metric at a point in time."""
    metric_type: MetricType
    value: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MonitoringSystem:
    """
    Central monitoring system for RALPH.

    Provides:
    - Real-time metric tracking
    - Threshold-based alerting
    - Historical metric storage
    - Alert management
    """

    # Default alert thresholds
    DEFAULT_THRESHOLDS = [
        AlertThreshold(MetricType.DRAWDOWN, 0.10, 0.15, 0.20, "gt", 15),
        AlertThreshold(MetricType.WIN_RATE, 0.45, 0.40, 0.35, "lt", 30),
        AlertThreshold(MetricType.SHARPE_RATIO, 0.8, 0.5, 0.3, "lt", 60),
        AlertThreshold(MetricType.CPU_USAGE, 70, 85, 95, "gt", 5),
        AlertThreshold(MetricType.MEMORY_USAGE, 70, 85, 95, "gt", 5),
        AlertThreshold(MetricType.API_ERROR_RATE, 0.05, 0.10, 0.20, "gt", 5),
        AlertThreshold(MetricType.API_RESPONSE_TIME, 1000, 2000, 5000, "gt", 5),  # ms
        AlertThreshold(MetricType.DATA_FRESHNESS, 60, 300, 600, "gt", 10),  # seconds
        AlertThreshold(MetricType.AGENT_ERROR_COUNT, 3, 5, 10, "gt", 5),
    ]

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.alerts_file = self.project_dir / "alerts.json"
        self.metrics_file = self.project_dir / "metrics_history.json"

        # Current metrics
        self.current_metrics: Dict[MetricType, MetricSnapshot] = {}

        # Metric history (in-memory, limited)
        self.metrics_history: Dict[MetricType, List[MetricSnapshot]] = {}
        self.history_limit = 1000  # Per metric

        # Alerts
        self.alerts: List[Alert] = []
        self.alert_counter = 0

        # Thresholds
        self.thresholds: Dict[MetricType, AlertThreshold] = {
            t.metric_type: t for t in self.DEFAULT_THRESHOLDS
        }

        # Cooldown tracking
        self._last_alert_times: Dict[str, datetime] = {}

        # Notification callbacks
        self._notification_callbacks: List[Callable] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Load persisted data
        self._load_data()

    def _load_data(self):
        """Load persisted alerts and metrics."""
        try:
            if self.alerts_file.exists():
                with open(self.alerts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.alerts = [Alert.from_dict(a) for a in data.get("alerts", [])]
                    self.alert_counter = data.get("counter", 0)
                    logger.info(f"Loaded {len(self.alerts)} alerts")

        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")

    async def _save_data(self):
        """Save alerts to file."""
        async with self._lock:
            try:
                data = {
                    "counter": self.alert_counter,
                    "alerts": [a.to_dict() for a in self.alerts[-500:]]  # Keep last 500
                }
                with open(self.alerts_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save alerts: {e}")

    def register_notification_callback(self, callback: Callable):
        """Register a callback for alert notifications."""
        self._notification_callbacks.append(callback)

    async def _notify(self, alert: Alert):
        """Send alert notifications."""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    # =========================================================================
    # METRIC TRACKING
    # =========================================================================

    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        metadata: Dict[str, Any] = None
    ) -> Optional[Alert]:
        """
        Record a metric value.

        Args:
            metric_type: Type of metric
            value: Metric value
            metadata: Optional additional data

        Returns:
            Alert if threshold was exceeded, else None
        """
        snapshot = MetricSnapshot(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )

        # Update current
        self.current_metrics[metric_type] = snapshot

        # Add to history
        if metric_type not in self.metrics_history:
            self.metrics_history[metric_type] = []
        self.metrics_history[metric_type].append(snapshot)

        # Trim history
        if len(self.metrics_history[metric_type]) > self.history_limit:
            self.metrics_history[metric_type] = self.metrics_history[metric_type][-self.history_limit:]

        # Check thresholds
        alert = await self._check_threshold(metric_type, value)
        return alert

    async def record_metrics_batch(self, metrics: Dict[MetricType, float]) -> List[Alert]:
        """Record multiple metrics at once."""
        alerts = []
        for metric_type, value in metrics.items():
            alert = await self.record_metric(metric_type, value)
            if alert:
                alerts.append(alert)
        return alerts

    async def _check_threshold(
        self,
        metric_type: MetricType,
        value: float
    ) -> Optional[Alert]:
        """Check if a metric exceeds its threshold."""
        if metric_type not in self.thresholds:
            return None

        threshold = self.thresholds[metric_type]

        # Determine severity based on value
        severity = None
        exceeded_threshold = None

        if threshold.comparison == "gt":
            if value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                exceeded_threshold = threshold.critical_threshold
            elif value >= threshold.error_threshold:
                severity = AlertSeverity.ERROR
                exceeded_threshold = threshold.error_threshold
            elif value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                exceeded_threshold = threshold.warning_threshold

        elif threshold.comparison == "lt":
            if value <= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                exceeded_threshold = threshold.critical_threshold
            elif value <= threshold.error_threshold:
                severity = AlertSeverity.ERROR
                exceeded_threshold = threshold.error_threshold
            elif value <= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                exceeded_threshold = threshold.warning_threshold

        if severity is None:
            return None

        # Check cooldown
        cooldown_key = f"{metric_type.value}_{severity.value}"
        last_alert = self._last_alert_times.get(cooldown_key)
        if last_alert:
            elapsed = datetime.utcnow() - last_alert
            if elapsed.total_seconds() < threshold.cooldown_minutes * 60:
                return None  # Still in cooldown

        # Create alert
        alert = await self._create_alert(
            severity=severity,
            category=self._get_category_for_metric(metric_type),
            title=f"{metric_type.value} threshold exceeded",
            message=f"{metric_type.value} = {value} (threshold: {exceeded_threshold})",
            metric_type=metric_type,
            metric_value=value,
            threshold=exceeded_threshold
        )

        self._last_alert_times[cooldown_key] = datetime.utcnow()
        return alert

    def _get_category_for_metric(self, metric_type: MetricType) -> AlertCategory:
        """Get the alert category for a metric type."""
        trading_metrics = [
            MetricType.PORTFOLIO_VALUE, MetricType.DAILY_PNL, MetricType.TOTAL_PNL,
            MetricType.WIN_RATE, MetricType.SHARPE_RATIO, MetricType.DRAWDOWN,
            MetricType.OPEN_POSITIONS, MetricType.TRADE_COUNT
        ]
        system_metrics = [
            MetricType.CPU_USAGE, MetricType.MEMORY_USAGE,
            MetricType.DISK_USAGE, MetricType.NETWORK_LATENCY
        ]
        agent_metrics = [
            MetricType.AGENT_ACTIVE_COUNT, MetricType.AGENT_ERROR_COUNT,
            MetricType.TASK_QUEUE_SIZE
        ]
        api_metrics = [
            MetricType.API_RESPONSE_TIME, MetricType.API_ERROR_RATE,
            MetricType.API_CALL_COUNT
        ]
        data_metrics = [
            MetricType.DATA_FRESHNESS, MetricType.DATA_QUALITY_SCORE
        ]

        if metric_type in trading_metrics:
            return AlertCategory.TRADING
        elif metric_type in system_metrics:
            return AlertCategory.SYSTEM
        elif metric_type in agent_metrics:
            return AlertCategory.AGENT
        elif metric_type in api_metrics:
            return AlertCategory.API
        elif metric_type in data_metrics:
            return AlertCategory.DATA
        else:
            return AlertCategory.PERFORMANCE

    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================

    async def _create_alert(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        metric_type: Optional[MetricType] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None
    ) -> Alert:
        """Create and store a new alert."""
        self.alert_counter += 1
        alert = Alert(
            alert_id=f"ALT-{self.alert_counter:05d}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            metric_type=metric_type,
            metric_value=metric_value,
            threshold=threshold,
            timestamp=datetime.utcnow().isoformat()
        )

        self.alerts.append(alert)
        await self._save_data()

        # Send notification
        await self._notify(alert)

        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else
            logging.ERROR if severity == AlertSeverity.ERROR else
            logging.WARNING if severity == AlertSeverity.WARNING else
            logging.INFO,
            f"Alert {alert.alert_id}: {title}"
        )

        return alert

    async def create_manual_alert(
        self,
        severity: str,
        category: str,
        title: str,
        message: str
    ) -> Alert:
        """Create a manual alert (not from threshold)."""
        return await self._create_alert(
            severity=AlertSeverity(severity),
            category=AlertCategory(category),
            title=title,
            message=message
        )

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow().isoformat()
                await self._save_data()
                return alert
        return None

    async def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: str = ""
    ) -> Optional[Alert]:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow().isoformat()
                alert.resolution_notes = resolution_notes
                await self._save_data()
                return alert
        return None

    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self.alerts if not a.resolved]

    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical unresolved alerts."""
        return [
            a for a in self.alerts
            if not a.resolved and a.severity == AlertSeverity.CRITICAL
        ]

    def get_alerts_by_category(self, category: AlertCategory) -> List[Alert]:
        """Get alerts by category."""
        return [a for a in self.alerts if a.category == category]

    # =========================================================================
    # THRESHOLD MANAGEMENT
    # =========================================================================

    def configure_threshold(
        self,
        metric_type: MetricType,
        warning: float = None,
        error: float = None,
        critical: float = None,
        comparison: str = None,
        cooldown: int = None
    ):
        """Update threshold configuration."""
        if metric_type not in self.thresholds:
            self.thresholds[metric_type] = AlertThreshold(
                metric_type=metric_type,
                warning_threshold=warning or 0,
                error_threshold=error or 0,
                critical_threshold=critical or 0,
                comparison=comparison or "gt"
            )
        else:
            threshold = self.thresholds[metric_type]
            if warning is not None:
                threshold.warning_threshold = warning
            if error is not None:
                threshold.error_threshold = error
            if critical is not None:
                threshold.critical_threshold = critical
            if comparison is not None:
                threshold.comparison = comparison
            if cooldown is not None:
                threshold.cooldown_minutes = cooldown

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_dashboard(self) -> str:
        """Get formatted monitoring dashboard for Discord."""
        output = ["## 📊 RALPH Monitoring Dashboard\n"]

        # Current metrics summary
        output.append("### Current Metrics")

        trading_metrics = [
            MetricType.PORTFOLIO_VALUE, MetricType.DAILY_PNL,
            MetricType.WIN_RATE, MetricType.SHARPE_RATIO, MetricType.DRAWDOWN
        ]

        for metric_type in trading_metrics:
            if metric_type in self.current_metrics:
                snapshot = self.current_metrics[metric_type]
                output.append(f"  {metric_type.value}: {snapshot.value:.4f}")

        # System health
        output.append("\n### System Health")
        system_metrics = [
            MetricType.CPU_USAGE, MetricType.MEMORY_USAGE, MetricType.API_RESPONSE_TIME
        ]

        for metric_type in system_metrics:
            if metric_type in self.current_metrics:
                snapshot = self.current_metrics[metric_type]
                output.append(f"  {metric_type.value}: {snapshot.value:.2f}")

        # Active alerts summary
        active = self.get_active_alerts()
        critical = [a for a in active if a.severity == AlertSeverity.CRITICAL]
        errors = [a for a in active if a.severity == AlertSeverity.ERROR]
        warnings = [a for a in active if a.severity == AlertSeverity.WARNING]

        output.append("\n### Alerts Summary")
        output.append(f"  🚨 Critical: {len(critical)}")
        output.append(f"  ❌ Errors: {len(errors)}")
        output.append(f"  ⚠️ Warnings: {len(warnings)}")

        if critical:
            output.append("\n**Critical Alerts:**")
            for alert in critical[:3]:
                output.append(f"  • `{alert.alert_id}`: {alert.title}")

        return "\n".join(output)

    def get_alerts_display(self, limit: int = 10) -> str:
        """Get formatted alerts list for Discord."""
        active = self.get_active_alerts()

        if not active:
            return "✅ No active alerts!"

        # Sort by severity (critical first)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
            AlertSeverity.DEBUG: 4
        }
        active.sort(key=lambda a: severity_order[a.severity])

        output = [f"## Active Alerts ({len(active)})\n"]

        for alert in active[:limit]:
            output.append(alert.format_for_discord())
            output.append("---")

        if len(active) > limit:
            output.append(f"*...and {len(active) - limit} more alerts*")

        output.append("\n**Commands:**")
        output.append("`!ack <id>` - Acknowledge alert")
        output.append("`!resolve <id> [notes]` - Resolve alert")

        return "\n".join(output)

    def get_metrics_history_display(
        self,
        metric_type: MetricType,
        limit: int = 20
    ) -> str:
        """Get formatted metric history for Discord."""
        if metric_type not in self.metrics_history:
            return f"No history for {metric_type.value}"

        history = self.metrics_history[metric_type][-limit:]

        output = [f"## {metric_type.value} History\n"]

        for snapshot in reversed(history):
            output.append(f"{snapshot.timestamp}: {snapshot.value:.4f}")

        # Basic stats
        values = [s.value for s in history]
        if values:
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            output.append(f"\n**Stats:** Avg={avg:.4f} Min={min_val:.4f} Max={max_val:.4f}")

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> MonitoringSystem:
    """Get or create the monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system


def set_monitoring_system(system: MonitoringSystem):
    """Set the monitoring system instance."""
    global _monitoring_system
    _monitoring_system = system
