"""
Data Quality Monitoring for RALPH Agent Ensemble

Monitors and validates data quality across all data sources:
- Data freshness and latency
- Missing value detection
- Outlier detection
- Schema validation
- Cross-source consistency
- Data drift detection

P1: Essential for reliable trading signals.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import statistics

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("data_quality")


class DataSource(Enum):
    """Data sources in the trading system."""
    POLYMARKET_API = "polymarket_api"
    COINBASE_WS = "coinbase_ws"
    BINANCE_WS = "binance_ws"
    MYSQL_CANDLES = "mysql_candles"
    TAAPI_INDICATORS = "taapi_indicators"
    INTERNAL_FEATURES = "internal_features"


class QualityStatus(Enum):
    """Data quality status levels."""
    HEALTHY = "healthy"          # All checks passing
    WARNING = "warning"          # Some minor issues
    DEGRADED = "degraded"        # Significant issues
    CRITICAL = "critical"        # Data unusable
    UNKNOWN = "unknown"          # No recent data


class CheckType(Enum):
    """Types of quality checks."""
    FRESHNESS = "freshness"        # Data age
    COMPLETENESS = "completeness"  # Missing values
    VALIDITY = "validity"          # Schema/type validation
    CONSISTENCY = "consistency"    # Cross-source consistency
    OUTLIER = "outlier"           # Statistical outliers
    DRIFT = "drift"               # Distribution drift


@dataclass
class QualityCheck:
    """A single quality check result."""
    check_id: str
    source: DataSource
    check_type: CheckType
    status: QualityStatus
    timestamp: str

    # Check details
    metric_name: str
    metric_value: float
    threshold: float
    message: str = ""

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "source": self.source.value,
            "check_type": self.check_type.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "message": self.message,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QualityCheck":
        return cls(
            check_id=data["check_id"],
            source=DataSource(data["source"]),
            check_type=CheckType(data["check_type"]),
            status=QualityStatus(data["status"]),
            timestamp=data["timestamp"],
            metric_name=data["metric_name"],
            metric_value=data["metric_value"],
            threshold=data["threshold"],
            message=data.get("message", ""),
            details=data.get("details", {})
        )


@dataclass
class DataSourceHealth:
    """Health status of a data source."""
    source: DataSource
    status: QualityStatus
    last_updated: str
    last_data_received: str

    # Metrics
    freshness_seconds: float = 0.0
    completeness_pct: float = 100.0
    error_rate: float = 0.0
    latency_ms: float = 0.0

    # Recent issues
    recent_issues: List[str] = field(default_factory=list)

    # Stats
    records_today: int = 0
    checks_passed: int = 0
    checks_failed: int = 0


@dataclass
class QualityThreshold:
    """Threshold configuration for quality checks."""
    source: DataSource
    check_type: CheckType

    warning_threshold: float
    critical_threshold: float

    # Direction: True if higher is worse
    higher_is_worse: bool = True


class DataQualityMonitor:
    """
    Central data quality monitoring system for RALPH.

    Provides:
    - Real-time quality monitoring
    - Configurable thresholds
    - Historical tracking
    - Alerting integration
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = [
        # Freshness thresholds (seconds)
        QualityThreshold(DataSource.COINBASE_WS, CheckType.FRESHNESS, 5, 30, True),
        QualityThreshold(DataSource.BINANCE_WS, CheckType.FRESHNESS, 5, 30, True),
        QualityThreshold(DataSource.POLYMARKET_API, CheckType.FRESHNESS, 60, 300, True),
        QualityThreshold(DataSource.MYSQL_CANDLES, CheckType.FRESHNESS, 120, 600, True),
        QualityThreshold(DataSource.TAAPI_INDICATORS, CheckType.FRESHNESS, 60, 300, True),

        # Completeness thresholds (percentage)
        QualityThreshold(DataSource.MYSQL_CANDLES, CheckType.COMPLETENESS, 95, 80, False),
        QualityThreshold(DataSource.INTERNAL_FEATURES, CheckType.COMPLETENESS, 98, 90, False),

        # Validity thresholds (error rate percentage)
        QualityThreshold(DataSource.POLYMARKET_API, CheckType.VALIDITY, 2, 10, True),
        QualityThreshold(DataSource.TAAPI_INDICATORS, CheckType.VALIDITY, 5, 15, True),
    ]

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.quality_file = self.project_dir / "data_quality.json"
        self.history_file = self.project_dir / "quality_history.json"

        # Current state
        self.source_health: Dict[DataSource, DataSourceHealth] = {}
        self.recent_checks: List[QualityCheck] = []
        self.check_counter = 0

        # Thresholds
        self.thresholds: Dict[tuple, QualityThreshold] = {}
        for t in self.DEFAULT_THRESHOLDS:
            self.thresholds[(t.source, t.check_type)] = t

        # Data tracking for drift detection
        self._data_stats: Dict[str, Dict[str, List[float]]] = {}  # source -> metric -> values
        self._stats_window = 1000  # Keep last N values

        # Callbacks
        self._alert_callbacks: List[Callable] = []

        self._lock = asyncio.Lock()
        self._load_state()

    def _load_state(self):
        """Load quality state from file."""
        try:
            if self.quality_file.exists():
                with open(self.quality_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.check_counter = data.get("check_counter", 0)

                    for check_data in data.get("recent_checks", []):
                        check = QualityCheck.from_dict(check_data)
                        self.recent_checks.append(check)

                    logger.info(f"Loaded {len(self.recent_checks)} quality checks")

        except Exception as e:
            logger.error(f"Failed to load quality state: {e}")

    async def _save_state(self):
        """Save quality state to file."""
        async with self._lock:
            try:
                data = {
                    "check_counter": self.check_counter,
                    "recent_checks": [c.to_dict() for c in self.recent_checks[-500:]],
                    "source_health": {
                        s.value: {
                            "status": h.status.value,
                            "last_updated": h.last_updated,
                            "freshness_seconds": h.freshness_seconds,
                            "completeness_pct": h.completeness_pct,
                            "error_rate": h.error_rate
                        }
                        for s, h in self.source_health.items()
                    }
                }

                with open(self.quality_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save quality state: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register a callback for quality alerts."""
        self._alert_callbacks.append(callback)

    async def _send_alert(self, check: QualityCheck):
        """Send alert for quality issue."""
        if check.status not in [QualityStatus.DEGRADED, QualityStatus.CRITICAL]:
            return

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(check)
                else:
                    callback(check)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    # =========================================================================
    # QUALITY CHECKS
    # =========================================================================

    async def check_freshness(
        self,
        source: DataSource,
        last_data_time: datetime
    ) -> QualityCheck:
        """
        Check data freshness for a source.

        Args:
            source: Data source
            last_data_time: Timestamp of most recent data

        Returns:
            Quality check result
        """
        self.check_counter += 1
        now = datetime.utcnow()
        age_seconds = (now - last_data_time).total_seconds()

        # Get threshold
        threshold_key = (source, CheckType.FRESHNESS)
        threshold = self.thresholds.get(threshold_key)

        if threshold:
            if age_seconds >= threshold.critical_threshold:
                status = QualityStatus.CRITICAL
            elif age_seconds >= threshold.warning_threshold:
                status = QualityStatus.WARNING
            else:
                status = QualityStatus.HEALTHY
            threshold_value = threshold.critical_threshold
        else:
            status = QualityStatus.HEALTHY if age_seconds < 60 else QualityStatus.WARNING
            threshold_value = 60

        check = QualityCheck(
            check_id=f"QC-{self.check_counter:06d}",
            source=source,
            check_type=CheckType.FRESHNESS,
            status=status,
            timestamp=now.isoformat(),
            metric_name="age_seconds",
            metric_value=age_seconds,
            threshold=threshold_value,
            message=f"Data is {age_seconds:.1f}s old"
        )

        await self._record_check(check)
        return check

    async def check_completeness(
        self,
        source: DataSource,
        total_expected: int,
        total_present: int,
        field_name: str = "all_fields"
    ) -> QualityCheck:
        """
        Check data completeness.

        Args:
            source: Data source
            total_expected: Expected number of values
            total_present: Actual number of non-null values
            field_name: Name of field being checked

        Returns:
            Quality check result
        """
        self.check_counter += 1
        completeness_pct = (total_present / total_expected * 100) if total_expected > 0 else 0

        threshold_key = (source, CheckType.COMPLETENESS)
        threshold = self.thresholds.get(threshold_key)

        if threshold:
            if completeness_pct <= threshold.critical_threshold:
                status = QualityStatus.CRITICAL
            elif completeness_pct <= threshold.warning_threshold:
                status = QualityStatus.WARNING
            else:
                status = QualityStatus.HEALTHY
            threshold_value = threshold.critical_threshold
        else:
            status = QualityStatus.HEALTHY if completeness_pct >= 95 else QualityStatus.WARNING
            threshold_value = 95

        check = QualityCheck(
            check_id=f"QC-{self.check_counter:06d}",
            source=source,
            check_type=CheckType.COMPLETENESS,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            metric_name=f"completeness_{field_name}",
            metric_value=completeness_pct,
            threshold=threshold_value,
            message=f"{field_name}: {completeness_pct:.1f}% complete ({total_present}/{total_expected})"
        )

        await self._record_check(check)
        return check

    async def check_validity(
        self,
        source: DataSource,
        total_records: int,
        invalid_records: int,
        validation_errors: List[str] = None
    ) -> QualityCheck:
        """
        Check data validity (schema/type validation).

        Args:
            source: Data source
            total_records: Total records checked
            invalid_records: Number of invalid records
            validation_errors: List of error messages

        Returns:
            Quality check result
        """
        self.check_counter += 1
        error_rate = (invalid_records / total_records * 100) if total_records > 0 else 0

        threshold_key = (source, CheckType.VALIDITY)
        threshold = self.thresholds.get(threshold_key)

        if threshold:
            if error_rate >= threshold.critical_threshold:
                status = QualityStatus.CRITICAL
            elif error_rate >= threshold.warning_threshold:
                status = QualityStatus.WARNING
            else:
                status = QualityStatus.HEALTHY
            threshold_value = threshold.critical_threshold
        else:
            status = QualityStatus.HEALTHY if error_rate < 5 else QualityStatus.WARNING
            threshold_value = 5

        check = QualityCheck(
            check_id=f"QC-{self.check_counter:06d}",
            source=source,
            check_type=CheckType.VALIDITY,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            metric_name="error_rate_pct",
            metric_value=error_rate,
            threshold=threshold_value,
            message=f"{invalid_records}/{total_records} invalid records ({error_rate:.2f}%)",
            details={"errors": validation_errors[:10] if validation_errors else []}
        )

        await self._record_check(check)
        return check

    async def check_outliers(
        self,
        source: DataSource,
        metric_name: str,
        value: float,
        z_score_threshold: float = 3.0
    ) -> QualityCheck:
        """
        Check for statistical outliers using z-score.

        Args:
            source: Data source
            metric_name: Name of the metric
            value: Current value
            z_score_threshold: Z-score threshold for outlier detection

        Returns:
            Quality check result
        """
        self.check_counter += 1

        # Track value for stats
        stats_key = f"{source.value}_{metric_name}"
        if stats_key not in self._data_stats:
            self._data_stats[stats_key] = []

        self._data_stats[stats_key].append(value)
        if len(self._data_stats[stats_key]) > self._stats_window:
            self._data_stats[stats_key] = self._data_stats[stats_key][-self._stats_window:]

        # Calculate z-score
        values = self._data_stats[stats_key]
        if len(values) < 30:  # Need minimum samples
            z_score = 0
            status = QualityStatus.UNKNOWN
        else:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values)
            z_score = abs(value - mean) / stdev if stdev > 0 else 0

            if z_score >= z_score_threshold * 1.5:
                status = QualityStatus.CRITICAL
            elif z_score >= z_score_threshold:
                status = QualityStatus.WARNING
            else:
                status = QualityStatus.HEALTHY

        check = QualityCheck(
            check_id=f"QC-{self.check_counter:06d}",
            source=source,
            check_type=CheckType.OUTLIER,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            metric_name=metric_name,
            metric_value=z_score,
            threshold=z_score_threshold,
            message=f"{metric_name}={value:.4f} (z-score={z_score:.2f})",
            details={"value": value, "mean": statistics.mean(values) if len(values) >= 30 else None}
        )

        await self._record_check(check)
        return check

    async def check_drift(
        self,
        source: DataSource,
        metric_name: str,
        current_values: List[float],
        baseline_mean: float,
        baseline_std: float
    ) -> QualityCheck:
        """
        Check for distribution drift.

        Args:
            source: Data source
            metric_name: Name of the metric
            current_values: Recent values
            baseline_mean: Historical mean
            baseline_std: Historical standard deviation

        Returns:
            Quality check result
        """
        self.check_counter += 1

        if not current_values or baseline_std == 0:
            status = QualityStatus.UNKNOWN
            drift_score = 0
        else:
            current_mean = statistics.mean(current_values)
            # Simple drift measure: normalized distance from baseline mean
            drift_score = abs(current_mean - baseline_mean) / baseline_std

            if drift_score >= 3.0:
                status = QualityStatus.CRITICAL
            elif drift_score >= 2.0:
                status = QualityStatus.WARNING
            else:
                status = QualityStatus.HEALTHY

        check = QualityCheck(
            check_id=f"QC-{self.check_counter:06d}",
            source=source,
            check_type=CheckType.DRIFT,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            metric_name=metric_name,
            metric_value=drift_score,
            threshold=2.0,
            message=f"{metric_name} drift score: {drift_score:.2f}",
            details={
                "current_mean": statistics.mean(current_values) if current_values else None,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std
            }
        )

        await self._record_check(check)
        return check

    async def _record_check(self, check: QualityCheck):
        """Record a quality check and update source health."""
        self.recent_checks.append(check)

        # Trim history
        if len(self.recent_checks) > 1000:
            self.recent_checks = self.recent_checks[-1000:]

        # Update source health
        if check.source not in self.source_health:
            self.source_health[check.source] = DataSourceHealth(
                source=check.source,
                status=QualityStatus.UNKNOWN,
                last_updated="",
                last_data_received=""
            )

        health = self.source_health[check.source]
        health.last_updated = check.timestamp

        if check.status in [QualityStatus.HEALTHY]:
            health.checks_passed += 1
        else:
            health.checks_failed += 1
            health.recent_issues.append(f"{check.check_type.value}: {check.message}")
            health.recent_issues = health.recent_issues[-10:]  # Keep last 10

        # Update source status based on recent checks
        recent_for_source = [
            c for c in self.recent_checks[-50:]
            if c.source == check.source
        ]

        if recent_for_source:
            critical_count = sum(1 for c in recent_for_source if c.status == QualityStatus.CRITICAL)
            warning_count = sum(1 for c in recent_for_source if c.status == QualityStatus.WARNING)

            if critical_count >= 3:
                health.status = QualityStatus.CRITICAL
            elif critical_count >= 1 or warning_count >= 5:
                health.status = QualityStatus.DEGRADED
            elif warning_count >= 2:
                health.status = QualityStatus.WARNING
            else:
                health.status = QualityStatus.HEALTHY

        # Update specific metrics
        if check.check_type == CheckType.FRESHNESS:
            health.freshness_seconds = check.metric_value
        elif check.check_type == CheckType.COMPLETENESS:
            health.completeness_pct = check.metric_value
        elif check.check_type == CheckType.VALIDITY:
            health.error_rate = check.metric_value

        await self._save_state()

        # Send alert if needed
        await self._send_alert(check)

    # =========================================================================
    # THRESHOLD MANAGEMENT
    # =========================================================================

    def configure_threshold(
        self,
        source: DataSource,
        check_type: CheckType,
        warning: float = None,
        critical: float = None
    ):
        """Update threshold configuration."""
        key = (source, check_type)
        if key not in self.thresholds:
            self.thresholds[key] = QualityThreshold(
                source=source,
                check_type=check_type,
                warning_threshold=warning or 0,
                critical_threshold=critical or 0
            )
        else:
            if warning is not None:
                self.thresholds[key].warning_threshold = warning
            if critical is not None:
                self.thresholds[key].critical_threshold = critical

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_dashboard(self) -> str:
        """Get data quality dashboard for Discord."""
        output = ["## 📊 Data Quality Dashboard\n"]

        if not self.source_health:
            return "No data quality checks recorded yet."

        status_emoji = {
            QualityStatus.HEALTHY: "🟢",
            QualityStatus.WARNING: "🟡",
            QualityStatus.DEGRADED: "🟠",
            QualityStatus.CRITICAL: "🔴",
            QualityStatus.UNKNOWN: "⚪"
        }

        output.append("### Source Health")
        for source, health in sorted(self.source_health.items(), key=lambda x: x[0].value):
            emoji = status_emoji.get(health.status, "❓")
            output.append(f"{emoji} **{source.value}** - {health.status.value}")
            output.append(f"   Freshness: {health.freshness_seconds:.1f}s | Completeness: {health.completeness_pct:.1f}%")

        # Recent issues summary
        all_issues = []
        for source, health in self.source_health.items():
            for issue in health.recent_issues[-3:]:
                all_issues.append(f"  • [{source.value}] {issue}")

        if all_issues:
            output.append("\n### Recent Issues")
            for issue in all_issues[:5]:
                output.append(issue)

        # Overall status
        critical_sources = [s for s, h in self.source_health.items() if h.status == QualityStatus.CRITICAL]
        if critical_sources:
            output.append(f"\n⚠️ **Critical issues in:** {', '.join(s.value for s in critical_sources)}")

        return "\n".join(output)

    def get_source_details(self, source: DataSource) -> str:
        """Get detailed quality info for a source."""
        if source not in self.source_health:
            return f"No data for {source.value}"

        health = self.source_health[source]

        output = [f"## Data Quality: {source.value}\n"]

        status_emoji = {
            QualityStatus.HEALTHY: "🟢",
            QualityStatus.WARNING: "🟡",
            QualityStatus.DEGRADED: "🟠",
            QualityStatus.CRITICAL: "🔴",
            QualityStatus.UNKNOWN: "⚪"
        }

        output.append(f"**Status:** {status_emoji.get(health.status, '❓')} {health.status.value}")
        output.append(f"**Last Updated:** {health.last_updated}")
        output.append("")
        output.append("**Metrics:**")
        output.append(f"  Freshness: {health.freshness_seconds:.1f}s")
        output.append(f"  Completeness: {health.completeness_pct:.1f}%")
        output.append(f"  Error Rate: {health.error_rate:.2f}%")
        output.append(f"  Latency: {health.latency_ms:.1f}ms")
        output.append("")
        output.append(f"**Checks:** {health.checks_passed} passed, {health.checks_failed} failed")

        # Recent checks for this source
        recent = [c for c in self.recent_checks[-20:] if c.source == source]
        if recent:
            output.append("\n**Recent Checks:**")
            for check in recent[-5:]:
                emoji = status_emoji.get(check.status, "❓")
                output.append(f"  {emoji} {check.check_type.value}: {check.message}")

        if health.recent_issues:
            output.append("\n**Recent Issues:**")
            for issue in health.recent_issues[-5:]:
                output.append(f"  • {issue}")

        return "\n".join(output)

    def get_quality_history(self, source: DataSource = None, limit: int = 20) -> str:
        """Get quality check history."""
        checks = self.recent_checks
        if source:
            checks = [c for c in checks if c.source == source]

        checks = list(reversed(checks[-limit:]))

        if not checks:
            return "No quality checks recorded."

        output = ["## Quality Check History\n"]

        for check in checks:
            status_emoji = {
                QualityStatus.HEALTHY: "🟢",
                QualityStatus.WARNING: "🟡",
                QualityStatus.DEGRADED: "🟠",
                QualityStatus.CRITICAL: "🔴",
                QualityStatus.UNKNOWN: "⚪"
            }
            emoji = status_emoji.get(check.status, "❓")
            output.append(
                f"{emoji} `{check.check_id}` [{check.source.value}] "
                f"{check.check_type.value}: {check.message}"
            )

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_data_quality_monitor: Optional[DataQualityMonitor] = None


def get_data_quality_monitor() -> DataQualityMonitor:
    """Get or create the data quality monitor instance."""
    global _data_quality_monitor
    if _data_quality_monitor is None:
        _data_quality_monitor = DataQualityMonitor()
    return _data_quality_monitor


def set_data_quality_monitor(monitor: DataQualityMonitor):
    """Set the data quality monitor instance."""
    global _data_quality_monitor
    _data_quality_monitor = monitor
