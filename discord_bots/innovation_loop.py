"""
Innovation Loop for RALPH Agent Ensemble

Autonomous self-improvement system that continuously monitors performance,
detects anomalies, generates improvement hypotheses, and proposes experiments.

Key Features:
- Periodic metrics collection from all sources
- Anomaly detection (regression, spikes, threshold breaches)
- Hypothesis generation for improvements
- Experiment proposal system with operator approval
- Learning documentation for continuous improvement

The innovation loop runs as a background task with configurable cycle time.
All experiments require operator approval before execution.

Usage:
    loop = InnovationLoop(orchestrator)
    await loop.start()  # Begins background innovation cycles
"""

import asyncio
import logging
import os
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger("innovation_loop")


class ExperimentStatus(Enum):
    """Lifecycle status of an experiment."""
    PROPOSED = "proposed"      # Awaiting operator approval
    APPROVED = "approved"      # Approved, ready to execute
    REJECTED = "rejected"      # Rejected by operator
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Finished successfully
    FAILED = "failed"          # Failed during execution
    ROLLED_BACK = "rolled_back"  # Rolled back due to failure


class AnomalyType(Enum):
    """Types of detected anomalies."""
    REGRESSION = "regression"       # Performance degradation
    SPIKE = "spike"                 # Sudden value spike
    THRESHOLD_BREACH = "threshold"  # Exceeded threshold
    PATTERN_CHANGE = "pattern"      # Behavioral pattern change
    STALE_DATA = "stale"           # Data freshness issue


@dataclass
class Metric:
    """A collected metric value."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected anomaly in metrics."""
    type: AnomalyType
    metric_name: str
    current_value: float
    expected_value: float
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Hypothesis:
    """An improvement hypothesis."""
    id: str
    title: str
    description: str
    trigger: str  # What caused this hypothesis
    expected_improvement: str
    implementation_steps: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Experiment:
    """An experiment proposal or execution record."""
    id: str
    hypothesis: Hypothesis
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": {
                "id": self.hypothesis.id,
                "title": self.hypothesis.title,
                "description": self.hypothesis.description,
                "trigger": self.hypothesis.trigger,
                "expected_improvement": self.hypothesis.expected_improvement,
                "implementation_steps": self.hypothesis.implementation_steps,
                "risk_level": self.hypothesis.risk_level
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "approved_by": self.approved_by,
            "rejection_reason": self.rejection_reason,
            "results": self.results,
            "baseline_metrics": self.baseline_metrics,
            "final_metrics": self.final_metrics
        }


class MetricAggregator:
    """
    Collects metrics from various sources.

    Sources:
    - Mission efficiency (completion time, success rate)
    - Agent performance (error rates, task durations)
    - System health (API latency, queue sizes)
    - Knowledge base (learning usage, staleness)
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._history: List[Dict[str, float]] = []
        self._max_history = 100

    async def collect_all(self) -> Dict[str, Metric]:
        """Collect all available metrics."""
        metrics = {}

        # Mission metrics
        mission_metrics = await self._collect_mission_metrics()
        metrics.update(mission_metrics)

        # Parallel execution metrics
        parallel_metrics = await self._collect_parallel_metrics()
        metrics.update(parallel_metrics)

        # Knowledge base metrics
        kb_metrics = await self._collect_kb_metrics()
        metrics.update(kb_metrics)

        # Token optimization metrics
        token_metrics = await self._collect_token_metrics()
        metrics.update(token_metrics)

        # Store in history
        self._history.append({
            name: m.value for name, m in metrics.items()
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return metrics

    async def _collect_mission_metrics(self) -> Dict[str, Metric]:
        """Collect mission-related metrics."""
        metrics = {}

        try:
            from mission_manager import get_mission_manager
            manager = get_mission_manager()

            if manager.current_mission:
                progress = manager.current_mission.get_progress()
                metrics["mission_completion_pct"] = Metric(
                    name="mission_completion_pct",
                    value=progress["percent"],
                    source="mission_manager"
                )
                metrics["mission_tasks_pending"] = Metric(
                    name="mission_tasks_pending",
                    value=progress["pending"],
                    source="mission_manager"
                )

        except Exception as e:
            logger.debug(f"Could not collect mission metrics: {e}")

        return metrics

    async def _collect_parallel_metrics(self) -> Dict[str, Metric]:
        """Collect parallel execution metrics."""
        metrics = {}

        try:
            from parallel_task_tracker import get_parallel_tracker
            tracker = get_parallel_tracker()

            if tracker:
                stats = tracker.get_stats()
                metrics["parallel_current_concurrent"] = Metric(
                    name="parallel_current_concurrent",
                    value=stats["current_concurrent"],
                    source="parallel_tracker"
                )
                metrics["parallel_max_concurrent"] = Metric(
                    name="parallel_max_concurrent",
                    value=stats["max_concurrent"],
                    source="parallel_tracker"
                )
                metrics["parallel_tasks_failed"] = Metric(
                    name="parallel_tasks_failed",
                    value=stats["tasks_failed"],
                    source="parallel_tracker"
                )

        except Exception as e:
            logger.debug(f"Could not collect parallel metrics: {e}")

        return metrics

    async def _collect_kb_metrics(self) -> Dict[str, Metric]:
        """Collect knowledge base metrics."""
        metrics = {}

        try:
            from knowledge_base import get_knowledge_base
            kb = get_knowledge_base()

            if kb:
                stats = kb.get_stats()
                metrics["kb_learnings_count"] = Metric(
                    name="kb_learnings_count",
                    value=stats.get("total_learnings", 0),
                    source="knowledge_base"
                )
                metrics["kb_missions_documented"] = Metric(
                    name="kb_missions_documented",
                    value=stats.get("missions_documented", 0),
                    source="knowledge_base"
                )

        except Exception as e:
            logger.debug(f"Could not collect KB metrics: {e}")

        return metrics

    async def _collect_token_metrics(self) -> Dict[str, Metric]:
        """Collect token optimization metrics."""
        metrics = {}

        try:
            from token_optimizer import get_token_optimizer
            optimizer = get_token_optimizer()

            if optimizer:
                stats = optimizer.get_stats()
                metrics["token_savings_pct"] = Metric(
                    name="token_savings_pct",
                    value=stats.get("savings_percentage", 0),
                    source="token_optimizer"
                )
                metrics["token_cache_hits"] = Metric(
                    name="token_cache_hits",
                    value=stats.get("cache_hits", 0),
                    source="token_optimizer"
                )

        except Exception as e:
            logger.debug(f"Could not collect token metrics: {e}")

        return metrics

    def get_metric_history(self, metric_name: str, window: int = 10) -> List[float]:
        """Get recent history for a metric."""
        return [
            h.get(metric_name, 0) for h in self._history[-window:]
            if metric_name in h
        ]


class AnomalyDetector:
    """
    Detects anomalies in collected metrics.

    Detection methods:
    - Threshold breaches (absolute limits)
    - Statistical anomalies (>2 std deviations)
    - Trend changes (regression detection)
    """

    def __init__(self):
        self.thresholds = {
            "parallel_tasks_failed": {"max": 5, "severity": "high"},
            "mission_tasks_pending": {"max": 20, "severity": "medium"},
            "token_savings_pct": {"min": 10, "severity": "low"},
        }

    async def analyze(
        self,
        metrics: Dict[str, Metric],
        history: List[Dict[str, float]] = None
    ) -> List[Anomaly]:
        """Analyze metrics for anomalies."""
        anomalies = []

        # Check thresholds
        for metric_name, thresholds in self.thresholds.items():
            if metric_name not in metrics:
                continue

            value = metrics[metric_name].value

            if "max" in thresholds and value > thresholds["max"]:
                anomalies.append(Anomaly(
                    type=AnomalyType.THRESHOLD_BREACH,
                    metric_name=metric_name,
                    current_value=value,
                    expected_value=thresholds["max"],
                    severity=thresholds.get("severity", "medium"),
                    description=f"{metric_name} exceeded max threshold: {value} > {thresholds['max']}"
                ))

            if "min" in thresholds and value < thresholds["min"]:
                anomalies.append(Anomaly(
                    type=AnomalyType.THRESHOLD_BREACH,
                    metric_name=metric_name,
                    current_value=value,
                    expected_value=thresholds["min"],
                    severity=thresholds.get("severity", "medium"),
                    description=f"{metric_name} below min threshold: {value} < {thresholds['min']}"
                ))

        # Statistical analysis if history available
        if history and len(history) >= 5:
            for metric_name, metric in metrics.items():
                values = [h.get(metric_name) for h in history if metric_name in h]
                if len(values) < 5:
                    continue

                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5

                if std_dev > 0:
                    z_score = (metric.value - mean) / std_dev
                    if abs(z_score) > 2:
                        anomalies.append(Anomaly(
                            type=AnomalyType.SPIKE if z_score > 0 else AnomalyType.REGRESSION,
                            metric_name=metric_name,
                            current_value=metric.value,
                            expected_value=mean,
                            severity="medium" if abs(z_score) < 3 else "high",
                            description=f"{metric_name} statistical anomaly: z-score={z_score:.2f}"
                        ))

        return anomalies


class HypothesisGenerator:
    """
    Generates improvement hypotheses based on anomalies and patterns.

    Templates:
    - High error rate → Add retry logic or error handling
    - Low parallelism → Increase concurrency limits
    - Stale learnings → Trigger revalidation
    - Slow missions → Identify bottlenecks
    """

    def __init__(self):
        self._hypothesis_counter = 0
        self.templates = {
            AnomalyType.THRESHOLD_BREACH: self._threshold_hypothesis,
            AnomalyType.REGRESSION: self._regression_hypothesis,
            AnomalyType.SPIKE: self._spike_hypothesis,
        }

    async def generate(
        self,
        metrics: Dict[str, Metric],
        anomalies: List[Anomaly]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on metrics and anomalies."""
        hypotheses = []

        for anomaly in anomalies:
            template = self.templates.get(anomaly.type)
            if template:
                hypothesis = template(anomaly, metrics)
                if hypothesis:
                    hypotheses.append(hypothesis)

        return hypotheses

    def _threshold_hypothesis(
        self,
        anomaly: Anomaly,
        metrics: Dict[str, Metric]
    ) -> Optional[Hypothesis]:
        """Generate hypothesis for threshold breach."""
        self._hypothesis_counter += 1

        if "failed" in anomaly.metric_name:
            return Hypothesis(
                id=f"H-{self._hypothesis_counter:04d}",
                title=f"Reduce {anomaly.metric_name}",
                description=f"High failure rate detected: {anomaly.current_value}",
                trigger=anomaly.description,
                expected_improvement="Reduce task failures by improving error handling",
                implementation_steps=[
                    "Review recent task failure logs",
                    "Identify common failure patterns",
                    "Add targeted retry logic or validation",
                    "Monitor for improvement"
                ],
                risk_level="medium"
            )

        return None

    def _regression_hypothesis(
        self,
        anomaly: Anomaly,
        metrics: Dict[str, Metric]
    ) -> Optional[Hypothesis]:
        """Generate hypothesis for regression."""
        self._hypothesis_counter += 1

        return Hypothesis(
            id=f"H-{self._hypothesis_counter:04d}",
            title=f"Investigate {anomaly.metric_name} regression",
            description=f"Performance regression detected: {anomaly.current_value:.2f} vs expected {anomaly.expected_value:.2f}",
            trigger=anomaly.description,
            expected_improvement="Restore performance to baseline levels",
            implementation_steps=[
                "Review recent code changes",
                "Check for configuration drift",
                "Profile bottlenecks",
                "Revert or optimize as needed"
            ],
            risk_level="low"
        )

    def _spike_hypothesis(
        self,
        anomaly: Anomaly,
        metrics: Dict[str, Metric]
    ) -> Optional[Hypothesis]:
        """Generate hypothesis for spike."""
        self._hypothesis_counter += 1

        return Hypothesis(
            id=f"H-{self._hypothesis_counter:04d}",
            title=f"Investigate {anomaly.metric_name} spike",
            description=f"Unusual spike detected: {anomaly.current_value:.2f}",
            trigger=anomaly.description,
            expected_improvement="Understand and address root cause",
            implementation_steps=[
                "Correlate with recent events",
                "Check for external factors",
                "Add monitoring if beneficial spike",
                "Add guards if detrimental spike"
            ],
            risk_level="low"
        )


class InnovationLoop:
    """
    Main innovation loop coordinator.

    Runs periodic cycles that:
    1. Collect metrics from all sources
    2. Detect anomalies
    3. Generate improvement hypotheses
    4. Propose experiments for operator approval
    5. Execute approved experiments
    6. Document learnings
    """

    def __init__(
        self,
        orchestrator=None,
        cycle_interval: int = None,
        max_proposals_per_day: int = None,
        project_dir: str = None
    ):
        """
        Initialize innovation loop.

        Args:
            orchestrator: AutonomousOrchestrator instance
            cycle_interval: Seconds between cycles (default 300 = 5 min)
            max_proposals_per_day: Max experiment proposals per day (default 3)
            project_dir: Project directory for persistence
        """
        self.orchestrator = orchestrator
        self.cycle_interval = cycle_interval or int(os.getenv("INNOVATION_CYCLE_INTERVAL", "300"))
        self.max_proposals_per_day = max_proposals_per_day or int(os.getenv("INNOVATION_MAX_PROPOSALS", "3"))
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))

        # Components
        self.aggregator = MetricAggregator(orchestrator)
        self.detector = AnomalyDetector()
        self.generator = HypothesisGenerator()

        # State
        self.experiments: Dict[str, Experiment] = {}
        self._proposals_today: int = 0
        self._last_proposal_date: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            "cycles_run": 0,
            "anomalies_detected": 0,
            "hypotheses_generated": 0,
            "experiments_proposed": 0,
            "experiments_approved": 0,
            "experiments_completed": 0
        }

        # Load existing experiments
        self._load_experiments()

        logger.info(f"InnovationLoop initialized: cycle={self.cycle_interval}s, max_proposals={self.max_proposals_per_day}/day")

    def _load_experiments(self):
        """Load experiments from persistence."""
        exp_file = self.project_dir / "experiments.json"
        if exp_file.exists():
            try:
                with open(exp_file, "r") as f:
                    data = json.load(f)
                    for exp_data in data.get("experiments", []):
                        # Recreate experiment (simplified for now)
                        logger.debug(f"Loaded experiment: {exp_data.get('id')}")
            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")

    def _save_experiments(self):
        """Save experiments to persistence."""
        exp_file = self.project_dir / "experiments.json"
        try:
            data = {
                "experiments": [exp.to_dict() for exp in self.experiments.values()],
                "stats": self.stats
            }
            with open(exp_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")

    async def start(self):
        """Start the innovation loop."""
        if self._running:
            logger.warning("Innovation loop already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Innovation loop started")

    async def stop(self):
        """Stop the innovation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save_experiments()
        logger.info("Innovation loop stopped")

    async def _run_loop(self):
        """Main loop that runs innovation cycles."""
        while self._running:
            try:
                await self._run_cycle()
                await asyncio.sleep(self.cycle_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in innovation cycle: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _run_cycle(self):
        """Run a single innovation cycle."""
        logger.debug("Starting innovation cycle")
        self.stats["cycles_run"] += 1

        # Reset daily counter
        today = datetime.utcnow().date()
        if self._last_proposal_date != today:
            self._proposals_today = 0
            self._last_proposal_date = today

        # Step 1: Collect metrics
        metrics = await self.aggregator.collect_all()
        logger.debug(f"Collected {len(metrics)} metrics")

        # Step 2: Detect anomalies
        history = self.aggregator._history
        anomalies = await self.detector.analyze(metrics, history)
        self.stats["anomalies_detected"] += len(anomalies)

        if anomalies:
            logger.info(f"Detected {len(anomalies)} anomalies")
            for a in anomalies:
                logger.info(f"  - {a.type.value}: {a.description}")

        # Step 3: Generate hypotheses
        hypotheses = await self.generator.generate(metrics, anomalies)
        self.stats["hypotheses_generated"] += len(hypotheses)

        # Step 4: Create experiment proposals (respecting daily limit)
        for hypothesis in hypotheses:
            if self._proposals_today >= self.max_proposals_per_day:
                logger.info("Daily proposal limit reached")
                break

            await self._propose_experiment(hypothesis, metrics)
            self._proposals_today += 1

        # Step 5: Process approved experiments
        await self._process_approved_experiments()

        # Save state
        self._save_experiments()

    async def _propose_experiment(self, hypothesis: Hypothesis, baseline_metrics: Dict[str, Metric]):
        """Create an experiment proposal."""
        exp_id = f"EXP-{len(self.experiments) + 1:04d}"

        experiment = Experiment(
            id=exp_id,
            hypothesis=hypothesis,
            baseline_metrics={name: m.value for name, m in baseline_metrics.items()}
        )

        self.experiments[exp_id] = experiment
        self.stats["experiments_proposed"] += 1

        logger.info(f"Proposed experiment {exp_id}: {hypothesis.title}")

        # Notify via Discord if orchestrator available
        if self.orchestrator:
            await self._notify_experiment_proposal(experiment)

    async def _notify_experiment_proposal(self, experiment: Experiment):
        """Notify operator of new experiment proposal via Discord."""
        try:
            from discord_embeds import RALPHEmbeds

            # Find an agent to post
            for agent in self.orchestrator.agents.values():
                if hasattr(agent, 'post_embed_to_team'):
                    embed = RALPHEmbeds.experiment_proposal(
                        experiment_id=experiment.id,
                        title=experiment.hypothesis.title,
                        description=experiment.hypothesis.description,
                        risk_level=experiment.hypothesis.risk_level,
                        steps=experiment.hypothesis.implementation_steps
                    )
                    await agent.post_embed_to_team(embed)
                    break
        except Exception as e:
            logger.error(f"Failed to notify experiment proposal: {e}")

    async def _process_approved_experiments(self):
        """Execute any approved experiments."""
        for exp in self.experiments.values():
            if exp.status == ExperimentStatus.APPROVED:
                await self._run_experiment(exp)

    async def _run_experiment(self, experiment: Experiment):
        """Execute an experiment."""
        logger.info(f"Running experiment: {experiment.id}")
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()

        try:
            # For now, experiments are documentation-only
            # Future: actual implementation execution

            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            self.stats["experiments_completed"] += 1

            logger.info(f"Experiment {experiment.id} completed")

        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.results = {"error": str(e)}
            logger.error(f"Experiment {experiment.id} failed: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_pending_experiments(self) -> List[Experiment]:
        """Get experiments awaiting approval."""
        return [
            exp for exp in self.experiments.values()
            if exp.status == ExperimentStatus.PROPOSED
        ]

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(exp_id)

    async def approve_experiment(self, exp_id: str, approved_by: str) -> bool:
        """Approve an experiment for execution."""
        exp = self.experiments.get(exp_id)
        if not exp or exp.status != ExperimentStatus.PROPOSED:
            return False

        exp.status = ExperimentStatus.APPROVED
        exp.approved_at = datetime.utcnow()
        exp.approved_by = approved_by
        self.stats["experiments_approved"] += 1

        self._save_experiments()
        logger.info(f"Experiment {exp_id} approved by {approved_by}")
        return True

    async def reject_experiment(self, exp_id: str, reason: str) -> bool:
        """Reject an experiment."""
        exp = self.experiments.get(exp_id)
        if not exp or exp.status != ExperimentStatus.PROPOSED:
            return False

        exp.status = ExperimentStatus.REJECTED
        exp.rejection_reason = reason

        self._save_experiments()
        logger.info(f"Experiment {exp_id} rejected: {reason}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get innovation loop statistics."""
        return {
            **self.stats,
            "pending_experiments": len(self.get_pending_experiments()),
            "total_experiments": len(self.experiments),
            "proposals_today": self._proposals_today,
            "running": self._running
        }

    def get_stats_report(self) -> str:
        """Get formatted statistics report."""
        stats = self.get_stats()
        return f"""**Innovation Loop Stats:**
- Cycles run: {stats['cycles_run']}
- Anomalies detected: {stats['anomalies_detected']}
- Hypotheses generated: {stats['hypotheses_generated']}
- Experiments proposed: {stats['experiments_proposed']}
- Experiments approved: {stats['experiments_approved']}
- Experiments completed: {stats['experiments_completed']}
- Pending approval: {stats['pending_experiments']}
- Proposals today: {stats['proposals_today']}/{self.max_proposals_per_day}
- Running: {'Yes' if stats['running'] else 'No'}"""


# Singleton instance
_innovation_loop: Optional[InnovationLoop] = None


def get_innovation_loop() -> Optional[InnovationLoop]:
    """Get the global innovation loop instance."""
    return _innovation_loop


def set_innovation_loop(loop: InnovationLoop):
    """Set the global innovation loop instance."""
    global _innovation_loop
    _innovation_loop = loop


async def initialize_innovation_loop(orchestrator) -> InnovationLoop:
    """Initialize and start the global innovation loop."""
    global _innovation_loop
    _innovation_loop = InnovationLoop(orchestrator)
    await _innovation_loop.start()
    return _innovation_loop
