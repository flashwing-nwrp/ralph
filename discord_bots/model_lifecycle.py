"""
Model Lifecycle Management for RALPH Agent Ensemble

Manages the complete lifecycle of ML models:
- Model versioning and registry
- Training run tracking
- Model deployment and rollback
- Performance monitoring and comparison
- A/B testing support

P1: Essential for managing ML model iterations.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("model_lifecycle")


class ModelStatus(Enum):
    """Status of a model in the lifecycle."""
    TRAINING = "training"      # Currently training
    TRAINED = "trained"        # Training complete, not validated
    VALIDATING = "validating"  # Being validated/backtested
    VALIDATED = "validated"    # Passed validation
    STAGED = "staged"          # Ready for deployment
    DEPLOYED = "deployed"      # Currently in production
    SHADOW = "shadow"          # Running in shadow mode (no real trades)
    DEPRECATED = "deprecated"  # Replaced by newer version
    ARCHIVED = "archived"      # No longer in use
    FAILED = "failed"          # Training or validation failed


class ModelType(Enum):
    """Types of models in the trading system."""
    SIGNAL_GENERATOR = "signal_generator"
    RISK_MODEL = "risk_model"
    CALIBRATION = "calibration"
    POSITION_SIZING = "position_sizing"
    MARKET_REGIME = "market_regime"
    FEATURE_SELECTOR = "feature_selector"
    ENSEMBLE = "ensemble"


@dataclass
class TrainingRun:
    """Record of a model training run."""
    run_id: str
    model_id: str
    started_at: str
    completed_at: str = ""
    status: str = "running"  # running, completed, failed

    # Training configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data: Dict[str, Any] = field(default_factory=dict)  # date range, features, etc.

    # Results
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Resource usage
    duration_seconds: float = 0.0
    gpu_hours: float = 0.0

    # Notes
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "training_data": self.training_data,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "duration_seconds": self.duration_seconds,
            "gpu_hours": self.gpu_hours,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingRun":
        return cls(**data)


@dataclass
class ModelVersion:
    """A specific version of a model."""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus

    # Metadata
    created_at: str
    created_by: str  # Agent or operator
    description: str = ""

    # Training
    training_run_id: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_set: List[str] = field(default_factory=list)

    # Performance metrics
    backtest_metrics: Dict[str, float] = field(default_factory=dict)
    live_metrics: Dict[str, float] = field(default_factory=dict)

    # Deployment
    deployed_at: str = ""
    deployed_by: str = ""
    deployment_notes: str = ""

    # Files
    model_path: str = ""
    config_path: str = ""
    checksum: str = ""

    # Lineage
    parent_version: str = ""
    experiment_id: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "description": self.description,
            "training_run_id": self.training_run_id,
            "hyperparameters": self.hyperparameters,
            "feature_set": self.feature_set,
            "backtest_metrics": self.backtest_metrics,
            "live_metrics": self.live_metrics,
            "deployed_at": self.deployed_at,
            "deployed_by": self.deployed_by,
            "deployment_notes": self.deployment_notes,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "checksum": self.checksum,
            "parent_version": self.parent_version,
            "experiment_id": self.experiment_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            model_type=ModelType(data["model_type"]),
            status=ModelStatus(data["status"]),
            created_at=data["created_at"],
            created_by=data["created_by"],
            description=data.get("description", ""),
            training_run_id=data.get("training_run_id", ""),
            hyperparameters=data.get("hyperparameters", {}),
            feature_set=data.get("feature_set", []),
            backtest_metrics=data.get("backtest_metrics", {}),
            live_metrics=data.get("live_metrics", {}),
            deployed_at=data.get("deployed_at", ""),
            deployed_by=data.get("deployed_by", ""),
            deployment_notes=data.get("deployment_notes", ""),
            model_path=data.get("model_path", ""),
            config_path=data.get("config_path", ""),
            checksum=data.get("checksum", ""),
            parent_version=data.get("parent_version", ""),
            experiment_id=data.get("experiment_id", "")
        )

    def format_for_discord(self) -> str:
        """Format model version for Discord display."""
        status_emoji = {
            ModelStatus.TRAINING: "🔄",
            ModelStatus.TRAINED: "📦",
            ModelStatus.VALIDATING: "🔍",
            ModelStatus.VALIDATED: "✔️",
            ModelStatus.STAGED: "📋",
            ModelStatus.DEPLOYED: "🚀",
            ModelStatus.SHADOW: "👤",
            ModelStatus.DEPRECATED: "📜",
            ModelStatus.ARCHIVED: "🗄️",
            ModelStatus.FAILED: "❌"
        }

        emoji = status_emoji.get(self.status, "❓")

        output = [
            f"{emoji} **{self.model_id}** v{self.version}",
            f"Type: {self.model_type.value} | Status: {self.status.value}",
            f"Created: {self.created_at} by {self.created_by}",
        ]

        if self.description:
            output.append(f"_{self.description}_")

        if self.backtest_metrics:
            metrics = ", ".join([f"{k}={v:.3f}" for k, v in list(self.backtest_metrics.items())[:3]])
            output.append(f"**Backtest:** {metrics}")

        if self.status == ModelStatus.DEPLOYED:
            output.append(f"**Deployed:** {self.deployed_at}")
            if self.live_metrics:
                metrics = ", ".join([f"{k}={v:.3f}" for k, v in list(self.live_metrics.items())[:3]])
                output.append(f"**Live:** {metrics}")

        return "\n".join(output)


class ModelRegistry:
    """
    Central registry for model versions and lifecycle management.

    Provides:
    - Model versioning and storage
    - Training run tracking
    - Deployment management
    - Performance comparison
    - Rollback capability
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.models_dir = self.project_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.registry_file = self.models_dir / "model_registry.json"
        self.training_runs_file = self.models_dir / "training_runs.json"

        # In-memory state
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # model_id -> {version -> ModelVersion}
        self.training_runs: List[TrainingRun] = []
        self.active_deployments: Dict[str, str] = {}  # model_id -> deployed version

        self.version_counter = 0
        self.run_counter = 0
        self._lock = asyncio.Lock()

        self._load_registry()

    def _load_registry(self):
        """Load model registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.version_counter = data.get("version_counter", 0)

                    for model_data in data.get("models", []):
                        model = ModelVersion.from_dict(model_data)
                        if model.model_id not in self.models:
                            self.models[model.model_id] = {}
                        self.models[model.model_id][model.version] = model

                        if model.status == ModelStatus.DEPLOYED:
                            self.active_deployments[model.model_id] = model.version

                    logger.info(f"Loaded {sum(len(v) for v in self.models.values())} model versions")

            if self.training_runs_file.exists():
                with open(self.training_runs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.run_counter = data.get("run_counter", 0)
                    self.training_runs = [
                        TrainingRun.from_dict(r) for r in data.get("runs", [])
                    ]
                    logger.info(f"Loaded {len(self.training_runs)} training runs")

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")

    async def _save_registry(self):
        """Save model registry to file."""
        async with self._lock:
            try:
                # Flatten models for storage
                all_models = []
                for model_id, versions in self.models.items():
                    for version, model in versions.items():
                        all_models.append(model.to_dict())

                data = {
                    "version_counter": self.version_counter,
                    "models": all_models,
                    "active_deployments": self.active_deployments
                }

                with open(self.registry_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                # Save training runs
                runs_data = {
                    "run_counter": self.run_counter,
                    "runs": [r.to_dict() for r in self.training_runs[-500:]]  # Keep last 500
                }

                with open(self.training_runs_file, "w", encoding="utf-8") as f:
                    json.dump(runs_data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save model registry: {e}")

    # =========================================================================
    # MODEL VERSION MANAGEMENT
    # =========================================================================

    async def register_model(
        self,
        model_id: str,
        model_type: ModelType,
        created_by: str,
        description: str = "",
        hyperparameters: Dict[str, Any] = None,
        feature_set: List[str] = None,
        parent_version: str = "",
        experiment_id: str = ""
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_id: Identifier for the model (e.g., "signal_v1")
            model_type: Type of model
            created_by: Agent or operator creating the model
            description: Description of this version
            hyperparameters: Model hyperparameters
            feature_set: List of features used
            parent_version: Version this is based on
            experiment_id: Associated experiment ID

        Returns:
            The created model version
        """
        self.version_counter += 1

        # Determine version number
        if model_id in self.models:
            existing_versions = len(self.models[model_id])
            version = f"{existing_versions + 1}.0.0"
        else:
            version = "1.0.0"
            self.models[model_id] = {}

        model = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            status=ModelStatus.TRAINED,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            description=description,
            hyperparameters=hyperparameters or {},
            feature_set=feature_set or [],
            parent_version=parent_version,
            experiment_id=experiment_id
        )

        self.models[model_id][version] = model
        await self._save_registry()

        logger.info(f"Registered model {model_id} v{version}")
        return model

    async def update_status(
        self,
        model_id: str,
        version: str,
        status: ModelStatus
    ) -> Optional[ModelVersion]:
        """Update model status."""
        if model_id not in self.models or version not in self.models[model_id]:
            return None

        model = self.models[model_id][version]
        old_status = model.status
        model.status = status

        # Handle deployment tracking
        if status == ModelStatus.DEPLOYED:
            # Deprecate currently deployed version
            if model_id in self.active_deployments:
                old_version = self.active_deployments[model_id]
                if old_version != version:
                    old_model = self.models[model_id].get(old_version)
                    if old_model:
                        old_model.status = ModelStatus.DEPRECATED
            self.active_deployments[model_id] = version
            model.deployed_at = datetime.utcnow().isoformat()

        elif old_status == ModelStatus.DEPLOYED:
            # Model was undeployed
            if model_id in self.active_deployments:
                if self.active_deployments[model_id] == version:
                    del self.active_deployments[model_id]

        await self._save_registry()
        logger.info(f"Model {model_id} v{version}: {old_status.value} -> {status.value}")
        return model

    async def update_metrics(
        self,
        model_id: str,
        version: str,
        backtest_metrics: Dict[str, float] = None,
        live_metrics: Dict[str, float] = None
    ) -> Optional[ModelVersion]:
        """Update model performance metrics."""
        if model_id not in self.models or version not in self.models[model_id]:
            return None

        model = self.models[model_id][version]

        if backtest_metrics:
            model.backtest_metrics.update(backtest_metrics)
        if live_metrics:
            model.live_metrics.update(live_metrics)

        await self._save_registry()
        return model

    def get_model(self, model_id: str, version: str = None) -> Optional[ModelVersion]:
        """Get a specific model version or the deployed version."""
        if model_id not in self.models:
            return None

        if version:
            return self.models[model_id].get(version)

        # Return deployed version if exists
        if model_id in self.active_deployments:
            deployed_version = self.active_deployments[model_id]
            return self.models[model_id].get(deployed_version)

        # Return latest version
        versions = sorted(self.models[model_id].keys(), reverse=True)
        if versions:
            return self.models[model_id][versions[0]]

        return None

    def get_all_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        if model_id not in self.models:
            return []
        return list(self.models[model_id].values())

    def get_deployed_models(self) -> Dict[str, ModelVersion]:
        """Get all currently deployed models."""
        deployed = {}
        for model_id, version in self.active_deployments.items():
            if model_id in self.models and version in self.models[model_id]:
                deployed[model_id] = self.models[model_id][version]
        return deployed

    # =========================================================================
    # DEPLOYMENT MANAGEMENT
    # =========================================================================

    async def deploy_model(
        self,
        model_id: str,
        version: str,
        deployed_by: str,
        notes: str = ""
    ) -> Optional[ModelVersion]:
        """
        Deploy a model version to production.

        Args:
            model_id: Model to deploy
            version: Version to deploy
            deployed_by: Who is deploying
            notes: Deployment notes

        Returns:
            The deployed model, or None if not found
        """
        if model_id not in self.models or version not in self.models[model_id]:
            return None

        model = self.models[model_id][version]

        # Validate model is ready for deployment
        if model.status not in [ModelStatus.VALIDATED, ModelStatus.STAGED, ModelStatus.DEPRECATED]:
            logger.warning(f"Model {model_id} v{version} not validated for deployment")

        model.deployed_by = deployed_by
        model.deployment_notes = notes

        return await self.update_status(model_id, version, ModelStatus.DEPLOYED)

    async def rollback_model(
        self,
        model_id: str,
        to_version: str,
        reason: str
    ) -> Optional[ModelVersion]:
        """
        Rollback to a previous model version.

        Args:
            model_id: Model to rollback
            to_version: Version to rollback to
            reason: Reason for rollback

        Returns:
            The rolled back model
        """
        if model_id not in self.active_deployments:
            return None

        current_version = self.active_deployments[model_id]

        # Deploy the old version
        result = await self.deploy_model(
            model_id=model_id,
            version=to_version,
            deployed_by="rollback",
            notes=f"Rollback from v{current_version}: {reason}"
        )

        if result:
            logger.warning(f"Rolled back {model_id} from v{current_version} to v{to_version}")

        return result

    async def enable_shadow_mode(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersion]:
        """Enable shadow mode for a model (runs alongside production without trading)."""
        return await self.update_status(model_id, version, ModelStatus.SHADOW)

    # =========================================================================
    # TRAINING RUN TRACKING
    # =========================================================================

    async def start_training_run(
        self,
        model_id: str,
        hyperparameters: Dict[str, Any] = None,
        training_data: Dict[str, Any] = None,
        notes: str = ""
    ) -> TrainingRun:
        """Start a new training run."""
        self.run_counter += 1

        run = TrainingRun(
            run_id=f"RUN-{datetime.utcnow().strftime('%Y%m%d')}-{self.run_counter:04d}",
            model_id=model_id,
            started_at=datetime.utcnow().isoformat(),
            hyperparameters=hyperparameters or {},
            training_data=training_data or {},
            notes=notes
        )

        self.training_runs.append(run)
        await self._save_registry()

        logger.info(f"Started training run {run.run_id} for {model_id}")
        return run

    async def complete_training_run(
        self,
        run_id: str,
        training_metrics: Dict[str, float],
        validation_metrics: Dict[str, float] = None,
        success: bool = True
    ) -> Optional[TrainingRun]:
        """Complete a training run."""
        for run in self.training_runs:
            if run.run_id == run_id:
                run.completed_at = datetime.utcnow().isoformat()
                run.status = "completed" if success else "failed"
                run.training_metrics = training_metrics
                run.validation_metrics = validation_metrics or {}

                # Calculate duration
                start = datetime.fromisoformat(run.started_at)
                end = datetime.fromisoformat(run.completed_at)
                run.duration_seconds = (end - start).total_seconds()

                await self._save_registry()
                logger.info(f"Completed training run {run_id}: {run.status}")
                return run

        return None

    def get_training_runs(self, model_id: str = None, limit: int = 20) -> List[TrainingRun]:
        """Get training runs, optionally filtered by model."""
        runs = self.training_runs
        if model_id:
            runs = [r for r in runs if r.model_id == model_id]
        return list(reversed(runs[-limit:]))

    # =========================================================================
    # COMPARISON AND ANALYSIS
    # =========================================================================

    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        if model_id not in self.models:
            return {"error": f"Model {model_id} not found"}

        m1 = self.models[model_id].get(version1)
        m2 = self.models[model_id].get(version2)

        if not m1 or not m2:
            return {"error": "One or both versions not found"}

        comparison = {
            "version1": version1,
            "version2": version2,
            "hyperparameter_diff": {},
            "metric_comparison": {}
        }

        # Compare hyperparameters
        all_params = set(m1.hyperparameters.keys()) | set(m2.hyperparameters.keys())
        for param in all_params:
            v1 = m1.hyperparameters.get(param)
            v2 = m2.hyperparameters.get(param)
            if v1 != v2:
                comparison["hyperparameter_diff"][param] = {"v1": v1, "v2": v2}

        # Compare metrics
        all_metrics = set(m1.backtest_metrics.keys()) | set(m2.backtest_metrics.keys())
        for metric in all_metrics:
            v1 = m1.backtest_metrics.get(metric, 0)
            v2 = m2.backtest_metrics.get(metric, 0)
            diff = v2 - v1 if v1 and v2 else None
            comparison["metric_comparison"][metric] = {
                "v1": v1,
                "v2": v2,
                "diff": diff,
                "pct_change": (diff / v1 * 100) if v1 and diff else None
            }

        return comparison

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_registry_display(self) -> str:
        """Get formatted registry display for Discord."""
        output = ["## 🤖 Model Registry\n"]

        if not self.models:
            return "No models registered yet."

        # Group by model type
        by_type: Dict[ModelType, List[ModelVersion]] = {}
        for model_id, versions in self.models.items():
            latest = sorted(versions.values(), key=lambda x: x.created_at, reverse=True)[0]
            if latest.model_type not in by_type:
                by_type[latest.model_type] = []
            by_type[latest.model_type].append(latest)

        for model_type, models in by_type.items():
            output.append(f"\n### {model_type.value}")
            for model in models:
                deployed = "🚀" if model.status == ModelStatus.DEPLOYED else ""
                output.append(f"  {deployed} **{model.model_id}** v{model.version} - {model.status.value}")

        # Deployed summary
        deployed = self.get_deployed_models()
        if deployed:
            output.append("\n### Currently Deployed")
            for model_id, model in deployed.items():
                output.append(f"  {model_id}: v{model.version}")

        return "\n".join(output)

    def get_model_details(self, model_id: str, version: str = None) -> str:
        """Get detailed model information."""
        model = self.get_model(model_id, version)

        if not model:
            return f"Model {model_id} not found"

        output = [f"## Model Details: {model.model_id}\n"]
        output.append(model.format_for_discord())

        if model.hyperparameters:
            output.append("\n**Hyperparameters:**")
            for k, v in list(model.hyperparameters.items())[:10]:
                output.append(f"  {k}: {v}")

        if model.feature_set:
            output.append(f"\n**Features:** {len(model.feature_set)} features")
            output.append(f"  {', '.join(model.feature_set[:5])}...")

        # Version history
        all_versions = self.get_all_versions(model_id)
        if len(all_versions) > 1:
            output.append(f"\n**Version History:** {len(all_versions)} versions")
            for v in sorted(all_versions, key=lambda x: x.created_at, reverse=True)[:5]:
                marker = "→" if v.version == model.version else "  "
                output.append(f"  {marker} v{v.version}: {v.status.value}")

        return "\n".join(output)

    def get_training_history(self, model_id: str = None) -> str:
        """Get training run history display."""
        runs = self.get_training_runs(model_id, limit=10)

        if not runs:
            return "No training runs recorded."

        output = ["## 🏋️ Training History\n"]

        for run in runs:
            status_emoji = "✅" if run.status == "completed" else "❌" if run.status == "failed" else "🔄"
            output.append(f"{status_emoji} **{run.run_id}** | {run.model_id}")
            output.append(f"   Started: {run.started_at}")
            if run.duration_seconds:
                output.append(f"   Duration: {run.duration_seconds:.1f}s")
            if run.training_metrics:
                metrics = ", ".join([f"{k}={v:.4f}" for k, v in list(run.training_metrics.items())[:3]])
                output.append(f"   Metrics: {metrics}")
            output.append("")

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create the model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def set_model_registry(registry: ModelRegistry):
    """Set the model registry instance."""
    global _model_registry
    _model_registry = registry
