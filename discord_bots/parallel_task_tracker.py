"""
Parallel Task Tracker for RALPH Agent Ensemble

This module provides concurrent task management for parallel execution
of agent tasks while respecting dependencies and per-agent concurrency limits.

Key Features:
- Track active tasks per agent type
- Enforce configurable concurrency limits
- Dependency resolution before dispatch
- Thread-safe state synchronization via asyncio locks
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Set, List, Optional, Any

logger = logging.getLogger("parallel_task_tracker")


class TaskExecutionState(Enum):
    """State of a task in the parallel execution pipeline."""
    QUEUED = "queued"          # Registered, waiting for dependencies
    READY = "ready"            # Dependencies met, waiting for agent slot
    RUNNING = "running"        # Currently executing
    COMPLETED = "completed"    # Finished successfully
    FAILED = "failed"          # Execution failed
    CANCELLED = "cancelled"    # Cancelled before completion


@dataclass
class TrackedTask:
    """A task being tracked for parallel execution."""
    task_id: str
    agent_type: str
    handoff: dict                                    # Original handoff data
    dependencies: Set[str] = field(default_factory=set)
    state: TaskExecutionState = TaskExecutionState.QUEUED
    asyncio_task: Optional[asyncio.Task] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    # Link to mission task if applicable
    mission_task_id: Optional[str] = None


@dataclass
class AgentConcurrencyConfig:
    """
    Per-agent concurrency limits.

    These limits balance parallelism with resource constraints:
    - tuning: Compute-heavy optimization, serialize to avoid resource contention
    - backtest: Long-running simulations, limit to avoid memory issues
    - risk: Quick audits, can run multiple
    - strategy: Planning tasks, can parallelize
    - data: I/O bound data tasks, highest parallelism
    """
    tuning: int = 1
    backtest: int = 1
    risk: int = 2
    strategy: int = 2
    data: int = 3
    default: int = 1

    @classmethod
    def from_env(cls) -> "AgentConcurrencyConfig":
        """Load configuration from environment variables."""
        return cls(
            tuning=int(os.getenv("PARALLEL_TUNING_LIMIT", "1")),
            backtest=int(os.getenv("PARALLEL_BACKTEST_LIMIT", "1")),
            risk=int(os.getenv("PARALLEL_RISK_LIMIT", "2")),
            strategy=int(os.getenv("PARALLEL_STRATEGY_LIMIT", "2")),
            data=int(os.getenv("PARALLEL_DATA_LIMIT", "3")),
            default=int(os.getenv("PARALLEL_DEFAULT_LIMIT", "1")),
        )


class ParallelTaskTracker:
    """
    Tracks concurrent task execution across agents.

    Maintains:
    - Active tasks per agent
    - Task dependency graph
    - Completion status for dependency resolution

    Thread Safety:
    - All state modifications protected by asyncio.Lock
    - Event signaling for task readiness
    """

    def __init__(self, concurrency_config: AgentConcurrencyConfig = None):
        self.config = concurrency_config or AgentConcurrencyConfig.from_env()

        # Task tracking
        self.tasks: Dict[str, TrackedTask] = {}
        self.active_by_agent: Dict[str, Set[str]] = {
            "tuning": set(),
            "backtest": set(),
            "risk": set(),
            "strategy": set(),
            "data": set()
        }

        # Dependency tracking
        self.completed_task_ids: Set[str] = set()

        # Synchronization
        self._lock = asyncio.Lock()
        self._task_ready_event = asyncio.Event()

        # Statistics
        self.stats = {
            "tasks_registered": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "max_concurrent": 0,
            "current_concurrent": 0
        }

        logger.info(f"ParallelTaskTracker initialized with limits: "
                   f"tuning={self.config.tuning}, backtest={self.config.backtest}, "
                   f"risk={self.config.risk}, strategy={self.config.strategy}, "
                   f"data={self.config.data}")

    def get_agent_limit(self, agent_type: str) -> int:
        """Get concurrency limit for an agent type."""
        return getattr(self.config, agent_type, self.config.default)

    async def can_start_task(self, agent_type: str) -> bool:
        """Check if agent has available capacity."""
        async with self._lock:
            limit = self.get_agent_limit(agent_type)
            current = len(self.active_by_agent.get(agent_type, set()))
            return current < limit

    async def register_task(
        self,
        task_id: str,
        agent_type: str,
        handoff: dict,
        dependencies: Set[str] = None,
        mission_task_id: str = None
    ) -> TrackedTask:
        """
        Register a new task for tracking.

        Args:
            task_id: Unique task identifier
            agent_type: Target agent (tuning, backtest, etc.)
            handoff: Original handoff data
            dependencies: Set of task_ids that must complete first
            mission_task_id: Optional link to mission task

        Returns:
            TrackedTask instance
        """
        async with self._lock:
            # Filter dependencies to only include known task IDs
            valid_deps = set()
            if dependencies:
                for dep in dependencies:
                    # Include if it's a registered task or already completed
                    if dep in self.tasks or dep in self.completed_task_ids:
                        valid_deps.add(dep)

            task = TrackedTask(
                task_id=task_id,
                agent_type=agent_type,
                handoff=handoff,
                dependencies=valid_deps,
                mission_task_id=mission_task_id
            )
            self.tasks[task_id] = task
            self.stats["tasks_registered"] += 1

            logger.debug(f"Registered task {task_id} for {agent_type} "
                        f"with {len(valid_deps)} dependencies")

            return task

    async def mark_task_running(self, task_id: str, asyncio_task: asyncio.Task = None):
        """
        Mark task as running and track the asyncio Task.

        Args:
            task_id: Task identifier
            asyncio_task: The asyncio.Task executing this task
        """
        async with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for mark_running")
                return

            task = self.tasks[task_id]
            task.state = TaskExecutionState.RUNNING
            task.asyncio_task = asyncio_task
            task.started_at = datetime.utcnow().isoformat()

            # Add to active set
            if task.agent_type not in self.active_by_agent:
                self.active_by_agent[task.agent_type] = set()
            self.active_by_agent[task.agent_type].add(task_id)

            # Update stats
            self.stats["current_concurrent"] = sum(
                len(s) for s in self.active_by_agent.values()
            )
            self.stats["max_concurrent"] = max(
                self.stats["max_concurrent"],
                self.stats["current_concurrent"]
            )

            logger.info(f"Task {task_id} running on {task.agent_type} "
                       f"(concurrent: {self.stats['current_concurrent']})")

    async def mark_task_completed(
        self,
        task_id: str,
        result: Any = None,
        success: bool = True,
        error: str = None
    ):
        """
        Mark task as completed and trigger dependency resolution.

        Args:
            task_id: Task identifier
            result: Task result data
            success: Whether task completed successfully
            error: Error message if failed
        """
        async with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for mark_completed")
                return

            task = self.tasks[task_id]
            task.state = TaskExecutionState.COMPLETED if success else TaskExecutionState.FAILED
            task.completed_at = datetime.utcnow().isoformat()
            task.result = result
            task.error = error

            # Remove from active set
            if task.agent_type in self.active_by_agent:
                self.active_by_agent[task.agent_type].discard(task_id)

            # Add to completed set for dependency resolution
            self.completed_task_ids.add(task_id)

            # Update stats
            if success:
                self.stats["tasks_completed"] += 1
            else:
                self.stats["tasks_failed"] += 1
            self.stats["current_concurrent"] = sum(
                len(s) for s in self.active_by_agent.values()
            )

            # Signal that tasks may be ready
            self._task_ready_event.set()

            logger.info(f"Task {task_id} {'completed' if success else 'failed'} "
                       f"(concurrent: {self.stats['current_concurrent']})")

    async def get_ready_tasks(self, max_tasks: int = 10) -> List[TrackedTask]:
        """
        Get tasks that are ready to execute.

        A task is ready when:
        1. State is QUEUED (not already running/completed)
        2. All dependencies are in completed_task_ids
        3. Agent has available capacity

        Args:
            max_tasks: Maximum number of ready tasks to return

        Returns:
            List of TrackedTask instances ready for dispatch
        """
        async with self._lock:
            ready = []

            for task in self.tasks.values():
                if len(ready) >= max_tasks:
                    break

                if task.state != TaskExecutionState.QUEUED:
                    continue

                # Check dependencies
                deps_met = task.dependencies.issubset(self.completed_task_ids)
                if not deps_met:
                    continue

                # Check agent capacity
                limit = self.get_agent_limit(task.agent_type)
                current = len(self.active_by_agent.get(task.agent_type, set()))
                if current >= limit:
                    continue

                # Mark as ready (prevents re-selection)
                task.state = TaskExecutionState.READY
                ready.append(task)

            # Clear the event (will be set again on next completion)
            self._task_ready_event.clear()

            if ready:
                logger.info(f"Found {len(ready)} ready tasks: "
                           f"{[t.task_id for t in ready]}")

            return ready

    async def wait_for_ready_tasks(self, timeout: float = 1.0) -> bool:
        """
        Wait for tasks to become ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if event was set (tasks may be ready), False on timeout
        """
        try:
            await asyncio.wait_for(
                self._task_ready_event.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it's running.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False otherwise
        """
        async with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]

            if task.state == TaskExecutionState.RUNNING and task.asyncio_task:
                task.asyncio_task.cancel()
                task.state = TaskExecutionState.CANCELLED
                task.completed_at = datetime.utcnow().isoformat()

                if task.agent_type in self.active_by_agent:
                    self.active_by_agent[task.agent_type].discard(task_id)

                logger.info(f"Cancelled task {task_id}")
                return True

            elif task.state == TaskExecutionState.QUEUED:
                task.state = TaskExecutionState.CANCELLED
                logger.info(f"Cancelled queued task {task_id}")
                return True

            return False

    async def get_task_status(self, task_id: str) -> Optional[TrackedTask]:
        """Get the current status of a task."""
        async with self._lock:
            return self.tasks.get(task_id)

    async def get_active_tasks(self, agent_type: str = None) -> List[TrackedTask]:
        """
        Get all currently running tasks.

        Args:
            agent_type: Optional filter by agent type

        Returns:
            List of running TrackedTask instances
        """
        async with self._lock:
            if agent_type:
                task_ids = self.active_by_agent.get(agent_type, set())
                return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
            else:
                active = []
                for task_ids in self.active_by_agent.values():
                    active.extend(
                        self.tasks[tid] for tid in task_ids if tid in self.tasks
                    )
                return active

    async def get_pending_count(self) -> Dict[str, int]:
        """Get count of pending tasks by agent type."""
        async with self._lock:
            counts = {}
            for task in self.tasks.values():
                if task.state == TaskExecutionState.QUEUED:
                    counts[task.agent_type] = counts.get(task.agent_type, 0) + 1
            return counts

    async def cleanup_completed(self, keep_recent: int = 100):
        """
        Remove old completed tasks to prevent memory growth.

        Args:
            keep_recent: Number of recent completed tasks to keep
        """
        async with self._lock:
            completed = [
                (tid, t) for tid, t in self.tasks.items()
                if t.state in (TaskExecutionState.COMPLETED,
                              TaskExecutionState.FAILED,
                              TaskExecutionState.CANCELLED)
            ]

            # Sort by completion time
            completed.sort(key=lambda x: x[1].completed_at or "", reverse=True)

            # Remove old ones
            for task_id, _ in completed[keep_recent:]:
                del self.tasks[task_id]

            if len(completed) > keep_recent:
                logger.debug(f"Cleaned up {len(completed) - keep_recent} old tasks")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self.stats,
            "active_by_agent": {
                agent: len(tasks)
                for agent, tasks in self.active_by_agent.items()
            },
            "pending_count": len([
                t for t in self.tasks.values()
                if t.state == TaskExecutionState.QUEUED
            ]),
            "total_tracked": len(self.tasks)
        }

    def get_stats_report(self) -> str:
        """Get formatted statistics report."""
        stats = self.get_stats()
        active = stats["active_by_agent"]

        return f"""**Parallel Execution Stats:**
- Tasks registered: {stats['tasks_registered']}
- Tasks completed: {stats['tasks_completed']}
- Tasks failed: {stats['tasks_failed']}
- Current concurrent: {stats['current_concurrent']}
- Max concurrent: {stats['max_concurrent']}
- Active by agent: {', '.join(f'{a}={c}' for a, c in active.items() if c > 0) or 'None'}
- Pending: {stats['pending_count']}"""


# Singleton instance
_parallel_tracker: Optional[ParallelTaskTracker] = None


def get_parallel_tracker() -> ParallelTaskTracker:
    """Get or create the global parallel task tracker."""
    global _parallel_tracker
    if _parallel_tracker is None:
        _parallel_tracker = ParallelTaskTracker()
    return _parallel_tracker


def set_parallel_tracker(tracker: ParallelTaskTracker):
    """Set the global parallel task tracker."""
    global _parallel_tracker
    _parallel_tracker = tracker
