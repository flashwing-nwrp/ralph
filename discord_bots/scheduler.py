"""
Scheduled Operations for RALPH Agent Ensemble

Provides cron-like scheduling for automated tasks:
- Model retraining schedules
- Data refresh intervals
- Report generation
- Health checks
- Backup operations

P2: Important for automated operations.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Coroutine
import re

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("scheduler")


class TaskStatus(Enum):
    """Status of a scheduled task."""
    ACTIVE = "active"          # Task is scheduled and running
    PAUSED = "paused"          # Task is paused
    DISABLED = "disabled"      # Task is disabled
    RUNNING = "running"        # Task is currently executing
    COMPLETED = "completed"    # One-time task completed
    FAILED = "failed"          # Task failed


class TaskFrequency(Enum):
    """Predefined frequency options."""
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ScheduleConfig:
    """Configuration for task scheduling."""
    frequency: TaskFrequency

    # For custom schedules (cron-like)
    minute: str = "*"      # 0-59
    hour: str = "*"        # 0-23
    day_of_month: str = "*"  # 1-31
    month: str = "*"       # 1-12
    day_of_week: str = "*"   # 0-6 (Sunday=0)

    # Simple interval (alternative to cron)
    interval_seconds: int = 0

    # Time window
    start_time: str = ""   # HH:MM format
    end_time: str = ""     # HH:MM format

    # Timezone
    timezone: str = "UTC"

    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency.value,
            "minute": self.minute,
            "hour": self.hour,
            "day_of_month": self.day_of_month,
            "month": self.month,
            "day_of_week": self.day_of_week,
            "interval_seconds": self.interval_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timezone": self.timezone
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduleConfig":
        return cls(
            frequency=TaskFrequency(data.get("frequency", "custom")),
            minute=data.get("minute", "*"),
            hour=data.get("hour", "*"),
            day_of_month=data.get("day_of_month", "*"),
            month=data.get("month", "*"),
            day_of_week=data.get("day_of_week", "*"),
            interval_seconds=data.get("interval_seconds", 0),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            timezone=data.get("timezone", "UTC")
        )

    @classmethod
    def every_minutes(cls, minutes: int) -> "ScheduleConfig":
        """Create a schedule that runs every N minutes."""
        return cls(
            frequency=TaskFrequency.CUSTOM,
            interval_seconds=minutes * 60
        )

    @classmethod
    def every_hours(cls, hours: int) -> "ScheduleConfig":
        """Create a schedule that runs every N hours."""
        return cls(
            frequency=TaskFrequency.CUSTOM,
            interval_seconds=hours * 3600
        )

    @classmethod
    def daily_at(cls, hour: int, minute: int = 0) -> "ScheduleConfig":
        """Create a schedule that runs daily at a specific time."""
        return cls(
            frequency=TaskFrequency.DAILY,
            hour=str(hour),
            minute=str(minute)
        )

    @classmethod
    def weekly_at(cls, day_of_week: int, hour: int, minute: int = 0) -> "ScheduleConfig":
        """Create a schedule that runs weekly on a specific day and time."""
        return cls(
            frequency=TaskFrequency.WEEKLY,
            day_of_week=str(day_of_week),
            hour=str(hour),
            minute=str(minute)
        )


@dataclass
class ScheduledTask:
    """A scheduled task."""
    task_id: str
    name: str
    description: str
    task_type: str  # Category of task (retraining, backup, report, etc.)

    schedule: ScheduleConfig
    status: TaskStatus = TaskStatus.ACTIVE

    # Handler
    handler_name: str = ""  # Name of registered handler function
    handler_args: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    created_at: str = ""
    last_run: str = ""
    last_result: str = ""
    next_run: str = ""
    run_count: int = 0
    failure_count: int = 0

    # Settings
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300
    notify_on_failure: bool = True

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "schedule": self.schedule.to_dict(),
            "status": self.status.value,
            "handler_name": self.handler_name,
            "handler_args": self.handler_args,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "last_result": self.last_result,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "failure_count": self.failure_count,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "notify_on_failure": self.notify_on_failure
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data["description"],
            task_type=data["task_type"],
            schedule=ScheduleConfig.from_dict(data.get("schedule", {})),
            status=TaskStatus(data.get("status", "active")),
            handler_name=data.get("handler_name", ""),
            handler_args=data.get("handler_args", {}),
            created_at=data.get("created_at", ""),
            last_run=data.get("last_run", ""),
            last_result=data.get("last_result", ""),
            next_run=data.get("next_run", ""),
            run_count=data.get("run_count", 0),
            failure_count=data.get("failure_count", 0),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            timeout_seconds=data.get("timeout_seconds", 300),
            notify_on_failure=data.get("notify_on_failure", True)
        )

    def format_for_discord(self) -> str:
        """Format task for Discord display."""
        status_emoji = {
            TaskStatus.ACTIVE: "🟢",
            TaskStatus.PAUSED: "⏸️",
            TaskStatus.DISABLED: "⚫",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌"
        }

        emoji = status_emoji.get(self.status, "❓")

        lines = [
            f"{emoji} **{self.name}** (`{self.task_id}`)",
            f"Type: {self.task_type} | Status: {self.status.value}",
            f"_{self.description}_",
        ]

        if self.next_run:
            lines.append(f"**Next Run:** {self.next_run}")

        if self.last_run:
            lines.append(f"**Last Run:** {self.last_run} ({self.last_result})")

        lines.append(f"**Runs:** {self.run_count} | Failures: {self.failure_count}")

        return "\n".join(lines)


@dataclass
class TaskExecution:
    """Record of a task execution."""
    execution_id: str
    task_id: str
    started_at: str
    completed_at: str = ""
    status: str = "running"  # running, success, failed, timeout
    result: str = ""
    error_message: str = ""
    duration_seconds: float = 0.0


class TaskScheduler:
    """
    Central task scheduler for RALPH.

    Provides:
    - Cron-like task scheduling
    - Handler registration
    - Execution tracking
    - Failure handling and retries
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.schedule_file = self.project_dir / "scheduled_tasks.json"
        self.history_file = self.project_dir / "task_history.json"

        # Tasks
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_counter = 0

        # Registered handlers
        self.handlers: Dict[str, Callable] = {}

        # Execution history
        self.execution_history: List[TaskExecution] = []
        self.execution_counter = 0

        # Running state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Callbacks
        self._notification_callbacks: List[Callable] = []

        self._load_tasks()

    def _load_tasks(self):
        """Load scheduled tasks from file."""
        try:
            if self.schedule_file.exists():
                with open(self.schedule_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.task_counter = data.get("task_counter", 0)

                    for task_data in data.get("tasks", []):
                        task = ScheduledTask.from_dict(task_data)
                        self.tasks[task.task_id] = task

                    logger.info(f"Loaded {len(self.tasks)} scheduled tasks")

            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.execution_counter = data.get("execution_counter", 0)
                    # History is loaded on-demand to save memory

        except Exception as e:
            logger.error(f"Failed to load scheduled tasks: {e}")

    async def _save_tasks(self):
        """Save scheduled tasks to file."""
        async with self._lock:
            try:
                data = {
                    "task_counter": self.task_counter,
                    "tasks": [t.to_dict() for t in self.tasks.values()]
                }

                with open(self.schedule_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save scheduled tasks: {e}")

    async def _save_execution(self, execution: TaskExecution):
        """Save execution to history."""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    history = data.get("executions", [])

            history.append({
                "execution_id": execution.execution_id,
                "task_id": execution.task_id,
                "started_at": execution.started_at,
                "completed_at": execution.completed_at,
                "status": execution.status,
                "result": execution.result,
                "error_message": execution.error_message,
                "duration_seconds": execution.duration_seconds
            })

            # Keep last 1000 executions
            history = history[-1000:]

            data = {
                "execution_counter": self.execution_counter,
                "executions": history
            }

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save execution history: {e}")

    def register_notification_callback(self, callback: Callable):
        """Register a callback for task notifications."""
        self._notification_callbacks.append(callback)

    async def _notify(self, task: ScheduledTask, message: str):
        """Send notification about task."""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, message)
                else:
                    callback(task, message)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    # =========================================================================
    # HANDLER REGISTRATION
    # =========================================================================

    def register_handler(self, name: str, handler: Callable):
        """
        Register a task handler function.

        Args:
            name: Unique name for the handler
            handler: Async or sync function to execute
        """
        self.handlers[name] = handler
        logger.info(f"Registered handler: {name}")

    def unregister_handler(self, name: str):
        """Unregister a task handler."""
        if name in self.handlers:
            del self.handlers[name]

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    async def create_task(
        self,
        name: str,
        description: str,
        task_type: str,
        schedule: ScheduleConfig,
        handler_name: str,
        handler_args: Dict[str, Any] = None,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        notify_on_failure: bool = True
    ) -> ScheduledTask:
        """
        Create a new scheduled task.

        Args:
            name: Task name
            description: Task description
            task_type: Category (retraining, backup, report, etc.)
            schedule: Schedule configuration
            handler_name: Name of registered handler
            handler_args: Arguments to pass to handler
            max_retries: Maximum retry attempts
            timeout_seconds: Task timeout
            notify_on_failure: Whether to notify on failure

        Returns:
            Created task
        """
        self.task_counter += 1
        task_id = f"TASK-{self.task_counter:04d}"

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            schedule=schedule,
            handler_name=handler_name,
            handler_args=handler_args or {},
            created_at=datetime.utcnow().isoformat(),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            notify_on_failure=notify_on_failure
        )

        # Calculate next run
        task.next_run = self._calculate_next_run(task)

        self.tasks[task_id] = task
        await self._save_tasks()

        logger.info(f"Created scheduled task: {task_id} - {name}")
        return task

    async def update_task(
        self,
        task_id: str,
        schedule: ScheduleConfig = None,
        status: TaskStatus = None,
        handler_args: Dict[str, Any] = None
    ) -> Optional[ScheduledTask]:
        """Update a scheduled task."""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        if schedule:
            task.schedule = schedule
            task.next_run = self._calculate_next_run(task)

        if status:
            task.status = status

        if handler_args is not None:
            task.handler_args = handler_args

        await self._save_tasks()
        return task

    async def delete_task(self, task_id: str) -> bool:
        """Delete a scheduled task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            await self._save_tasks()
            return True
        return False

    async def pause_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Pause a scheduled task."""
        return await self.update_task(task_id, status=TaskStatus.PAUSED)

    async def resume_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Resume a paused task."""
        return await self.update_task(task_id, status=TaskStatus.ACTIVE)

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_tasks_by_type(self, task_type: str) -> List[ScheduledTask]:
        """Get all tasks of a specific type."""
        return [t for t in self.tasks.values() if t.task_type == task_type]

    def get_active_tasks(self) -> List[ScheduledTask]:
        """Get all active tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.ACTIVE]

    # =========================================================================
    # SCHEDULING LOGIC
    # =========================================================================

    def _calculate_next_run(self, task: ScheduledTask) -> str:
        """Calculate the next run time for a task."""
        now = datetime.utcnow()
        schedule = task.schedule

        # Simple interval-based scheduling
        if schedule.interval_seconds > 0:
            if task.last_run:
                last = datetime.fromisoformat(task.last_run)
                next_run = last + timedelta(seconds=schedule.interval_seconds)
                if next_run <= now:
                    next_run = now + timedelta(seconds=schedule.interval_seconds)
            else:
                next_run = now + timedelta(seconds=schedule.interval_seconds)
            return next_run.isoformat()

        # Frequency-based scheduling
        if schedule.frequency == TaskFrequency.MINUTELY:
            next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

        elif schedule.frequency == TaskFrequency.HOURLY:
            minute = int(schedule.minute) if schedule.minute != "*" else 0
            next_run = now.replace(minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)

        elif schedule.frequency == TaskFrequency.DAILY:
            hour = int(schedule.hour) if schedule.hour != "*" else 0
            minute = int(schedule.minute) if schedule.minute != "*" else 0
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

        elif schedule.frequency == TaskFrequency.WEEKLY:
            target_dow = int(schedule.day_of_week) if schedule.day_of_week != "*" else 0
            hour = int(schedule.hour) if schedule.hour != "*" else 0
            minute = int(schedule.minute) if schedule.minute != "*" else 0

            days_ahead = target_dow - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7

            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)

        else:
            # Default: 1 hour from now
            next_run = now + timedelta(hours=1)

        return next_run.isoformat()

    def _should_run(self, task: ScheduledTask) -> bool:
        """Check if a task should run now."""
        if task.status != TaskStatus.ACTIVE:
            return False

        if not task.next_run:
            return False

        now = datetime.utcnow()
        next_run = datetime.fromisoformat(task.next_run)

        return now >= next_run

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def run_task(self, task_id: str, force: bool = False) -> Optional[TaskExecution]:
        """
        Run a task immediately.

        Args:
            task_id: Task to run
            force: Run even if not scheduled

        Returns:
            Execution record
        """
        task = self.tasks.get(task_id)
        if not task:
            return None

        if not force and task.status not in [TaskStatus.ACTIVE, TaskStatus.PAUSED]:
            return None

        # Check handler exists
        if task.handler_name not in self.handlers:
            logger.error(f"Handler not found: {task.handler_name}")
            return None

        # Create execution record
        self.execution_counter += 1
        execution = TaskExecution(
            execution_id=f"EXEC-{self.execution_counter:06d}",
            task_id=task_id,
            started_at=datetime.utcnow().isoformat()
        )

        # Update task status
        task.status = TaskStatus.RUNNING
        task.run_count += 1

        start_time = datetime.utcnow()

        try:
            handler = self.handlers[task.handler_name]

            # Run with timeout
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(**task.handler_args),
                    timeout=task.timeout_seconds
                )
            else:
                result = handler(**task.handler_args)

            execution.status = "success"
            execution.result = str(result) if result else "OK"
            task.last_result = "success"

        except asyncio.TimeoutError:
            execution.status = "timeout"
            execution.error_message = f"Task timed out after {task.timeout_seconds}s"
            task.failure_count += 1
            task.last_result = "timeout"

            if task.notify_on_failure:
                await self._notify(task, f"Task {task.name} timed out")

        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            task.failure_count += 1
            task.last_result = "failed"

            if task.notify_on_failure:
                await self._notify(task, f"Task {task.name} failed: {e}")

            logger.error(f"Task {task_id} failed: {e}")

        # Complete execution
        execution.completed_at = datetime.utcnow().isoformat()
        execution.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        # Update task
        task.status = TaskStatus.ACTIVE
        task.last_run = execution.started_at
        task.next_run = self._calculate_next_run(task)

        await self._save_tasks()
        await self._save_execution(execution)

        self.execution_history.append(execution)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]

        logger.info(f"Task {task_id} completed: {execution.status}")
        return execution

    # =========================================================================
    # SCHEDULER LOOP
    # =========================================================================

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler started")

    async def stop(self):
        """Stop the scheduler loop."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                # Check all tasks
                for task_id, task in list(self.tasks.items()):
                    if self._should_run(task):
                        asyncio.create_task(self.run_task(task_id))

                # Sleep for a bit
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_schedule_display(self) -> str:
        """Get scheduled tasks display for Discord."""
        if not self.tasks:
            return "No scheduled tasks."

        output = ["## 📅 Scheduled Tasks\n"]

        # Group by type
        by_type: Dict[str, List[ScheduledTask]] = {}
        for task in self.tasks.values():
            if task.task_type not in by_type:
                by_type[task.task_type] = []
            by_type[task.task_type].append(task)

        for task_type, tasks in sorted(by_type.items()):
            output.append(f"\n### {task_type}")
            for task in sorted(tasks, key=lambda t: t.next_run or ""):
                status_emoji = {
                    TaskStatus.ACTIVE: "🟢",
                    TaskStatus.PAUSED: "⏸️",
                    TaskStatus.DISABLED: "⚫",
                    TaskStatus.RUNNING: "🔄"
                }
                emoji = status_emoji.get(task.status, "❓")
                next_run = task.next_run[:16] if task.next_run else "N/A"
                output.append(f"  {emoji} `{task.task_id}` {task.name}")
                output.append(f"      Next: {next_run} | Runs: {task.run_count}")

        return "\n".join(output)

    def get_task_details(self, task_id: str) -> str:
        """Get detailed task information."""
        task = self.tasks.get(task_id)
        if not task:
            return f"Task {task_id} not found."

        output = [f"## Task Details: {task.name}\n"]
        output.append(task.format_for_discord())

        output.append("\n**Schedule:**")
        if task.schedule.interval_seconds:
            output.append(f"  Every {task.schedule.interval_seconds}s")
        else:
            output.append(f"  Frequency: {task.schedule.frequency.value}")
            output.append(f"  Hour: {task.schedule.hour} | Minute: {task.schedule.minute}")

        output.append(f"\n**Handler:** {task.handler_name}")
        if task.handler_args:
            output.append(f"**Args:** {json.dumps(task.handler_args)[:100]}")

        output.append(f"\n**Settings:**")
        output.append(f"  Timeout: {task.timeout_seconds}s")
        output.append(f"  Max Retries: {task.max_retries}")
        output.append(f"  Notify on Failure: {task.notify_on_failure}")

        # Recent executions
        recent = [e for e in self.execution_history if e.task_id == task_id][-5:]
        if recent:
            output.append("\n**Recent Executions:**")
            for exec in reversed(recent):
                status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏱️"}
                emoji = status_emoji.get(exec.status, "❓")
                output.append(f"  {emoji} {exec.started_at[:16]} ({exec.duration_seconds:.1f}s)")

        return "\n".join(output)

    def get_execution_history(self, task_id: str = None, limit: int = 20) -> str:
        """Get execution history display."""
        executions = self.execution_history
        if task_id:
            executions = [e for e in executions if e.task_id == task_id]

        executions = list(reversed(executions[-limit:]))

        if not executions:
            return "No execution history."

        output = ["## Execution History\n"]

        for exec in executions:
            status_emoji = {"success": "✅", "failed": "❌", "timeout": "⏱️", "running": "🔄"}
            emoji = status_emoji.get(exec.status, "❓")

            output.append(
                f"{emoji} `{exec.execution_id}` | {exec.task_id}\n"
                f"   {exec.started_at[:16]} | {exec.duration_seconds:.1f}s | {exec.status}"
            )
            if exec.error_message:
                output.append(f"   Error: {exec.error_message[:50]}...")

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get or create the scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


def set_scheduler(scheduler: TaskScheduler):
    """Set the scheduler instance."""
    global _scheduler
    _scheduler = scheduler
