"""
Mission Manager for RALPH Agent Ensemble

Handles the initial goal/mission system where:
1. User provides a high-level mission via !mission command
2. Strategy Agent receives and breaks down the mission
3. Tasks are delegated to appropriate agents
4. Progress is tracked and reported

The mission persists across agent sessions via the mission file.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("mission_manager")


class MissionStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MissionTask:
    """A subtask within a mission."""
    task_id: str
    description: str
    assigned_to: str  # agent type
    status: str = "pending"
    priority: str = "medium"
    dependencies: List[str] = field(default_factory=list)
    output: str = ""
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "output": self.output,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MissionTask":
        return cls(**data)


@dataclass
class Mission:
    """A high-level mission/goal for the agent ensemble."""
    mission_id: str
    objective: str
    status: MissionStatus = MissionStatus.PENDING
    created_by: str = "operator"
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    tasks: List[MissionTask] = field(default_factory=list)
    context: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tasks": [t.to_dict() for t in self.tasks],
            "context": self.context,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mission":
        tasks = [MissionTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            mission_id=data["mission_id"],
            objective=data["objective"],
            status=MissionStatus(data.get("status", "pending")),
            created_by=data.get("created_by", "operator"),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            tasks=tasks,
            context=data.get("context", ""),
            notes=data.get("notes", [])
        )

    def add_task(self, description: str, assigned_to: str, priority: str = "medium", dependencies: List[str] = None) -> MissionTask:
        """Add a new task to the mission."""
        task_id = f"{self.mission_id}-T{len(self.tasks) + 1:02d}"
        task = MissionTask(
            task_id=task_id,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            dependencies=dependencies or [],
            created_at=datetime.utcnow().isoformat()
        )
        self.tasks.append(task)
        return task

    def get_pending_tasks(self, agent_type: str = None) -> List[MissionTask]:
        """Get pending tasks, optionally filtered by agent."""
        pending = [t for t in self.tasks if t.status == "pending"]
        if agent_type:
            pending = [t for t in pending if t.assigned_to == agent_type]
        return pending

    def get_next_task(self, agent_type: str = None) -> Optional[MissionTask]:
        """Get the next task that can be executed (dependencies met)."""
        pending = self.get_pending_tasks(agent_type)

        for task in pending:
            # Check if all dependencies are completed
            deps_met = all(
                any(t.task_id == dep and t.status == "completed" for t in self.tasks)
                for dep in task.dependencies
            )
            if deps_met:
                return task
        return None

    def complete_task(self, task_id: str, output: str = ""):
        """Mark a task as completed."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = "completed"
                task.output = output
                task.completed_at = datetime.utcnow().isoformat()
                break

        # Check if all tasks completed
        if all(t.status == "completed" for t in self.tasks):
            self.status = MissionStatus.COMPLETED
            self.completed_at = datetime.utcnow().isoformat()

    def get_progress(self) -> Dict[str, int]:
        """Get mission progress statistics."""
        total = len(self.tasks)
        completed = len([t for t in self.tasks if t.status == "completed"])
        in_progress = len([t for t in self.tasks if t.status == "in_progress"])
        pending = len([t for t in self.tasks if t.status == "pending"])

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "percent": int((completed / total) * 100) if total > 0 else 0
        }


class MissionManager:
    """
    Manages missions for the RALPH agent ensemble.

    Responsibilities:
    - Create and track missions
    - Persist mission state to file
    - Coordinate task delegation
    - Report progress
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.mission_file = self.project_dir / "current_mission.json"
        self.mission_history_file = self.project_dir / "mission_history.json"

        self.current_mission: Optional[Mission] = None
        self.mission_counter = 0
        self._lock = asyncio.Lock()

        # Load existing mission if present
        self._load_current_mission()

    def _load_current_mission(self):
        """Load the current mission from file."""
        try:
            if self.mission_file.exists():
                with open(self.mission_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.current_mission = Mission.from_dict(data)
                    # Extract counter from mission ID
                    try:
                        self.mission_counter = int(self.current_mission.mission_id.split("-")[1])
                    except (ValueError, IndexError):
                        self.mission_counter = 1
                    logger.info(f"Loaded mission: {self.current_mission.mission_id}")
        except Exception as e:
            logger.error(f"Failed to load mission: {e}")
            self.current_mission = None

    async def _save_current_mission(self):
        """Save the current mission to file."""
        if not self.current_mission:
            return

        async with self._lock:
            try:
                with open(self.mission_file, "w", encoding="utf-8") as f:
                    json.dump(self.current_mission.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save mission: {e}")

    async def _archive_mission(self, mission: Mission):
        """Archive a completed mission to history."""
        try:
            history = []
            if self.mission_history_file.exists():
                with open(self.mission_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)

            history.append(mission.to_dict())

            with open(self.mission_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to archive mission: {e}")

    async def create_mission(self, objective: str, context: str = "", created_by: str = "operator") -> Mission:
        """
        Create a new mission.

        Args:
            objective: The high-level goal/mission statement
            context: Additional context or constraints
            created_by: Who created the mission (operator or agent)

        Returns:
            The created Mission object
        """
        # Archive current mission if exists and completed
        if self.current_mission and self.current_mission.status == MissionStatus.COMPLETED:
            await self._archive_mission(self.current_mission)

        # Create new mission
        self.mission_counter += 1
        mission_id = f"M-{self.mission_counter:04d}"

        self.current_mission = Mission(
            mission_id=mission_id,
            objective=objective,
            status=MissionStatus.PENDING,
            created_by=created_by,
            created_at=datetime.utcnow().isoformat(),
            context=context
        )

        await self._save_current_mission()
        logger.info(f"Created mission: {mission_id} - {objective[:50]}...")

        return self.current_mission

    async def start_mission(self):
        """Mark the current mission as in progress."""
        if self.current_mission:
            self.current_mission.status = MissionStatus.IN_PROGRESS
            self.current_mission.started_at = datetime.utcnow().isoformat()
            await self._save_current_mission()

    async def add_task_to_mission(
        self,
        description: str,
        assigned_to: str,
        priority: str = "medium",
        dependencies: List[str] = None
    ) -> Optional[MissionTask]:
        """Add a task to the current mission."""
        if not self.current_mission:
            logger.warning("No active mission to add task to")
            return None

        task = self.current_mission.add_task(description, assigned_to, priority, dependencies)
        await self._save_current_mission()
        return task

    async def complete_task(self, task_id: str, output: str = ""):
        """Mark a task as completed."""
        if not self.current_mission:
            return

        self.current_mission.complete_task(task_id, output)
        await self._save_current_mission()

        # Add note about completion
        self.current_mission.notes.append(
            f"[{datetime.utcnow().strftime('%H:%M:%S')}] Task {task_id} completed"
        )

    async def update_task_status(self, task_id: str, status: str):
        """Update a task's status."""
        if not self.current_mission:
            return

        for task in self.current_mission.tasks:
            if task.task_id == task_id:
                task.status = status
                break

        await self._save_current_mission()

    def get_current_mission(self) -> Optional[Mission]:
        """Get the current active mission."""
        return self.current_mission

    def get_mission_summary(self) -> str:
        """Get a formatted summary of the current mission."""
        if not self.current_mission:
            return "No active mission. Use `!mission <objective>` to set one."

        m = self.current_mission
        progress = m.get_progress()

        summary = [
            f"**Mission {m.mission_id}**: {m.objective}",
            f"**Status**: {m.status.value}",
            f"**Progress**: {progress['completed']}/{progress['total']} tasks ({progress['percent']}%)",
            ""
        ]

        if m.tasks:
            summary.append("**Tasks:**")
            for task in m.tasks:
                status_icon = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "failed": "❌"
                }.get(task.status, "❓")

                summary.append(
                    f"  {status_icon} `{task.task_id}` [{task.assigned_to}] {task.description[:60]}"
                )

        return "\n".join(summary)

    async def pause_mission(self):
        """Pause the current mission."""
        if self.current_mission:
            self.current_mission.status = MissionStatus.PAUSED
            self.current_mission.notes.append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] Mission paused"
            )
            await self._save_current_mission()

    async def resume_mission(self):
        """Resume the current mission."""
        if self.current_mission:
            self.current_mission.status = MissionStatus.IN_PROGRESS
            self.current_mission.notes.append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] Mission resumed"
            )
            await self._save_current_mission()


# Singleton instance
_mission_manager: Optional[MissionManager] = None


def get_mission_manager() -> MissionManager:
    """Get or create the mission manager instance."""
    global _mission_manager
    if _mission_manager is None:
        _mission_manager = MissionManager()
    return _mission_manager


def set_mission_manager(manager: MissionManager):
    """Set the mission manager instance."""
    global _mission_manager
    _mission_manager = manager
