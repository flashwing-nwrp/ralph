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

    def find_similar_completed_task(self, description: str, similarity_threshold: float = 0.7) -> Optional[MissionTask]:
        """
        Find a completed task with similar description to avoid duplicate work.

        Uses keyword overlap to detect similarity. Returns the completed task if found.
        """
        # Extract keywords from new task description (words > 3 chars)
        new_keywords = set(
            word.lower().strip('.,;:()[]{}"\'-')
            for word in description.split()
            if len(word) > 3
        )

        if not new_keywords:
            return None

        for task in self.tasks:
            if task.status != "completed":
                continue

            # Extract keywords from completed task
            task_keywords = set(
                word.lower().strip('.,;:()[]{}"\'-')
                for word in task.description.split()
                if len(word) > 3
            )

            if not task_keywords:
                continue

            # Calculate Jaccard similarity
            intersection = len(new_keywords & task_keywords)
            union = len(new_keywords | task_keywords)
            similarity = intersection / union if union > 0 else 0

            if similarity >= similarity_threshold:
                return task

        return None

    def is_task_assigned(self, description: str, agent_type: str = None) -> bool:
        """
        Check if a similar task is already assigned (pending or in_progress).

        Prevents duplicate handoffs for the same work.
        """
        # Extract key identifiers (file paths, function names, etc.)
        import re
        file_pattern = r'[`"\']([^`"\']+\.(py|json|txt|md))[`"\']'
        files_in_new = set(re.findall(file_pattern, description.lower()))

        for task in self.tasks:
            if task.status not in ["pending", "in_progress"]:
                continue

            # If checking specific agent, skip others
            if agent_type and task.assigned_to != agent_type:
                continue

            # Check file overlap
            files_in_task = set(re.findall(file_pattern, task.description.lower()))
            if files_in_new and files_in_task and files_in_new & files_in_task:
                return True

            # Check keyword overlap (stricter threshold for pending tasks)
            new_keywords = set(w.lower() for w in description.split() if len(w) > 4)
            task_keywords = set(w.lower() for w in task.description.split() if len(w) > 4)

            if new_keywords and task_keywords:
                overlap = len(new_keywords & task_keywords) / min(len(new_keywords), len(task_keywords))
                if overlap > 0.5:
                    return True

        return False


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

        # Recover any tasks that were in_progress when we crashed
        self._recover_crashed_tasks()

    def _recover_crashed_tasks(self):
        """
        Recover tasks that were in_progress when the bot crashed.

        On startup, any task marked as 'in_progress' is assumed to have
        been interrupted by a crash. We reset them to 'pending' so they
        can be picked up and retried.
        """
        if not self.current_mission:
            return

        recovered_count = 0
        for task in self.current_mission.tasks:
            if task.status == "in_progress":
                task.status = "pending"
                recovered_count += 1
                logger.info(f"Recovered crashed task: {task.task_id} -> pending")

        if recovered_count > 0:
            self.current_mission.notes.append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] Recovered {recovered_count} crashed task(s)"
            )
            # Save immediately (synchronous for startup)
            try:
                with open(self.mission_file, "w", encoding="utf-8") as f:
                    json.dump(self.current_mission.to_dict(), f, indent=2)
                logger.info(f"Recovered {recovered_count} crashed tasks to pending state")
            except Exception as e:
                logger.error(f"Failed to save recovered tasks: {e}")

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
        """Mark the current mission as in progress (thread-safe)."""
        async with self._lock:
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
        """Add a task to the current mission (thread-safe)."""
        async with self._lock:
            if not self.current_mission:
                logger.warning("No active mission to add task to")
                return None

            task = self.current_mission.add_task(description, assigned_to, priority, dependencies)

        await self._save_current_mission()

        # Sync to OpenProject if enabled
        await self._sync_task_to_openproject(task)

        return task

    async def complete_task(self, task_id: str, output: str = "", agent: str = None):
        """Mark a task as completed (thread-safe for parallel execution)."""
        async with self._lock:
            if not self.current_mission:
                return

            self.current_mission.complete_task(task_id, output)

            # Get agent from task if not provided
            if not agent:
                for task in self.current_mission.tasks:
                    if task.task_id == task_id:
                        agent = task.assigned_to
                        break

            # Add note about completion
            self.current_mission.notes.append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] Task {task_id} completed"
            )

        await self._save_current_mission()

        # Sync status to OpenProject with output as comment
        await self._sync_task_status_to_openproject(task_id, "completed", output, agent)

    async def update_task_status(self, task_id: str, status: str, message: str = None):
        """Update a task's status (thread-safe for parallel execution)."""
        agent = None
        async with self._lock:
            if not self.current_mission:
                return

            for task in self.current_mission.tasks:
                if task.task_id == task_id:
                    task.status = status
                    agent = task.assigned_to
                    break

        await self._save_current_mission()

        # Sync status to OpenProject with optional message
        await self._sync_task_status_to_openproject(task_id, status, message, agent)

    async def _sync_task_to_openproject(self, task: MissionTask):
        """Sync a task to OpenProject (creates work package)."""
        if not os.getenv("OPENPROJECT_API_KEY"):
            return  # OpenProject not configured

        try:
            from openproject_service import get_openproject_service
            service = get_openproject_service()

            await service.sync_task_to_openproject(
                task_id=task.task_id,
                description=task.description,
                agent=task.assigned_to,
                mission_id=self.current_mission.mission_id if self.current_mission else "",
                status=task.status.value if hasattr(task.status, 'value') else task.status,
                priority=task.priority if hasattr(task, 'priority') else "medium"
            )
            logger.debug(f"Synced task {task.task_id} to OpenProject")
        except Exception as e:
            logger.warning(f"Failed to sync task to OpenProject: {e}")

    async def _sync_task_status_to_openproject(self, task_id: str, status: str, output: str = None, agent: str = None):
        """Sync task status change to OpenProject, optionally with output as comment."""
        if not os.getenv("OPENPROJECT_API_KEY"):
            return  # OpenProject not configured

        try:
            from openproject_service import get_openproject_service
            service = get_openproject_service()

            # Get the WP ID for this task
            wp_id = service.get_wp_id(task_id)
            if not wp_id:
                logger.debug(f"No OpenProject WP linked to task {task_id}")
                return

            # Add output as comment if provided
            if output and status == "completed":
                # Truncate if too long
                comment_output = output[:3000] if len(output) > 3000 else output
                comment = f"**Task Completed**\n\n{comment_output}"
                await service.add_comment(wp_id, comment, agent)
                logger.debug(f"Added completion comment to WP #{wp_id}")

            # Update status
            status_map = {
                "completed": "Closed",
                "failed": "Rejected",
                "in_progress": "In progress",
                "pending": "New"
            }
            op_status = status_map.get(status, "New")
            await service.update_work_package(wp_id, status_name=op_status)
            logger.debug(f"Synced task {task_id} status to OpenProject WP #{wp_id}: {op_status}")
        except Exception as e:
            logger.warning(f"Failed to sync task status to OpenProject: {e}")

    def get_current_mission(self) -> Optional[Mission]:
        """Get the current active mission."""
        return self.current_mission

    def get_mission_summary(self) -> str:
        """Get a formatted summary of the current mission."""
        chunks = self.get_mission_summary_chunks()
        return "\n".join(chunks)

    def get_mission_summary_chunks(self, max_chunk_size: int = 1900) -> list:
        """
        Get mission summary as a list of chunks for Discord's message limit.

        Args:
            max_chunk_size: Max characters per chunk (Discord limit is 2000)

        Returns:
            List of message strings, each under max_chunk_size
        """
        if not self.current_mission:
            return ["No active mission. Use `!mission <objective>` to set one."]

        m = self.current_mission
        progress = m.get_progress()

        # Build header - show full objective
        objective = m.objective
        if len(objective) > 500:
            truncate_at = objective.rfind(' ', 0, 500)
            if truncate_at > 400:
                objective = objective[:truncate_at] + "..."
            else:
                objective = objective[:500] + "..."

        header = (
            f"**Mission {m.mission_id}**: {objective}\n"
            f"**Status**: {m.status.value}\n"
            f"**Progress**: {progress['completed']}/{progress['total']} tasks ({progress['percent']}%)"
        )

        # Build task lines with full descriptions (will be chunked if too long)
        task_lines = []
        for task in m.tasks:
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(task.status, "❓")

            # Show full description - chunking will handle overflow
            task_lines.append(f"  {status_icon} `{task.task_id}` [{task.assigned_to}] {task.description}")

        # Combine into chunks
        chunks = []
        current_content = header

        if task_lines:
            current_content += "\n\n**Tasks:**"

            for line in task_lines:
                # Check if adding this line would exceed limit
                if len(current_content) + len(line) + 1 > max_chunk_size:
                    # Save current chunk
                    chunks.append(current_content)
                    # Start new chunk (continuation)
                    current_content = f"**Tasks (continued):**\n{line}"
                else:
                    current_content += f"\n{line}"

        # Add final chunk
        if current_content:
            chunks.append(current_content)

        return chunks

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

    async def complete_mission(self):
        """Mark the current mission as completed."""
        if self.current_mission:
            self.current_mission.status = MissionStatus.COMPLETED
            self.current_mission.completed_at = datetime.utcnow().isoformat()
            self.current_mission.notes.append(
                f"[{datetime.utcnow().strftime('%H:%M:%S')}] Mission completed"
            )
            await self._save_current_mission()
            await self._archive_mission(self.current_mission)
            logger.info(f"Mission {self.current_mission.mission_id} completed and archived")


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
