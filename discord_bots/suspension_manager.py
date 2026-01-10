"""
Suspension Manager - Graceful Wind Down for RALPH Agents

Handles graceful suspension of agent activities for scenarios like:
- Approaching token limits
- End of work session
- Maintenance windows
- Weekly resets

Ensures:
- Current tasks complete cleanly
- Documentation is updated
- State is saved for resumption
- No broken code left behind
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SuspensionMode(str, Enum):
    GRACEFUL = "graceful"      # Complete tasks, run cleanup, then stop
    IMMEDIATE = "immediate"    # Stop after current task, skip cleanup
    EMERGENCY = "emergency"    # Stop immediately (use killswitch instead)


class SuspensionPhase(str, Enum):
    ACTIVE = "active"              # Normal operations
    WINDING_DOWN = "winding_down"  # Completing current tasks
    CLEANUP = "cleanup"            # Running cleanup tasks
    SUSPENDED = "suspended"        # Fully suspended


@dataclass
class SuspensionState:
    suspended: bool = False
    mode: Optional[str] = None
    phase: str = SuspensionPhase.ACTIVE.value
    started_at: Optional[str] = None
    reason: Optional[str] = None
    triggered_by: Optional[str] = None
    mission_id: Optional[str] = None
    pending_tasks: List[str] = None
    completed_cleanup: bool = False
    resumed_at: Optional[str] = None
    resumed_by: Optional[str] = None

    def __post_init__(self):
        if self.pending_tasks is None:
            self.pending_tasks = []


class SuspensionManager:
    """Manages graceful suspension and resumption of agent operations."""

    def __init__(self, state_file: str = None):
        if state_file is None:
            state_file = os.path.join(
                os.path.dirname(__file__),
                "data",
                "suspension_state.json"
            )
        self.state_file = state_file
        self._ensure_data_dir()
        self.state = SuspensionState()
        self._load_state()
        self._cleanup_callbacks: List[callable] = []
        self._orchestrator = None

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        data_dir = os.path.dirname(self.state_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    def _load_state(self):
        """Load suspension state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state = SuspensionState(**data)
            except (json.JSONDecodeError, IOError, TypeError) as e:
                logger.warning(f"Could not load suspension state: {e}")
                self.state = SuspensionState()

    def _save_state(self):
        """Save suspension state to file."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.state), f, indent=2)
        except IOError as e:
            logger.error(f"Could not save suspension state: {e}")

    def set_orchestrator(self, orchestrator):
        """Set reference to the orchestrator for task management."""
        self._orchestrator = orchestrator

    def register_cleanup_callback(self, callback: callable):
        """Register a callback to run during cleanup phase."""
        self._cleanup_callbacks.append(callback)

    def is_suspended(self) -> bool:
        """Check if operations are suspended."""
        return self.state.suspended

    def is_winding_down(self) -> bool:
        """Check if system is in wind-down phase."""
        return self.state.phase in [
            SuspensionPhase.WINDING_DOWN.value,
            SuspensionPhase.CLEANUP.value
        ]

    def should_accept_new_tasks(self) -> bool:
        """Check if new tasks should be accepted."""
        return self.state.phase == SuspensionPhase.ACTIVE.value

    def get_status(self) -> Dict[str, Any]:
        """Get current suspension status."""
        return {
            "suspended": self.state.suspended,
            "mode": self.state.mode,
            "phase": self.state.phase,
            "started_at": self.state.started_at,
            "reason": self.state.reason,
            "pending_tasks": len(self.state.pending_tasks),
            "cleanup_done": self.state.completed_cleanup,
            "mission_id": self.state.mission_id,
        }

    async def initiate_suspension(
        self,
        mode: SuspensionMode,
        reason: str,
        triggered_by: str
    ) -> Dict[str, Any]:
        """
        Initiate graceful suspension of operations.

        Args:
            mode: SuspensionMode (graceful, immediate)
            reason: Reason for suspension
            triggered_by: Who triggered the suspension

        Returns:
            Status dict with suspension details
        """
        logger.info(f"Initiating suspension: mode={mode.value}, reason={reason}")

        # Update state
        self.state.suspended = True
        self.state.mode = mode.value
        self.state.phase = SuspensionPhase.WINDING_DOWN.value
        self.state.started_at = datetime.now().isoformat()
        self.state.reason = reason
        self.state.triggered_by = triggered_by
        self.state.completed_cleanup = False

        # Capture current mission state
        await self._capture_mission_state()

        self._save_state()

        # Handle based on mode
        if mode == SuspensionMode.IMMEDIATE:
            # Skip to suspended state
            self.state.phase = SuspensionPhase.SUSPENDED.value
            self._save_state()
            return {"success": True, "phase": "suspended"}

        # For graceful mode, start wind-down process
        asyncio.create_task(self._run_graceful_suspension())

        return {
            "success": True,
            "phase": self.state.phase,
            "pending_tasks": len(self.state.pending_tasks)
        }

    async def _capture_mission_state(self):
        """Capture current mission and task state for later resumption."""
        try:
            from mission_manager import get_mission_manager
            mission_mgr = get_mission_manager()

            if mission_mgr and hasattr(mission_mgr, 'current_mission'):
                current = mission_mgr.get_current_mission()
                if current:
                    self.state.mission_id = current.get('id')

                    # Get pending tasks
                    tasks = current.get('tasks', [])
                    pending = [
                        t.get('task_id')
                        for t in tasks
                        if t.get('status') in ['pending', 'in_progress']
                    ]
                    self.state.pending_tasks = pending

        except Exception as e:
            logger.warning(f"Could not capture mission state: {e}")

    async def _run_graceful_suspension(self):
        """Run the graceful suspension process."""
        logger.info("Starting graceful suspension process...")

        # Phase 1: Wait for current tasks to complete (with timeout)
        await self._wait_for_tasks_completion(timeout_minutes=10)

        # Phase 2: Run cleanup
        self.state.phase = SuspensionPhase.CLEANUP.value
        self._save_state()

        await self._run_cleanup_tasks()

        # Phase 3: Mark as suspended
        self.state.phase = SuspensionPhase.SUSPENDED.value
        self.state.completed_cleanup = True
        self._save_state()

        logger.info("Graceful suspension complete")

    async def _wait_for_tasks_completion(self, timeout_minutes: int = 2):
        """Wait for in-progress tasks to complete."""
        logger.info(f"Waiting for tasks to complete (timeout: {timeout_minutes}m)...")

        start_time = datetime.now()
        timeout = timedelta(minutes=timeout_minutes)

        while datetime.now() - start_time < timeout:
            # Check if orchestrator has active tasks
            if self._orchestrator:
                active = getattr(self._orchestrator, 'active_workflows', {})
                running = [
                    w for w in active.values()
                    if w.get('status') == 'running'
                ]
                if not running:
                    logger.info("All tasks completed")
                    return
            else:
                # No orchestrator reference - just wait briefly and proceed
                logger.info("No orchestrator reference, proceeding with cleanup")
                await asyncio.sleep(5)
                return

            await asyncio.sleep(5)  # Check every 5 seconds

        logger.warning(f"Task completion timeout after {timeout_minutes} minutes")

    async def _run_cleanup_tasks(self):
        """Run cleanup tasks before full suspension."""
        logger.info("Running cleanup tasks...")

        cleanup_tasks = [
            self._save_mission_documentation(),
            self._save_learnings_summary(),
            self._generate_suspension_report(),
        ]

        # Run registered callbacks
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    cleanup_tasks.append(callback())
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Cleanup callback error: {e}")

        # Run all cleanup tasks
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("Cleanup tasks completed")

    async def _save_mission_documentation(self):
        """Document current mission state."""
        try:
            from mission_manager import get_mission_manager
            mission_mgr = get_mission_manager()

            if mission_mgr:
                # Mission manager should have its own state persistence
                # This just ensures it's saved
                if hasattr(mission_mgr, 'save_state'):
                    await mission_mgr.save_state()

            logger.info("Mission documentation saved")
        except Exception as e:
            logger.warning(f"Could not save mission documentation: {e}")

    async def _save_learnings_summary(self):
        """Save summary of learnings from current session."""
        try:
            from knowledge_base import get_knowledge_base
            kb = get_knowledge_base()

            if kb and hasattr(kb, 'save'):
                kb.save()

            logger.info("Learnings summary saved")
        except Exception as e:
            logger.warning(f"Could not save learnings: {e}")

    async def _generate_suspension_report(self):
        """Generate a report of the suspension."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "mode": self.state.mode,
                "reason": self.state.reason,
                "triggered_by": self.state.triggered_by,
                "mission_id": self.state.mission_id,
                "pending_tasks": self.state.pending_tasks,
                "summary": "Agent operations suspended gracefully."
            }

            report_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                f"suspension_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Suspension report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Could not generate suspension report: {e}")

    async def resume_operations(self, resumed_by: str) -> Dict[str, Any]:
        """
        Resume operations after suspension.

        Args:
            resumed_by: Who is resuming operations

        Returns:
            Status dict with resumption details
        """
        if not self.state.suspended:
            return {"success": False, "error": "Not suspended"}

        logger.info(f"Resuming operations, triggered by {resumed_by}")

        # Calculate suspended duration
        started = datetime.fromisoformat(self.state.started_at)
        duration = datetime.now() - started
        duration_str = str(duration).split('.')[0]  # Remove microseconds

        # Capture info before clearing
        mission_id = self.state.mission_id
        pending_tasks = self.state.pending_tasks

        # Clear suspension state
        self.state.suspended = False
        self.state.phase = SuspensionPhase.ACTIVE.value
        self.state.resumed_at = datetime.now().isoformat()
        self.state.resumed_by = resumed_by
        self._save_state()

        return {
            "success": True,
            "suspended_duration": duration_str,
            "mission_id": mission_id,
            "pending_tasks": len(pending_tasks) if pending_tasks else 0
        }

    def clear_state(self):
        """Clear suspension state (for testing or reset)."""
        self.state = SuspensionState()
        self._save_state()


# Singleton instance
_suspension_manager: Optional[SuspensionManager] = None


def get_suspension_manager() -> SuspensionManager:
    """Get the singleton suspension manager instance."""
    global _suspension_manager
    if _suspension_manager is None:
        _suspension_manager = SuspensionManager()
    return _suspension_manager
