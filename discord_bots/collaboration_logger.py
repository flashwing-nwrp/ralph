"""
Collaboration Logger for RALPH Agent Ensemble

Captures detailed logs of agent collaboration for post-hoc analysis.
Logs are structured JSON for easy parsing and analysis.

Log events:
- Mission lifecycle (start, planning, complete, failed)
- Task creation and assignment
- Agent handoffs
- Claude Code executions (prompt, response, duration)
- Errors and retries
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """Types of events to log."""
    MISSION_START = "mission_start"
    MISSION_PLANNING = "mission_planning"
    MISSION_COMPLETE = "mission_complete"
    MISSION_FAILED = "mission_failed"

    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    AGENT_HANDOFF = "agent_handoff"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"

    CLAUDE_EXECUTION = "claude_execution"
    CLAUDE_RESPONSE = "claude_response"

    FILE_DISCOVERY = "file_discovery"
    PROMPT_SENT = "prompt_sent"

    ERROR = "error"
    WARNING = "warning"


@dataclass
class LogEvent:
    """A single log event."""
    timestamp: str
    event_type: str
    agent: Optional[str]
    mission_id: Optional[str]
    task_id: Optional[str]
    data: Dict[str, Any]
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class CollaborationLogger:
    """
    Logs all agent collaboration events for analysis.

    Creates two log files:
    1. collaboration_log.jsonl - Structured JSON lines for programmatic analysis
    2. collaboration_log.txt - Human-readable formatted log
    """

    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log files
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"collaboration_{timestamp}.jsonl"
        self.txt_path = self.log_dir / f"collaboration_{timestamp}.txt"

        # Also maintain a "latest" symlink/copy
        self.latest_jsonl = self.log_dir / "collaboration_latest.jsonl"
        self.latest_txt = self.log_dir / "collaboration_latest.txt"

        # In-memory buffer for quick access
        self.events: List[LogEvent] = []

        # Session tracking
        self.session_start = datetime.utcnow()
        self.current_mission_id: Optional[str] = None

        # Write header
        self._write_header()

        # Standard logger for console output
        self.logger = logging.getLogger("collaboration")

    def _write_header(self):
        """Write log file headers."""
        header = f"""
================================================================================
RALPH Agent Ensemble - Collaboration Log
Session Started: {self.session_start.isoformat()}
================================================================================

"""
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(header)

        # Copy to latest
        with open(self.latest_txt, "w", encoding="utf-8") as f:
            f.write(header)

    def _append_jsonl(self, event: LogEvent):
        """Append event to JSONL file."""
        line = json.dumps(event.to_dict()) + "\n"
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
        with open(self.latest_jsonl, "a", encoding="utf-8") as f:
            f.write(line)

    def _append_txt(self, formatted: str):
        """Append formatted text to log file."""
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        with open(self.latest_txt, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    def _format_event(self, event: LogEvent) -> str:
        """Format event for human-readable log."""
        timestamp = event.timestamp[:19]  # Trim microseconds
        agent = f"[{event.agent}]" if event.agent else ""
        mission = f"({event.mission_id})" if event.mission_id else ""
        task = f"<{event.task_id}>" if event.task_id else ""
        duration = f" [{event.duration_seconds:.1f}s]" if event.duration_seconds else ""

        header = f"{timestamp} | {event.event_type:20} | {agent:15} {mission} {task}{duration}"

        # Format data based on event type
        data_str = ""
        if event.data:
            if "message" in event.data:
                data_str = f"\n    {event.data['message']}"
            elif "summary" in event.data:
                data_str = f"\n    Summary: {event.data['summary'][:200]}"
            elif "files" in event.data:
                data_str = f"\n    Files: {', '.join(event.data['files'][:5])}"
            elif "prompt_preview" in event.data:
                preview = event.data['prompt_preview'][:150].replace('\n', ' ')
                data_str = f"\n    Prompt: {preview}..."
            elif "response_preview" in event.data:
                preview = event.data['response_preview'][:150].replace('\n', ' ')
                data_str = f"\n    Response: {preview}..."
            elif "error" in event.data:
                data_str = f"\n    ERROR: {event.data['error']}"
            elif "tasks" in event.data:
                tasks = event.data['tasks']
                if isinstance(tasks, list):
                    data_str = f"\n    Tasks ({len(tasks)}): " + ", ".join(
                        [f"{t.get('agent', '?')}" for t in tasks[:5]]
                    )

        return header + data_str

    def log(
        self,
        event_type: EventType,
        agent: str = None,
        mission_id: str = None,
        task_id: str = None,
        duration_seconds: float = None,
        **data
    ):
        """Log an event."""
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type.value,
            agent=agent,
            mission_id=mission_id or self.current_mission_id,
            task_id=task_id,
            data=data,
            duration_seconds=duration_seconds
        )

        # Store in memory
        self.events.append(event)

        # Write to files
        self._append_jsonl(event)
        self._append_txt(self._format_event(event))

        # Also log to console at debug level
        self.logger.debug(f"{event_type.value}: {agent or 'system'} - {data.get('message', '')[:100]}")

    # Convenience methods for common events

    def mission_start(self, mission_id: str, objective: str, initiated_by: str):
        """Log mission start."""
        self.current_mission_id = mission_id
        self.log(
            EventType.MISSION_START,
            mission_id=mission_id,
            objective=objective,
            initiated_by=initiated_by,
            message=f"Mission started: {objective[:100]}"
        )

    def mission_planning(self, mission_id: str, status: str, details: str = None):
        """Log mission planning phase."""
        self.log(
            EventType.MISSION_PLANNING,
            mission_id=mission_id,
            status=status,
            details=details,
            message=f"Planning: {status}"
        )

    def mission_complete(self, mission_id: str, total_tasks: int, duration_seconds: float):
        """Log mission completion."""
        self.log(
            EventType.MISSION_COMPLETE,
            mission_id=mission_id,
            duration_seconds=duration_seconds,
            total_tasks=total_tasks,
            message=f"Mission complete: {total_tasks} tasks in {duration_seconds:.1f}s"
        )
        self.current_mission_id = None

    def mission_failed(self, mission_id: str, error: str):
        """Log mission failure."""
        self.log(
            EventType.MISSION_FAILED,
            mission_id=mission_id,
            error=error,
            message=f"Mission failed: {error[:100]}"
        )
        self.current_mission_id = None

    def task_created(self, task_id: str, agent: str, description: str):
        """Log task creation."""
        self.log(
            EventType.TASK_CREATED,
            agent=agent,
            task_id=task_id,
            description=description[:200],
            message=f"Task created for {agent}: {description[:80]}"
        )

    def task_started(self, task_id: str, agent: str, description: str):
        """Log task start."""
        self.log(
            EventType.TASK_STARTED,
            agent=agent,
            task_id=task_id,
            description=description[:200],
            message=f"{agent} starting task"
        )

    def task_completed(self, task_id: str, agent: str, duration_seconds: float, summary: str):
        """Log task completion."""
        self.log(
            EventType.TASK_COMPLETED,
            agent=agent,
            task_id=task_id,
            duration_seconds=duration_seconds,
            summary=summary[:500],
            message=f"{agent} completed task in {duration_seconds:.1f}s"
        )

    def task_failed(self, task_id: str, agent: str, error: str):
        """Log task failure."""
        self.log(
            EventType.TASK_FAILED,
            agent=agent,
            task_id=task_id,
            error=error,
            message=f"{agent} task failed: {error[:100]}"
        )

    def agent_handoff(self, from_agent: str, to_agent: str, task_description: str):
        """Log agent-to-agent handoff."""
        self.log(
            EventType.AGENT_HANDOFF,
            agent=to_agent,
            from_agent=from_agent,
            to_agent=to_agent,
            task=task_description[:200],
            message=f"Handoff: {from_agent} -> {to_agent}"
        )

    def claude_execution(
        self,
        agent: str,
        prompt_preview: str,
        prompt_length: int,
        task_id: str = None
    ):
        """Log Claude Code execution start."""
        self.log(
            EventType.CLAUDE_EXECUTION,
            agent=agent,
            task_id=task_id,
            prompt_length=prompt_length,
            prompt_preview=prompt_preview[:300],
            message=f"Claude execution started ({prompt_length} chars)"
        )

    def claude_response(
        self,
        agent: str,
        response_preview: str,
        response_length: int,
        duration_seconds: float,
        task_id: str = None,
        tasks_found: int = 0
    ):
        """Log Claude Code response."""
        self.log(
            EventType.CLAUDE_RESPONSE,
            agent=agent,
            task_id=task_id,
            duration_seconds=duration_seconds,
            response_length=response_length,
            response_preview=response_preview[:300],
            tasks_found=tasks_found,
            message=f"Claude response ({response_length} chars, {duration_seconds:.1f}s, {tasks_found} tasks)"
        )

    def file_discovery(self, files: List[str], total_chars: int):
        """Log file discovery phase."""
        self.log(
            EventType.FILE_DISCOVERY,
            files=files,
            file_count=len(files),
            total_chars=total_chars,
            message=f"Discovered {len(files)} files ({total_chars} chars)"
        )

    def error(self, agent: str, error: str, context: str = None):
        """Log an error."""
        self.log(
            EventType.ERROR,
            agent=agent,
            error=error,
            context=context,
            message=f"Error in {agent}: {error[:100]}"
        )

    def warning(self, agent: str, message: str):
        """Log a warning."""
        self.log(
            EventType.WARNING,
            agent=agent,
            message=message
        )

    # Analysis helpers

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self.events:
            return {"message": "No events logged yet"}

        mission_events = [e for e in self.events if e.event_type.startswith("mission")]
        task_events = [e for e in self.events if e.event_type.startswith("task")]
        claude_events = [e for e in self.events if e.event_type.startswith("claude")]
        error_events = [e for e in self.events if e.event_type == "error"]

        # Calculate durations
        task_durations = [
            e.duration_seconds for e in task_events
            if e.duration_seconds and e.event_type == "task_completed"
        ]
        claude_durations = [
            e.duration_seconds for e in claude_events
            if e.duration_seconds
        ]

        return {
            "session_duration_seconds": (datetime.utcnow() - self.session_start).total_seconds(),
            "total_events": len(self.events),
            "missions": len([e for e in mission_events if e.event_type == "mission_start"]),
            "tasks_created": len([e for e in task_events if e.event_type == "task_created"]),
            "tasks_completed": len([e for e in task_events if e.event_type == "task_completed"]),
            "tasks_failed": len([e for e in task_events if e.event_type == "task_failed"]),
            "handoffs": len([e for e in self.events if e.event_type == "agent_handoff"]),
            "claude_executions": len(claude_events),
            "errors": len(error_events),
            "avg_task_duration": sum(task_durations) / len(task_durations) if task_durations else 0,
            "avg_claude_duration": sum(claude_durations) / len(claude_durations) if claude_durations else 0,
            "log_files": {
                "jsonl": str(self.jsonl_path),
                "txt": str(self.txt_path)
            }
        }

    def get_events_by_type(self, event_type: EventType) -> List[LogEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type.value]

    def get_events_by_agent(self, agent: str) -> List[LogEvent]:
        """Get all events for a specific agent."""
        return [e for e in self.events if e.agent == agent]

    def get_events_by_mission(self, mission_id: str) -> List[LogEvent]:
        """Get all events for a specific mission."""
        return [e for e in self.events if e.mission_id == mission_id]


# Global singleton
_collaboration_logger: Optional[CollaborationLogger] = None


def get_collaboration_logger() -> CollaborationLogger:
    """Get or create the collaboration logger."""
    global _collaboration_logger
    if _collaboration_logger is None:
        _collaboration_logger = CollaborationLogger()
    return _collaboration_logger


def set_collaboration_logger(logger: CollaborationLogger):
    """Set the collaboration logger instance."""
    global _collaboration_logger
    _collaboration_logger = logger
