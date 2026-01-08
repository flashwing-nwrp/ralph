"""
Claude Code Executor

This module provides the execution engine that allows Discord agents
to invoke Claude Code CLI for actual work on the codebase.

The key insight from Ralph: each invocation gets fresh context,
but memory persists via git history, progress.txt, and structured files.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("claude_executor")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result from a Claude Code execution."""
    task_id: str
    status: TaskStatus
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp
        }


class ClaudeExecutor:
    """
    Executes tasks using Claude Code CLI.

    Each agent uses this executor to perform actual work on the codebase.
    The executor manages:
    - CLI invocation with proper arguments
    - Output capture and parsing
    - Timeout handling
    - Progress file updates (Ralph pattern)
    """

    def __init__(
        self,
        project_dir: str = None,
        claude_cmd: str = None,
        timeout: int = 300,  # 5 minutes default
        progress_file: str = "agent_progress.txt"
    ):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.claude_cmd = claude_cmd or os.getenv("CLAUDE_CMD", "claude")
        self.timeout = timeout
        self.progress_file = self.project_dir / progress_file

        # Task tracking
        self.task_counter = 0
        self.running_tasks: dict[str, asyncio.Task] = {}

    def _generate_task_id(self, agent_name: str) -> str:
        """Generate unique task ID."""
        self.task_counter += 1
        timestamp = datetime.utcnow().strftime("%H%M%S")
        return f"{agent_name[:3].upper()}-{timestamp}-{self.task_counter:04d}"

    async def execute(
        self,
        agent_name: str,
        agent_role: str,
        task_prompt: str,
        context: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None
    ) -> TaskResult:
        """
        Execute a task using Claude Code CLI.

        Args:
            agent_name: Name of the agent (e.g., "Tuning Agent")
            agent_role: Role description for context
            task_prompt: The specific task to perform
            context: Additional context (e.g., from other agents)
            on_progress: Callback for streaming progress updates
            timeout: Override default timeout

        Returns:
            TaskResult with output and status
        """
        task_id = self._generate_task_id(agent_name)
        start_time = datetime.utcnow()

        # Build the full prompt with agent persona
        full_prompt = self._build_prompt(agent_name, agent_role, task_prompt, context)

        logger.info(f"[{task_id}] Starting task for {agent_name}")
        logger.debug(f"[{task_id}] Prompt: {task_prompt[:200]}...")

        try:
            # Build command arguments
            # -p/--print: Non-interactive mode (required for automation)
            # --dangerously-skip-permissions: Bypass permission prompts (for autonomous operation)
            cmd = [
                self.claude_cmd,
                "--print",  # Non-interactive mode
                "--dangerously-skip-permissions",  # Required for autonomous operation
                full_prompt,
            ]

            # Execute Claude Code
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL
            )

            # Wait with timeout
            effective_timeout = timeout or self.timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                duration = (datetime.utcnow() - start_time).total_seconds()
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    output="",
                    error=f"Task timed out after {effective_timeout} seconds",
                    duration_seconds=duration,
                    timestamp=start_time.isoformat()
                )

            duration = (datetime.utcnow() - start_time).total_seconds()
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # Check for success
            if process.returncode == 0:
                status = TaskStatus.COMPLETED
                # Log to progress file (Ralph pattern)
                await self._log_progress(task_id, agent_name, task_prompt, output)
            else:
                status = TaskStatus.FAILED
                output = f"{output}\n\nSTDERR:\n{error_output}" if error_output else output

            logger.info(f"[{task_id}] Completed with status {status.value} in {duration:.1f}s")

            return TaskResult(
                task_id=task_id,
                status=status,
                output=output,
                error=error_output if status == TaskStatus.FAILED else None,
                duration_seconds=duration,
                timestamp=start_time.isoformat()
            )

        except FileNotFoundError:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                output="",
                error=f"Claude CLI not found: {self.claude_cmd}. Install with: npm install -g @anthropic-ai/claude-code",
                duration_seconds=0,
                timestamp=start_time.isoformat()
            )
        except Exception as e:
            logger.exception(f"[{task_id}] Execution error")
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                output="",
                error=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=start_time.isoformat()
            )

    def _build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        task_prompt: str,
        context: Optional[str] = None
    ) -> str:
        """Build the full prompt with agent persona and context."""

        prompt_parts = [
            f"# Agent Identity",
            f"You are the **{agent_name}** for the RALPH autonomous trading system.",
            f"",
            f"## Your Role",
            f"{agent_role}",
            f"",
            f"## Working Guidelines",
            f"- Focus ONLY on your area of expertise",
            f"- Be concise and actionable in your outputs",
            f"- If you need input from another agent, clearly state what you need",
            f"- Update relevant files when making changes",
            f"- Summarize your findings/actions at the end",
            f"",
        ]

        if context:
            prompt_parts.extend([
                f"## Context from Other Agents",
                f"{context}",
                f"",
            ])

        prompt_parts.extend([
            f"## Current Task",
            f"{task_prompt}",
            f"",
            f"Execute this task now. Be thorough but focused.",
        ])

        return "\n".join(prompt_parts)

    async def _log_progress(
        self,
        task_id: str,
        agent_name: str,
        task: str,
        output: str
    ):
        """
        Log task completion to progress file (Ralph pattern).
        This provides memory across fresh context windows.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Extract summary (first 500 chars of output)
        summary = output[:500].replace("\n", " ").strip()
        if len(output) > 500:
            summary += "..."

        entry = f"""
---
[{timestamp}] {agent_name} | Task: {task_id}
Task: {task[:200]}
Summary: {summary}
---
"""

        try:
            async with asyncio.Lock():
                with open(self.progress_file, "a", encoding="utf-8") as f:
                    f.write(entry)
        except Exception as e:
            logger.error(f"Failed to log progress: {e}")

    async def execute_in_background(
        self,
        agent_name: str,
        agent_role: str,
        task_prompt: str,
        context: Optional[str] = None,
        callback: Optional[Callable[[TaskResult], None]] = None
    ) -> str:
        """
        Execute task in background, return task_id immediately.

        Args:
            callback: Called when task completes with TaskResult

        Returns:
            task_id for tracking
        """
        task_id = self._generate_task_id(agent_name)

        async def _run():
            result = await self.execute(
                agent_name, agent_role, task_prompt, context
            )
            result.task_id = task_id  # Ensure consistent ID

            if callback:
                callback(result)

            # Cleanup
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            return result

        task = asyncio.create_task(_run())
        self.running_tasks[task_id] = task

        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Check if a task is still running."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if task.done():
                return TaskStatus.COMPLETED
            return TaskStatus.RUNNING
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return True
        return False


class AgentCoordinator:
    """
    Coordinates task handoffs between agents.

    Implements the autonomous workflow where agents can trigger
    each other without user intervention.
    """

    def __init__(self, executor: ClaudeExecutor):
        self.executor = executor
        self.context_store: dict[str, str] = {}  # agent -> latest output
        self.workflow_queue: asyncio.Queue = asyncio.Queue()

    def store_context(self, agent_name: str, output: str):
        """Store agent output for other agents to reference."""
        # Keep last output per agent, trimmed for context efficiency
        self.context_store[agent_name] = output[:2000]

    def get_context_for(self, requesting_agent: str, from_agents: list[str]) -> str:
        """Get relevant context from other agents."""
        context_parts = []
        for agent in from_agents:
            if agent in self.context_store:
                context_parts.append(f"### From {agent}:\n{self.context_store[agent]}")

        return "\n\n".join(context_parts) if context_parts else ""

    async def queue_handoff(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        context: str
    ):
        """Queue a task handoff from one agent to another."""
        await self.workflow_queue.put({
            "from": from_agent,
            "to": to_agent,
            "task": task,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        })

    async def get_next_handoff(self) -> Optional[dict]:
        """Get next queued handoff (non-blocking)."""
        try:
            return self.workflow_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
