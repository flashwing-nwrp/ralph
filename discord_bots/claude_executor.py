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

# Import knowledge base for context injection
try:
    from knowledge_base import get_knowledge_base
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False

logger = logging.getLogger("claude_executor")

# Import token optimizer for cost reduction
try:
    from token_optimizer import get_token_optimizer, TokenOptimizer
    TOKEN_OPTIMIZER_AVAILABLE = True
except ImportError:
    TOKEN_OPTIMIZER_AVAILABLE = False
    logger.warning("TokenOptimizer not available - using verbose prompts")


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

    # Keywords that indicate compute-heavy tasks needing longer timeouts
    LONG_RUNNING_KEYWORDS = [
        "backtest", "walk-forward", "validation", "train", "retrain", "re-train",
        "optimize", "sweep", "grid search", "bayesian", "cross-validation",
        "full analysis", "comprehensive", "all markets", "all symbols",
        "feature", "model performance", "compare", "regime"
    ]

    def __init__(
        self,
        project_dir: str = None,
        claude_cmd: str = None,
        timeout: int = 300,  # 5 minutes default
        long_timeout: int = 1800,  # 30 minutes for compute-heavy tasks (backtests can span years)
        progress_file: str = "agent_progress.txt"
    ):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        # Normalize Claude command path for Windows compatibility
        raw_claude_cmd = claude_cmd or os.getenv("CLAUDE_CMD", "claude")
        self.claude_cmd = str(Path(raw_claude_cmd)) if raw_claude_cmd else "claude"
        self.timeout = timeout
        self.long_timeout = long_timeout
        self.progress_file = self.project_dir / progress_file

        # Task tracking
        self.task_counter = 0
        self.running_tasks: dict[str, asyncio.Task] = {}

    def _get_task_timeout(self, task_prompt: str, agent_name: str = "") -> int:
        """Determine appropriate timeout based on task content and agent type."""
        # Backtest Agent always gets extended timeout
        if "backtest" in agent_name.lower():
            logger.info(f"Backtest Agent task - using extended timeout ({self.long_timeout}s)")
            return self.long_timeout

        # Check for long-running keywords in task prompt
        task_lower = task_prompt.lower()
        for keyword in self.LONG_RUNNING_KEYWORDS:
            if keyword in task_lower:
                logger.info(f"Detected long-running task (keyword: {keyword}), using extended timeout ({self.long_timeout}s)")
                return self.long_timeout

        logger.debug(f"Using default timeout ({self.timeout}s)")
        return self.timeout

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
        timeout: Optional[int] = None,
        model: Optional[str] = None,
        is_user_facing: bool = False,
        skip_cache: bool = False
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
            model: Claude model to use (e.g., "claude-opus-4-5-20250101", "claude-sonnet-4-20250514")
            is_user_facing: If True, use verbose prompts for quality (default: False for internal work)
            skip_cache: If True, skip cache lookup and always execute

        Returns:
            TaskResult with output and status
        """
        task_id = self._generate_task_id(agent_name)
        start_time = datetime.utcnow()

        # Extract agent type from name (e.g., "Tuning Agent" -> "tuning")
        agent_type = agent_name.lower().replace(" agent", "").strip()

        # =================================================================
        # TOKEN OPTIMIZATION: Check cache for non-user-facing work
        # =================================================================
        if TOKEN_OPTIMIZER_AVAILABLE and not skip_cache and not is_user_facing:
            optimizer = get_token_optimizer()

            # Check if we have a cached result
            cached_output = optimizer.check_cache(task_prompt, agent_type)
            if cached_output:
                logger.info(f"[{task_id}] Cache hit for {agent_name} - returning cached result")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output=f"[CACHED] {cached_output}",
                    duration_seconds=0.0,
                    timestamp=start_time.isoformat()
                )

        # =================================================================
        # TOKEN OPTIMIZATION: Build optimized prompt with session priming
        # =================================================================
        session_needs_priming = False
        if TOKEN_OPTIMIZER_AVAILABLE and not is_user_facing:
            optimizer = get_token_optimizer()
            full_prompt, tokens_saved, session_needs_priming = optimizer.build_optimized_prompt(
                agent_type=agent_type,
                task=task_prompt,
                context=context or "",
                is_user_facing=is_user_facing,
                full_role=agent_role
            )
            if session_needs_priming:
                logger.info(f"[{task_id}] Session priming for {agent_name} (full context)")
            elif tokens_saved > 0:
                logger.info(f"[{task_id}] Using minimal prompt (est. {tokens_saved} tokens saved)")
        else:
            # Build the full prompt with agent persona (verbose mode)
            full_prompt = self._build_prompt(agent_name, agent_role, task_prompt, context)

        logger.info(f"[{task_id}] Starting task for {agent_name}")
        logger.debug(f"[{task_id}] Prompt: {task_prompt[:200]}...")

        try:
            # Build command arguments
            # -p/--print: Non-interactive mode (required for automation)
            # --dangerously-skip-permissions: Bypass permission prompts (for autonomous operation)

            # On Windows, .cmd files need to go through cmd.exe
            import sys
            is_windows_cmd = sys.platform == "win32" and self.claude_cmd.lower().endswith(".cmd")
            logger.debug(f"Platform: {sys.platform}, claude_cmd: {self.claude_cmd}, is_windows_cmd: {is_windows_cmd}")

            # Use stdin for prompt to avoid Windows command-line length limit (~8192 chars)
            # Claude Code accepts prompt from stdin with --print flag
            if is_windows_cmd:
                # Use cmd.exe /c to execute .cmd files on Windows
                cmd = [
                    "cmd.exe", "/c",
                    self.claude_cmd,
                    "--print",
                    "--dangerously-skip-permissions",
                ]
                # Add model selection if specified
                if model:
                    cmd.extend(["--model", model])
                    logger.info(f"[{task_id}] Using model: {model}")
                cmd.append("-")  # Read prompt from stdin
            else:
                cmd = [
                    self.claude_cmd,
                    "--print",  # Non-interactive mode
                    "--dangerously-skip-permissions",  # Required for autonomous operation
                ]
                # Add model selection if specified
                if model:
                    cmd.extend(["--model", model])
                    logger.info(f"[{task_id}] Using model: {model}")
                cmd.append("-")  # Read prompt from stdin

            logger.debug(f"Command: {cmd}")
            logger.debug(f"Prompt length: {len(full_prompt)} chars")

            # Execute Claude Code with prompt via stdin
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.project_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.PIPE  # Enable stdin for prompt
                )
                logger.debug(f"Process created successfully, PID: {process.pid}")
            except FileNotFoundError as e:
                logger.error(f"FileNotFoundError creating subprocess: {e}")
                raise

            # Wait with timeout, sending prompt via stdin
            # Use intelligent timeout detection for long-running tasks
            effective_timeout = timeout or self._get_task_timeout(task_prompt, agent_name)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=full_prompt.encode("utf-8")),
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

                # =================================================================
                # TOKEN OPTIMIZATION: Cache successful output for reuse
                # =================================================================
                if TOKEN_OPTIMIZER_AVAILABLE and not is_user_facing:
                    optimizer = get_token_optimizer()
                    # Estimate tokens used (rough: 1 token ≈ 4 chars)
                    estimated_tokens = (len(full_prompt) + len(output)) // 4
                    optimizer.cache_output(task_prompt, agent_type, output, estimated_tokens)
                    # Mark session as primed if this was a priming prompt
                    optimizer.update_session(
                        agent_type,
                        tokens_used=estimated_tokens,
                        mark_primed=session_needs_priming
                    )
            else:
                status = TaskStatus.FAILED
                # Try to extract error from Claude Code JSON response
                extracted_error = self._extract_claude_error(output)
                if extracted_error:
                    error_output = extracted_error
                    logger.warning(f"[{task_id}] Claude Code error: {extracted_error[:200]}")
                elif error_output:
                    output = f"{output}\n\nSTDERR:\n{error_output}"

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

    def _extract_claude_error(self, output: str) -> Optional[str]:
        """
        Extract error message from Claude Code JSON response.

        Claude Code returns JSON like:
        {"type":"result","subtype":"error_during_execution","is_error":true,...}
        """
        if not output:
            return None

        try:
            # Try to parse as JSON
            import json
            data = json.loads(output)

            if isinstance(data, dict):
                # Check for Claude Code error response
                if data.get("is_error") or data.get("subtype") == "error_during_execution":
                    # Try to get error message from various fields
                    error_msg = (
                        data.get("error") or
                        data.get("error_message") or
                        data.get("message") or
                        data.get("subtype", "Unknown Claude Code error")
                    )

                    # Include result field if present
                    result = data.get("result", "")
                    if result and isinstance(result, str) and len(result) < 500:
                        error_msg = f"{error_msg}: {result}"

                    return str(error_msg)
        except (json.JSONDecodeError, TypeError, KeyError):
            # Not JSON or unexpected format - check for error patterns in text
            pass

        # Check for common error patterns in raw output
        error_patterns = [
            "error_during_execution",
            "Error:",
            "Exception:",
            "Traceback",
            "FAILED",
        ]
        for pattern in error_patterns:
            if pattern in output:
                # Extract a snippet around the error
                idx = output.find(pattern)
                snippet = output[max(0, idx-50):min(len(output), idx+200)]
                return f"Execution error: ...{snippet}..."

        return None

    def _build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        task_prompt: str,
        context: Optional[str] = None
    ) -> str:
        """Build the full prompt with agent persona, context, and relevant learnings."""

        # Build a more action-oriented prompt
        prompt_parts = [
            f"You are {agent_name}. Your working directory is set to the project.",
            f"",
            f"TASK: {task_prompt}",
            f"",
        ]

        if context:
            prompt_parts.extend([
                f"CONTEXT: {context}",
                f"",
            ])

        # Inject relevant learnings from knowledge base (keeps context small)
        if KNOWLEDGE_BASE_AVAILABLE:
            try:
                kb = get_knowledge_base()
                # Extract agent type from name (e.g., "Tuning Agent" -> "tuning")
                agent_type = agent_name.lower().replace(" agent", "").strip()
                learnings_context = kb.get_context_for_agent(agent_type, task_prompt)
                if learnings_context:
                    prompt_parts.extend([
                        learnings_context,
                        f"",
                    ])
            except Exception as e:
                logger.debug(f"Failed to fetch learnings context: {e}")

        prompt_parts.extend([
            f"INSTRUCTIONS:",
            f"1. First, use Glob and Read tools to explore relevant files",
            f"2. Check if the requested work is ALREADY DONE - if so, output [ALREADY_DONE] and a brief explanation, then stop",
            f"3. If work is needed, complete the task",
            f"4. Output any follow-up tasks in format: [TASK: agent_type] specific task description",
            f"5. Output any handoffs in format: [HANDOFF: agent_type] description",
            f"",
            f"IMPORTANT: If the file/feature/change already exists as requested, output [ALREADY_DONE] immediately.",
            f"",
            f"Valid agent_types: tuning, backtest, risk, data, strategy",
            f"",
            f"Begin by exploring the codebase.",
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

            # Cleanup (use pop for thread safety)
            self.running_tasks.pop(task_id, None)

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

    Thread Safety:
    - context_store protected by _context_lock for parallel execution
    - handoff operations protected by _handoff_lock
    """

    def __init__(self, executor: ClaudeExecutor):
        self.executor = executor
        self.context_store: dict[str, str] = {}  # agent -> latest output
        self.workflow_queue: asyncio.Queue = asyncio.Queue()
        self.recent_handoffs: set = set()  # Track recent handoffs to prevent duplicates
        self.handoff_history: list = []  # Full history for debugging

        # Locks for thread-safe parallel execution
        self._context_lock = asyncio.Lock()
        self._handoff_lock = asyncio.Lock()

    def _get_handoff_key(self, to_agent: str, task: str) -> str:
        """Generate a key for handoff deduplication based on target and task keywords."""
        # Extract first 100 chars for matching
        task_key = task[:100].lower().strip()
        return f"{to_agent}:{task_key}"

    async def clear_handoff_cache(self):
        """Clear the handoff deduplication cache (call at mission start)."""
        async with self._handoff_lock:
            self.recent_handoffs.clear()
            logger.debug("Cleared handoff deduplication cache")

    async def store_context(self, agent_name: str, output: str):
        """Store agent output for other agents to reference (thread-safe)."""
        async with self._context_lock:
            # Keep last output per agent, trimmed for context efficiency
            self.context_store[agent_name] = output[:2000]

    async def get_context_for(self, requesting_agent: str, from_agents: list[str]) -> str:
        """Get relevant context from other agents (thread-safe)."""
        async with self._context_lock:
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
        """Queue a task handoff from one agent to another with deduplication (thread-safe)."""
        async with self._handoff_lock:
            # Check for duplicate handoff
            handoff_key = self._get_handoff_key(to_agent, task)
            if handoff_key in self.recent_handoffs:
                logger.info(f"Skipping duplicate handoff to {to_agent}: {task[:50]}...")
                return

            # Track this handoff
            self.recent_handoffs.add(handoff_key)
            handoff = {
                "from": from_agent,
                "to": to_agent,
                "task": task,
                "context": context,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.handoff_history.append(handoff)

        # Put outside lock - asyncio.Queue is already thread-safe
        await self.workflow_queue.put(handoff)
        logger.debug(f"Queued handoff: {from_agent} -> {to_agent}")

    async def get_next_handoff(self, timeout: float = None) -> Optional[dict]:
        """
        Get next queued handoff.

        Args:
            timeout: Optional timeout in seconds. If None, non-blocking.
                     If specified, waits up to timeout for a handoff.

        Returns:
            Handoff dict or None if queue empty/timeout
        """
        if timeout is None:
            # Non-blocking
            try:
                return self.workflow_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
        else:
            # Wait with timeout
            try:
                return await asyncio.wait_for(
                    self.workflow_queue.get(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return None
            except asyncio.QueueEmpty:
                return None


# =============================================================================
# TOKEN OPTIMIZATION UTILITIES
# =============================================================================

def get_token_optimization_stats() -> str:
    """Get token optimization statistics report."""
    if not TOKEN_OPTIMIZER_AVAILABLE:
        return "Token optimization not available (module not loaded)"

    optimizer = get_token_optimizer()
    return optimizer.get_stats_report()


def clear_token_cache():
    """Clear the token optimizer cache."""
    if TOKEN_OPTIMIZER_AVAILABLE:
        optimizer = get_token_optimizer()
        optimizer.cache.clear()
        return "Token cache cleared"
    return "Token optimization not available"


def reset_token_stats():
    """Reset token optimization statistics."""
    if TOKEN_OPTIMIZER_AVAILABLE:
        optimizer = get_token_optimizer()
        optimizer.reset_stats()
        return "Token stats reset"
    return "Token optimization not available"
