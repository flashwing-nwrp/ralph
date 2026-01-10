"""
Autonomous Orchestrator for RALPH Agent Ensemble

This orchestrator manages the autonomous workflow where agents can
trigger each other without user intervention. It implements the
"Ralph pattern" where:

1. Each agent has focused context (stays accurate)
2. Agents hand off to each other automatically
3. Memory persists via git history and progress files
4. User only needs to kick off the initial task

The orchestrator runs a background loop that:
- Monitors the handoff queue
- Dispatches tasks to appropriate agents IN PARALLEL
- Respects per-agent concurrency limits
- Tracks workflow progress with dependency resolution
- Posts rate-limited updates to Discord

Parallel Execution Strategy:
- Each agent type has a concurrency limit (tuning=1, backtest=1, risk=2, strategy=2, data=3)
- Tasks with satisfied dependencies are dispatched concurrently up to limits
- DiscordEmbedQueue prevents flooding with rate-limited embed posting
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List, Set

from dotenv import load_dotenv

load_dotenv()

# Import parallel execution modules
try:
    from parallel_task_tracker import (
        ParallelTaskTracker,
        TrackedTask,
        TaskExecutionState,
        get_parallel_tracker,
        set_parallel_tracker
    )
    PARALLEL_TRACKER_AVAILABLE = True
except ImportError:
    PARALLEL_TRACKER_AVAILABLE = False

try:
    from discord_rate_limiter import (
        DiscordEmbedQueue,
        EmbedPriority,
        get_embed_queue,
        initialize_embed_queue
    )
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# Import innovation loop for autonomous improvement
try:
    from innovation_loop import (
        InnovationLoop,
        get_innovation_loop,
        set_innovation_loop,
        initialize_innovation_loop
    )
    INNOVATION_LOOP_AVAILABLE = True
except ImportError:
    INNOVATION_LOOP_AVAILABLE = False

from claude_executor import ClaudeExecutor, AgentCoordinator, TaskStatus
from base_agent import BaseAgentBot
from mission_manager import MissionManager, MissionStatus, set_mission_manager, get_mission_manager
from discord_embeds import RALPHEmbeds
from collaboration_logger import get_collaboration_logger, EventType
from agent_prompts import validate_task_assignment
from mission_summary import get_summary_generator
from agents import (
    TuningAgent,
    BacktestAgent,
    RiskAgent,
    StrategyAgent,
    DataAgent,
)

logger = logging.getLogger("orchestrator")


class AutonomousOrchestrator:
    """
    Manages autonomous agent workflows.

    Key responsibilities:
    - Initialize shared executor and coordinator
    - Process handoff queue
    - Track workflow state
    - Enable agents to trigger each other
    """

    def __init__(self, project_dir: str = None):
        # Initialize execution infrastructure
        self.executor = ClaudeExecutor(project_dir=project_dir)
        self.coordinator = AgentCoordinator(self.executor)

        # Initialize mission manager
        self.mission_manager = MissionManager(project_dir=project_dir)
        set_mission_manager(self.mission_manager)

        # Set shared instances on base class
        BaseAgentBot.set_executor(self.executor)
        BaseAgentBot.set_coordinator(self.coordinator)

        # Agent instances (will be populated)
        self.agents: Dict[str, BaseAgentBot] = {}

        # Workflow tracking
        self.active_workflows: Dict[str, dict] = {}
        self.workflow_counter = 0

        # Control
        self.running = False
        self._process_task: Optional[asyncio.Task] = None

        # Parallel execution components
        self.parallel_tracker: Optional[ParallelTaskTracker] = None
        self.embed_queue: Optional[DiscordEmbedQueue] = None
        self._parallel_enabled = PARALLEL_TRACKER_AVAILABLE and RATE_LIMITER_AVAILABLE

        if self._parallel_enabled:
            self.parallel_tracker = ParallelTaskTracker()
            set_parallel_tracker(self.parallel_tracker)
            logger.info("Parallel execution engine initialized")
        else:
            logger.warning("Parallel execution not available - running in sequential mode")

        # Innovation loop for autonomous improvement
        self.innovation_loop: Optional[InnovationLoop] = None
        self._innovation_enabled = INNOVATION_LOOP_AVAILABLE

    def register_agent(self, agent_type: str, agent: BaseAgentBot):
        """Register an agent instance for task dispatch."""
        self.agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type}")

    async def start(self):
        """Start the autonomous processing loop."""
        self.running = True

        # Initialize embed queue with send callback
        if self._parallel_enabled and RATE_LIMITER_AVAILABLE:
            self.embed_queue = await initialize_embed_queue(self._send_embed_callback)
            logger.info("Discord embed queue started")

        # Start the appropriate processing loop
        if self._parallel_enabled:
            self._process_task = asyncio.create_task(self._process_handoffs_parallel())
        else:
            self._process_task = asyncio.create_task(self._process_handoffs_sequential())

        # Start innovation loop for autonomous improvement (disabled by default)
        # Enable via INNOVATION_LOOP_ENABLED=true environment variable
        import os
        innovation_enabled = os.environ.get("INNOVATION_LOOP_ENABLED", "false").lower() == "true"
        if self._innovation_enabled and innovation_enabled:
            self.innovation_loop = await initialize_innovation_loop(self)
            logger.info("Innovation loop started (5-minute cycle)")
        elif self._innovation_enabled:
            logger.info("Innovation loop available but disabled (set INNOVATION_LOOP_ENABLED=true to enable)")

        logger.info(f"Autonomous orchestrator started (parallel={self._parallel_enabled})")

    async def stop(self):
        """Stop the processing loop."""
        self.running = False

        # Stop innovation loop
        if self.innovation_loop:
            await self.innovation_loop.stop()
            logger.info("Innovation loop stopped")

        # Stop embed queue gracefully
        if self.embed_queue:
            await self.embed_queue.stop(drain=True)
            logger.info("Discord embed queue stopped")

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomous orchestrator stopped")

    async def _send_embed_callback(self, embed, channel_id: int, content: str = ""):
        """Callback for embed queue to send embeds to Discord."""
        # Find any agent to send the embed (they all share the team channel)
        for agent in self.agents.values():
            if hasattr(agent, 'team_channel') and agent.team_channel:
                try:
                    if content:
                        await agent.team_channel.send(content=content, embed=embed)
                    else:
                        await agent.team_channel.send(embed=embed)
                    return
                except Exception as e:
                    logger.error(f"Failed to send embed via callback: {e}")
                    raise

    async def _process_handoffs_sequential(self):
        """
        Sequential background loop that processes handoffs one at a time.

        Fallback mode when parallel execution is not available.
        """
        logger.info("Sequential handoff processor started")

        while self.running:
            try:
                # Check for pending handoffs
                handoff = await self.coordinator.get_next_handoff()

                if handoff:
                    await self._dispatch_handoff(handoff)

                # Small delay to prevent busy-waiting
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error processing handoff: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def _process_handoffs_parallel(self):
        """
        Parallel background loop that dispatches multiple tasks concurrently.

        Strategy:
        1. Collect all pending handoffs from queue
        2. Register them with ParallelTaskTracker
        3. Get tasks that have satisfied dependencies AND agent capacity
        4. Dispatch ready tasks concurrently using asyncio.gather()
        5. Tasks self-report completion, triggering dependency resolution
        6. Repeat until no more handoffs or stopped
        """
        logger.info("Parallel handoff processor started")

        while self.running:
            try:
                # Check suspension status - don't dispatch new tasks if winding down
                try:
                    from suspension_manager import get_suspension_manager
                    suspension = get_suspension_manager()
                    if suspension.is_suspended():
                        logger.info("Operations suspended - processor stopped")
                        break
                    if suspension.is_winding_down():
                        # Let current tasks complete but don't start new ones
                        await asyncio.sleep(1.0)
                        continue
                except ImportError:
                    pass  # Suspension manager not available

                # Step 1: Drain pending handoffs into tracker
                handoffs_registered = await self._register_pending_handoffs()

                # Step 2: Get tasks ready for dispatch
                ready_tasks = await self.parallel_tracker.get_ready_tasks(max_tasks=10)

                if ready_tasks:
                    logger.info(f"Dispatching {len(ready_tasks)} tasks in parallel")

                    # Step 3: Dispatch all ready tasks concurrently
                    dispatch_coroutines = [
                        self._dispatch_tracked_task(task)
                        for task in ready_tasks
                    ]

                    # Run all dispatches concurrently
                    # Use return_exceptions=True to handle individual failures
                    results = await asyncio.gather(*dispatch_coroutines, return_exceptions=True)

                    # Log any exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Task {ready_tasks[i].task_id} dispatch failed: {result}")

                # Step 4: Wait for tasks to complete or new handoffs
                if not ready_tasks and handoffs_registered == 0:
                    # Nothing to do, wait for completion events or new work
                    await self.parallel_tracker.wait_for_ready_tasks(timeout=1.0)
                else:
                    # Small delay to let tasks make progress
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in parallel processor: {e}")
                await asyncio.sleep(5)  # Back off on error

        logger.info("Parallel handoff processor stopped")

    async def _register_pending_handoffs(self) -> int:
        """
        Drain the handoff queue and register tasks with the tracker.

        Returns:
            Number of handoffs registered
        """
        registered = 0

        while True:
            # Non-blocking check for handoff
            handoff = await self.coordinator.get_next_handoff(timeout=0.1)

            if not handoff:
                break

            target_type = handoff["to"]
            task_desc = handoff["task"]

            # Generate unique task ID
            task_id = f"PT-{self.workflow_counter:04d}"
            self.workflow_counter += 1

            # Extract dependencies from context if specified
            dependencies: Set[str] = set()
            if "depends_on" in handoff:
                dependencies = set(handoff["depends_on"])

            # Register with tracker
            await self.parallel_tracker.register_task(
                task_id=task_id,
                agent_type=target_type,
                handoff=handoff,
                dependencies=dependencies,
                mission_task_id=handoff.get("mission_task_id")
            )

            registered += 1
            logger.debug(f"Registered task {task_id} for {target_type}")

        return registered

    async def _dispatch_tracked_task(self, task: TrackedTask):
        """
        Dispatch a tracked task to its target agent.

        This runs as part of asyncio.gather() for parallel execution.
        Updates task state in tracker upon completion.
        """
        handoff = task.handoff
        target_type = task.agent_type
        from_type = handoff.get("from", "unknown")
        task_desc = handoff["task"]
        context = handoff.get("context", "")

        logger.info(f"[{task.task_id}] Dispatching: {from_type} → {target_type}")

        # Mark as running in tracker
        asyncio_task = asyncio.current_task()
        await self.parallel_tracker.mark_task_running(task.task_id, asyncio_task)

        collab_logger = get_collaboration_logger()
        collab_logger.agent_handoff(
            from_agent=from_type,
            to_agent=target_type,
            task_description=task_desc[:200]
        )

        target_agent = self.agents.get(target_type)
        if not target_agent:
            logger.error(f"[{task.task_id}] Target agent not found: {target_type}")
            await self.parallel_tracker.mark_task_completed(
                task.task_id,
                success=False,
                error=f"Agent not found: {target_type}"
            )
            return

        # Create workflow tracking entry
        self.active_workflows[task.task_id] = {
            "from": from_type,
            "to": target_type,
            "task": task_desc,
            "started": datetime.utcnow().isoformat(),
            "status": "running"
        }

        # Post working embed via queue (rate-limited)
        working_embed = RALPHEmbeds.agent_working(
            agent_type=target_type,
            task_description=task_desc[:500],
            task_id=task.task_id,
            handoff_from=from_type
        )
        await self._post_embed(working_embed, target_agent, priority=EmbedPriority.WORKING)

        try:
            # Execute the task via the target agent
            result = await target_agent.execute_task(
                task=task_desc,
                context=context,
                notify_channel=False,  # We handle embeds via queue
                force_claude=True
            )

            # Update workflow status
            self.active_workflows[task.task_id]["status"] = result.status.value
            self.active_workflows[task.task_id]["completed"] = datetime.utcnow().isoformat()

            # Mark complete in tracker
            success = result.status == TaskStatus.COMPLETED
            await self.parallel_tracker.mark_task_completed(
                task.task_id,
                result=result.output,
                success=success,
                error=result.error if not success else None
            )

            # Parse backlog items from agent response
            if result.output and "[BACKLOG]" in result.output:
                await self._parse_backlog_items(
                    response=result.output,
                    agent_type=target_type,
                    task_id=task.task_id
                )

            # Post result embed via queue
            if success:
                complete_embeds = RALPHEmbeds.agent_complete_chunked(
                    agent_type=target_type,
                    task_description=task_desc[:500],
                    result_summary=result.output if result.output else "Task completed successfully",
                    duration_seconds=result.duration_seconds,
                    task_id=task.task_id
                )
                for embed in complete_embeds:
                    await self._post_embed(embed, target_agent, priority=EmbedPriority.COMPLETION)
            else:
                error_embed = RALPHEmbeds.agent_error(
                    agent_type=target_type,
                    task_description=task_desc[:200],
                    error_message=result.error or "Unknown error",
                    task_id=task.task_id
                )
                await self._post_embed(error_embed, target_agent, priority=EmbedPriority.ERROR)

            # Check for auto-handoffs from result
            await self._check_auto_handoffs(target_type, result)

            # Check mission completion
            await self._check_mission_completion(target_agent)

        except Exception as e:
            logger.exception(f"[{task.task_id}] Error executing task: {e}")

            self.active_workflows[task.task_id]["status"] = "failed"
            self.active_workflows[task.task_id]["error"] = str(e)

            await self.parallel_tracker.mark_task_completed(
                task.task_id,
                success=False,
                error=str(e)
            )

            collab_logger.error(
                agent=target_type,
                error=str(e)[:200],
                context=f"Handoff from {from_type}: {task_desc[:100]}"
            )

            error_embed = RALPHEmbeds.agent_error(
                agent_type=target_type,
                task_description=task_desc[:200],
                error_message=str(e)[:300],
                task_id=task.task_id
            )
            await self._post_embed(error_embed, target_agent, priority=EmbedPriority.ERROR)

    async def _post_embed(self, embed, agent: BaseAgentBot, priority: EmbedPriority = None):
        """
        Post an embed, using queue if available or direct send otherwise.

        Args:
            embed: Discord embed to post
            agent: Agent to send from
            priority: Priority level (only used with queue)
        """
        if self.embed_queue and self.embed_queue.is_running():
            # Get channel ID from agent
            channel_id = 0
            if hasattr(agent, 'team_channel') and agent.team_channel:
                channel_id = agent.team_channel.id

            await self.embed_queue.enqueue(
                embed=embed,
                channel_id=channel_id,
                priority=priority or EmbedPriority.COMPLETION
            )
        else:
            # Direct send
            await agent.post_embed_to_team(embed)

    async def _dispatch_handoff(self, handoff: dict):
        """Dispatch a handoff to the target agent."""
        target_type = handoff["to"]
        from_type = handoff["from"]
        task = handoff["task"]
        context = handoff["context"]

        logger.info(f"Dispatching handoff: {from_type} → {target_type}")
        collab_logger = get_collaboration_logger()

        # Log the handoff
        collab_logger.agent_handoff(
            from_agent=from_type,
            to_agent=target_type,
            task_description=task[:200]
        )

        target_agent = self.agents.get(target_type)
        if not target_agent:
            logger.error(f"Target agent not found: {target_type}")
            return

        # Create workflow tracking entry
        workflow_id = f"WF-{self.workflow_counter:04d}"
        self.workflow_counter += 1
        self.active_workflows[workflow_id] = {
            "from": from_type,
            "to": target_type,
            "task": task,
            "started": datetime.utcnow().isoformat(),
            "status": "running"
        }

        # Post combined handoff + working embed (single message instead of two)
        working_embed = RALPHEmbeds.agent_working(
            agent_type=target_type,
            task_description=task[:500],
            task_id=workflow_id,
            handoff_from=from_type  # Add handoff info to working embed
        )
        await target_agent.post_embed_to_team(working_embed)

        try:
            # Execute the task via the target agent
            result = await target_agent.execute_task(
                task=task,
                context=context,
                notify_channel=True,
                force_claude=True  # Handoffs go to Claude Code
            )

            # Update workflow status
            self.active_workflows[workflow_id]["status"] = result.status.value
            self.active_workflows[workflow_id]["completed"] = datetime.utcnow().isoformat()

            # Post result embed
            if result.status == TaskStatus.COMPLETED:
                # Use chunked embeds for long outputs
                complete_embeds = RALPHEmbeds.agent_complete_chunked(
                    agent_type=target_type,
                    task_description=task[:500],
                    result_summary=result.output if result.output else "Task completed successfully",
                    duration_seconds=result.duration_seconds,
                    task_id=workflow_id
                )
                for embed in complete_embeds:
                    await target_agent.post_embed_to_team(embed)
            else:
                error_embed = RALPHEmbeds.agent_error(
                    agent_type=target_type,
                    task_description=task[:200],
                    error_message=result.error or "Unknown error",
                    task_id=workflow_id
                )
                await target_agent.post_embed_to_team(error_embed)

            # Check if this triggers further handoffs
            await self._check_auto_handoffs(target_type, result)

            # Check if mission is now complete
            await self._check_mission_completion(target_agent)

        except Exception as e:
            logger.exception(f"Error executing handoff task: {e}")
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

            # Log the error
            collab_logger.error(
                agent=target_type,
                error=str(e)[:200],
                context=f"Handoff from {from_type}: {task[:100]}"
            )

            # Post error embed
            error_embed = RALPHEmbeds.agent_error(
                agent_type=target_type,
                task_description=task[:200],
                error_message=str(e)[:300],
                task_id=workflow_id
            )
            await target_agent.post_embed_to_team(error_embed)

    async def _parse_backlog_items(self, response: str, agent_type: str, task_id: str):
        """
        Parse [BACKLOG] entries from agent response and add to team backlog.

        Agents can add items to the backlog using:
        [BACKLOG] type: bug|improvement|idea|tech_debt
        Title: Brief description
        Priority: low|medium|high
        Rationale: Why this matters
        """
        try:
            from backlog_manager import get_backlog_manager

            mgr = get_backlog_manager()
            mission_id = None

            # Get current mission ID if available
            if hasattr(self, 'mission_manager') and self.mission_manager:
                current = self.mission_manager.get_current_mission()
                if current:
                    mission_id = current.get('id')

            # Parse and add items
            added_items = mgr.parse_and_add_from_response(
                response=response,
                created_by=agent_type,
                mission_id=mission_id
            )

            if added_items:
                logger.info(f"[{task_id}] {agent_type} added {len(added_items)} backlog items")

                # Notify team channel about new backlog items
                for item in added_items:
                    type_emoji = {"bug": "🐛", "improvement": "💡", "idea": "💭", "tech_debt": "🔧"}
                    emoji = type_emoji.get(item.item_type, "📝")

                    # Post to team channel
                    if self.team_channel:
                        msg = (
                            f"{emoji} **New Backlog Item** ({item.id})\n"
                            f"**{item.title}**\n"
                            f"Type: {item.item_type} | Priority: {item.priority} | By: {agent_type}\n"
                            f"Use `!approve_backlog {item.id}` or `!reject_backlog {item.id} <reason>`"
                        )
                        try:
                            await self.team_channel.send(msg)
                        except Exception as e:
                            logger.warning(f"Could not post backlog notification: {e}")

        except ImportError:
            logger.warning("backlog_manager not available, skipping backlog parsing")
        except Exception as e:
            logger.warning(f"Error parsing backlog items: {e}")

    async def _check_auto_handoffs(self, agent_type: str, result):
        """
        Check if the completed task should trigger automatic handoffs.

        Parses output for:
        1. [TASK: agent_type] format - adds to mission and queues handoffs
        2. [HANDOFF: agent_type] format - queues direct handoffs
        3. Keyword-based triggers from HANDOFF_RULES
        """
        import re
        from agent_prompts import HANDOFF_RULES

        if result.status != TaskStatus.COMPLETED or not result.output:
            return

        output = result.output
        output_lower = output.lower()

        # ============================================================
        # Parse [TASK: agent_type] format and add to mission
        # ============================================================
        task_pattern = r'\[TASK:\s*(\w+)\]\s*(.+?)(?=\[TASK:|$)'
        task_matches = re.findall(task_pattern, output, re.IGNORECASE | re.DOTALL)

        if task_matches:
            logger.info(f"Found {len(task_matches)} tasks in output")
            collab_logger = get_collaboration_logger()

            # Get the strategy agent to post updates
            strategy_agent = self.agents.get("strategy")

            # Get current mission
            mission = self.mission_manager.current_mission

            if mission and strategy_agent:
                # Build task list for embed
                tasks_for_embed = []

                for agent_type_task, task_desc in task_matches:
                    agent_type_task = agent_type_task.lower().strip()
                    task_desc = task_desc.strip()[:500]  # Limit task length

                    if agent_type_task in ["tuning", "backtest", "risk", "data", "strategy"]:
                        # Validate domain boundary - suggest correct agent if misassigned
                        is_valid, suggested_agent = validate_task_assignment(task_desc, agent_type_task)
                        if not is_valid:
                            logger.info(f"Redirecting task from {agent_type_task} to {suggested_agent} (domain mismatch)")
                            agent_type_task = suggested_agent

                        # Check for duplicate: similar task already completed
                        similar_completed = mission.find_similar_completed_task(task_desc)
                        if similar_completed:
                            logger.info(f"Skipping duplicate task (similar to {similar_completed.task_id}): {task_desc[:80]}...")
                            continue

                        # Check for duplicate: similar task already assigned
                        if mission.is_task_assigned(task_desc, agent_type_task):
                            logger.info(f"Skipping duplicate task (already assigned to {agent_type_task}): {task_desc[:80]}...")
                            continue

                        tasks_for_embed.append({
                            "agent": agent_type_task,
                            "task": task_desc
                        })

                        # Add task to mission
                        task = await self.mission_manager.add_task_to_mission(
                            description=task_desc,
                            assigned_to=agent_type_task,
                            priority="medium"
                        )

                        if task:
                            logger.info(f"Added mission task: {task.task_id} -> {agent_type_task}")

                            # Log task creation
                            collab_logger.task_created(
                                task_id=task.task_id,
                                agent=agent_type_task,
                                description=task_desc[:200]
                            )

                            # Queue handoff to execute immediately
                            await self.coordinator.queue_handoff(
                                from_agent="strategy",
                                to_agent=agent_type_task,
                                task=task_desc,
                                context=f"Mission: {mission.mission_id}\nTask: {task.task_id}\nObjective: {mission.objective}"
                            )

                # Post task breakdown embed
                if tasks_for_embed:
                    task_embed = RALPHEmbeds.task_breakdown(
                        tasks=tasks_for_embed,
                        mission_objective=mission.objective
                    )
                    await strategy_agent.post_embed_to_team(task_embed)

                    # Post detailed task embeds per agent
                    detailed_embeds = RALPHEmbeds.task_list_detailed(tasks_for_embed)
                    for embed in detailed_embeds[:5]:  # Limit to 5 to avoid spam
                        await strategy_agent.post_embed_to_team(embed)
                        await asyncio.sleep(0.3)

                # Start the mission
                await self.mission_manager.start_mission()

                # Post progress embed
                progress_embed = RALPHEmbeds.mission_progress(
                    completed=0,
                    total=len(tasks_for_embed),
                    current_agent="strategy",
                    current_task="Delegating tasks to agents"
                )
                await strategy_agent.post_embed_to_team(progress_embed)

            return  # Tasks were parsed, skip other checks

        # ============================================================
        # Parse [HANDOFF: agent_type] format for direct handoffs
        # ============================================================
        handoff_pattern = r'\[HANDOFF:\s*(\w+)\]\s*(.+?)(?=\[HANDOFF:|$)'
        handoff_matches = re.findall(handoff_pattern, output, re.IGNORECASE | re.DOTALL)

        for target_agent, task in handoff_matches:
            target_agent = target_agent.lower().strip()
            task = task.strip()

            if target_agent in ["tuning", "backtest", "risk", "strategy", "data"]:
                logger.info(f"Parsed handoff to {target_agent}: {task[:50]}...")
                await self.coordinator.queue_handoff(
                    from_agent=agent_type,
                    to_agent=target_agent,
                    task=task,
                    context=result.output[:1000]
                )

        # ============================================================
        # Keyword-based triggers from HANDOFF_RULES
        # ============================================================
        rules = HANDOFF_RULES.get(agent_type, {})

        # Strategy agent completed with proposal
        if agent_type == "strategy" and "proposal" in output_lower:
            targets = rules.get("on_proposal", [])
            for target in targets:
                await self.coordinator.queue_handoff(
                    from_agent=agent_type,
                    to_agent=target,
                    task="Prepare features for the proposed strategy",
                    context=result.output[:1000]
                )

        # Backtest completed → always audit
        elif agent_type == "backtest":
            targets = rules.get("on_complete", [])
            for target in targets:
                await self.coordinator.queue_handoff(
                    from_agent=agent_type,
                    to_agent=target,
                    task="Audit the backtest results",
                    context=result.output[:1000]
                )

        # Risk audit completed
        elif agent_type == "risk":
            if "approved" in output_lower:
                targets = rules.get("on_approved", [])
                for target in targets:
                    await self.coordinator.queue_handoff(
                        from_agent=agent_type,
                        to_agent=target,
                        task="Strategy approved - proceed with implementation",
                        context=result.output[:1000]
                    )
            elif "rejected" in output_lower:
                targets = rules.get("on_rejected", [])
                for target in targets:
                    await self.coordinator.queue_handoff(
                        from_agent=agent_type,
                        to_agent=target,
                        task="Strategy rejected - review and fix issues",
                        context=result.output[:1000]
                    )

        # Tuning completed → validate
        elif agent_type == "tuning":
            targets = rules.get("on_proposal", [])
            for target in targets:
                await self.coordinator.queue_handoff(
                    from_agent=agent_type,
                    to_agent=target,
                    task="Validate the proposed parameter changes",
                    context=result.output[:1000]
                )

    async def kickoff_workflow(
        self,
        task: str,
        starting_agent: str = "strategy",
        context: str = ""
    ) -> str:
        """
        Kick off a new autonomous workflow.

        Args:
            task: The initial task description
            starting_agent: Which agent starts the workflow
            context: Initial context

        Returns:
            workflow_id for tracking
        """
        workflow_id = f"WF-{self.workflow_counter:04d}"
        self.workflow_counter += 1

        self.active_workflows[workflow_id] = {
            "task": task,
            "starting_agent": starting_agent,
            "started": datetime.utcnow().isoformat(),
            "status": "running"
        }

        # Log workflow start
        collab_logger = get_collaboration_logger()
        collab_logger.mission_start(
            mission_id=workflow_id,
            objective=task[:200],
            initiated_by="user"
        )

        # Queue the initial task
        await self.coordinator.queue_handoff(
            from_agent="user",
            to_agent=starting_agent,
            task=task,
            context=context
        )

        logger.info(f"Kicked off workflow {workflow_id}: {task[:100]}")
        return workflow_id

    def get_workflow_status(self, workflow_id: str) -> Optional[dict]:
        """Get the status of a workflow."""
        return self.active_workflows.get(workflow_id)

    def get_all_active_workflows(self) -> Dict[str, dict]:
        """Get all active workflows."""
        return {
            wid: wf for wid, wf in self.active_workflows.items()
            if wf.get("status") == "running"
        }

    async def _check_mission_completion(self, agent: "BaseAgentBot"):
        """
        Check if the current mission is complete and trigger retro ceremony.

        Called after each task completes to monitor mission progress.
        """
        mission = self.mission_manager.current_mission
        if not mission or mission.status == MissionStatus.COMPLETED:
            return

        # Check if all tasks are complete
        progress = mission.get_progress()
        pending_handoffs = not self.coordinator.workflow_queue.empty()

        logger.info(f"Mission progress: {progress['completed']}/{progress['total']} tasks, pending handoffs: {pending_handoffs}")

        # Only complete if all tasks done AND no pending handoffs
        if progress['completed'] == progress['total'] and not pending_handoffs:
            logger.info(f"Mission {mission.mission_id} complete! Running retro ceremony...")
            await self._run_retro_ceremony(mission, agent)

    async def _run_retro_ceremony(self, mission, agent: "BaseAgentBot"):
        """
        Run the mission retrospective ceremony.

        Steps:
        1. Post confirmation of completion
        2. Generate and post executive summary
        3. Run SCRUM retrospective
        4. Document learnings for future missions
        5. Archive mission
        """
        collab_logger = get_collaboration_logger()

        try:
            # Calculate duration
            start_time = datetime.fromisoformat(mission.started_at) if mission.started_at else datetime.utcnow()
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Count tasks by agent
            tasks_by_agent = {}
            for task in mission.tasks:
                agent_type = task.assigned_to
                tasks_by_agent[agent_type] = tasks_by_agent.get(agent_type, 0) + 1

            # ========================================
            # Step 1: Confirmation of Completion
            # ========================================
            confirmation_embed = RALPHEmbeds.mission_complete(
                mission_objective=f"[{mission.mission_id}] {mission.objective[:250]}",
                total_tasks=len([t for t in mission.tasks if t.status == "completed"]),
                duration_seconds=duration,
                tasks_by_agent=tasks_by_agent
            )
            await agent.post_embed_to_team(confirmation_embed)
            collab_logger.mission_complete(
                mission_id=mission.mission_id,
                total_tasks=len(mission.tasks),
                duration_seconds=duration
            )

            # Small delay for visual separation
            await asyncio.sleep(1)

            # ========================================
            # Step 2: Generate Executive Summary
            # ========================================
            summary_gen = get_summary_generator()
            summary_data = await summary_gen.generate_summary(mission)

            summary_embeds = RALPHEmbeds.executive_summary(
                mission_id=mission.mission_id,
                mission_objective=mission.objective,
                total_tasks=len(mission.tasks),
                duration_minutes=summary_data.get("stats", {}).get("duration_minutes", 0),
                key_findings=summary_data.get("key_findings", []),
                work_summary=summary_data.get("work_summary", ""),
                suggestions=summary_data.get("suggestions", []),
                owner_mention=mission.created_by if hasattr(mission, 'created_by') else None
            )

            for embed in summary_embeds:
                await agent.post_embed_to_team(embed)
                await asyncio.sleep(0.5)

            # ========================================
            # Step 3: SCRUM Retrospective
            # ========================================
            retro_data = await summary_gen.generate_retrospective(mission)

            retro_embed = RALPHEmbeds.retrospective(
                mission_id=mission.mission_id,
                what_went_well=retro_data.get("what_went_well", []),
                what_could_improve=retro_data.get("what_could_improve", []),
                learnings=retro_data.get("learnings", []),
                action_items=retro_data.get("action_items", [])
            )
            await agent.post_embed_to_team(retro_embed)

            # ========================================
            # Step 4: Document Learnings
            # ========================================
            await self._document_mission_learnings(mission, summary_data, retro_data)

            # ========================================
            # Step 5: Mark Mission Complete & Archive
            # ========================================
            await self.mission_manager.complete_mission()

            # Clear handoff cache for next mission
            self.coordinator.clear_handoff_cache()

            # Final message with owner mention
            owner_mention = mission.created_by if hasattr(mission, 'created_by') else None
            if owner_mention:
                await agent.post_to_team_channel(
                    f"<@{owner_mention}> 🎉 **Mission {mission.mission_id} Complete!**\n\n"
                    f"The retro ceremony is complete. Key learnings have been documented for future missions.\n"
                    f"Use `!mission <objective>` to start a new mission when ready."
                )

            logger.info(f"Retro ceremony complete for mission {mission.mission_id}")

        except Exception as e:
            logger.exception(f"Error during retro ceremony: {e}")
            # Fallback notification
            await agent.post_to_team_channel(
                f"**Mission {mission.mission_id} Complete!** (Retro ceremony encountered an error: {str(e)[:100]})"
            )

    async def _document_mission_learnings(self, mission, summary_data: dict, retro_data: dict):
        """
        Document mission learnings for use in future missions.

        Saves to:
        1. Knowledge base (for agent context injection)
        2. Learnings directory (JSON archive)
        3. Consolidated learnings file (JSONL for analysis)
        """
        from knowledge_base import get_knowledge_base
        import json
        from pathlib import Path

        try:
            # Compile comprehensive learnings document
            learnings_doc = {
                "mission_id": mission.mission_id,
                "objective": mission.objective,
                "completed_at": datetime.utcnow().isoformat(),
                "duration_minutes": summary_data.get("stats", {}).get("duration_minutes", 0),
                "tasks_summary": {
                    "total": len(mission.tasks),
                    "completed": len([t for t in mission.tasks if t.status == "completed"]),
                    "failed": len([t for t in mission.tasks if t.status == "failed"]),
                    "by_agent": summary_data.get("stats", {}).get("tasks_by_agent", {})
                },
                "work_summary": summary_data.get("work_summary", ""),
                "key_findings": summary_data.get("key_findings", []),
                "suggestions": summary_data.get("suggestions", []),
                "retrospective": {
                    "what_went_well": retro_data.get("what_went_well", []),
                    "what_could_improve": retro_data.get("what_could_improve", []),
                    "learnings": retro_data.get("learnings", []),
                    "action_items": retro_data.get("action_items", [])
                },
                "task_outputs": [
                    {
                        "task_id": t.task_id,
                        "agent": t.assigned_to,
                        "description": t.description[:200],
                        "output_preview": t.output[:500] if t.output else ""
                    }
                    for t in mission.tasks if t.status == "completed"
                ]
            }

            # Save to knowledge base for future agent context
            kb = get_knowledge_base()
            kb.add_learnings_from_mission(mission.mission_id, {
                **summary_data,
                "retrospective": retro_data
            })

            # Save to learnings directory
            learnings_dir = Path(self.mission_manager.project_dir) / "learnings"
            learnings_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{mission.mission_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = learnings_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(learnings_doc, f, indent=2)

            # Append to consolidated learnings
            consolidated_path = learnings_dir / "consolidated_learnings.jsonl"
            with open(consolidated_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(learnings_doc) + "\n")

            logger.info(f"Documented learnings for mission {mission.mission_id} to {filepath}")

        except Exception as e:
            logger.error(f"Failed to document mission learnings: {e}")

    def get_parallel_stats(self) -> str:
        """
        Get parallel execution statistics report.

        Returns:
            Formatted statistics string
        """
        if not self._parallel_enabled:
            return "**Parallel Execution**: Not available (running in sequential mode)"

        lines = ["**Parallel Execution Stats:**"]

        # Tracker stats
        if self.parallel_tracker:
            tracker_stats = self.parallel_tracker.get_stats()
            lines.append(f"- Tasks registered: {tracker_stats['tasks_registered']}")
            lines.append(f"- Tasks completed: {tracker_stats['tasks_completed']}")
            lines.append(f"- Tasks failed: {tracker_stats['tasks_failed']}")
            lines.append(f"- Current concurrent: {tracker_stats['current_concurrent']}")
            lines.append(f"- Max concurrent: {tracker_stats['max_concurrent']}")

            active_str = ', '.join(
                f"{a}={c}" for a, c in tracker_stats['active_by_agent'].items() if c > 0
            ) or 'None'
            lines.append(f"- Active by agent: {active_str}")
            lines.append(f"- Pending: {tracker_stats['pending_count']}")

        # Embed queue stats
        if self.embed_queue:
            queue_stats = self.embed_queue.get_stats()
            lines.append("")
            lines.append("**Discord Embed Queue:**")
            lines.append(f"- Queued: {queue_stats['embeds_queued']}")
            lines.append(f"- Sent: {queue_stats['embeds_sent']}")
            lines.append(f"- Dropped: {queue_stats['embeds_dropped']}")
            lines.append(f"- Rate limited: {queue_stats['rate_limited_count']} times")
            lines.append(f"- Current queue: {queue_stats['queue_size']}")

        return "\n".join(lines)


# Singleton instance for global access
_orchestrator: Optional[AutonomousOrchestrator] = None


def get_orchestrator() -> Optional[AutonomousOrchestrator]:
    """Get the global orchestrator instance."""
    return _orchestrator


def set_orchestrator(orchestrator: AutonomousOrchestrator):
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator
