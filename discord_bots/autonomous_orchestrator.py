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
- Dispatches tasks to appropriate agents
- Tracks workflow progress
- Posts updates to Discord
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

from claude_executor import ClaudeExecutor, AgentCoordinator, TaskStatus
from base_agent import BaseAgentBot
from mission_manager import MissionManager, set_mission_manager, get_mission_manager
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

    def register_agent(self, agent_type: str, agent: BaseAgentBot):
        """Register an agent instance for task dispatch."""
        self.agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type}")

    async def start(self):
        """Start the autonomous processing loop."""
        self.running = True
        self._process_task = asyncio.create_task(self._process_handoffs())
        logger.info("Autonomous orchestrator started")

    async def stop(self):
        """Stop the processing loop."""
        self.running = False
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomous orchestrator stopped")

    async def _process_handoffs(self):
        """
        Background loop that processes the handoff queue.

        This is the heart of autonomous operation - it takes
        handoffs queued by agents and dispatches them to the
        appropriate target agent.
        """
        logger.info("Handoff processor started")

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

    async def _dispatch_handoff(self, handoff: dict):
        """Dispatch a handoff to the target agent."""
        target_type = handoff["to"]
        from_type = handoff["from"]
        task = handoff["task"]
        context = handoff["context"]

        logger.info(f"Dispatching handoff: {from_type} → {target_type}")

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

        try:
            # Execute the task via the target agent
            result = await target_agent.execute_task(
                task=task,
                context=context,
                notify_channel=True
            )

            # Update workflow status
            self.active_workflows[workflow_id]["status"] = result.status.value
            self.active_workflows[workflow_id]["completed"] = datetime.utcnow().isoformat()

            # Check if this triggers further handoffs
            await self._check_auto_handoffs(target_type, result)

        except Exception as e:
            logger.exception(f"Error executing handoff task: {e}")
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

    async def _check_auto_handoffs(self, agent_type: str, result):
        """
        Check if the completed task should trigger automatic handoffs.

        This implements the autonomous workflow progression based on
        the handoff rules defined in agent_prompts.py
        """
        from agent_prompts import HANDOFF_RULES

        rules = HANDOFF_RULES.get(agent_type, {})

        # Determine which trigger applies based on result
        if result.status == TaskStatus.COMPLETED:
            # Check output for keywords to determine next action
            output_lower = result.output.lower()

            # Strategy agent completed → trigger data prep
            if agent_type == "strategy" and "proposal" in output_lower:
                targets = rules.get("on_proposal", [])
                for target in targets:
                    await self.coordinator.queue_handoff(
                        from_agent=agent_type,
                        to_agent=target,
                        task="Prepare features for the proposed strategy",
                        context=result.output[:1000]
                    )

            # Data agent completed → notify backtest
            elif agent_type == "data" and "features ready" in output_lower:
                targets = rules.get("on_features_ready", [])
                for target in targets:
                    if target != "*":
                        await self.coordinator.queue_handoff(
                            from_agent=agent_type,
                            to_agent=target,
                            task="Features are ready for use",
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


# Singleton instance for global access
_orchestrator: Optional[AutonomousOrchestrator] = None


def get_orchestrator() -> Optional[AutonomousOrchestrator]:
    """Get the global orchestrator instance."""
    return _orchestrator


def set_orchestrator(orchestrator: AutonomousOrchestrator):
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator
