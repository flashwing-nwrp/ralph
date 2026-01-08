#!/usr/bin/env python3
"""
RALPH Discord Bot Launcher - Run All Agents

This script starts all 5 RALPH agent bots concurrently using asyncio.
Each bot runs in its own task with independent error handling.

Usage:
    python run_all.py
"""

import asyncio
import logging
import os
import signal
import sys
from typing import List

from dotenv import load_dotenv

# Load environment variables before importing agents
load_dotenv()

from agents import (
    TuningAgent,
    BacktestAgent,
    RiskAgent,
    StrategyAgent,
    DataAgent,
)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ORCHESTRATOR | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("orchestrator")


class BotOrchestrator:
    """Manages multiple Discord bot instances concurrently."""

    def __init__(self):
        self.agents = []
        self.tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

    def create_agents(self):
        """Instantiate all agent bots."""
        logger.info("Creating agent instances...")

        agent_classes = [
            TuningAgent,
            BacktestAgent,
            RiskAgent,
            StrategyAgent,
            DataAgent,
        ]

        for AgentClass in agent_classes:
            try:
                agent = AgentClass()
                self.agents.append(agent)
                logger.info(f"Created {agent.agent_name}")
            except ValueError as e:
                logger.error(f"Failed to create {AgentClass.__name__}: {e}")
                raise

        logger.info(f"Successfully created {len(self.agents)} agents")

    async def run_agent(self, agent):
        """Run a single agent with error isolation."""
        try:
            await agent.run()
        except Exception as e:
            logger.error(f"{agent.agent_name} crashed: {e}")
            # Don't re-raise - let other agents continue

    async def run_all(self):
        """Run all agents concurrently."""
        logger.info("Starting all agents...")

        # Create tasks for each agent
        self.tasks = [
            asyncio.create_task(self.run_agent(agent), name=agent.agent_name)
            for agent in self.agents
        ]

        # Wait for shutdown signal or all tasks to complete
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")

    async def shutdown(self):
        """Gracefully shutdown all agents."""
        logger.info("Initiating graceful shutdown...")

        # Shutdown each agent
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {agent.agent_name}: {e}")

        # Cancel any remaining tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        logger.info("All agents shut down")


async def main():
    """Main entry point."""
    orchestrator = BotOrchestrator()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(orchestrator.shutdown())

    # Windows doesn't support add_signal_handler the same way
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

    try:
        # Create and run all agents
        orchestrator.create_agents()
        await orchestrator.run_all()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await orchestrator.shutdown()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await orchestrator.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           RALPH Discord Agent Ensemble                        ║
║                                                               ║
║   Starting all 5 agents:                                      ║
║   • Tuning Agent    - Parameter optimization                  ║
║   • Backtest Agent  - Simulation & metrics                    ║
║   • Risk Agent      - Safety auditing                         ║
║   • Strategy Agent  - Logic & features                        ║
║   • Data Agent      - Preprocessing & denoising               ║
║                                                               ║
║   Press Ctrl+C to shutdown gracefully                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
