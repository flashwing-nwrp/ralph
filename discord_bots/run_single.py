#!/usr/bin/env python3
"""
RALPH Discord Bot Launcher - Run Single Agent

This script starts a single RALPH agent bot for testing or development.

Usage:
    python run_single.py <agent_name>

    agent_name: tuning | backtest | risk | strategy | data

Examples:
    python run_single.py tuning
    python run_single.py risk
"""

import asyncio
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents import AGENTS


def print_usage():
    """Print usage information."""
    print("""
Usage: python run_single.py <agent_name>

Available agents:
    tuning    - Parameter optimization agent
    backtest  - Simulation & metrics agent
    risk      - Safety auditing agent
    strategy  - Logic & features agent
    data      - Preprocessing & denoising agent

Examples:
    python run_single.py tuning
    python run_single.py risk
    """)


async def main(agent_name: str):
    """Run a single agent."""
    if agent_name not in AGENTS:
        print(f"Error: Unknown agent '{agent_name}'")
        print_usage()
        sys.exit(1)

    AgentClass = AGENTS[agent_name]

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           RALPH Discord Agent - Single Mode                   ║
║                                                               ║
║   Starting: {AgentClass.__name__:<45}  ║
║                                                               ║
║   Press Ctrl+C to shutdown gracefully                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    try:
        agent = AgentClass()
        await agent.run()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nMake sure your .env file contains the required token.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    agent_name = sys.argv[1].lower()
    asyncio.run(main(agent_name))
