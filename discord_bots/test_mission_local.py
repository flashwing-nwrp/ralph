"""
Local Mission Testing Script

This script allows you to test the full mission workflow locally
without needing Discord. It simulates:
1. The ConversationalMissionManager driving Claude Code
2. Task parsing and handoff logic
3. Agent task execution (optionally)

Usage:
    python test_mission_local.py --mission "Improve the ML model accuracy"
    python test_mission_local.py --mission "Add risk controls" --dry-run
    python test_mission_local.py --list-files  # Just show what files would be found
"""

import asyncio
import argparse
import logging
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_mission")


def print_banner(text: str, char: str = "="):
    """Print a banner for visual separation."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def print_update(message: str):
    """Print an update (simulates Discord post)."""
    # Strip markdown for cleaner console output
    clean = message.replace("**", "").replace("```", "")
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {clean}")


async def print_update_async(message: str):
    """Async version of print_update for callback."""
    print_update(message)


class LocalMissionTester:
    """
    Tests mission workflow locally without Discord.
    """

    def __init__(self, project_dir: str, dry_run: bool = False):
        self.project_dir = Path(project_dir)
        self.dry_run = dry_run
        self.executor = None
        self.orch = None

    def _get_relevant_files(self) -> List[Path]:
        """Find relevant files in the project."""
        patterns = [
            "**/regime*.py", "**/*model*.py", "**/*ml*.py",
            "**/paper*.py", "**/trading*.py", "**/inference*.py",
            "**/train*.py", "**/predict*.py", "**/backtest*.py",
            "**/strategy*.py", "**/risk*.py", "**/data*.py",
            "**/config*.yaml", "**/config*.json"
        ]

        relevant_files = []
        exclude_dirs = ["__pycache__", "venv", ".venv", "node_modules", ".git"]
        for pattern in patterns:
            matches = list(self.project_dir.glob(pattern))
            for match in matches[:5]:  # Limit per pattern
                match_str = str(match)
                if match.is_file() and not any(ex in match_str for ex in exclude_dirs):
                    relevant_files.append(match)

        # Deduplicate and limit
        return list(set(relevant_files))[:15]

    def list_files(self):
        """Just list what files would be found."""
        print_banner("FILE DISCOVERY")
        print(f"Project directory: {self.project_dir}")

        files = self._get_relevant_files()

        if not files:
            print("No relevant files found!")
            return

        print(f"\nFound {len(files)} relevant files:\n")
        for f in sorted(files):
            rel_path = f.relative_to(self.project_dir)
            size = f.stat().st_size
            print(f"  - {rel_path} ({size:,} bytes)")

    def _read_file_contents(self, files: List[Path]) -> Dict[str, str]:
        """Read contents of files."""
        contents = {}
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                # Truncate long files
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                rel_path = str(file_path.relative_to(self.project_dir))
                contents[rel_path] = content
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        return contents

    async def test_file_exploration(self):
        """Test the file exploration step."""
        print_banner("FILE EXPLORATION TEST")

        files = self._get_relevant_files()
        print(f"Found {len(files)} files")

        contents = self._read_file_contents(files)
        print(f"Successfully read {len(contents)} files")

        # Show summary
        total_chars = sum(len(c) for c in contents.values())
        print(f"Total content: {total_chars:,} characters")

        # Show file summary
        for name, content in list(contents.items())[:5]:
            lines = content.count("\n")
            print(f"  - {name}: {lines} lines, {len(content):,} chars")

        return contents

    async def test_prompt_building(self, mission: str, file_contents: Dict[str, str]):
        """Test building the prompt for Claude."""
        print_banner("PROMPT BUILDING TEST")

        # Build context like drive_mission does
        context_parts = ["# CODEBASE ANALYSIS\n"]
        for file_name, content in file_contents.items():
            context_parts.append(f"\n## File: {file_name}\n```python\n{content}\n```\n")

        codebase_context = "".join(context_parts)

        prompt = f"""MISSION: {mission}

I've explored the codebase and found these relevant files. Based on this code analysis, create a task breakdown.

{codebase_context}

Based on the code above, create specific implementation tasks. Output EACH task on its own line using this EXACT format:

[TASK: data] specific task with file references
[TASK: tuning] specific task with file references
[TASK: backtest] specific task with file references
[TASK: risk] specific task with file references
[TASK: strategy] specific task with file references

Focus on the mission objective: {mission}

Be specific - reference the actual files and functions you see above."""

        print(f"Prompt length: {len(prompt):,} characters")
        print(f"Would fit in context: {'Yes' if len(prompt) < 100000 else 'No (too long)'}")

        # Show first 500 chars
        print("\nPrompt preview:")
        print("-" * 40)
        print(prompt[:500] + "...")

        return prompt

    async def test_claude_execution(self, mission: str, prompt: str):
        """Test executing with Claude Code."""
        print_banner("CLAUDE CODE EXECUTION TEST")

        if self.dry_run:
            print("DRY RUN - Skipping actual Claude Code execution")
            # Return mock response
            return """Based on the codebase analysis, here are the implementation tasks:

[TASK: data] Update data preprocessing in data_loader.py to add feature normalization
[TASK: tuning] Optimize learning rate in model_config.yaml using grid search
[TASK: backtest] Add walk-forward validation in backtest_runner.py
[TASK: risk] Add position sizing limits in risk_manager.py
[TASK: strategy] Implement momentum signal in strategy_v2.py

These tasks will improve the ML model accuracy as requested."""

        # Import and initialize executor
        from claude_executor import ClaudeExecutor

        if not self.executor:
            self.executor = ClaudeExecutor(project_dir=str(self.project_dir))

        print(f"Claude command: {self.executor.claude_cmd}")
        print(f"Project dir: {self.executor.project_dir}")
        print("Executing with Claude Code...")
        print("(This may take 30-60 seconds)")

        result = await self.executor.execute(
            agent_name="Strategy Agent",
            agent_role="Mission planner and code analyst",
            task_prompt=prompt,
            context=None
        )

        print(f"\nExecution completed in {result.duration_seconds:.1f}s")
        print(f"Status: {result.status.value}")

        if result.error:
            print(f"Error: {result.error}")

        return result.output if result.output else ""

    async def test_task_parsing(self, response: str):
        """Test parsing tasks from Claude's response."""
        print_banner("TASK PARSING TEST")

        task_pattern = r'\[TASK:\s*(\w+)\]\s*(.+?)(?=\[TASK:|$)'
        matches = re.findall(task_pattern, response, re.IGNORECASE | re.DOTALL)

        print(f"Found {len(matches)} tasks in response")

        tasks = []
        for agent_type, task_desc in matches:
            task = {
                "agent": agent_type.lower().strip(),
                "task": task_desc.strip()[:500]
            }
            tasks.append(task)
            print(f"\n  [{task['agent'].upper()}]")
            print(f"  {task['task'][:100]}...")

        return tasks

    async def test_full_workflow(self, mission: str):
        """Test the complete workflow."""
        print_banner(f"FULL WORKFLOW TEST: {mission}", "=")

        # Step 1: File exploration
        file_contents = await self.test_file_exploration()
        if not file_contents:
            print("ERROR: No files found!")
            return None

        # Step 2: Build prompt
        prompt = await self.test_prompt_building(mission, file_contents)

        # Step 3: Execute with Claude
        response = await self.test_claude_execution(mission, prompt)

        # Step 4: Parse tasks
        tasks = await self.test_task_parsing(response)

        # Summary
        print_banner("WORKFLOW SUMMARY")
        print(f"Mission: {mission}")
        print(f"Files analyzed: {len(file_contents)}")
        print(f"Tasks generated: {len(tasks)}")

        if tasks:
            print("\nTask breakdown:")
            for task in tasks:
                print(f"  - {task['agent']}: {task['task'][:60]}...")
            print("\n[OK] Workflow completed successfully!")
        else:
            print("\n[FAIL] No tasks were generated - Claude may need a better prompt")

        return tasks

    async def test_conversational_manager(self, mission: str):
        """Test using the full ConversationalMissionManager."""
        print_banner("CONVERSATIONAL MANAGER TEST")

        from orchestration_layer import OrchestrationLayer, ConversationalMissionManager
        from claude_executor import ClaudeExecutor

        # Initialize
        if not self.orch:
            self.orch = OrchestrationLayer()

        if not self.executor:
            self.executor = ClaudeExecutor(project_dir=str(self.project_dir))

        manager = ConversationalMissionManager(self.orch)

        print(f"Starting mission: {mission}")
        print(f"Project dir: {self.project_dir}")
        print(f"Orchestration provider: {self.orch.provider.value}")
        print()

        result = await manager.drive_mission(
            mission_objective=mission,
            project_dir=str(self.project_dir),
            claude_executor=self.executor,
            post_update=print_update_async
        )

        print_banner("MANAGER RESULT")
        print(f"Turns used: {result['turns_used']}")
        print(f"Complete: {result['complete']}")
        print(f"Tasks found: {len(result['tasks'])}")

        if result['tasks']:
            print("\nTasks:")
            for task in result['tasks']:
                print(f"  [{task['agent'].upper()}] {task['task'][:80]}...")

        return result


async def main():
    parser = argparse.ArgumentParser(description="Test mission workflow locally")
    parser.add_argument("--mission", "-m", type=str, help="Mission objective to test")
    parser.add_argument("--project-dir", "-p", type=str,
                       default=os.getenv("POLYMARKET_PROJECT_DIR", "."),
                       help="Project directory to analyze")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Skip actual Claude Code execution")
    parser.add_argument("--list-files", "-l", action="store_true",
                       help="Just list what files would be found")
    parser.add_argument("--use-manager", action="store_true",
                       help="Use ConversationalMissionManager instead of direct test")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve project directory
    project_dir = Path(args.project_dir).resolve()
    if not project_dir.exists():
        print(f"ERROR: Project directory does not exist: {project_dir}")
        sys.exit(1)

    print_banner("LOCAL MISSION TESTER")
    print(f"Project: {project_dir}")
    print(f"Dry run: {args.dry_run}")

    tester = LocalMissionTester(
        project_dir=str(project_dir),
        dry_run=args.dry_run
    )

    if args.list_files:
        tester.list_files()
        return

    if not args.mission:
        # Default test mission
        args.mission = "Improve the ML model accuracy and add proper risk controls"

    if args.use_manager:
        result = await tester.test_conversational_manager(args.mission)
    else:
        result = await tester.test_full_workflow(args.mission)

    if result:
        print("\n[OK] Test completed successfully")
    else:
        print("\n[FAIL] Test failed")


if __name__ == "__main__":
    asyncio.run(main())
