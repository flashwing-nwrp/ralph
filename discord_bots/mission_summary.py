"""
Mission Summary Generator for RALPH Agent Ensemble

Generates executive summaries, retrospectives, and learnings documentation
when missions complete. Uses the orchestration layer's LLM for analysis.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from knowledge_base import get_knowledge_base

logger = logging.getLogger("mission_summary")


class MissionSummaryGenerator:
    """
    Generates comprehensive summaries when missions complete.

    Features:
    - Executive summary with key findings
    - Suggested next steps
    - SCRUM retrospective
    - Learnings documentation for knowledge base
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("PROJECT_DIR", "."))
        self.learnings_dir = self.project_dir / "learnings"
        self.learnings_dir.mkdir(parents=True, exist_ok=True)

    async def generate_summary(
        self,
        mission: Any,
        orchestrator: Any = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive mission summary.

        Args:
            mission: The completed mission object
            orchestrator: OrchestrationLayer for LLM analysis (optional)

        Returns:
            Dictionary with summary components
        """
        # Gather all task outputs
        task_outputs = self._collect_task_outputs(mission)

        # Calculate statistics
        stats = self._calculate_stats(mission)

        # Generate summaries (with or without LLM)
        if orchestrator:
            summary_data = await self._generate_with_llm(
                mission, task_outputs, stats, orchestrator
            )
        else:
            summary_data = self._generate_basic_summary(mission, task_outputs, stats)

        # Save learnings to knowledge base
        await self._save_learnings(mission, summary_data)

        return summary_data

    def _collect_task_outputs(self, mission: Any) -> List[Dict[str, str]]:
        """Collect outputs from all completed tasks."""
        outputs = []
        for task in mission.tasks:
            if task.status == "completed" and task.output:
                outputs.append({
                    "task_id": task.task_id,
                    "agent": task.assigned_to,
                    "description": task.description[:200],
                    "output": task.output[:500]  # Truncate for analysis
                })
        return outputs

    def _calculate_stats(self, mission: Any) -> Dict[str, Any]:
        """Calculate mission statistics."""
        tasks_by_agent = {}
        for task in mission.tasks:
            agent = task.assigned_to
            tasks_by_agent[agent] = tasks_by_agent.get(agent, 0) + 1

        # Calculate duration
        duration_minutes = 0
        if mission.started_at:
            try:
                start = datetime.fromisoformat(mission.started_at)
                end = datetime.fromisoformat(mission.completed_at) if mission.completed_at else datetime.utcnow()
                duration_minutes = (end - start).total_seconds() / 60
            except:
                pass

        return {
            "total_tasks": len(mission.tasks),
            "completed_tasks": len([t for t in mission.tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in mission.tasks if t.status == "failed"]),
            "tasks_by_agent": tasks_by_agent,
            "duration_minutes": duration_minutes
        }

    async def _generate_with_llm(
        self,
        mission: Any,
        task_outputs: List[Dict],
        stats: Dict,
        orchestrator: Any
    ) -> Dict[str, Any]:
        """Generate summary using the orchestrator's LLM."""

        # Build context from task outputs
        outputs_text = "\n\n".join([
            f"[{t['agent'].upper()}] {t['description']}\nResult: {t['output']}"
            for t in task_outputs[:10]  # Limit to first 10 for context
        ])

        # Generate executive summary
        summary_prompt = f"""Analyze this completed mission and provide a brief executive summary.

Mission: {mission.objective}
Tasks Completed: {stats['completed_tasks']}/{stats['total_tasks']}
Duration: {stats['duration_minutes']:.1f} minutes

Task Outputs:
{outputs_text}

Provide:
1. A 2-3 sentence work summary
2. 3-5 key findings (bullet points)
3. 3-5 suggested next steps/experiments

Format as JSON:
{{"work_summary": "...", "key_findings": ["...", "..."], "suggestions": ["...", "..."]}}"""

        try:
            response, _ = await orchestrator._call_cheap_llm(
                summary_prompt,
                system="You are an AI project manager summarizing mission results. Be concise and actionable.",
                max_tokens=1000
            )

            # Parse JSON response
            if response:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "work_summary": parsed.get("work_summary", "Mission completed successfully."),
                        "key_findings": parsed.get("key_findings", []),
                        "suggestions": parsed.get("suggestions", []),
                        "stats": stats
                    }
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")

        # Fallback to basic summary
        return self._generate_basic_summary(mission, task_outputs, stats)

    def _generate_basic_summary(
        self,
        mission: Any,
        task_outputs: List[Dict],
        stats: Dict
    ) -> Dict[str, Any]:
        """Generate basic summary without LLM."""

        # Extract key findings from task outputs
        findings = []
        for output in task_outputs[:5]:
            # Look for key patterns in outputs
            text = output.get("output", "")
            if "found" in text.lower() or "discovered" in text.lower():
                findings.append(f"{output['agent'].title()} Agent: {text[:100]}...")
            elif "implemented" in text.lower() or "added" in text.lower():
                findings.append(f"{output['agent'].title()} Agent: {text[:100]}...")

        # Default suggestions
        suggestions = [
            "Review the task outputs for detailed results",
            "Run validation tests on any code changes",
            "Consider a follow-up mission for items marked as future work",
            "Update documentation if architecture changed",
            "Monitor performance metrics for regressions"
        ]

        return {
            "work_summary": f"Completed {stats['completed_tasks']} tasks across {len(stats['tasks_by_agent'])} agents in {stats['duration_minutes']:.1f} minutes.",
            "key_findings": findings or ["Mission completed - review individual task outputs for details"],
            "suggestions": suggestions,
            "stats": stats
        }

    async def generate_retrospective(
        self,
        mission: Any,
        orchestrator: Any = None
    ) -> Dict[str, List[str]]:
        """
        Generate a SCRUM retrospective for the mission.

        Returns:
            Dictionary with what_went_well, what_could_improve, learnings, action_items
        """
        task_outputs = self._collect_task_outputs(mission)
        stats = self._calculate_stats(mission)

        if orchestrator:
            return await self._generate_retro_with_llm(mission, task_outputs, stats, orchestrator)

        return self._generate_basic_retro(mission, task_outputs, stats)

    async def _generate_retro_with_llm(
        self,
        mission: Any,
        task_outputs: List[Dict],
        stats: Dict,
        orchestrator: Any
    ) -> Dict[str, List[str]]:
        """Generate retrospective using LLM analysis."""

        outputs_text = "\n".join([
            f"[{t['agent']}] {t['description'][:80]}: {t['output'][:150]}"
            for t in task_outputs[:8]
        ])

        retro_prompt = f"""Conduct a SCRUM retrospective for this completed mission.

Mission: {mission.objective}
Stats: {stats['completed_tasks']}/{stats['total_tasks']} tasks, {stats['failed_tasks']} failed, {stats['duration_minutes']:.0f} min

Sample outputs:
{outputs_text}

Provide retrospective in JSON format:
{{"what_went_well": ["...", "..."], "what_could_improve": ["...", "..."], "learnings": ["...", "..."], "action_items": ["...", "..."]}}

Focus on actionable insights for future missions."""

        try:
            response, _ = await orchestrator._call_cheap_llm(
                retro_prompt,
                system="You are an agile coach conducting a sprint retrospective. Be constructive and specific.",
                max_tokens=800
            )

            if response:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"LLM retro generation failed: {e}")

        return self._generate_basic_retro(mission, task_outputs, stats)

    def _generate_basic_retro(
        self,
        mission: Any,
        task_outputs: List[Dict],
        stats: Dict
    ) -> Dict[str, List[str]]:
        """Generate basic retrospective without LLM."""

        what_went_well = []
        what_could_improve = []

        # Analyze based on stats
        if stats['failed_tasks'] == 0:
            what_went_well.append("All tasks completed successfully - no failures")
        else:
            what_could_improve.append(f"{stats['failed_tasks']} task(s) failed - investigate root cause")

        if stats['duration_minutes'] < 30:
            what_went_well.append("Mission completed in under 30 minutes - good efficiency")

        # Count agent participation
        active_agents = len(stats['tasks_by_agent'])
        if active_agents >= 3:
            what_went_well.append(f"Good collaboration across {active_agents} agents")

        return {
            "what_went_well": what_went_well or ["Mission completed"],
            "what_could_improve": what_could_improve or ["Review for optimization opportunities"],
            "learnings": ["Document patterns that worked well", "Note any blockers encountered"],
            "action_items": ["Archive mission for future reference", "Update best practices if applicable"]
        }

    async def _save_learnings(self, mission: Any, summary_data: Dict[str, Any]):
        """Save learnings to the knowledge base."""

        learning_doc = {
            "mission_id": mission.mission_id,
            "objective": mission.objective,
            "completed_at": datetime.utcnow().isoformat(),
            "stats": summary_data.get("stats", {}),
            "key_findings": summary_data.get("key_findings", []),
            "suggestions": summary_data.get("suggestions", []),
            "work_summary": summary_data.get("work_summary", "")
        }

        # Save to learnings directory (JSON file backup)
        filename = f"{mission.mission_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.learnings_dir / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(learning_doc, f, indent=2)
            logger.info(f"Saved learnings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save learnings: {e}")

        # Also append to consolidated learnings file
        consolidated_path = self.learnings_dir / "consolidated_learnings.jsonl"
        try:
            with open(consolidated_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(learning_doc) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to consolidated learnings: {e}")

        # Save to vector-searchable knowledge base for agent context injection
        try:
            kb = get_knowledge_base()
            kb.add_learnings_from_mission(mission.mission_id, summary_data)
            logger.info(f"Added learnings to knowledge base for mission {mission.mission_id}")
        except Exception as e:
            logger.error(f"Failed to add learnings to knowledge base: {e}")


# Singleton instance
_summary_generator: Optional[MissionSummaryGenerator] = None


def get_summary_generator() -> MissionSummaryGenerator:
    """Get or create the summary generator."""
    global _summary_generator
    if _summary_generator is None:
        _summary_generator = MissionSummaryGenerator()
    return _summary_generator
