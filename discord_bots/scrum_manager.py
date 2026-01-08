"""
SCRUM Methodology Framework for RALPH Agent Ensemble

Implements agile SCRUM practices for structured iterative development:
- Product Backlog: Prioritized list of user stories and tasks
- Sprint Planning: Select items for the sprint
- Sprint Execution: Daily standups, progress tracking
- Sprint Review: Demo completed work
- Sprint Retrospective: Lessons learned, improvements
- Backlog Grooming: Refine and prioritize

Mission is complete when all backlog items are done and deployed to production.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("scrum_manager")


class StoryStatus(Enum):
    BACKLOG = "backlog"           # In product backlog
    SPRINT = "sprint"             # Selected for current sprint
    IN_PROGRESS = "in_progress"   # Being worked on
    IN_REVIEW = "in_review"       # Awaiting review/validation
    DONE = "done"                 # Completed
    BLOCKED = "blocked"           # Cannot proceed


class StoryType(Enum):
    FEATURE = "feature"           # New functionality
    BUG = "bug"                   # Bug fix
    IMPROVEMENT = "improvement"   # Enhancement
    RESEARCH = "research"         # Investigation/spike
    TECHNICAL = "technical"       # Tech debt, refactoring


class SprintStatus(Enum):
    PLANNING = "planning"         # Sprint being planned
    ACTIVE = "active"             # Sprint in progress
    REVIEW = "review"             # Sprint review phase
    RETRO = "retro"               # Retrospective phase
    COMPLETED = "completed"       # Sprint done


@dataclass
class UserStory:
    """A user story or task in the backlog."""
    story_id: str
    title: str
    description: str
    story_type: StoryType
    status: StoryStatus = StoryStatus.BACKLOG

    # Estimation and assignment
    story_points: int = 0         # Fibonacci: 1, 2, 3, 5, 8, 13
    assigned_to: str = ""         # Agent type
    priority: int = 0             # Lower = higher priority

    # Acceptance criteria
    acceptance_criteria: List[str] = field(default_factory=list)

    # Tracking
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    sprint_id: str = ""

    # Dependencies
    blocked_by: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "story_id": self.story_id,
            "title": self.title,
            "description": self.description,
            "story_type": self.story_type.value,
            "status": self.status.value,
            "story_points": self.story_points,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "acceptance_criteria": self.acceptance_criteria,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "sprint_id": self.sprint_id,
            "blocked_by": self.blocked_by,
            "blocks": self.blocks
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserStory":
        return cls(
            story_id=data["story_id"],
            title=data["title"],
            description=data["description"],
            story_type=StoryType(data["story_type"]),
            status=StoryStatus(data.get("status", "backlog")),
            story_points=data.get("story_points", 0),
            assigned_to=data.get("assigned_to", ""),
            priority=data.get("priority", 0),
            acceptance_criteria=data.get("acceptance_criteria", []),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            sprint_id=data.get("sprint_id", ""),
            blocked_by=data.get("blocked_by", []),
            blocks=data.get("blocks", [])
        )

    def format_card(self) -> str:
        """Format as a card for Discord."""
        status_emoji = {
            StoryStatus.BACKLOG: "📋",
            StoryStatus.SPRINT: "📌",
            StoryStatus.IN_PROGRESS: "🔄",
            StoryStatus.IN_REVIEW: "👀",
            StoryStatus.DONE: "✅",
            StoryStatus.BLOCKED: "🚫"
        }
        type_emoji = {
            StoryType.FEATURE: "✨",
            StoryType.BUG: "🐛",
            StoryType.IMPROVEMENT: "📈",
            StoryType.RESEARCH: "🔬",
            StoryType.TECHNICAL: "🔧"
        }

        points = f"[{self.story_points}pts]" if self.story_points else ""
        assignee = f"@{self.assigned_to}" if self.assigned_to else "unassigned"

        return f"{status_emoji[self.status]} {type_emoji[self.story_type]} **{self.story_id}** {points}\n{self.title}\n*{assignee}*"


@dataclass
class Sprint:
    """A development sprint."""
    sprint_id: str
    name: str
    goal: str
    status: SprintStatus = SprintStatus.PLANNING

    # Timing
    start_date: str = ""
    end_date: str = ""
    duration_days: int = 14       # 2-week sprints default

    # Capacity and velocity
    team_capacity: int = 0        # Total story points available
    committed_points: int = 0     # Points committed to sprint
    completed_points: int = 0     # Points actually completed

    # Stories in this sprint
    story_ids: List[str] = field(default_factory=list)

    # Retrospective
    went_well: List[str] = field(default_factory=list)
    to_improve: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sprint_id": self.sprint_id,
            "name": self.name,
            "goal": self.goal,
            "status": self.status.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "duration_days": self.duration_days,
            "team_capacity": self.team_capacity,
            "committed_points": self.committed_points,
            "completed_points": self.completed_points,
            "story_ids": self.story_ids,
            "went_well": self.went_well,
            "to_improve": self.to_improve,
            "action_items": self.action_items
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Sprint":
        return cls(
            sprint_id=data["sprint_id"],
            name=data["name"],
            goal=data["goal"],
            status=SprintStatus(data.get("status", "planning")),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            duration_days=data.get("duration_days", 14),
            team_capacity=data.get("team_capacity", 0),
            committed_points=data.get("committed_points", 0),
            completed_points=data.get("completed_points", 0),
            story_ids=data.get("story_ids", []),
            went_well=data.get("went_well", []),
            to_improve=data.get("to_improve", []),
            action_items=data.get("action_items", [])
        )

    def get_progress(self) -> Dict[str, int]:
        """Calculate sprint progress."""
        if self.committed_points == 0:
            return {"percent": 0, "completed": 0, "committed": 0}

        return {
            "percent": int((self.completed_points / self.committed_points) * 100),
            "completed": self.completed_points,
            "committed": self.committed_points
        }

    def days_remaining(self) -> int:
        """Calculate days remaining in sprint."""
        if not self.end_date:
            return self.duration_days

        end = datetime.fromisoformat(self.end_date)
        now = datetime.utcnow()
        remaining = (end - now).days
        return max(0, remaining)


class ScrumManager:
    """
    Manages SCRUM workflow for the agent ensemble.

    Provides:
    - Product backlog management
    - Sprint planning and execution
    - Progress tracking
    - Retrospectives
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.backlog_file = self.project_dir / "product_backlog.json"
        self.sprints_file = self.project_dir / "sprints.json"

        self.stories: Dict[str, UserStory] = {}
        self.sprints: Dict[str, Sprint] = {}
        self.current_sprint_id: Optional[str] = None
        self.story_counter = 0
        self.sprint_counter = 0
        self._lock = asyncio.Lock()

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load backlog and sprints from files."""
        try:
            if self.backlog_file.exists():
                with open(self.backlog_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stories = {
                        s["story_id"]: UserStory.from_dict(s)
                        for s in data.get("stories", [])
                    }
                    self.story_counter = data.get("story_counter", 0)

            if self.sprints_file.exists():
                with open(self.sprints_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sprints = {
                        s["sprint_id"]: Sprint.from_dict(s)
                        for s in data.get("sprints", [])
                    }
                    self.sprint_counter = data.get("sprint_counter", 0)
                    self.current_sprint_id = data.get("current_sprint_id")

            logger.info(f"Loaded {len(self.stories)} stories, {len(self.sprints)} sprints")
        except Exception as e:
            logger.error(f"Failed to load SCRUM data: {e}")

    async def _save_data(self):
        """Save backlog and sprints to files."""
        async with self._lock:
            try:
                # Save backlog
                backlog_data = {
                    "story_counter": self.story_counter,
                    "stories": [s.to_dict() for s in self.stories.values()]
                }
                with open(self.backlog_file, "w", encoding="utf-8") as f:
                    json.dump(backlog_data, f, indent=2)

                # Save sprints
                sprints_data = {
                    "sprint_counter": self.sprint_counter,
                    "current_sprint_id": self.current_sprint_id,
                    "sprints": [s.to_dict() for s in self.sprints.values()]
                }
                with open(self.sprints_file, "w", encoding="utf-8") as f:
                    json.dump(sprints_data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save SCRUM data: {e}")

    # =========================================================================
    # BACKLOG MANAGEMENT
    # =========================================================================

    async def create_story(
        self,
        title: str,
        description: str,
        story_type: str = "feature",
        story_points: int = 0,
        priority: int = 100,
        acceptance_criteria: List[str] = None
    ) -> UserStory:
        """Create a new user story in the backlog."""
        self.story_counter += 1
        story_id = f"US-{self.story_counter:04d}"

        story = UserStory(
            story_id=story_id,
            title=title,
            description=description,
            story_type=StoryType(story_type),
            status=StoryStatus.BACKLOG,
            story_points=story_points,
            priority=priority,
            acceptance_criteria=acceptance_criteria or [],
            created_at=datetime.utcnow().isoformat()
        )

        self.stories[story_id] = story
        await self._save_data()

        logger.info(f"Created story {story_id}: {title}")
        return story

    async def update_story_status(
        self,
        story_id: str,
        status: str,
        assigned_to: str = None
    ) -> Optional[UserStory]:
        """Update a story's status."""
        story = self.stories.get(story_id)
        if not story:
            return None

        old_status = story.status
        story.status = StoryStatus(status)

        if assigned_to:
            story.assigned_to = assigned_to

        if status == "in_progress" and not story.started_at:
            story.started_at = datetime.utcnow().isoformat()

        if status == "done":
            story.completed_at = datetime.utcnow().isoformat()

            # Update sprint completed points
            if story.sprint_id and story.sprint_id in self.sprints:
                self.sprints[story.sprint_id].completed_points += story.story_points

        await self._save_data()
        return story

    def get_backlog(self) -> List[UserStory]:
        """Get prioritized product backlog."""
        backlog = [s for s in self.stories.values() if s.status == StoryStatus.BACKLOG]
        return sorted(backlog, key=lambda s: s.priority)

    def get_sprint_stories(self, sprint_id: str = None) -> List[UserStory]:
        """Get stories in a sprint."""
        sid = sprint_id or self.current_sprint_id
        if not sid:
            return []

        return [
            s for s in self.stories.values()
            if s.sprint_id == sid and s.status != StoryStatus.BACKLOG
        ]

    # =========================================================================
    # SPRINT MANAGEMENT
    # =========================================================================

    async def create_sprint(
        self,
        name: str,
        goal: str,
        duration_days: int = 14,
        team_capacity: int = 40
    ) -> Sprint:
        """Create a new sprint."""
        self.sprint_counter += 1
        sprint_id = f"SPRINT-{self.sprint_counter:02d}"

        sprint = Sprint(
            sprint_id=sprint_id,
            name=name,
            goal=goal,
            status=SprintStatus.PLANNING,
            duration_days=duration_days,
            team_capacity=team_capacity
        )

        self.sprints[sprint_id] = sprint
        await self._save_data()

        logger.info(f"Created sprint {sprint_id}: {name}")
        return sprint

    async def start_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """Start a sprint."""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            return None

        sprint.status = SprintStatus.ACTIVE
        sprint.start_date = datetime.utcnow().isoformat()
        sprint.end_date = (
            datetime.utcnow() + timedelta(days=sprint.duration_days)
        ).isoformat()

        self.current_sprint_id = sprint_id

        # Update story statuses
        for story_id in sprint.story_ids:
            if story_id in self.stories:
                self.stories[story_id].status = StoryStatus.SPRINT

        await self._save_data()
        return sprint

    async def add_to_sprint(
        self,
        sprint_id: str,
        story_id: str
    ) -> Optional[UserStory]:
        """Add a story to a sprint."""
        sprint = self.sprints.get(sprint_id)
        story = self.stories.get(story_id)

        if not sprint or not story:
            return None

        if story_id not in sprint.story_ids:
            sprint.story_ids.append(story_id)
            sprint.committed_points += story.story_points

        story.sprint_id = sprint_id
        story.status = StoryStatus.SPRINT

        await self._save_data()
        return story

    async def end_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """End a sprint and move to review."""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            return None

        sprint.status = SprintStatus.REVIEW

        # Move incomplete stories back to backlog
        incomplete = []
        for story_id in sprint.story_ids:
            story = self.stories.get(story_id)
            if story and story.status != StoryStatus.DONE:
                story.status = StoryStatus.BACKLOG
                story.sprint_id = ""
                incomplete.append(story_id)

        await self._save_data()
        return sprint

    async def run_retrospective(
        self,
        sprint_id: str,
        went_well: List[str],
        to_improve: List[str],
        action_items: List[str]
    ) -> Optional[Sprint]:
        """Record sprint retrospective."""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            return None

        sprint.status = SprintStatus.RETRO
        sprint.went_well = went_well
        sprint.to_improve = to_improve
        sprint.action_items = action_items

        await self._save_data()
        return sprint

    async def complete_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """Mark sprint as completed."""
        sprint = self.sprints.get(sprint_id)
        if not sprint:
            return None

        sprint.status = SprintStatus.COMPLETED

        if self.current_sprint_id == sprint_id:
            self.current_sprint_id = None

        await self._save_data()
        return sprint

    def get_current_sprint(self) -> Optional[Sprint]:
        """Get the current active sprint."""
        if not self.current_sprint_id:
            return None
        return self.sprints.get(self.current_sprint_id)

    # =========================================================================
    # VIEWS AND REPORTS
    # =========================================================================

    def get_sprint_board(self, sprint_id: str = None) -> str:
        """Get a Kanban-style sprint board."""
        sid = sprint_id or self.current_sprint_id
        if not sid:
            return "No active sprint. Use `!sprint create <name> <goal>` to create one."

        sprint = self.sprints.get(sid)
        if not sprint:
            return f"Sprint {sid} not found."

        stories = self.get_sprint_stories(sid)

        # Group by status
        columns = {
            "To Do": [s for s in stories if s.status == StoryStatus.SPRINT],
            "In Progress": [s for s in stories if s.status == StoryStatus.IN_PROGRESS],
            "In Review": [s for s in stories if s.status == StoryStatus.IN_REVIEW],
            "Done": [s for s in stories if s.status == StoryStatus.DONE],
            "Blocked": [s for s in stories if s.status == StoryStatus.BLOCKED]
        }

        progress = sprint.get_progress()
        days_left = sprint.days_remaining()

        output = [
            f"## 🏃 Sprint: {sprint.name}",
            f"**Goal:** {sprint.goal}",
            f"**Progress:** {progress['completed']}/{progress['committed']} points ({progress['percent']}%)",
            f"**Days Remaining:** {days_left}",
            ""
        ]

        for col_name, col_stories in columns.items():
            if col_stories:
                output.append(f"### {col_name} ({len(col_stories)})")
                for story in col_stories:
                    output.append(story.format_card())
                output.append("")

        return "\n".join(output)

    def get_backlog_view(self, limit: int = 20) -> str:
        """Get formatted backlog view."""
        backlog = self.get_backlog()[:limit]

        if not backlog:
            return "📋 Backlog is empty. Use `!story add <title>` to add stories."

        output = [
            "## 📋 Product Backlog",
            f"*{len(self.stories)} total stories*\n"
        ]

        for i, story in enumerate(backlog, 1):
            points = f"[{story.story_points}pts]" if story.story_points else "[?pts]"
            output.append(f"{i}. **{story.story_id}** {points} {story.title}")

        if len(self.get_backlog()) > limit:
            output.append(f"\n*...and {len(self.get_backlog()) - limit} more*")

        return "\n".join(output)

    def get_daily_standup(self) -> str:
        """Generate daily standup summary."""
        sprint = self.get_current_sprint()
        if not sprint:
            return "No active sprint for standup."

        stories = self.get_sprint_stories()

        # Group by agent
        by_agent = {}
        for story in stories:
            agent = story.assigned_to or "Unassigned"
            if agent not in by_agent:
                by_agent[agent] = {"in_progress": [], "done": [], "blocked": []}

            if story.status == StoryStatus.IN_PROGRESS:
                by_agent[agent]["in_progress"].append(story)
            elif story.status == StoryStatus.DONE:
                by_agent[agent]["done"].append(story)
            elif story.status == StoryStatus.BLOCKED:
                by_agent[agent]["blocked"].append(story)

        output = [
            "## 🌅 Daily Standup",
            f"**Sprint:** {sprint.name}",
            f"**Days Left:** {sprint.days_remaining()}",
            ""
        ]

        for agent, work in by_agent.items():
            if any(work.values()):
                output.append(f"### {agent.title()} Agent")

                if work["done"]:
                    output.append("**Completed:**")
                    for s in work["done"]:
                        output.append(f"  ✅ {s.story_id}: {s.title}")

                if work["in_progress"]:
                    output.append("**Working on:**")
                    for s in work["in_progress"]:
                        output.append(f"  🔄 {s.story_id}: {s.title}")

                if work["blocked"]:
                    output.append("**Blocked:**")
                    for s in work["blocked"]:
                        output.append(f"  🚫 {s.story_id}: {s.title}")

                output.append("")

        return "\n".join(output)

    def get_velocity_report(self) -> str:
        """Get velocity report across sprints."""
        completed_sprints = [
            s for s in self.sprints.values()
            if s.status == SprintStatus.COMPLETED
        ]

        if not completed_sprints:
            return "No completed sprints yet. Complete a sprint to see velocity."

        output = ["## 📊 Velocity Report\n"]

        total_committed = 0
        total_completed = 0

        for sprint in completed_sprints:
            total_committed += sprint.committed_points
            total_completed += sprint.completed_points

            completion_rate = int((sprint.completed_points / sprint.committed_points * 100)) if sprint.committed_points else 0
            output.append(
                f"**{sprint.sprint_id}**: {sprint.completed_points}/{sprint.committed_points} pts ({completion_rate}%)"
            )

        avg_velocity = total_completed / len(completed_sprints)
        output.append(f"\n**Average Velocity:** {avg_velocity:.1f} points/sprint")

        return "\n".join(output)


# Singleton instance
_scrum_manager: Optional[ScrumManager] = None


def get_scrum_manager() -> ScrumManager:
    """Get or create the SCRUM manager instance."""
    global _scrum_manager
    if _scrum_manager is None:
        _scrum_manager = ScrumManager()
    return _scrum_manager


def set_scrum_manager(manager: ScrumManager):
    """Set the SCRUM manager instance."""
    global _scrum_manager
    _scrum_manager = manager
