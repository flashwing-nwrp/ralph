"""
Improvement Proposal System for RALPH Agent Ensemble

Enables agents to propose self-improvements when they notice opportunities.
Proposals are queued and presented to the operator for approval at the end
of missions or cycles, so they don't block ongoing work.

Workflow:
1. Agent notices an opportunity during task execution
2. Agent submits a proposal with Problem/Solution statement
3. Proposals queue up during the mission
4. At mission completion, operator reviews proposals
5. Approved proposals become new missions or tasks
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("improvement_proposals")


class ProposalStatus(Enum):
    PENDING = "pending"          # Awaiting review
    APPROVED = "approved"        # Operator approved
    REJECTED = "rejected"        # Operator rejected
    IMPLEMENTED = "implemented"  # Changes made
    DEFERRED = "deferred"        # Postponed for later


class ProposalCategory(Enum):
    PERFORMANCE = "performance"      # Speed, efficiency improvements
    ACCURACY = "accuracy"            # Model, prediction improvements
    RISK = "risk"                    # Safety, risk management
    DATA = "data"                    # Data quality, new sources
    ARCHITECTURE = "architecture"    # Code structure, design
    STRATEGY = "strategy"            # Trading logic improvements
    AUTOMATION = "automation"        # Workflow, process improvements
    BUG_FIX = "bug_fix"             # Issues discovered during work
    FEATURE = "feature"              # New capabilities


class ProposalPriority(Enum):
    LOW = "low"           # Nice to have
    MEDIUM = "medium"     # Should do eventually
    HIGH = "high"         # Important improvement
    CRITICAL = "critical" # Blocking or urgent issue


@dataclass
class ImprovementProposal:
    """A proposal for system improvement from an agent."""

    proposal_id: str
    submitted_by: str           # Agent type that submitted
    category: ProposalCategory
    priority: ProposalPriority

    # Core proposal content
    problem: str                # What's the issue or opportunity?
    solution: str               # What should be done?
    expected_impact: str        # What improvement do we expect?

    # Context
    discovered_during: str = "" # Task/mission where this was noticed
    affected_files: List[str] = field(default_factory=list)
    related_metrics: Dict[str, float] = field(default_factory=dict)

    # Status tracking
    status: ProposalStatus = ProposalStatus.PENDING
    submitted_at: str = ""
    reviewed_at: str = ""
    reviewer_notes: str = ""

    # If approved, track implementation
    implementation_mission_id: str = ""

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "submitted_by": self.submitted_by,
            "category": self.category.value,
            "priority": self.priority.value,
            "problem": self.problem,
            "solution": self.solution,
            "expected_impact": self.expected_impact,
            "discovered_during": self.discovered_during,
            "affected_files": self.affected_files,
            "related_metrics": self.related_metrics,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "reviewed_at": self.reviewed_at,
            "reviewer_notes": self.reviewer_notes,
            "implementation_mission_id": self.implementation_mission_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImprovementProposal":
        return cls(
            proposal_id=data["proposal_id"],
            submitted_by=data["submitted_by"],
            category=ProposalCategory(data["category"]),
            priority=ProposalPriority(data["priority"]),
            problem=data["problem"],
            solution=data["solution"],
            expected_impact=data["expected_impact"],
            discovered_during=data.get("discovered_during", ""),
            affected_files=data.get("affected_files", []),
            related_metrics=data.get("related_metrics", {}),
            status=ProposalStatus(data.get("status", "pending")),
            submitted_at=data.get("submitted_at", ""),
            reviewed_at=data.get("reviewed_at", ""),
            reviewer_notes=data.get("reviewer_notes", ""),
            implementation_mission_id=data.get("implementation_mission_id", "")
        )

    def format_for_discord(self) -> str:
        """Format proposal for Discord display."""
        priority_emoji = {
            ProposalPriority.LOW: "🟢",
            ProposalPriority.MEDIUM: "🟡",
            ProposalPriority.HIGH: "🟠",
            ProposalPriority.CRITICAL: "🔴"
        }

        status_emoji = {
            ProposalStatus.PENDING: "⏳",
            ProposalStatus.APPROVED: "✅",
            ProposalStatus.REJECTED: "❌",
            ProposalStatus.IMPLEMENTED: "🚀",
            ProposalStatus.DEFERRED: "⏸️"
        }

        return f"""
{status_emoji[self.status]} **{self.proposal_id}** | {priority_emoji[self.priority]} {self.priority.value.upper()}
**Category:** {self.category.value} | **From:** {self.submitted_by}

**🔍 Problem:**
{self.problem}

**💡 Solution:**
{self.solution}

**📈 Expected Impact:**
{self.expected_impact}
"""


class ProposalManager:
    """
    Manages improvement proposals from agents.

    Proposals are collected during mission execution and presented
    to the operator for review at appropriate checkpoints.
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.proposals_file = self.project_dir / "improvement_proposals.json"
        self.proposals_history_file = self.project_dir / "proposals_history.json"

        self.proposals: List[ImprovementProposal] = []
        self.proposal_counter = 0
        self._lock = asyncio.Lock()

        # Load existing proposals
        self._load_proposals()

    def _load_proposals(self):
        """Load proposals from file."""
        try:
            if self.proposals_file.exists():
                with open(self.proposals_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.proposals = [
                        ImprovementProposal.from_dict(p)
                        for p in data.get("proposals", [])
                    ]
                    self.proposal_counter = data.get("counter", 0)
                    logger.info(f"Loaded {len(self.proposals)} proposals")
        except Exception as e:
            logger.error(f"Failed to load proposals: {e}")
            self.proposals = []

    async def _save_proposals(self):
        """Save proposals to file."""
        async with self._lock:
            try:
                data = {
                    "counter": self.proposal_counter,
                    "proposals": [p.to_dict() for p in self.proposals]
                }
                with open(self.proposals_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save proposals: {e}")

    async def submit_proposal(
        self,
        submitted_by: str,
        category: str,
        priority: str,
        problem: str,
        solution: str,
        expected_impact: str,
        discovered_during: str = "",
        affected_files: List[str] = None,
        related_metrics: Dict[str, float] = None
    ) -> ImprovementProposal:
        """
        Submit a new improvement proposal.

        Args:
            submitted_by: Agent type (tuning, backtest, risk, strategy, data)
            category: Category from ProposalCategory
            priority: Priority from ProposalPriority
            problem: Description of the problem or opportunity
            solution: Proposed solution
            expected_impact: What improvement we expect
            discovered_during: Task/mission context
            affected_files: List of files that would be modified
            related_metrics: Relevant metrics (e.g., {"sharpe": 1.2})

        Returns:
            The created proposal
        """
        self.proposal_counter += 1
        proposal_id = f"IMP-{self.proposal_counter:04d}"

        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            submitted_by=submitted_by,
            category=ProposalCategory(category),
            priority=ProposalPriority(priority),
            problem=problem,
            solution=solution,
            expected_impact=expected_impact,
            discovered_during=discovered_during,
            affected_files=affected_files or [],
            related_metrics=related_metrics or {},
            status=ProposalStatus.PENDING,
            submitted_at=datetime.utcnow().isoformat()
        )

        self.proposals.append(proposal)
        await self._save_proposals()

        logger.info(f"New proposal {proposal_id} from {submitted_by}: {problem[:50]}...")
        return proposal

    def get_pending_proposals(self) -> List[ImprovementProposal]:
        """Get all pending proposals."""
        return [p for p in self.proposals if p.status == ProposalStatus.PENDING]

    def get_proposals_by_agent(self, agent_type: str) -> List[ImprovementProposal]:
        """Get proposals submitted by a specific agent."""
        return [p for p in self.proposals if p.submitted_by == agent_type]

    def get_proposals_by_priority(self, priority: ProposalPriority) -> List[ImprovementProposal]:
        """Get proposals of a specific priority."""
        return [p for p in self.proposals if p.priority == priority]

    async def approve_proposal(
        self,
        proposal_id: str,
        reviewer_notes: str = ""
    ) -> Optional[ImprovementProposal]:
        """Approve a proposal for implementation."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.status = ProposalStatus.APPROVED
                proposal.reviewed_at = datetime.utcnow().isoformat()
                proposal.reviewer_notes = reviewer_notes
                await self._save_proposals()
                return proposal
        return None

    async def reject_proposal(
        self,
        proposal_id: str,
        reviewer_notes: str = ""
    ) -> Optional[ImprovementProposal]:
        """Reject a proposal."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.status = ProposalStatus.REJECTED
                proposal.reviewed_at = datetime.utcnow().isoformat()
                proposal.reviewer_notes = reviewer_notes
                await self._save_proposals()
                return proposal
        return None

    async def defer_proposal(
        self,
        proposal_id: str,
        reviewer_notes: str = ""
    ) -> Optional[ImprovementProposal]:
        """Defer a proposal for later consideration."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.status = ProposalStatus.DEFERRED
                proposal.reviewed_at = datetime.utcnow().isoformat()
                proposal.reviewer_notes = reviewer_notes
                await self._save_proposals()
                return proposal
        return None

    async def mark_implemented(
        self,
        proposal_id: str,
        mission_id: str = ""
    ) -> Optional[ImprovementProposal]:
        """Mark a proposal as implemented."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.status = ProposalStatus.IMPLEMENTED
                proposal.implementation_mission_id = mission_id
                await self._save_proposals()

                # Archive to history
                await self._archive_proposal(proposal)
                return proposal
        return None

    async def _archive_proposal(self, proposal: ImprovementProposal):
        """Archive an implemented proposal to history."""
        try:
            history = []
            if self.proposals_history_file.exists():
                with open(self.proposals_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)

            history.append(proposal.to_dict())

            with open(self.proposals_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to archive proposal: {e}")

    def get_summary(self) -> str:
        """Get a summary of current proposals."""
        pending = self.get_pending_proposals()

        if not pending:
            return "No pending improvement proposals."

        by_priority = {
            ProposalPriority.CRITICAL: [],
            ProposalPriority.HIGH: [],
            ProposalPriority.MEDIUM: [],
            ProposalPriority.LOW: []
        }

        for p in pending:
            by_priority[p.priority].append(p)

        summary = [f"**{len(pending)} Pending Improvement Proposals**\n"]

        for priority, proposals in by_priority.items():
            if proposals:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                summary.append(f"\n{emoji[priority.value]} **{priority.value.upper()}** ({len(proposals)})")
                for p in proposals[:3]:  # Show max 3 per priority
                    summary.append(f"  • `{p.proposal_id}` [{p.submitted_by}] {p.problem[:50]}...")
                if len(proposals) > 3:
                    summary.append(f"  • ...and {len(proposals) - 3} more")

        summary.append(f"\n*Use `!proposals` to view details or `!approve <id>` to approve*")
        return "\n".join(summary)

    def get_review_queue(self) -> str:
        """Get formatted review queue for Discord."""
        pending = self.get_pending_proposals()

        if not pending:
            return "🎉 No proposals to review! All caught up."

        # Sort by priority (critical first)
        priority_order = {
            ProposalPriority.CRITICAL: 0,
            ProposalPriority.HIGH: 1,
            ProposalPriority.MEDIUM: 2,
            ProposalPriority.LOW: 3
        }
        pending.sort(key=lambda p: priority_order[p.priority])

        output = ["## 📋 Improvement Proposals for Review\n"]

        for proposal in pending:
            output.append(proposal.format_for_discord())
            output.append("---")

        output.append("\n**Commands:**")
        output.append("`!approve <id>` - Approve and create mission")
        output.append("`!reject <id> [reason]` - Reject proposal")
        output.append("`!defer <id> [reason]` - Defer for later")

        return "\n".join(output)


# Singleton instance
_proposal_manager: Optional[ProposalManager] = None


def get_proposal_manager() -> ProposalManager:
    """Get or create the proposal manager instance."""
    global _proposal_manager
    if _proposal_manager is None:
        _proposal_manager = ProposalManager()
    return _proposal_manager


def set_proposal_manager(manager: ProposalManager):
    """Set the proposal manager instance."""
    global _proposal_manager
    _proposal_manager = manager
