"""
Backlog Manager - Agile Backlog System for RALPH Agents

Manages a persistent backlog of observations, ideas, bugs, and improvements
that agents notice during work. Enables operator participation in ceremonies
like backlog grooming and sprint planning.
"""

import json
import os
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum


class BacklogItemType(str, Enum):
    BUG = "bug"
    IMPROVEMENT = "improvement"
    IDEA = "idea"
    TECH_DEBT = "tech_debt"


class BacklogPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BacklogEffort(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class BacklogStatus(str, Enum):
    PENDING = "pending"        # Awaiting operator review
    APPROVED = "approved"      # Operator approved for future sprint
    REJECTED = "rejected"      # Operator rejected
    IN_PROGRESS = "in_progress"  # Being worked on
    COMPLETED = "completed"    # Done


@dataclass
class BacklogItem:
    id: str
    item_type: str
    title: str
    priority: str
    effort: str
    rationale: str
    status: str
    created_at: str
    created_by: str  # Agent type that created it
    mission_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    completed_at: Optional[str] = None
    sprint_id: Optional[str] = None


class BacklogManager:
    """Manages the team backlog with persistence."""

    def __init__(self, backlog_file: str = None):
        if backlog_file is None:
            backlog_file = os.path.join(
                os.path.dirname(__file__),
                "data",
                "backlog.json"
            )
        self.backlog_file = backlog_file
        self._ensure_data_dir()
        self.items: Dict[str, BacklogItem] = {}
        self._load()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        data_dir = os.path.dirname(self.backlog_file)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    def _load(self):
        """Load backlog from file."""
        if os.path.exists(self.backlog_file):
            try:
                with open(self.backlog_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item_data in data.get("items", []):
                        item = BacklogItem(**item_data)
                        self.items[item.id] = item
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load backlog: {e}")
                self.items = {}

    def _save(self):
        """Save backlog to file."""
        try:
            data = {
                "items": [asdict(item) for item in self.items.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(self.backlog_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save backlog: {e}")

    def _generate_id(self) -> str:
        """Generate a unique backlog item ID."""
        existing_nums = []
        for item_id in self.items.keys():
            if item_id.startswith("BL-"):
                try:
                    num = int(item_id.split("-")[1])
                    existing_nums.append(num)
                except (IndexError, ValueError):
                    pass
        next_num = max(existing_nums, default=0) + 1
        return f"BL-{next_num:04d}"

    def add_item(
        self,
        item_type: str,
        title: str,
        priority: str,
        effort: str,
        rationale: str,
        created_by: str,
        mission_id: Optional[str] = None
    ) -> BacklogItem:
        """Add a new backlog item."""
        item = BacklogItem(
            id=self._generate_id(),
            item_type=item_type.lower(),
            title=title,
            priority=priority.lower(),
            effort=effort.lower(),
            rationale=rationale,
            status=BacklogStatus.PENDING.value,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            mission_id=mission_id
        )
        self.items[item.id] = item
        self._save()
        return item

    def parse_and_add_from_response(
        self,
        response: str,
        created_by: str,
        mission_id: Optional[str] = None
    ) -> List[BacklogItem]:
        """Parse [BACKLOG] entries from agent response and add them."""
        added = []

        # Pattern to match backlog entries
        pattern = r'\[BACKLOG\]\s*type:\s*(\w+)\s*\n\s*Title:\s*(.+?)(?:\s*\|\s*Priority:\s*(\w+))?(?:\s*\|\s*Effort:\s*(\w+))?\s*\n\s*(?:Priority:\s*(\w+)\s*\n)?(?:Rationale:\s*)?(.+?)(?=\n\n|\n\[BACKLOG\]|$)'

        # Also try simpler format
        simple_pattern = r'\[BACKLOG\]\s*type:\s*(\w+)\s*\nTitle:\s*(.+)\nPriority:\s*(\w+)\nRationale:\s*(.+?)(?:\nEffort:\s*(\w+))?(?=\n\n|\n\[BACKLOG\]|$)'

        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
        for match in matches:
            groups = match.groups()
            item_type = groups[0]
            title = groups[1].strip()
            # Priority could be in position 2 or 4
            priority = (groups[2] or groups[4] or "medium").strip()
            effort = (groups[3] or "medium").strip()
            rationale = groups[5].strip() if groups[5] else ""

            if item_type and title:
                item = self.add_item(
                    item_type=item_type,
                    title=title,
                    priority=priority,
                    effort=effort,
                    rationale=rationale,
                    created_by=created_by,
                    mission_id=mission_id
                )
                added.append(item)

        # Try simpler format if no matches
        if not added:
            matches = re.finditer(simple_pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                groups = match.groups()
                item = self.add_item(
                    item_type=groups[0],
                    title=groups[1].strip(),
                    priority=groups[2].strip(),
                    effort=(groups[4] or "medium").strip(),
                    rationale=groups[3].strip(),
                    created_by=created_by,
                    mission_id=mission_id
                )
                added.append(item)

        return added

    def get_item(self, item_id: str) -> Optional[BacklogItem]:
        """Get a backlog item by ID."""
        return self.items.get(item_id)

    def approve_item(self, item_id: str) -> Optional[BacklogItem]:
        """Approve a backlog item for future sprint."""
        item = self.items.get(item_id)
        if item and item.status == BacklogStatus.PENDING.value:
            item.status = BacklogStatus.APPROVED.value
            self._save()
            return item
        return None

    def reject_item(self, item_id: str, reason: str) -> Optional[BacklogItem]:
        """Reject a backlog item with reason."""
        item = self.items.get(item_id)
        if item and item.status == BacklogStatus.PENDING.value:
            item.status = BacklogStatus.REJECTED.value
            item.rejection_reason = reason
            self._save()
            return item
        return None

    def start_item(self, item_id: str, sprint_id: str) -> Optional[BacklogItem]:
        """Mark item as in progress for a sprint."""
        item = self.items.get(item_id)
        if item and item.status == BacklogStatus.APPROVED.value:
            item.status = BacklogStatus.IN_PROGRESS.value
            item.sprint_id = sprint_id
            self._save()
            return item
        return None

    def complete_item(self, item_id: str) -> Optional[BacklogItem]:
        """Mark item as completed."""
        item = self.items.get(item_id)
        if item and item.status == BacklogStatus.IN_PROGRESS.value:
            item.status = BacklogStatus.COMPLETED.value
            item.completed_at = datetime.now().isoformat()
            self._save()
            return item
        return None

    def list_items(
        self,
        status: Optional[str] = None,
        item_type: Optional[str] = None,
        priority: Optional[str] = None
    ) -> List[BacklogItem]:
        """List backlog items with optional filters."""
        items = list(self.items.values())

        if status:
            items = [i for i in items if i.status == status]
        if item_type:
            items = [i for i in items if i.item_type == item_type]
        if priority:
            items = [i for i in items if i.priority == priority]

        # Sort by priority (high first), then by created_at
        priority_order = {"high": 0, "medium": 1, "low": 2}
        items.sort(key=lambda x: (priority_order.get(x.priority, 1), x.created_at))

        return items

    def get_pending_items(self) -> List[BacklogItem]:
        """Get all pending items for grooming."""
        return self.list_items(status=BacklogStatus.PENDING.value)

    def get_approved_items(self) -> List[BacklogItem]:
        """Get all approved items ready for sprint planning."""
        return self.list_items(status=BacklogStatus.APPROVED.value)

    def get_sprint_candidates(self) -> List[BacklogItem]:
        """Get approved items prioritized for sprint planning."""
        items = self.get_approved_items()
        # High priority first, then medium, then low
        return items

    def format_item_summary(self, item: BacklogItem) -> str:
        """Format a single item for display."""
        status_emoji = {
            "pending": "⏳",
            "approved": "✅",
            "rejected": "❌",
            "in_progress": "🔄",
            "completed": "✔️"
        }
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        type_emoji = {"bug": "🐛", "improvement": "💡", "idea": "💭", "tech_debt": "🔧"}

        emoji = status_emoji.get(item.status, "")
        prio = priority_emoji.get(item.priority, "")
        typ = type_emoji.get(item.item_type, "")

        return f"{emoji} **{item.id}** {typ} {prio} {item.title}"

    def format_backlog_embed(self, items: List[BacklogItem], title: str = "Backlog") -> dict:
        """Format backlog items for Discord embed."""
        if not items:
            return {
                "title": title,
                "description": "No items found.",
                "color": 0x808080
            }

        # Group by status
        grouped = {}
        for item in items:
            status = item.status
            if status not in grouped:
                grouped[status] = []
            grouped[status].append(item)

        fields = []
        for status, status_items in grouped.items():
            if status_items:
                value = "\n".join([
                    self.format_item_summary(i) for i in status_items[:10]
                ])
                if len(status_items) > 10:
                    value += f"\n... and {len(status_items) - 10} more"
                fields.append({
                    "name": f"{status.title()} ({len(status_items)})",
                    "value": value,
                    "inline": False
                })

        return {
            "title": title,
            "fields": fields,
            "color": 0x3498db,
            "footer": {"text": f"Total: {len(items)} items"}
        }

    def get_statistics(self) -> dict:
        """Get backlog statistics."""
        items = list(self.items.values())
        return {
            "total": len(items),
            "by_status": {
                status.value: len([i for i in items if i.status == status.value])
                for status in BacklogStatus
            },
            "by_type": {
                t.value: len([i for i in items if i.item_type == t.value])
                for t in BacklogItemType
            },
            "by_priority": {
                p.value: len([i for i in items if i.priority == p.value])
                for p in BacklogPriority
            }
        }


# Singleton instance
_backlog_manager: Optional[BacklogManager] = None


def get_backlog_manager() -> BacklogManager:
    """Get the singleton backlog manager instance."""
    global _backlog_manager
    if _backlog_manager is None:
        _backlog_manager = BacklogManager()
    return _backlog_manager
