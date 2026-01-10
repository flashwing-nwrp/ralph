"""
OpenProject Integration Service for RALPH Agent Ensemble

Provides bi-directional sync between RALPH missions/tasks and OpenProject:
- Create/update work packages from mission tasks
- Sync task status changes
- Manage sprints and backlogs
- Allow agents to organize boards freely

Uses OpenProject API v3 with HAL+JSON responses.

Configuration:
    OPENPROJECT_URL=http://46.252.192.140:8080
    OPENPROJECT_API_KEY=<your-api-key>
    OPENPROJECT_PROJECT_ID=ralph-agents (or numeric ID)
"""

import asyncio
import aiohttp
import json
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkPackageType(Enum):
    """OpenProject work package types."""
    TASK = "Task"
    USER_STORY = "User story"
    BUG = "Bug"
    FEATURE = "Feature"
    EPIC = "Epic"
    MILESTONE = "Milestone"


class WorkPackageStatus(Enum):
    """Standard OpenProject statuses (IDs may vary by installation)."""
    NEW = "New"
    IN_PROGRESS = "In progress"
    CLOSED = "Closed"
    ON_HOLD = "On hold"
    REJECTED = "Rejected"


@dataclass
class OpenProjectWorkPackage:
    """Represents an OpenProject work package."""
    id: Optional[int] = None
    subject: str = ""
    description: str = ""
    type_name: str = "Task"
    status_name: str = "New"
    priority_name: str = "Normal"
    assignee: Optional[str] = None
    project_id: Optional[int] = None
    parent_id: Optional[int] = None

    # RALPH-specific metadata stored in custom fields or description
    ralph_task_id: Optional[str] = None
    ralph_mission_id: Optional[str] = None
    ralph_agent: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Links for HAL navigation
    _links: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenProjectProject:
    """Represents an OpenProject project."""
    id: Optional[int] = None
    identifier: str = ""
    name: str = ""
    description: str = ""
    public: bool = False
    active: bool = True


class OpenProjectService:
    """
    Service for interacting with OpenProject API.

    Handles:
    - Project management
    - Work package CRUD
    - Status synchronization
    - Sprint/version management
    - Agent user attribution
    """

    # Agent user IDs in OpenProject (created via API)
    AGENT_USER_IDS = {
        "strategy": 5,
        "tuning": 6,
        "backtest": 7,
        "risk": 8,
        "data": 9,
    }

    def __init__(self):
        self.base_url = os.getenv("OPENPROJECT_URL", "http://46.252.192.140:8080")
        self.api_key = os.getenv("OPENPROJECT_API_KEY", "")
        self.project_identifier = os.getenv("OPENPROJECT_PROJECT_ID", "ralph-agents")

        # Per-agent API keys (optional - falls back to main key)
        self.agent_api_keys = {
            "strategy": os.getenv("OPENPROJECT_STRATEGY_KEY", ""),
            "tuning": os.getenv("OPENPROJECT_TUNING_KEY", ""),
            "backtest": os.getenv("OPENPROJECT_BACKTEST_KEY", ""),
            "risk": os.getenv("OPENPROJECT_RISK_KEY", ""),
            "data": os.getenv("OPENPROJECT_DATA_KEY", ""),
        }

        # Remove trailing slash
        self.base_url = self.base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v3"

        # Cache for lookups
        self._type_cache: Dict[str, int] = {}
        self._status_cache: Dict[str, int] = {}
        self._priority_cache: Dict[str, int] = {}
        self._project_cache: Dict[str, int] = {}
        self._user_cache: Dict[str, int] = self.AGENT_USER_IDS.copy()

        # Mapping file for RALPH task ID -> OpenProject WP ID
        self.mapping_file = Path(__file__).parent / "data" / "openproject_mapping.json"
        self._load_mappings()

    def _load_mappings(self):
        """Load task ID to work package ID mappings."""
        self.task_to_wp: Dict[str, int] = {}
        self.wp_to_task: Dict[int, str] = {}

        if self.mapping_file.exists():
            try:
                data = json.loads(self.mapping_file.read_text())
                self.task_to_wp = data.get("task_to_wp", {})
                self.wp_to_task = {int(v): k for k, v in self.task_to_wp.items()}
            except Exception as e:
                logger.warning(f"Failed to load OpenProject mappings: {e}")

    def _save_mappings(self):
        """Save task ID to work package ID mappings."""
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "task_to_wp": self.task_to_wp,
                "updated_at": datetime.now().isoformat()
            }
            self.mapping_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save OpenProject mappings: {e}")

    def _get_auth(self) -> aiohttp.BasicAuth:
        """Get Basic Auth for API requests."""
        return aiohttp.BasicAuth("apikey", self.api_key)

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/hal+json"
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make an API request to OpenProject."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    auth=self._get_auth(),
                    headers=self._get_headers(),
                    json=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status in (200, 201):
                        return await response.json()
                    elif response.status == 204:
                        return {"success": True}
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenProject API error {response.status}: {error_text[:500]}")
                        return None
        except Exception as e:
            logger.error(f"OpenProject request failed: {e}")
            return None

    # =========================================================================
    # PROJECT MANAGEMENT
    # =========================================================================

    async def get_project(self, identifier: str = None) -> Optional[OpenProjectProject]:
        """Get a project by identifier."""
        identifier = identifier or self.project_identifier
        result = await self._request("GET", f"/projects/{identifier}")

        if result:
            return OpenProjectProject(
                id=result.get("id"),
                identifier=result.get("identifier", ""),
                name=result.get("name", ""),
                description=result.get("description", {}).get("raw", ""),
                public=result.get("public", False),
                active=result.get("active", True)
            )
        return None

    async def create_project(
        self,
        name: str,
        identifier: str,
        description: str = ""
    ) -> Optional[OpenProjectProject]:
        """Create a new project."""
        data = {
            "name": name,
            "identifier": identifier,
            "description": {"raw": description},
            "public": False,
            "active": True
        }

        result = await self._request("POST", "/projects", data)

        if result:
            return OpenProjectProject(
                id=result.get("id"),
                identifier=result.get("identifier", identifier),
                name=result.get("name", name),
                description=description,
                public=False,
                active=True
            )
        return None

    async def ensure_project_exists(self) -> Optional[int]:
        """Ensure the RALPH project exists, create if not."""
        project = await self.get_project()

        if project:
            self._project_cache[self.project_identifier] = project.id
            return project.id

        # Create the project
        project = await self.create_project(
            name="RALPH Agent Ensemble",
            identifier=self.project_identifier,
            description="Autonomous AI agent collaboration for trading bot development"
        )

        if project:
            self._project_cache[self.project_identifier] = project.id
            return project.id

        return None

    # =========================================================================
    # TYPE/STATUS/PRIORITY LOOKUPS
    # =========================================================================

    async def _populate_caches(self, project_id: int):
        """Populate type, status, and priority caches from project schema."""
        # Get available types
        types_result = await self._request("GET", f"/projects/{project_id}/types")
        if types_result and "_embedded" in types_result:
            for t in types_result["_embedded"].get("elements", []):
                self._type_cache[t["name"]] = t["id"]

        # Get available statuses
        statuses_result = await self._request("GET", "/statuses")
        if statuses_result and "_embedded" in statuses_result:
            for s in statuses_result["_embedded"].get("elements", []):
                self._status_cache[s["name"]] = s["id"]

        # Get available priorities
        priorities_result = await self._request("GET", "/priorities")
        if priorities_result and "_embedded" in priorities_result:
            for p in priorities_result["_embedded"].get("elements", []):
                self._priority_cache[p["name"]] = p["id"]

    async def get_type_id(self, type_name: str, project_id: int) -> Optional[int]:
        """Get type ID by name."""
        if not self._type_cache:
            await self._populate_caches(project_id)
        return self._type_cache.get(type_name)

    async def get_status_id(self, status_name: str, project_id: int) -> Optional[int]:
        """Get status ID by name."""
        if not self._status_cache:
            await self._populate_caches(project_id)
        return self._status_cache.get(status_name)

    async def get_priority_id(self, priority_name: str, project_id: int) -> Optional[int]:
        """Get priority ID by name."""
        if not self._priority_cache:
            await self._populate_caches(project_id)
        return self._priority_cache.get(priority_name)

    # =========================================================================
    # WORK PACKAGE MANAGEMENT
    # =========================================================================

    async def create_work_package(
        self,
        subject: str,
        description: str = "",
        type_name: str = "Task",
        status_name: str = "New",
        priority_name: str = "Normal",
        ralph_task_id: str = None,
        ralph_mission_id: str = None,
        ralph_agent: str = None,
        parent_id: int = None
    ) -> Optional[OpenProjectWorkPackage]:
        """Create a new work package."""
        project_id = await self.ensure_project_exists()
        if not project_id:
            logger.error("Cannot create work package: project not available")
            return None

        # Build description with RALPH metadata
        full_description = description
        if ralph_task_id or ralph_mission_id or ralph_agent:
            metadata = []
            if ralph_task_id:
                metadata.append(f"**RALPH Task ID:** {ralph_task_id}")
            if ralph_mission_id:
                metadata.append(f"**Mission:** {ralph_mission_id}")
            if ralph_agent:
                metadata.append(f"**Agent:** {ralph_agent}")
            full_description = "\n".join(metadata) + "\n\n---\n\n" + description

        # Get IDs for type, status, priority
        type_id = await self.get_type_id(type_name, project_id)
        status_id = await self.get_status_id(status_name, project_id)
        priority_id = await self.get_priority_id(priority_name, project_id)

        # Build request data
        data = {
            "subject": subject[:255],  # OpenProject limit
            "description": {"raw": full_description},
            "_links": {
                "project": {"href": f"/api/v3/projects/{project_id}"}
            }
        }

        if type_id:
            data["_links"]["type"] = {"href": f"/api/v3/types/{type_id}"}
        if status_id:
            data["_links"]["status"] = {"href": f"/api/v3/statuses/{status_id}"}
        if priority_id:
            data["_links"]["priority"] = {"href": f"/api/v3/priorities/{priority_id}"}
        if parent_id:
            data["_links"]["parent"] = {"href": f"/api/v3/work_packages/{parent_id}"}

        # Assign to agent user if specified
        if ralph_agent and ralph_agent.lower() in self._user_cache:
            agent_user_id = self._user_cache[ralph_agent.lower()]
            data["_links"]["assignee"] = {"href": f"/api/v3/users/{agent_user_id}"}
            data["_links"]["responsible"] = {"href": f"/api/v3/users/{agent_user_id}"}

        result = await self._request("POST", "/work_packages", data)

        if result:
            wp = OpenProjectWorkPackage(
                id=result.get("id"),
                subject=result.get("subject", subject),
                description=description,
                type_name=type_name,
                status_name=status_name,
                priority_name=priority_name,
                project_id=project_id,
                ralph_task_id=ralph_task_id,
                ralph_mission_id=ralph_mission_id,
                ralph_agent=ralph_agent,
                created_at=result.get("createdAt"),
                updated_at=result.get("updatedAt")
            )

            # Save mapping
            if ralph_task_id and wp.id:
                self.task_to_wp[ralph_task_id] = wp.id
                self.wp_to_task[wp.id] = ralph_task_id
                self._save_mappings()

            return wp

        return None

    async def get_work_package(self, wp_id: int) -> Optional[OpenProjectWorkPackage]:
        """Get a work package by ID."""
        result = await self._request("GET", f"/work_packages/{wp_id}")

        if result:
            # Extract type, status, priority names from _links
            type_name = result.get("_links", {}).get("type", {}).get("title", "Task")
            status_name = result.get("_links", {}).get("status", {}).get("title", "New")
            priority_name = result.get("_links", {}).get("priority", {}).get("title", "Normal")

            return OpenProjectWorkPackage(
                id=result.get("id"),
                subject=result.get("subject", ""),
                description=result.get("description", {}).get("raw", ""),
                type_name=type_name,
                status_name=status_name,
                priority_name=priority_name,
                project_id=result.get("_links", {}).get("project", {}).get("href", "").split("/")[-1],
                created_at=result.get("createdAt"),
                updated_at=result.get("updatedAt"),
                _links=result.get("_links", {})
            )
        return None

    async def update_work_package(
        self,
        wp_id: int,
        subject: str = None,
        description: str = None,
        status_name: str = None,
        priority_name: str = None
    ) -> Optional[OpenProjectWorkPackage]:
        """Update an existing work package."""
        # First get current state for lock version
        current = await self._request("GET", f"/work_packages/{wp_id}")
        if not current:
            return None

        lock_version = current.get("lockVersion", 0)
        project_id = current.get("_links", {}).get("project", {}).get("href", "").split("/")[-1]

        data = {"lockVersion": lock_version}

        if subject:
            data["subject"] = subject[:255]
        if description:
            data["description"] = {"raw": description}

        if status_name or priority_name:
            data["_links"] = {}
            if status_name:
                status_id = await self.get_status_id(status_name, int(project_id))
                if status_id:
                    data["_links"]["status"] = {"href": f"/api/v3/statuses/{status_id}"}
            if priority_name:
                priority_id = await self.get_priority_id(priority_name, int(project_id))
                if priority_id:
                    data["_links"]["priority"] = {"href": f"/api/v3/priorities/{priority_id}"}

        result = await self._request("PATCH", f"/work_packages/{wp_id}", data)

        if result:
            return await self.get_work_package(wp_id)
        return None

    async def list_work_packages(
        self,
        status_filter: str = None,
        type_filter: str = None,
        limit: int = 1000
    ) -> List[OpenProjectWorkPackage]:
        """List work packages with optional filters."""
        project_id = await self.ensure_project_exists()
        if not project_id:
            return []

        # Build filter query
        filters = [{"project": {"operator": "=", "values": [str(project_id)]}}]

        if status_filter:
            status_id = await self.get_status_id(status_filter, project_id)
            if status_id:
                filters.append({"status": {"operator": "=", "values": [str(status_id)]}})

        if type_filter:
            type_id = await self.get_type_id(type_filter, project_id)
            if type_id:
                filters.append({"type": {"operator": "=", "values": [str(type_id)]}})

        params = {
            "filters": json.dumps(filters),
            "pageSize": limit
        }

        result = await self._request("GET", "/work_packages", params=params)

        packages = []
        if result and "_embedded" in result:
            for wp in result["_embedded"].get("elements", []):
                packages.append(OpenProjectWorkPackage(
                    id=wp.get("id"),
                    subject=wp.get("subject", ""),
                    description=wp.get("description", {}).get("raw", ""),
                    type_name=wp.get("_links", {}).get("type", {}).get("title", "Task"),
                    status_name=wp.get("_links", {}).get("status", {}).get("title", "New"),
                    priority_name=wp.get("_links", {}).get("priority", {}).get("title", "Normal"),
                    created_at=wp.get("createdAt"),
                    updated_at=wp.get("updatedAt")
                ))

        return packages

    async def add_comment(
        self,
        wp_id: int,
        comment: str,
        agent: str = None
    ) -> bool:
        """
        Add a comment to a work package.

        Args:
            wp_id: Work package ID
            comment: Comment text
            agent: Optional agent name for attribution in comment

        Returns:
            True if comment added successfully
        """
        # Add agent attribution to comment if specified
        if agent:
            agent_title = agent.title()
            comment = f"**{agent_title} Agent:**\n\n{comment}"

        data = {
            "comment": {"raw": comment}
        }

        result = await self._request("POST", f"/work_packages/{wp_id}/activities", data)
        return result is not None

    async def get_agent_api_key(self, agent: str) -> str:
        """Get API key for specific agent, fallback to main key."""
        agent_key = self.agent_api_keys.get(agent.lower(), "")
        return agent_key if agent_key else self.api_key

    # =========================================================================
    # RALPH INTEGRATION
    # =========================================================================

    async def sync_task_to_openproject(
        self,
        task_id: str,
        description: str,
        agent: str,
        mission_id: str,
        status: str = "pending",
        priority: str = "medium"
    ) -> Optional[OpenProjectWorkPackage]:
        """
        Sync a RALPH task to OpenProject.
        Creates new WP if doesn't exist, updates if it does.
        """
        # Map RALPH status to OpenProject status
        status_map = {
            "pending": "New",
            "in_progress": "In progress",
            "completed": "Closed",
            "failed": "Rejected"
        }
        op_status = status_map.get(status, "New")

        # Map priority
        priority_map = {
            "high": "High",
            "medium": "Normal",
            "low": "Low"
        }
        op_priority = priority_map.get(priority, "Normal")

        # Check if already synced
        if task_id in self.task_to_wp:
            wp_id = self.task_to_wp[task_id]
            return await self.update_work_package(
                wp_id=wp_id,
                status_name=op_status,
                priority_name=op_priority
            )

        # Create new work package
        return await self.create_work_package(
            subject=f"[{task_id}] {description[:200]}",
            description=description,
            type_name="Task",
            status_name=op_status,
            priority_name=op_priority,
            ralph_task_id=task_id,
            ralph_mission_id=mission_id,
            ralph_agent=agent
        )

    async def sync_mission_to_openproject(
        self,
        mission_id: str,
        objective: str,
        tasks: List[Dict]
    ) -> Optional[OpenProjectWorkPackage]:
        """
        Sync an entire mission to OpenProject.
        Creates an Epic for the mission with child tasks.
        """
        # Create Epic for mission
        epic = await self.create_work_package(
            subject=f"[{mission_id}] {objective[:180]}",
            description=f"**Mission Objective:**\n\n{objective}",
            type_name="Epic",
            status_name="New",
            ralph_mission_id=mission_id
        )

        if not epic:
            return None

        # Create tasks as children
        for task in tasks:
            await self.create_work_package(
                subject=f"[{task.get('task_id', '')}] {task.get('description', '')[:180]}",
                description=task.get('description', ''),
                type_name="Task",
                status_name="New",
                ralph_task_id=task.get('task_id'),
                ralph_mission_id=mission_id,
                ralph_agent=task.get('assigned_to'),
                parent_id=epic.id
            )

        return epic

    async def sync_backlog_item_to_openproject(
        self,
        item_id: str,
        title: str,
        item_type: str,
        rationale: str,
        priority: str = "medium",
        effort: str = "medium"
    ) -> Optional[OpenProjectWorkPackage]:
        """Sync a backlog item to OpenProject."""
        # Map backlog item type to OpenProject type
        type_map = {
            "bug": "Bug",
            "improvement": "Feature",
            "idea": "User story",
            "tech_debt": "Task"
        }
        op_type = type_map.get(item_type, "Task")

        # Map priority
        priority_map = {
            "high": "High",
            "medium": "Normal",
            "low": "Low"
        }
        op_priority = priority_map.get(priority, "Normal")

        # Check if already synced
        backlog_key = f"BL:{item_id}"
        if backlog_key in self.task_to_wp:
            return await self.get_work_package(self.task_to_wp[backlog_key])

        # Build description
        description = f"**{title}**\n\n{rationale}\n\n---\n**Effort:** {effort}\n**Source:** RALPH Backlog"

        wp = await self.create_work_package(
            subject=f"[{item_id}] {title[:200]}",
            description=description,
            type_name=op_type,
            status_name="New",
            priority_name=op_priority,
            ralph_task_id=backlog_key
        )

        return wp

    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status in OpenProject when RALPH status changes."""
        if task_id not in self.task_to_wp:
            return False

        status_map = {
            "pending": "New",
            "in_progress": "In progress",
            "completed": "Closed",
            "failed": "Rejected"
        }
        op_status = status_map.get(status, "New")

        result = await self.update_work_package(
            wp_id=self.task_to_wp[task_id],
            status_name=op_status
        )

        return result is not None

    # =========================================================================
    # BOARD/ORGANIZATION
    # =========================================================================

    async def get_board_summary(self) -> Dict[str, Any]:
        """Get a summary of work packages organized by status (Kanban-style)."""
        packages = await self.list_work_packages(limit=1000)

        board = {
            "New": [],
            "In progress": [],
            "Closed": [],
            "On hold": [],
            "Rejected": []
        }

        for wp in packages:
            status = wp.status_name
            if status not in board:
                board[status] = []
            board[status].append({
                "id": wp.id,
                "subject": wp.subject,
                "type": wp.type_name,
                "priority": wp.priority_name
            })

        return {
            "columns": board,
            "total": len(packages),
            "url": f"{self.base_url}/projects/{self.project_identifier}/work_packages"
        }

    async def test_connection(self) -> Dict[str, Any]:
        """Test the OpenProject connection and return status."""
        try:
            # Test basic connectivity
            result = await self._request("GET", "/")
            if not result:
                return {
                    "success": False,
                    "error": "Failed to connect to OpenProject API",
                    "url": self.base_url
                }

            # Check project
            project = await self.get_project()

            return {
                "success": True,
                "url": self.base_url,
                "api_version": result.get("_type", "unknown"),
                "instance_name": result.get("instanceName", "unknown"),
                "project_exists": project is not None,
                "project_name": project.name if project else None,
                "project_url": f"{self.base_url}/projects/{self.project_identifier}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": self.base_url
            }


# Singleton instance
_openproject_service: Optional[OpenProjectService] = None


def get_openproject_service() -> OpenProjectService:
    """Get the singleton OpenProject service instance."""
    global _openproject_service
    if _openproject_service is None:
        _openproject_service = OpenProjectService()
    return _openproject_service
