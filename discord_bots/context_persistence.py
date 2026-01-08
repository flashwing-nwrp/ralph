"""
Context Persistence for RALPH Agent Ensemble

Provides session and context management for agents:
- Agent memory and context storage
- Session management across restarts
- Context sharing between agents
- Conversation history
- Knowledge accumulation

P2: Important for maintaining continuity.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import hashlib

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("context_persistence")


class ContextType(Enum):
    """Types of context that can be stored."""
    CONVERSATION = "conversation"      # Chat history
    WORKING_MEMORY = "working_memory"  # Current task context
    LONG_TERM = "long_term"           # Persistent knowledge
    SHARED = "shared"                  # Cross-agent shared context
    SESSION = "session"                # Session-specific data


class MemoryPriority(Enum):
    """Priority levels for memories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    memory_id: str
    agent: str
    context_type: ContextType
    priority: MemoryPriority

    # Content
    key: str
    value: Any
    summary: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    accessed_at: str = ""
    access_count: int = 0

    # Expiration
    expires_at: str = ""
    ttl_seconds: int = 0

    # Linking
    related_memories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "agent": self.agent,
            "context_type": self.context_type.value,
            "priority": self.priority.value,
            "key": self.key,
            "value": self.value,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "related_memories": self.related_memories,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            memory_id=data["memory_id"],
            agent=data["agent"],
            context_type=ContextType(data["context_type"]),
            priority=MemoryPriority(data["priority"]),
            key=data["key"],
            value=data["value"],
            summary=data.get("summary", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            accessed_at=data.get("accessed_at", ""),
            access_count=data.get("access_count", 0),
            expires_at=data.get("expires_at", ""),
            ttl_seconds=data.get("ttl_seconds", 0),
            related_memories=data.get("related_memories", []),
            tags=data.get("tags", [])
        )

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= datetime.fromisoformat(self.expires_at)


@dataclass
class ConversationMessage:
    """A message in a conversation."""
    message_id: str
    agent: str
    role: str  # "user", "agent", "system"
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSession:
    """Session state for an agent."""
    session_id: str
    agent: str
    started_at: str
    last_active: str
    status: str = "active"  # active, paused, ended

    # Session data
    current_task: str = ""
    current_context: Dict[str, Any] = field(default_factory=dict)

    # Conversation
    message_count: int = 0
    last_message_id: str = ""

    # Stats
    actions_taken: int = 0
    errors_occurred: int = 0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent": self.agent,
            "started_at": self.started_at,
            "last_active": self.last_active,
            "status": self.status,
            "current_task": self.current_task,
            "current_context": self.current_context,
            "message_count": self.message_count,
            "last_message_id": self.last_message_id,
            "actions_taken": self.actions_taken,
            "errors_occurred": self.errors_occurred
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSession":
        return cls(**data)


class ContextStore:
    """
    Central context and memory store for RALPH agents.

    Provides:
    - Memory storage and retrieval
    - Session management
    - Conversation history
    - Cross-agent context sharing
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.context_dir = self.project_dir / "context"
        self.context_dir.mkdir(exist_ok=True)

        self.memories_file = self.context_dir / "memories.json"
        self.sessions_file = self.context_dir / "sessions.json"
        self.conversations_file = self.context_dir / "conversations.json"

        # In-memory state
        self.memories: Dict[str, MemoryEntry] = {}
        self.sessions: Dict[str, AgentSession] = {}
        self.conversations: Dict[str, List[ConversationMessage]] = {}  # agent -> messages

        self.memory_counter = 0
        self.session_counter = 0
        self.message_counter = 0

        self._lock = asyncio.Lock()
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        try:
            if self.memories_file.exists():
                with open(self.memories_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memory_counter = data.get("counter", 0)
                    for mem_data in data.get("memories", []):
                        mem = MemoryEntry.from_dict(mem_data)
                        if not mem.is_expired():
                            self.memories[mem.memory_id] = mem
                    logger.info(f"Loaded {len(self.memories)} memories")

            if self.sessions_file.exists():
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.session_counter = data.get("counter", 0)
                    for sess_data in data.get("sessions", []):
                        sess = AgentSession.from_dict(sess_data)
                        self.sessions[sess.session_id] = sess

            if self.conversations_file.exists():
                with open(self.conversations_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.message_counter = data.get("counter", 0)
                    for agent, messages in data.get("conversations", {}).items():
                        self.conversations[agent] = [
                            ConversationMessage(**m) for m in messages[-100:]  # Keep last 100
                        ]

        except Exception as e:
            logger.error(f"Failed to load context state: {e}")

    async def _save_state(self):
        """Save state to files."""
        async with self._lock:
            try:
                # Save memories
                memories_data = {
                    "counter": self.memory_counter,
                    "memories": [m.to_dict() for m in self.memories.values()]
                }
                with open(self.memories_file, "w", encoding="utf-8") as f:
                    json.dump(memories_data, f, indent=2)

                # Save sessions
                sessions_data = {
                    "counter": self.session_counter,
                    "sessions": [s.to_dict() for s in self.sessions.values()]
                }
                with open(self.sessions_file, "w", encoding="utf-8") as f:
                    json.dump(sessions_data, f, indent=2)

                # Save conversations
                conv_data = {
                    "counter": self.message_counter,
                    "conversations": {
                        agent: [
                            {
                                "message_id": m.message_id,
                                "agent": m.agent,
                                "role": m.role,
                                "content": m.content,
                                "timestamp": m.timestamp,
                                "metadata": m.metadata
                            }
                            for m in messages[-100:]
                        ]
                        for agent, messages in self.conversations.items()
                    }
                }
                with open(self.conversations_file, "w", encoding="utf-8") as f:
                    json.dump(conv_data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save context state: {e}")

    # =========================================================================
    # MEMORY OPERATIONS
    # =========================================================================

    async def store_memory(
        self,
        agent: str,
        key: str,
        value: Any,
        context_type: ContextType = ContextType.WORKING_MEMORY,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        summary: str = "",
        ttl_seconds: int = 0,
        tags: List[str] = None,
        related_to: List[str] = None
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            agent: Agent storing the memory
            key: Memory key
            value: Memory value (JSON-serializable)
            context_type: Type of context
            priority: Priority level
            summary: Brief summary
            ttl_seconds: Time-to-live (0 = no expiry)
            tags: Tags for searching
            related_to: Related memory IDs

        Returns:
            Created memory entry
        """
        self.memory_counter += 1
        now = datetime.utcnow()

        memory = MemoryEntry(
            memory_id=f"MEM-{self.memory_counter:06d}",
            agent=agent,
            context_type=context_type,
            priority=priority,
            key=key,
            value=value,
            summary=summary,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            accessed_at=now.isoformat(),
            ttl_seconds=ttl_seconds,
            expires_at=(now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else "",
            tags=tags or [],
            related_memories=related_to or []
        )

        self.memories[memory.memory_id] = memory
        await self._save_state()

        logger.debug(f"Stored memory {memory.memory_id}: {key}")
        return memory

    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            if memory.is_expired():
                del self.memories[memory_id]
                return None
            memory.accessed_at = datetime.utcnow().isoformat()
            memory.access_count += 1
        return memory

    async def get_memory_by_key(self, agent: str, key: str) -> Optional[MemoryEntry]:
        """Get a memory by agent and key."""
        for memory in self.memories.values():
            if memory.agent == agent and memory.key == key:
                if memory.is_expired():
                    del self.memories[memory.memory_id]
                    return None
                memory.accessed_at = datetime.utcnow().isoformat()
                memory.access_count += 1
                return memory
        return None

    async def update_memory(
        self,
        memory_id: str,
        value: Any = None,
        summary: str = None,
        priority: MemoryPriority = None
    ) -> Optional[MemoryEntry]:
        """Update an existing memory."""
        memory = self.memories.get(memory_id)
        if not memory:
            return None

        if value is not None:
            memory.value = value
        if summary is not None:
            memory.summary = summary
        if priority is not None:
            memory.priority = priority

        memory.updated_at = datetime.utcnow().isoformat()
        await self._save_state()

        return memory

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            await self._save_state()
            return True
        return False

    def search_memories(
        self,
        agent: str = None,
        context_type: ContextType = None,
        priority: MemoryPriority = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """
        Search memories by criteria.

        Args:
            agent: Filter by agent
            context_type: Filter by context type
            priority: Filter by priority
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            Matching memories
        """
        results = []

        for memory in self.memories.values():
            if memory.is_expired():
                continue

            if agent and memory.agent != agent:
                continue
            if context_type and memory.context_type != context_type:
                continue
            if priority and memory.priority != priority:
                continue
            if tags and not any(t in memory.tags for t in tags):
                continue

            results.append(memory)

        # Sort by priority and recency
        priority_order = {
            MemoryPriority.CRITICAL: 0,
            MemoryPriority.HIGH: 1,
            MemoryPriority.MEDIUM: 2,
            MemoryPriority.LOW: 3
        }
        results.sort(key=lambda m: (priority_order[m.priority], m.updated_at), reverse=True)

        return results[:limit]

    def get_agent_context(self, agent: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get compiled context for an agent.

        Returns a dictionary with all relevant context organized by type.
        """
        context = {
            "working_memory": {},
            "long_term": {},
            "shared": {},
            "recent_conversation": []
        }

        # Get memories
        for memory in self.search_memories(agent=agent, limit=100):
            if memory.context_type == ContextType.WORKING_MEMORY:
                context["working_memory"][memory.key] = memory.value
            elif memory.context_type == ContextType.LONG_TERM:
                context["long_term"][memory.key] = memory.value
            elif memory.context_type == ContextType.SHARED:
                context["shared"][memory.key] = memory.value

        # Get recent conversation
        if agent in self.conversations:
            context["recent_conversation"] = [
                {"role": m.role, "content": m.content[:200]}
                for m in self.conversations[agent][-limit:]
            ]

        return context

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def start_session(self, agent: str) -> AgentSession:
        """Start a new session for an agent."""
        self.session_counter += 1
        now = datetime.utcnow().isoformat()

        session = AgentSession(
            session_id=f"SESS-{self.session_counter:05d}",
            agent=agent,
            started_at=now,
            last_active=now
        )

        self.sessions[session.session_id] = session
        await self._save_state()

        logger.info(f"Started session {session.session_id} for {agent}")
        return session

    async def get_active_session(self, agent: str) -> Optional[AgentSession]:
        """Get the active session for an agent."""
        for session in self.sessions.values():
            if session.agent == agent and session.status == "active":
                return session
        return None

    async def update_session(
        self,
        session_id: str,
        current_task: str = None,
        context_update: Dict[str, Any] = None
    ) -> Optional[AgentSession]:
        """Update session state."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.last_active = datetime.utcnow().isoformat()

        if current_task is not None:
            session.current_task = current_task

        if context_update:
            session.current_context.update(context_update)

        session.actions_taken += 1
        await self._save_state()

        return session

    async def end_session(self, session_id: str) -> Optional[AgentSession]:
        """End a session."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.status = "ended"
        session.last_active = datetime.utcnow().isoformat()
        await self._save_state()

        logger.info(f"Ended session {session_id}")
        return session

    async def pause_session(self, session_id: str) -> Optional[AgentSession]:
        """Pause a session (can be resumed)."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        session.status = "paused"
        session.last_active = datetime.utcnow().isoformat()
        await self._save_state()

        return session

    async def resume_session(self, session_id: str) -> Optional[AgentSession]:
        """Resume a paused session."""
        session = self.sessions.get(session_id)
        if not session or session.status != "paused":
            return None

        session.status = "active"
        session.last_active = datetime.utcnow().isoformat()
        await self._save_state()

        return session

    # =========================================================================
    # CONVERSATION HISTORY
    # =========================================================================

    async def add_message(
        self,
        agent: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> ConversationMessage:
        """Add a message to conversation history."""
        self.message_counter += 1

        message = ConversationMessage(
            message_id=f"MSG-{self.message_counter:07d}",
            agent=agent,
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )

        if agent not in self.conversations:
            self.conversations[agent] = []

        self.conversations[agent].append(message)

        # Trim conversation history
        if len(self.conversations[agent]) > 200:
            self.conversations[agent] = self.conversations[agent][-200:]

        # Update session if exists
        session = await self.get_active_session(agent)
        if session:
            session.message_count += 1
            session.last_message_id = message.message_id

        await self._save_state()
        return message

    def get_conversation_history(
        self,
        agent: str,
        limit: int = 50
    ) -> List[ConversationMessage]:
        """Get conversation history for an agent."""
        if agent not in self.conversations:
            return []
        return self.conversations[agent][-limit:]

    def clear_conversation(self, agent: str):
        """Clear conversation history for an agent."""
        if agent in self.conversations:
            self.conversations[agent] = []

    # =========================================================================
    # SHARED CONTEXT
    # =========================================================================

    async def share_context(
        self,
        from_agent: str,
        key: str,
        value: Any,
        summary: str = "",
        for_agents: List[str] = None
    ) -> MemoryEntry:
        """
        Share context between agents.

        Args:
            from_agent: Agent sharing the context
            key: Context key
            value: Context value
            summary: Brief summary
            for_agents: Specific agents to share with (None = all)

        Returns:
            Created shared memory
        """
        return await self.store_memory(
            agent=from_agent,
            key=key,
            value={
                "data": value,
                "shared_by": from_agent,
                "for_agents": for_agents
            },
            context_type=ContextType.SHARED,
            priority=MemoryPriority.HIGH,
            summary=summary,
            tags=["shared"] + (for_agents or [])
        )

    def get_shared_context(self, agent: str) -> Dict[str, Any]:
        """Get all shared context available to an agent."""
        shared = {}

        for memory in self.memories.values():
            if memory.context_type != ContextType.SHARED:
                continue
            if memory.is_expired():
                continue

            value = memory.value
            if isinstance(value, dict):
                for_agents = value.get("for_agents")
                if for_agents is None or agent in for_agents:
                    shared[memory.key] = {
                        "data": value.get("data"),
                        "shared_by": value.get("shared_by"),
                        "summary": memory.summary
                    }

        return shared

    # =========================================================================
    # CLEANUP
    # =========================================================================

    async def cleanup_expired(self) -> int:
        """Remove expired memories."""
        expired = [
            mid for mid, mem in self.memories.items()
            if mem.is_expired()
        ]

        for mid in expired:
            del self.memories[mid]

        if expired:
            await self._save_state()
            logger.info(f"Cleaned up {len(expired)} expired memories")

        return len(expired)

    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """End sessions older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        ended = 0

        for session in list(self.sessions.values()):
            if session.status == "active":
                last_active = datetime.fromisoformat(session.last_active)
                if last_active < cutoff:
                    session.status = "ended"
                    ended += 1

        if ended:
            await self._save_state()
            logger.info(f"Ended {ended} stale sessions")

        return ended

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_context_summary(self) -> str:
        """Get context store summary for Discord."""
        output = ["## 🧠 Context Store Summary\n"]

        # Memory stats
        output.append("**Memories:**")
        by_type: Dict[ContextType, int] = {}
        by_agent: Dict[str, int] = {}

        for memory in self.memories.values():
            by_type[memory.context_type] = by_type.get(memory.context_type, 0) + 1
            by_agent[memory.agent] = by_agent.get(memory.agent, 0) + 1

        for ct, count in sorted(by_type.items(), key=lambda x: x[0].value):
            output.append(f"  {ct.value}: {count}")

        # Session stats
        active_sessions = [s for s in self.sessions.values() if s.status == "active"]
        output.append(f"\n**Active Sessions:** {len(active_sessions)}")
        for session in active_sessions[:5]:
            output.append(f"  {session.agent}: {session.current_task or 'No task'}")

        # By agent
        output.append(f"\n**By Agent:**")
        for agent, count in sorted(by_agent.items()):
            conv_count = len(self.conversations.get(agent, []))
            output.append(f"  {agent}: {count} memories, {conv_count} messages")

        return "\n".join(output)

    def get_agent_memory_summary(self, agent: str) -> str:
        """Get memory summary for a specific agent."""
        memories = self.search_memories(agent=agent, limit=100)
        conversation = self.get_conversation_history(agent, limit=10)
        session = None
        for s in self.sessions.values():
            if s.agent == agent and s.status == "active":
                session = s
                break

        output = [f"## Memory Summary: {agent}\n"]

        if session:
            output.append(f"**Session:** {session.session_id}")
            output.append(f"  Task: {session.current_task or 'None'}")
            output.append(f"  Actions: {session.actions_taken}")
            output.append("")

        output.append(f"**Memories:** {len(memories)}")
        by_type: Dict[ContextType, List[MemoryEntry]] = {}
        for mem in memories:
            if mem.context_type not in by_type:
                by_type[mem.context_type] = []
            by_type[mem.context_type].append(mem)

        for ct, mems in sorted(by_type.items(), key=lambda x: x[0].value):
            output.append(f"\n  **{ct.value}:**")
            for mem in mems[:5]:
                output.append(f"    • {mem.key}: {mem.summary or str(mem.value)[:30]}...")

        if conversation:
            output.append(f"\n**Recent Messages:** {len(conversation)}")
            for msg in conversation[-3:]:
                content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                output.append(f"  [{msg.role}] {content}")

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_context_store: Optional[ContextStore] = None


def get_context_store() -> ContextStore:
    """Get or create the context store instance."""
    global _context_store
    if _context_store is None:
        _context_store = ContextStore()
    return _context_store


def set_context_store(store: ContextStore):
    """Set the context store instance."""
    global _context_store
    _context_store = store
