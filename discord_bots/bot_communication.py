"""
Inter-Bot Communication System for RALPH Agent Ensemble

Enables bots to communicate with each other via Discord @mentions.
Bots can:
- Register themselves on startup
- Recognize @mentions from other RALPH bots
- Parse structured messages (tasks, responses, alerts)
- Track conversation threads between bots

Message Format:
  @TargetBot [ACTION] message content

Actions:
  [TASK] - Request the target bot to do something
  [RESPONSE] - Reply to a previous task
  [HANDOFF] - Transfer ownership of work
  [ALERT] - Urgent notification
  [INFO] - Informational message
  [QUESTION] - Ask for input/decision
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("bot_communication")


class MessageType(Enum):
    """Types of inter-bot messages."""
    TASK = "TASK"           # Request to do something
    RESPONSE = "RESPONSE"   # Reply to a task
    HANDOFF = "HANDOFF"     # Transfer work ownership
    ALERT = "ALERT"         # Urgent notification
    INFO = "INFO"           # Informational
    QUESTION = "QUESTION"   # Request for input
    ACK = "ACK"             # Acknowledgment


@dataclass
class BotInfo:
    """Information about a registered bot."""
    agent_type: str         # tuning, backtest, risk, strategy, data
    agent_name: str         # Full name like "Tuning Agent"
    bot_user_id: int        # Discord user ID
    bot_username: str       # Discord username
    primary_channel: str    # Primary channel name
    registered_at: str = ""
    is_online: bool = False


@dataclass
class InterBotMessage:
    """A structured message between bots."""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: str

    # Tracking
    timestamp: str = ""
    discord_message_id: int = 0
    channel_id: int = 0

    # For responses
    in_reply_to: str = ""  # Original message_id being responded to

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "discord_message_id": self.discord_message_id,
            "channel_id": self.channel_id,
            "in_reply_to": self.in_reply_to,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InterBotMessage":
        return cls(
            message_id=data["message_id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=data.get("timestamp", ""),
            discord_message_id=data.get("discord_message_id", 0),
            channel_id=data.get("channel_id", 0),
            in_reply_to=data.get("in_reply_to", ""),
            context=data.get("context", {})
        )


class BotRegistry:
    """
    Registry of all RALPH bots for inter-bot communication.

    Bots register themselves on startup, allowing other bots
    to identify and communicate with them.
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.registry_file = self.project_dir / "bot_registry.json"

        # Registered bots
        self.bots: Dict[str, BotInfo] = {}  # agent_type -> BotInfo
        self.bots_by_id: Dict[int, BotInfo] = {}  # Discord user ID -> BotInfo

        # Message tracking
        self.messages: Dict[str, InterBotMessage] = {}  # message_id -> message
        self.pending_tasks: Dict[str, InterBotMessage] = {}  # Tasks awaiting response
        self.message_counter = 0

        self._lock = asyncio.Lock()
        self._load_registry()

    def _load_registry(self):
        """Load registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.message_counter = data.get("message_counter", 0)

                    for bot_data in data.get("bots", []):
                        bot = BotInfo(
                            agent_type=bot_data["agent_type"],
                            agent_name=bot_data["agent_name"],
                            bot_user_id=bot_data["bot_user_id"],
                            bot_username=bot_data["bot_username"],
                            primary_channel=bot_data["primary_channel"],
                            registered_at=bot_data.get("registered_at", ""),
                            is_online=False  # Will be set when bot comes online
                        )
                        self.bots[bot.agent_type] = bot
                        self.bots_by_id[bot.bot_user_id] = bot

                    logger.info(f"Loaded {len(self.bots)} bots from registry")

        except Exception as e:
            logger.error(f"Failed to load bot registry: {e}")

    async def _save_registry(self):
        """Save registry to file."""
        async with self._lock:
            try:
                data = {
                    "message_counter": self.message_counter,
                    "bots": [
                        {
                            "agent_type": bot.agent_type,
                            "agent_name": bot.agent_name,
                            "bot_user_id": bot.bot_user_id,
                            "bot_username": bot.bot_username,
                            "primary_channel": bot.primary_channel,
                            "registered_at": bot.registered_at,
                        }
                        for bot in self.bots.values()
                    ]
                }

                with open(self.registry_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save bot registry: {e}")

    async def register_bot(
        self,
        agent_type: str,
        agent_name: str,
        bot_user_id: int,
        bot_username: str,
        primary_channel: str
    ) -> BotInfo:
        """
        Register a bot in the registry.

        Called by each bot on startup.
        """
        bot = BotInfo(
            agent_type=agent_type,
            agent_name=agent_name,
            bot_user_id=bot_user_id,
            bot_username=bot_username,
            primary_channel=primary_channel,
            registered_at=datetime.utcnow().isoformat(),
            is_online=True
        )

        self.bots[agent_type] = bot
        self.bots_by_id[bot_user_id] = bot

        await self._save_registry()
        logger.info(f"Registered bot: {agent_name} ({agent_type}) - ID: {bot_user_id}")

        return bot

    def set_online(self, agent_type: str, is_online: bool):
        """Update bot online status."""
        if agent_type in self.bots:
            self.bots[agent_type].is_online = is_online

    def get_bot(self, agent_type: str) -> Optional[BotInfo]:
        """Get bot info by agent type."""
        return self.bots.get(agent_type)

    def get_bot_by_id(self, user_id: int) -> Optional[BotInfo]:
        """Get bot info by Discord user ID."""
        return self.bots_by_id.get(user_id)

    def is_ralph_bot(self, user_id: int) -> bool:
        """Check if a user ID belongs to a RALPH bot."""
        return user_id in self.bots_by_id

    def get_all_bot_ids(self) -> List[int]:
        """Get all registered bot user IDs."""
        return list(self.bots_by_id.keys())

    def get_online_bots(self) -> List[BotInfo]:
        """Get all currently online bots."""
        return [bot for bot in self.bots.values() if bot.is_online]

    def generate_message_id(self) -> str:
        """Generate unique message ID."""
        self.message_counter += 1
        return f"MSG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self.message_counter:05d}"

    def track_message(self, message: InterBotMessage):
        """Track an inter-bot message."""
        self.messages[message.message_id] = message

        # If it's a task, add to pending
        if message.message_type == MessageType.TASK:
            self.pending_tasks[message.message_id] = message

    def complete_task(self, task_message_id: str, response: InterBotMessage):
        """Mark a task as complete with its response."""
        if task_message_id in self.pending_tasks:
            del self.pending_tasks[task_message_id]
        self.messages[response.message_id] = response

    def get_pending_tasks_for(self, agent_type: str) -> List[InterBotMessage]:
        """Get pending tasks assigned to an agent."""
        return [
            msg for msg in self.pending_tasks.values()
            if msg.to_agent == agent_type
        ]


class MessageParser:
    """
    Parses inter-bot messages from Discord message content.

    Format: [ACTION] content
    or just: content (defaults to INFO)
    """

    # Pattern to match [ACTION] at the start of a message
    ACTION_PATTERN = re.compile(r'^\[([A-Z]+)\]\s*(.*)$', re.DOTALL)

    @classmethod
    def parse_content(cls, content: str) -> tuple[MessageType, str]:
        """
        Parse message content to extract action type and actual content.

        Returns:
            (MessageType, content)
        """
        match = cls.ACTION_PATTERN.match(content.strip())

        if match:
            action_str = match.group(1)
            actual_content = match.group(2).strip()

            try:
                message_type = MessageType(action_str)
            except ValueError:
                # Unknown action, treat as INFO
                message_type = MessageType.INFO
                actual_content = content

            return message_type, actual_content

        # No action specified, default based on content
        content_lower = content.lower()
        if content_lower.startswith("please ") or "can you" in content_lower:
            return MessageType.TASK, content
        elif "?" in content:
            return MessageType.QUESTION, content
        else:
            return MessageType.INFO, content

    @classmethod
    def format_message(cls, message_type: MessageType, content: str) -> str:
        """Format a message with action prefix."""
        return f"[{message_type.value}] {content}"

    @classmethod
    def format_task(cls, task: str, context: str = "") -> str:
        """Format a task message."""
        msg = f"[TASK] {task}"
        if context:
            msg += f"\n\n**Context:**\n{context}"
        return msg

    @classmethod
    def format_response(cls, result: str, original_task_id: str = "") -> str:
        """Format a response message."""
        msg = f"[RESPONSE] {result}"
        if original_task_id:
            msg = f"[RESPONSE to {original_task_id}] {result}"
        return msg

    @classmethod
    def format_handoff(cls, task: str, reason: str = "") -> str:
        """Format a handoff message."""
        msg = f"[HANDOFF] {task}"
        if reason:
            msg += f"\n**Reason:** {reason}"
        return msg

    @classmethod
    def format_alert(cls, alert: str, severity: str = "warning") -> str:
        """Format an alert message."""
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }
        emoji = severity_emoji.get(severity, "⚠️")
        return f"[ALERT] {emoji} {alert}"


class BotCommunicator:
    """
    High-level interface for inter-bot communication.

    Used by agents to send messages to other agents.
    """

    def __init__(self, registry: BotRegistry, bot_instance):
        """
        Args:
            registry: The shared bot registry
            bot_instance: The Discord bot instance (for sending messages)
        """
        self.registry = registry
        self.bot = bot_instance
        self.agent_type: str = ""  # Set by the agent
        self.guild_id: int = 0  # Set by the agent

        # Callbacks for handling incoming messages
        self._task_handlers: List[Callable] = []
        self._response_handlers: List[Callable] = []

    def set_agent_info(self, agent_type: str, guild_id: int):
        """Set the agent info for this communicator."""
        self.agent_type = agent_type
        self.guild_id = guild_id

    def register_task_handler(self, handler: Callable):
        """Register a handler for incoming tasks."""
        self._task_handlers.append(handler)

    def register_response_handler(self, handler: Callable):
        """Register a handler for incoming responses."""
        self._response_handlers.append(handler)

    async def send_task(
        self,
        to_agent: str,
        task: str,
        context: str = "",
        channel_name: str = "ralph-team"
    ) -> Optional[InterBotMessage]:
        """
        Send a task to another agent via @mention.

        Args:
            to_agent: Target agent type (tuning, backtest, risk, strategy, data)
            task: Task description
            context: Additional context
            channel_name: Channel to post in

        Returns:
            The sent message, or None if failed
        """
        target_bot = self.registry.get_bot(to_agent)
        if not target_bot:
            logger.warning(f"Target bot not found: {to_agent}")
            return None

        # Get the channel
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return None

        channel = None
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

        if not channel:
            logger.warning(f"Channel not found: {channel_name}")
            return None

        # Create message
        message_id = self.registry.generate_message_id()
        content = MessageParser.format_task(task, context)

        # Send with @mention
        discord_content = f"<@{target_bot.bot_user_id}> {content}"

        try:
            discord_msg = await channel.send(discord_content)

            inter_msg = InterBotMessage(
                message_id=message_id,
                from_agent=self.agent_type,
                to_agent=to_agent,
                message_type=MessageType.TASK,
                content=task,
                timestamp=datetime.utcnow().isoformat(),
                discord_message_id=discord_msg.id,
                channel_id=channel.id,
                context={"full_context": context}
            )

            self.registry.track_message(inter_msg)
            logger.info(f"Sent task {message_id} to {to_agent}")

            return inter_msg

        except Exception as e:
            logger.error(f"Failed to send task: {e}")
            return None

    async def send_response(
        self,
        to_agent: str,
        result: str,
        in_reply_to: str = "",
        channel_name: str = "ralph-team"
    ) -> Optional[InterBotMessage]:
        """
        Send a response to another agent.

        Args:
            to_agent: Target agent type
            result: Response content
            in_reply_to: Message ID being responded to
            channel_name: Channel to post in

        Returns:
            The sent message, or None if failed
        """
        target_bot = self.registry.get_bot(to_agent)
        if not target_bot:
            return None

        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return None

        channel = None
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

        if not channel:
            return None

        message_id = self.registry.generate_message_id()
        content = MessageParser.format_response(result, in_reply_to)

        discord_content = f"<@{target_bot.bot_user_id}> {content}"

        try:
            discord_msg = await channel.send(discord_content)

            inter_msg = InterBotMessage(
                message_id=message_id,
                from_agent=self.agent_type,
                to_agent=to_agent,
                message_type=MessageType.RESPONSE,
                content=result,
                timestamp=datetime.utcnow().isoformat(),
                discord_message_id=discord_msg.id,
                channel_id=channel.id,
                in_reply_to=in_reply_to
            )

            # Complete the original task if this is a response
            if in_reply_to:
                self.registry.complete_task(in_reply_to, inter_msg)

            logger.info(f"Sent response {message_id} to {to_agent}")
            return inter_msg

        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            return None

    async def send_handoff(
        self,
        to_agent: str,
        task: str,
        reason: str = "",
        context: str = "",
        channel_name: str = "ralph-team"
    ) -> Optional[InterBotMessage]:
        """
        Send a handoff to another agent.

        A handoff transfers ownership of work to another agent.
        """
        target_bot = self.registry.get_bot(to_agent)
        if not target_bot:
            return None

        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return None

        channel = None
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

        if not channel:
            return None

        message_id = self.registry.generate_message_id()
        content = MessageParser.format_handoff(task, reason)
        if context:
            content += f"\n\n**Context:**\n{context}"

        discord_content = f"<@{target_bot.bot_user_id}> {content}"

        try:
            discord_msg = await channel.send(discord_content)

            inter_msg = InterBotMessage(
                message_id=message_id,
                from_agent=self.agent_type,
                to_agent=to_agent,
                message_type=MessageType.HANDOFF,
                content=task,
                timestamp=datetime.utcnow().isoformat(),
                discord_message_id=discord_msg.id,
                channel_id=channel.id,
                context={"reason": reason, "full_context": context}
            )

            self.registry.track_message(inter_msg)
            logger.info(f"Sent handoff {message_id} to {to_agent}")

            return inter_msg

        except Exception as e:
            logger.error(f"Failed to send handoff: {e}")
            return None

    async def send_alert(
        self,
        to_agents: List[str],
        alert: str,
        severity: str = "warning",
        channel_name: str = "ralph-team"
    ) -> List[InterBotMessage]:
        """
        Send an alert to multiple agents.

        Args:
            to_agents: List of target agent types (or ["*"] for all)
            alert: Alert content
            severity: Alert severity
            channel_name: Channel to post in

        Returns:
            List of sent messages
        """
        if "*" in to_agents:
            to_agents = list(self.registry.bots.keys())

        # Remove self from recipients
        to_agents = [a for a in to_agents if a != self.agent_type]

        if not to_agents:
            return []

        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return []

        channel = None
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

        if not channel:
            return []

        # Build mentions
        mentions = []
        for agent_type in to_agents:
            bot = self.registry.get_bot(agent_type)
            if bot:
                mentions.append(f"<@{bot.bot_user_id}>")

        if not mentions:
            return []

        content = MessageParser.format_alert(alert, severity)
        discord_content = f"{' '.join(mentions)} {content}"

        messages = []
        try:
            discord_msg = await channel.send(discord_content)

            for agent_type in to_agents:
                message_id = self.registry.generate_message_id()
                inter_msg = InterBotMessage(
                    message_id=message_id,
                    from_agent=self.agent_type,
                    to_agent=agent_type,
                    message_type=MessageType.ALERT,
                    content=alert,
                    timestamp=datetime.utcnow().isoformat(),
                    discord_message_id=discord_msg.id,
                    channel_id=channel.id,
                    context={"severity": severity}
                )
                self.registry.track_message(inter_msg)
                messages.append(inter_msg)

            logger.info(f"Sent alert to {len(to_agents)} agents")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

        return messages

    async def broadcast(
        self,
        content: str,
        message_type: MessageType = MessageType.INFO,
        channel_name: str = "ralph-team"
    ) -> Optional[InterBotMessage]:
        """
        Broadcast a message to all agents (no specific @mention, just team channel).
        """
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return None

        channel = None
        for ch in guild.text_channels:
            if ch.name == channel_name:
                channel = ch
                break

        if not channel:
            return None

        formatted = MessageParser.format_message(message_type, content)

        try:
            discord_msg = await channel.send(f"**{self.agent_type.title()} Agent:** {formatted}")

            message_id = self.registry.generate_message_id()
            inter_msg = InterBotMessage(
                message_id=message_id,
                from_agent=self.agent_type,
                to_agent="*",
                message_type=message_type,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                discord_message_id=discord_msg.id,
                channel_id=channel.id
            )

            return inter_msg

        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
            return None

    def parse_incoming_message(
        self,
        from_user_id: int,
        content: str,
        discord_message_id: int,
        channel_id: int
    ) -> Optional[InterBotMessage]:
        """
        Parse an incoming Discord message into an InterBotMessage.

        Returns None if the message is not from a RALPH bot.
        """
        # Check if from a RALPH bot
        from_bot = self.registry.get_bot_by_id(from_user_id)
        if not from_bot:
            return None

        # Remove any @mentions from the content to get the actual message
        # Pattern: <@123456789> or <@!123456789>
        clean_content = re.sub(r'<@!?\d+>\s*', '', content).strip()

        # Parse the action and content
        message_type, actual_content = MessageParser.parse_content(clean_content)

        # Check for reply reference
        in_reply_to = ""
        reply_match = re.search(r'\[RESPONSE to (MSG-[\w-]+)\]', content)
        if reply_match:
            in_reply_to = reply_match.group(1)

        message_id = self.registry.generate_message_id()

        return InterBotMessage(
            message_id=message_id,
            from_agent=from_bot.agent_type,
            to_agent=self.agent_type,
            message_type=message_type,
            content=actual_content,
            timestamp=datetime.utcnow().isoformat(),
            discord_message_id=discord_message_id,
            channel_id=channel_id,
            in_reply_to=in_reply_to
        )


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_bot_registry: Optional[BotRegistry] = None


def get_bot_registry() -> BotRegistry:
    """Get or create the bot registry instance."""
    global _bot_registry
    if _bot_registry is None:
        _bot_registry = BotRegistry()
    return _bot_registry


def set_bot_registry(registry: BotRegistry):
    """Set the bot registry instance."""
    global _bot_registry
    _bot_registry = registry
