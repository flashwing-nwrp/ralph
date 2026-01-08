"""
Base Agent Bot Class for RALPH Discord Ensemble

This module provides the abstract base class that all RALPH agent bots inherit from.
Includes reconnection logic, logging, error handling, and common utilities.

Now with Claude Code execution capabilities for autonomous task completion.
"""

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Callable

import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import execution engine
from claude_executor import ClaudeExecutor, TaskResult, TaskStatus, AgentCoordinator
from agent_prompts import AGENT_ROLES, HANDOFF_RULES, build_agent_role
from vps_deployer import get_deployer, DeploymentStatus
from mission_manager import get_mission_manager, MissionStatus
from improvement_proposals import get_proposal_manager, ProposalStatus
from scrum_manager import get_scrum_manager, StoryStatus, SprintStatus

# P0/P1/P2 Operational Systems
from emergency_controls import get_emergency_system, TradingState, CircuitBreakerType
from monitoring_alerts import get_monitoring_system, AlertSeverity, AlertCategory, MetricType
from decision_logger import get_decision_logger, DecisionType, DecisionOutcome
from model_lifecycle import get_model_registry, ModelStatus, ModelType
from data_quality import get_data_quality_monitor, DataSource, QualityStatus
from scheduler import get_scheduler, TaskFrequency, ScheduleConfig
from testing_framework import get_testing_framework, TestType, TestStatus
from context_persistence import get_context_store, ContextType, MemoryPriority

# Inter-bot Communication
from bot_communication import (
    get_bot_registry, BotRegistry, BotCommunicator, MessageType,
    InterBotMessage, MessageParser
)

# Tiered LLM Orchestration (cost optimization)
from orchestration_layer import (
    get_orchestration_layer, OrchestrationLayer, TaskComplexity,
    OrchestrationResult
)


def setup_logging(agent_name: str, log_level: str = None) -> logging.Logger:
    """Configure logging for an agent with colored output and optional file logging."""

    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_dir = os.getenv("LOG_DIR", "logs")

    logger = logging.getLogger(agent_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Try to use colored logging if available
    try:
        import colorlog
        console_format = colorlog.ColoredFormatter(
            f"%(log_color)s%(asctime)s | {agent_name} | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        )
        console_handler.setFormatter(console_format)
    except ImportError:
        console_format = logging.Formatter(
            f"%(asctime)s | {agent_name} | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{agent_name.lower().replace(' ', '_')}.log"),
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class BaseAgentBot(ABC):
    """
    Abstract base class for RALPH agent bots.

    Provides:
    - Discord client setup with minimal intents
    - Automatic reconnection with exponential backoff
    - Structured logging
    - Error handling and recovery
    - Common utility methods for cross-agent communication
    """

    # Shared executor and coordinator (class-level, set by orchestrator)
    _executor: Optional[ClaudeExecutor] = None
    _coordinator: Optional[AgentCoordinator] = None

    def __init__(
        self,
        agent_name: str,
        token_env_var: str,
        primary_channel_name: str,
        agent_type: str,  # e.g., "tuning", "backtest", "risk", "strategy", "data"
        description: str = ""
    ):
        self.agent_name = agent_name
        self.token_env_var = token_env_var
        self.primary_channel_name = primary_channel_name
        self.agent_type = agent_type
        self.description = description

        # Get role from prompts (with project-specific context)
        self.agent_role = build_agent_role(agent_type, include_project_context=True)

        # Load token
        self.token = os.getenv(token_env_var)
        if not self.token:
            raise ValueError(f"Missing token: {token_env_var} not found in environment")

        # Load guild ID
        self.guild_id = os.getenv("DISCORD_GUILD_ID")
        if not self.guild_id:
            raise ValueError("Missing DISCORD_GUILD_ID in environment")
        self.guild_id = int(self.guild_id)

        # Load owner ID (optional but recommended)
        owner_id = os.getenv("OWNER_USER_ID")
        self.owner_id = int(owner_id) if owner_id else None

        # Setup logging
        self.logger = setup_logging(agent_name)

        # Reconnection settings
        self.reconnect_delay = int(os.getenv("RECONNECT_DELAY", "5"))
        self.max_reconnect_attempts = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "10"))
        self.reconnect_attempts = 0

        # Task tracking
        self.running_tasks: dict[str, str] = {}  # task_id -> description
        self.completed_tasks: list[dict] = []

        # Inter-bot communication
        self._bot_registry = get_bot_registry()
        self._communicator: Optional[BotCommunicator] = None  # Initialized in on_ready

        # Tiered LLM orchestration (cost optimization)
        # Uses cheap models (GPT-4o-mini/Haiku) for routing & simple tasks
        # Only escalates to Claude Code for complex work
        self._orchestrator = get_orchestration_layer()

        # Discord client setup with minimal intents
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.guilds = True
        intents.guild_messages = True

        self.bot = commands.Bot(
            command_prefix="!",
            intents=intents,
            help_command=None  # We'll implement our own
        )

        # Register event handlers
        self._register_events()
        self._register_commands()
        self._register_execution_commands()
        self._register_user_commands()
        self._register_mission_commands()
        self._register_proposal_commands()
        self._register_scrum_commands()
        self._register_operational_commands()
        self._register_interbot_commands()

    @classmethod
    def set_executor(cls, executor: ClaudeExecutor):
        """Set the shared executor for all agents."""
        cls._executor = executor

    @classmethod
    def set_coordinator(cls, coordinator: AgentCoordinator):
        """Set the shared coordinator for all agents."""
        cls._coordinator = coordinator

    def _register_events(self):
        """Register Discord event handlers."""

        @self.bot.event
        async def on_ready():
            """Called when bot successfully connects to Discord."""
            self.reconnect_attempts = 0  # Reset on successful connection
            self.logger.info(f"Connected as {self.bot.user.name} ({self.bot.user.id})")
            self.logger.info(f"Serving guild: {self.guild_id}")

            # Register with bot registry for inter-bot communication
            await self._bot_registry.register_bot(
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                bot_user_id=self.bot.user.id,
                bot_username=self.bot.user.name,
                primary_channel=self.primary_channel_name
            )

            # Initialize the communicator now that we have the bot user
            self._communicator = BotCommunicator(self._bot_registry, self.bot)
            self._communicator.set_agent_info(self.agent_type, self.guild_id)

            self.logger.info(f"Registered with bot registry: {self.agent_type} (ID: {self.bot.user.id})")

            # Post ready message to bot-logs channel
            await self._post_to_channel("bot-logs", f"**{self.agent_name}** is online and ready!")

            # Call agent-specific on_ready
            await self.on_agent_ready()

        @self.bot.event
        async def on_disconnect():
            """Called when bot disconnects from Discord."""
            self.logger.warning("Disconnected from Discord")

        @self.bot.event
        async def on_resumed():
            """Called when bot successfully resumes a session."""
            self.logger.info("Session resumed")

        @self.bot.event
        async def on_error(event: str, *args, **kwargs):
            """Global error handler for events."""
            self.logger.exception(f"Error in event {event}")
            await self._post_to_channel(
                "error-logs",
                f"**{self.agent_name}** error in `{event}`: Check logs for details"
            )

        @self.bot.event
        async def on_message(message: discord.Message):
            """Handle incoming messages."""
            # Ignore messages from self
            if message.author == self.bot.user:
                return

            # Ignore DMs
            if not message.guild:
                return

            # Only process messages from our guild
            if message.guild.id != self.guild_id:
                return

            # =========================================================
            # CHANNEL-BASED COMMAND ROUTING
            # Prevents all bots from responding to the same command
            # =========================================================
            channel_id = message.channel.id
            channel_name = message.channel.name.lower() if hasattr(message.channel, 'name') else ""
            should_process_commands = False

            # Get team channel ID from env
            team_channel_id = int(os.getenv("CHANNEL_RALPH_TEAM", "0"))

            # Team channel: Only Strategy Agent handles commands (mission lead)
            if channel_id == team_channel_id or "ralph" in channel_name:
                if self.agent_type == "strategy":
                    should_process_commands = True
                # Other bots still listen for @mentions in team channel

            # Agent-specific channels: Only that agent responds
            elif channel_name == self.primary_channel_name or self.primary_channel_name in channel_name:
                should_process_commands = True

            # Other channels (bot-logs, error-logs, etc): No command processing
            # but @mentions still work

            # Process commands only if this bot should handle this channel
            if should_process_commands:
                await self.bot.process_commands(message)

            # Check if this is a message from another RALPH bot mentioning us
            if self._bot_registry.is_ralph_bot(message.author.id):
                if self.bot.user.mentioned_in(message):
                    await self._handle_interbot_message(message)
                    return  # Don't process as regular mention

            # Check if owner is @mentioning this agent
            if self.bot.user.mentioned_in(message):
                await self._handle_mention(message)

            # Then handle agent-specific message processing
            await self.on_agent_message(message)

    def _is_owner(self, user_id: int) -> bool:
        """Check if user is the owner/operator."""
        return self.owner_id is not None and user_id == self.owner_id

    async def _handle_mention(self, message: discord.Message):
        """Handle when this agent is @mentioned."""
        # Remove the bot mention from the message to get the actual content
        content = message.content
        for mention in message.mentions:
            if mention == self.bot.user:
                content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
        content = content.strip()

        if not content:
            # Just a mention with no content
            await message.reply(
                f"You called? I'm the **{self.agent_name}**. "
                f"Ask me something or give me a task with `!do <task>`"
            )
            return

        # Check if this is from the owner - give priority
        is_owner = self._is_owner(message.author.id)

        if is_owner:
            self.logger.info(f"Owner directive: {content[:100]}")

            # If it looks like a question, answer it
            if "?" in content:
                await self._respond_to_question(message, content)
            # If it looks like a directive, execute it
            elif any(word in content.lower() for word in ["stop", "pause", "cancel", "wait"]):
                await message.reply(f"Understood. **{self.agent_name}** standing by.")
            elif any(word in content.lower() for word in ["resume", "continue", "proceed", "go"]):
                await message.reply(f"**{self.agent_name}** resuming operations.")
            else:
                # Treat as a task directive
                await message.reply(f"On it! Executing: {content[:100]}...")
                result = await self.execute_task(content)
                await self._post_task_result(message.channel, result)
        else:
            # Non-owner mention - still respond but don't auto-execute
            await message.reply(
                f"I heard you! For task execution, use `!do <task>` or ask the operator."
            )

    async def _respond_to_question(self, message: discord.Message, question: str):
        """Respond to a question using Claude Code."""
        if not self._executor:
            await message.reply("Executor not available. Please try again later.")
            return

        # Frame as a question-answering task
        task = f"Answer this question based on your expertise as {self.agent_name}: {question}"

        await message.reply(f"Let me think about that...")
        result = await self.execute_task(task, notify_channel=False)

        if result.status == TaskStatus.COMPLETED:
            # Truncate for Discord
            response = result.output[:1900] if len(result.output) > 1900 else result.output
            await message.reply(response)
        else:
            await message.reply(f"I had trouble with that: {result.error or 'Unknown error'}")

    async def _handle_interbot_message(self, message: discord.Message):
        """
        Handle a message from another RALPH bot.

        This enables inter-bot collaboration where bots can:
        - Send tasks to each other with [TASK]
        - Respond to tasks with [RESPONSE]
        - Hand off work with [HANDOFF]
        - Send alerts with [ALERT]
        - Ask questions with [QUESTION]
        """
        # Get info about the sending bot
        from_bot = self._bot_registry.get_bot_by_id(message.author.id)
        if not from_bot:
            return  # Not a registered bot

        # Parse the message
        if self._communicator:
            inter_msg = self._communicator.parse_incoming_message(
                from_user_id=message.author.id,
                content=message.content,
                discord_message_id=message.id,
                channel_id=message.channel.id
            )

            if not inter_msg:
                return

            self.logger.info(
                f"Inter-bot message from {from_bot.agent_name}: "
                f"[{inter_msg.message_type.value}] {inter_msg.content[:100]}"
            )

            # Handle based on message type
            if inter_msg.message_type == MessageType.TASK:
                await self._handle_bot_task(message, inter_msg, from_bot)

            elif inter_msg.message_type == MessageType.HANDOFF:
                await self._handle_bot_handoff(message, inter_msg, from_bot)

            elif inter_msg.message_type == MessageType.QUESTION:
                await self._handle_bot_question(message, inter_msg, from_bot)

            elif inter_msg.message_type == MessageType.RESPONSE:
                # Log that we received a response
                self.logger.info(f"Received response from {from_bot.agent_name}: {inter_msg.content[:100]}")
                # Could trigger callbacks here if registered

            elif inter_msg.message_type == MessageType.ALERT:
                # Log alert from another bot
                self.logger.warning(f"Alert from {from_bot.agent_name}: {inter_msg.content}")
                # Optionally take action based on alert

            elif inter_msg.message_type == MessageType.ACK:
                self.logger.info(f"ACK from {from_bot.agent_name}")

    async def _handle_bot_task(self, message: discord.Message, inter_msg: InterBotMessage, from_bot):
        """Handle a task request from another bot."""
        # Acknowledge receipt
        await message.reply(f"[ACK] Received task from **{from_bot.agent_name}**. Processing...")

        # Execute the task
        context = inter_msg.context.get("full_context", "")
        result = await self.execute_task(
            task=inter_msg.content,
            context=f"Task from {from_bot.agent_name}:\n{context}",
            notify_channel=True
        )

        # Send response back to the requesting bot
        if self._communicator:
            response_text = result.output if result.status == TaskStatus.COMPLETED else f"Failed: {result.error}"
            await self._communicator.send_response(
                to_agent=from_bot.agent_type,
                result=response_text[:1500],  # Truncate for Discord
                in_reply_to=inter_msg.message_id,
                channel_name=message.channel.name
            )

    async def _handle_bot_handoff(self, message: discord.Message, inter_msg: InterBotMessage, from_bot):
        """Handle a handoff from another bot."""
        reason = inter_msg.context.get("reason", "")
        context = inter_msg.context.get("full_context", "")

        self.logger.info(f"Handoff received from {from_bot.agent_name}: {inter_msg.content}")

        # Acknowledge the handoff
        await message.reply(
            f"[ACK] Accepted handoff from **{from_bot.agent_name}**.\n"
            f"Task: {inter_msg.content[:200]}"
        )

        # Queue the handoff task
        if self._coordinator:
            await self._coordinator.queue_handoff(
                from_agent=from_bot.agent_type,
                to_agent=self.agent_type,
                task=inter_msg.content,
                context=context
            )
        else:
            # Execute directly if no coordinator
            await self.execute_task(
                task=inter_msg.content,
                context=f"Handoff from {from_bot.agent_name}:\nReason: {reason}\n{context}"
            )

    async def _handle_bot_question(self, message: discord.Message, inter_msg: InterBotMessage, from_bot):
        """Handle a question from another bot."""
        # Respond to the question
        task = f"Answer this question from {from_bot.agent_name}: {inter_msg.content}"
        result = await self.execute_task(task, notify_channel=False)

        if self._communicator:
            response_text = result.output if result.status == TaskStatus.COMPLETED else f"Couldn't answer: {result.error}"
            await self._communicator.send_response(
                to_agent=from_bot.agent_type,
                result=response_text[:1500],
                in_reply_to=inter_msg.message_id,
                channel_name=message.channel.name
            )

    def _register_commands(self):
        """Register common bot commands."""

        @self.bot.command(name="ping")
        async def ping(ctx: commands.Context):
            """Check if bot is responsive."""
            latency = round(self.bot.latency * 1000)
            await ctx.reply(f"Pong! Latency: {latency}ms")

        @self.bot.command(name="status")
        async def status(ctx: commands.Context):
            """Get agent status."""
            status_info = await self.get_status()
            embed = discord.Embed(
                title=f"{self.agent_name} Status",
                color=discord.Color.green(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(name="Agent", value=self.agent_name, inline=True)
            embed.add_field(name="Primary Channel", value=f"#{self.primary_channel_name}", inline=True)
            embed.add_field(name="Latency", value=f"{round(self.bot.latency * 1000)}ms", inline=True)

            # Add agent-specific status fields
            for key, value in status_info.items():
                embed.add_field(name=key, value=str(value), inline=True)

            embed.set_footer(text=self.description)
            await ctx.reply(embed=embed)

        @self.bot.command(name="help")
        async def help_cmd(ctx: commands.Context):
            """Show available commands."""
            agent_commands = await self.get_commands()

            embed = discord.Embed(
                title=f"{self.agent_name} Commands",
                color=discord.Color.blue()
            )

            # Common commands
            embed.add_field(
                name="Common Commands",
                value=(
                    "`!ping` - Check bot latency\n"
                    "`!status` - Get agent status\n"
                    "`!help` - Show this message\n"
                    "`!mission <goal>` - Set a new mission (owner)\n"
                    "`!mission_status` - Check mission progress"
                ),
                inline=False
            )

            # Agent-specific commands
            if agent_commands:
                cmd_text = "\n".join([f"`!{cmd}` - {desc}" for cmd, desc in agent_commands.items()])
                embed.add_field(
                    name=f"{self.agent_name} Commands",
                    value=cmd_text,
                    inline=False
                )

            await ctx.reply(embed=embed)

    def _register_execution_commands(self):
        """Register commands for Claude Code execution."""

        @self.bot.command(name="do")
        async def do_task(ctx: commands.Context, *, task: str = None):
            """Execute a task using Claude Code."""
            if not task:
                await ctx.reply(
                    f"Usage: `!do <task description>`\n"
                    f"Example: `!do Analyze the momentum strategy and suggest improvements`"
                )
                return

            if not self._executor:
                await ctx.reply("Executor not initialized. Please wait for system startup.")
                return

            # Acknowledge the task
            status_msg = await ctx.reply(
                f"**{self.agent_name}** executing task...\n"
                f"```\n{task[:200]}{'...' if len(task) > 200 else ''}\n```"
            )

            # Execute the task
            result = await self.execute_task(task)

            # Post results
            await self._post_task_result(ctx.channel, result)

        @self.bot.command(name="handoff")
        async def handoff(ctx: commands.Context, target_agent: str = None, *, task: str = None):
            """Hand off a task to another agent."""
            if not target_agent or not task:
                await ctx.reply(
                    "Usage: `!handoff <agent> <task>`\n"
                    "Agents: `tuning`, `backtest`, `risk`, `strategy`, `data`\n"
                    "Example: `!handoff backtest Validate the new momentum parameters`"
                )
                return

            target_agent = target_agent.lower()
            valid_agents = ["tuning", "backtest", "risk", "strategy", "data"]

            if target_agent not in valid_agents:
                await ctx.reply(f"Unknown agent: `{target_agent}`. Valid: {', '.join(valid_agents)}")
                return

            # Queue the handoff
            if self._coordinator:
                await self._coordinator.queue_handoff(
                    from_agent=self.agent_type,
                    to_agent=target_agent,
                    task=task,
                    context=self._coordinator.context_store.get(self.agent_type, "")
                )

            # Post to team channel
            await self.post_to_team_channel(
                f"**{self.agent_name}** → **{target_agent.title()} Agent**\n"
                f"Task: {task}\n\n"
                f"*Handoff queued for autonomous execution*"
            )

            await ctx.reply(f"Handed off to **{target_agent.title()} Agent**")

        @self.bot.command(name="tasks")
        async def tasks(ctx: commands.Context):
            """Show running and recent tasks."""
            embed = discord.Embed(
                title=f"{self.agent_name} Tasks",
                color=discord.Color.blue()
            )

            # Running tasks
            if self.running_tasks:
                running_text = "\n".join([
                    f"• `{tid}`: {desc[:50]}..."
                    for tid, desc in self.running_tasks.items()
                ])
            else:
                running_text = "No tasks running"
            embed.add_field(name="Running", value=running_text, inline=False)

            # Recent completed
            if self.completed_tasks:
                recent = self.completed_tasks[-5:]
                completed_text = "\n".join([
                    f"• `{t['task_id']}`: {t['status']}"
                    for t in recent
                ])
            else:
                completed_text = "No completed tasks"
            embed.add_field(name="Recent Completed", value=completed_text, inline=False)

            await ctx.reply(embed=embed)

    def _register_user_commands(self):
        """Register commands for user/operator interaction."""

        @self.bot.command(name="ask")
        async def ask(ctx: commands.Context, *, question: str = None):
            """Ask this agent a question."""
            if not question:
                await ctx.reply(
                    f"Usage: `!ask <question>`\n"
                    f"Example: `!ask What parameters should I tune for momentum strategies?`"
                )
                return

            await self._respond_to_question(ctx.message, question)

        @self.bot.command(name="stop")
        async def stop_agent(ctx: commands.Context):
            """Tell this agent to pause (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can stop agents.")
                return

            await ctx.reply(f"**{self.agent_name}** pausing. Use `!resume` to continue.")
            # Note: Full pause implementation would require task queue management

        @self.bot.command(name="resume")
        async def resume_agent(ctx: commands.Context):
            """Tell this agent to resume (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can resume agents.")
                return

            await ctx.reply(f"**{self.agent_name}** resuming operations.")

        @self.bot.command(name="redirect")
        async def redirect(ctx: commands.Context, target_agent: str = None, *, message: str = None):
            """Send a message to another agent (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can redirect messages.")
                return

            if not target_agent or not message:
                await ctx.reply(
                    "Usage: `!redirect <agent> <message>`\n"
                    "Example: `!redirect risk Please review the latest backtest`"
                )
                return

            # Post to the target agent's channel
            valid_channels = {
                "tuning": "tuning",
                "backtest": "backtesting",
                "risk": "risk",
                "strategy": "strategy",
                "data": "data"
            }

            target = target_agent.lower()
            if target not in valid_channels:
                await ctx.reply(f"Unknown agent: `{target}`. Valid: {', '.join(valid_channels.keys())}")
                return

            # Post message as if from the operator
            channel_name = valid_channels[target]
            await self._post_to_channel(
                channel_name,
                f"**Operator message via {self.agent_name}:**\n{message}"
            )
            await ctx.reply(f"Message sent to #{channel_name}")

        @self.bot.command(name="broadcast")
        async def broadcast(ctx: commands.Context, *, message: str = None):
            """Broadcast a message to all agents (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can broadcast.")
                return

            if not message:
                await ctx.reply("Usage: `!broadcast <message>`")
                return

            await self.post_to_team_channel(f"**OPERATOR BROADCAST:**\n{message}")
            await ctx.reply("Broadcast sent to #ralph_team")

        # VPS Deployment Commands

        @self.bot.command(name="deploy")
        async def deploy(ctx: commands.Context):
            """Deploy latest code to VPS (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can deploy.")
                return

            deployer = get_deployer()
            if not deployer.config:
                await ctx.reply("VPS not configured. Check .env file.")
                return

            await ctx.reply("Deploying to VPS... This may take a minute.")

            status, message = await deployer.deploy()

            if status == DeploymentStatus.SUCCESS:
                embed = discord.Embed(
                    title="Deployment Successful",
                    description=message,
                    color=discord.Color.green()
                )
            else:
                embed = discord.Embed(
                    title="Deployment Failed",
                    description=message,
                    color=discord.Color.red()
                )

            await ctx.reply(embed=embed)
            await self.post_to_team_channel(
                f"**{self.agent_name}** triggered deployment: {status.value}"
            )

        @self.bot.command(name="vps")
        async def vps_status(ctx: commands.Context):
            """Check VPS status."""
            deployer = get_deployer()
            if not deployer.config:
                await ctx.reply("VPS not configured. Check .env file.")
                return

            await ctx.reply("Checking VPS status...")
            status = await deployer.get_status()

            embed = discord.Embed(
                title="VPS Status",
                color=discord.Color.green() if status["connection"] else discord.Color.red()
            )
            embed.add_field(name="Connection", value="OK" if status["connection"] else "FAILED", inline=True)
            embed.add_field(name="Service", value=status["service"], inline=True)
            embed.add_field(name="Uptime", value=status["uptime"], inline=True)

            if status["last_log"]:
                embed.add_field(
                    name="Recent Log",
                    value=f"```\n{status['last_log'][-500:]}\n```",
                    inline=False
                )

            await ctx.reply(embed=embed)

        @self.bot.command(name="logs")
        async def vps_logs(ctx: commands.Context, lines: int = 50):
            """Get VPS logs (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can view logs.")
                return

            deployer = get_deployer()
            if not deployer.config:
                await ctx.reply("VPS not configured.")
                return

            logs = await deployer.get_logs(lines)

            # Split into chunks for Discord
            if len(logs) > 1900:
                logs = logs[-1900:]

            await ctx.reply(f"```\n{logs}\n```")

        @self.bot.command(name="restart")
        async def restart_vps(ctx: commands.Context):
            """Restart VPS service (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can restart services.")
                return

            deployer = get_deployer()
            if not deployer.config:
                await ctx.reply("VPS not configured.")
                return

            success, message = await deployer.restart_service()

            if success:
                await ctx.reply(f"Service restarted successfully")
                await self.post_to_team_channel(
                    f"**{self.agent_name}** restarted VPS service"
                )
            else:
                await ctx.reply(f"Restart failed: {message}")

    def _register_mission_commands(self):
        """Register commands for mission/goal management."""

        @self.bot.command(name="mission")
        async def mission(ctx: commands.Context, *, objective: str = None):
            """Set a new mission/goal for the agent ensemble (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can set missions.")
                return

            if not objective:
                # Show current mission status
                manager = get_mission_manager()
                summary = manager.get_mission_summary()
                await ctx.reply(summary)
                return

            # Create new mission
            manager = get_mission_manager()
            mission = await manager.create_mission(
                objective=objective,
                created_by=str(ctx.author)
            )

            # Acknowledge creation
            embed = discord.Embed(
                title=f"Mission Created: {mission.mission_id}",
                description=objective,
                color=discord.Color.gold(),
                timestamp=datetime.utcnow()
            )
            embed.add_field(
                name="Status",
                value="Routing to **Strategy Agent** for planning...",
                inline=False
            )
            embed.set_footer(text="Use !mission to check progress")
            await ctx.reply(embed=embed)

            # Notify the team
            await self.post_to_team_channel(
                f"**NEW MISSION** from {ctx.author.mention}\n\n"
                f"**{mission.mission_id}**: {objective}\n\n"
                f"**@Strategy Agent** - Please break down this mission into actionable tasks."
            )

            # Trigger Strategy Agent to plan the mission
            if self._coordinator:
                await self._coordinator.queue_handoff(
                    from_agent="operator",
                    to_agent="strategy",
                    task=f"""NEW MISSION: {objective}

You are receiving a new mission from the operator. Your job is to:

1. Analyze the mission objective
2. Break it down into specific, actionable tasks
3. Assign each task to the appropriate agent:
   - tuning: Parameter optimization, hyperparameter search
   - backtest: Simulation, validation, performance testing
   - risk: Safety audits, risk assessment, compliance
   - data: Data preprocessing, feature engineering, data quality
   - strategy: Strategy logic, signal generation (yourself)

4. Identify dependencies between tasks (what must complete before what)
5. Post a mission plan to #ralph_team

After planning, begin executing by handing off tasks to the appropriate agents.

Respond with your mission breakdown and begin autonomous execution.""",
                    context=f"Mission ID: {mission.mission_id}\nCreated by: {ctx.author}\nObjective: {objective}"
                )

        @self.bot.command(name="mission_status")
        async def mission_status(ctx: commands.Context):
            """Check the current mission status."""
            manager = get_mission_manager()
            summary = manager.get_mission_summary()
            await ctx.reply(summary)

        @self.bot.command(name="pause_mission")
        async def pause_mission(ctx: commands.Context):
            """Pause the current mission (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can pause missions.")
                return

            manager = get_mission_manager()
            if not manager.current_mission:
                await ctx.reply("No active mission to pause.")
                return

            await manager.pause_mission()
            await ctx.reply(f"Mission **{manager.current_mission.mission_id}** paused.")
            await self.post_to_team_channel(
                f"**MISSION PAUSED** by {ctx.author.mention}\n"
                f"All agents should complete current tasks and stand by."
            )

        @self.bot.command(name="resume_mission")
        async def resume_mission(ctx: commands.Context):
            """Resume a paused mission (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can resume missions.")
                return

            manager = get_mission_manager()
            if not manager.current_mission:
                await ctx.reply("No mission to resume.")
                return

            await manager.resume_mission()
            await ctx.reply(f"Mission **{manager.current_mission.mission_id}** resumed.")
            await self.post_to_team_channel(
                f"**MISSION RESUMED** by {ctx.author.mention}\n"
                f"All agents resume normal operations."
            )

        @self.bot.command(name="abort_mission")
        async def abort_mission(ctx: commands.Context):
            """Abort the current mission (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can abort missions.")
                return

            manager = get_mission_manager()
            if not manager.current_mission:
                await ctx.reply("No active mission to abort.")
                return

            mission_id = manager.current_mission.mission_id
            manager.current_mission = None
            # Remove the mission file
            if manager.mission_file.exists():
                manager.mission_file.unlink()

            await ctx.reply(f"Mission **{mission_id}** aborted.")
            await self.post_to_team_channel(
                f"**MISSION ABORTED** by {ctx.author.mention}\n"
                f"Mission {mission_id} has been cancelled. All agents stand by."
            )

    def _register_proposal_commands(self):
        """Register commands for improvement proposals."""

        @self.bot.command(name="propose")
        async def propose(ctx: commands.Context, category: str = None, priority: str = None, *, description: str = None):
            """
            Submit an improvement proposal.

            Usage: !propose <category> <priority> <problem> | <solution> | <impact>

            Categories: performance, accuracy, risk, data, architecture, strategy, automation, bug_fix, feature
            Priorities: low, medium, high, critical
            """
            if not category or not priority or not description:
                embed = discord.Embed(
                    title="Submit Improvement Proposal",
                    description="Noticed something that could be improved? Submit a proposal!",
                    color=discord.Color.blue()
                )
                embed.add_field(
                    name="Usage",
                    value="`!propose <category> <priority> <problem> | <solution> | <impact>`",
                    inline=False
                )
                embed.add_field(
                    name="Categories",
                    value="`performance`, `accuracy`, `risk`, `data`, `architecture`, `strategy`, `automation`, `bug_fix`, `feature`",
                    inline=False
                )
                embed.add_field(
                    name="Priorities",
                    value="`low`, `medium`, `high`, `critical`",
                    inline=False
                )
                embed.add_field(
                    name="Example",
                    value="`!propose accuracy high Model predictions biased toward YES | Add isotonic calibration | Reduce prediction bias from 65% to 50%`",
                    inline=False
                )
                await ctx.reply(embed=embed)
                return

            # Parse the description (problem | solution | impact)
            parts = [p.strip() for p in description.split("|")]
            if len(parts) < 3:
                await ctx.reply(
                    "Please separate problem, solution, and impact with `|`\n"
                    "Example: `!propose accuracy high Model is biased | Add calibration | Fix 15% bias`"
                )
                return

            problem, solution, impact = parts[0], parts[1], parts[2]

            # Validate category and priority
            valid_categories = ["performance", "accuracy", "risk", "data", "architecture", "strategy", "automation", "bug_fix", "feature"]
            valid_priorities = ["low", "medium", "high", "critical"]

            if category.lower() not in valid_categories:
                await ctx.reply(f"Invalid category. Choose from: {', '.join(valid_categories)}")
                return

            if priority.lower() not in valid_priorities:
                await ctx.reply(f"Invalid priority. Choose from: {', '.join(valid_priorities)}")
                return

            # Get current mission context if available
            mission_manager = get_mission_manager()
            discovered_during = ""
            if mission_manager.current_mission:
                discovered_during = f"Mission {mission_manager.current_mission.mission_id}"

            # Submit the proposal
            proposal_manager = get_proposal_manager()
            proposal = await proposal_manager.submit_proposal(
                submitted_by=self.agent_type,
                category=category.lower(),
                priority=priority.lower(),
                problem=problem,
                solution=solution,
                expected_impact=impact,
                discovered_during=discovered_during
            )

            embed = discord.Embed(
                title=f"Proposal Submitted: {proposal.proposal_id}",
                color=discord.Color.green()
            )
            embed.add_field(name="Problem", value=problem[:200], inline=False)
            embed.add_field(name="Solution", value=solution[:200], inline=False)
            embed.add_field(name="Expected Impact", value=impact[:200], inline=False)
            embed.add_field(name="Category", value=category, inline=True)
            embed.add_field(name="Priority", value=priority, inline=True)
            embed.set_footer(text=f"Submitted by {self.agent_name}")

            await ctx.reply(embed=embed)

            # Notify team channel
            await self.post_to_team_channel(
                f"💡 **New Improvement Proposal** from {self.agent_name}\n\n"
                f"**{proposal.proposal_id}** | {priority.upper()}\n"
                f"**Problem:** {problem[:100]}...\n\n"
                f"*Use `!proposals` to review*"
            )

        @self.bot.command(name="proposals")
        async def proposals(ctx: commands.Context):
            """View pending improvement proposals."""
            proposal_manager = get_proposal_manager()
            review_queue = proposal_manager.get_review_queue()

            # Discord has 2000 char limit, split if needed
            if len(review_queue) > 1900:
                # Send first part
                await ctx.reply(review_queue[:1900] + "...")
            else:
                await ctx.reply(review_queue)

        @self.bot.command(name="approve")
        async def approve(ctx: commands.Context, proposal_id: str = None, *, notes: str = ""):
            """Approve an improvement proposal (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can approve proposals.")
                return

            if not proposal_id:
                await ctx.reply("Usage: `!approve <proposal_id> [notes]`")
                return

            proposal_manager = get_proposal_manager()
            proposal = await proposal_manager.approve_proposal(proposal_id.upper(), notes)

            if not proposal:
                await ctx.reply(f"Proposal `{proposal_id}` not found.")
                return

            # Create a mission for the approved proposal
            mission_manager = get_mission_manager()
            mission = await mission_manager.create_mission(
                objective=f"[{proposal.proposal_id}] {proposal.solution}",
                context=f"Problem: {proposal.problem}\nExpected Impact: {proposal.expected_impact}",
                created_by=f"Proposal from {proposal.submitted_by}"
            )

            proposal.implementation_mission_id = mission.mission_id
            await proposal_manager._save_proposals()

            embed = discord.Embed(
                title=f"✅ Proposal Approved: {proposal.proposal_id}",
                color=discord.Color.green()
            )
            embed.add_field(name="Solution", value=proposal.solution[:500], inline=False)
            embed.add_field(name="New Mission", value=f"`{mission.mission_id}`", inline=True)
            embed.add_field(name="Assigned To", value=proposal.submitted_by, inline=True)
            if notes:
                embed.add_field(name="Operator Notes", value=notes, inline=False)

            await ctx.reply(embed=embed)

            # Notify team
            await self.post_to_team_channel(
                f"✅ **Proposal Approved!**\n\n"
                f"**{proposal.proposal_id}** → Mission **{mission.mission_id}**\n"
                f"**Solution:** {proposal.solution[:200]}\n\n"
                f"**@{proposal.submitted_by.title()} Agent** - Please proceed with implementation."
            )

            # Queue handoff to the originating agent
            if self._coordinator:
                await self._coordinator.queue_handoff(
                    from_agent="operator",
                    to_agent=proposal.submitted_by,
                    task=f"""APPROVED IMPROVEMENT: {proposal.solution}

Your proposal {proposal.proposal_id} has been approved!

Problem: {proposal.problem}
Solution: {proposal.solution}
Expected Impact: {proposal.expected_impact}

Please implement this improvement now. When complete, hand off to the appropriate
agent for validation (usually Backtest for testing, Risk for safety audit).""",
                    context=f"Mission: {mission.mission_id}\nOperator notes: {notes}"
                )

        @self.bot.command(name="reject")
        async def reject(ctx: commands.Context, proposal_id: str = None, *, reason: str = ""):
            """Reject an improvement proposal (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can reject proposals.")
                return

            if not proposal_id:
                await ctx.reply("Usage: `!reject <proposal_id> [reason]`")
                return

            proposal_manager = get_proposal_manager()
            proposal = await proposal_manager.reject_proposal(proposal_id.upper(), reason)

            if not proposal:
                await ctx.reply(f"Proposal `{proposal_id}` not found.")
                return

            embed = discord.Embed(
                title=f"❌ Proposal Rejected: {proposal.proposal_id}",
                color=discord.Color.red()
            )
            embed.add_field(name="Problem", value=proposal.problem[:200], inline=False)
            if reason:
                embed.add_field(name="Reason", value=reason, inline=False)

            await ctx.reply(embed=embed)

        @self.bot.command(name="defer")
        async def defer(ctx: commands.Context, proposal_id: str = None, *, reason: str = ""):
            """Defer a proposal for later consideration (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can defer proposals.")
                return

            if not proposal_id:
                await ctx.reply("Usage: `!defer <proposal_id> [reason]`")
                return

            proposal_manager = get_proposal_manager()
            proposal = await proposal_manager.defer_proposal(proposal_id.upper(), reason)

            if not proposal:
                await ctx.reply(f"Proposal `{proposal_id}` not found.")
                return

            embed = discord.Embed(
                title=f"⏸️ Proposal Deferred: {proposal.proposal_id}",
                color=discord.Color.gold()
            )
            embed.add_field(name="Problem", value=proposal.problem[:200], inline=False)
            if reason:
                embed.add_field(name="Reason", value=reason, inline=False)
            embed.set_footer(text="This proposal will be reconsidered later")

            await ctx.reply(embed=embed)

    def _register_scrum_commands(self):
        """Register SCRUM methodology commands."""

        # =====================================================================
        # BACKLOG COMMANDS
        # =====================================================================

        @self.bot.command(name="story")
        async def story(ctx: commands.Context, action: str = None, *, args: str = None):
            """
            Manage user stories in the backlog.

            Usage:
              !story add <title> | <description> | <type> | <points>
              !story view <story_id>
              !story update <story_id> <status>
              !story assign <story_id> <agent>
            """
            if not action:
                await ctx.reply(
                    "**Story Commands:**\n"
                    "`!story add <title> | <description> | <type> | <points>`\n"
                    "`!story view <story_id>`\n"
                    "`!story update <story_id> <status>`\n"
                    "`!story assign <story_id> <agent>`\n\n"
                    "Types: feature, bug, improvement, research, technical\n"
                    "Status: backlog, sprint, in_progress, in_review, done, blocked"
                )
                return

            scrum = get_scrum_manager()

            if action == "add":
                if not args:
                    await ctx.reply("Usage: `!story add <title> | <description> | <type> | <points>`")
                    return

                parts = [p.strip() for p in args.split("|")]
                title = parts[0]
                description = parts[1] if len(parts) > 1 else ""
                story_type = parts[2] if len(parts) > 2 else "feature"
                points = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0

                story = await scrum.create_story(
                    title=title,
                    description=description,
                    story_type=story_type.lower(),
                    story_points=points
                )

                embed = discord.Embed(
                    title=f"Story Created: {story.story_id}",
                    color=discord.Color.green()
                )
                embed.add_field(name="Title", value=title, inline=False)
                embed.add_field(name="Type", value=story_type, inline=True)
                embed.add_field(name="Points", value=str(points) or "Not estimated", inline=True)

                await ctx.reply(embed=embed)

            elif action == "view":
                story_id = args.upper() if args else None
                if not story_id:
                    await ctx.reply("Usage: `!story view <story_id>`")
                    return

                story = scrum.stories.get(story_id)
                if not story:
                    await ctx.reply(f"Story `{story_id}` not found.")
                    return

                embed = discord.Embed(
                    title=f"{story.story_id}: {story.title}",
                    description=story.description,
                    color=discord.Color.blue()
                )
                embed.add_field(name="Status", value=story.status.value, inline=True)
                embed.add_field(name="Type", value=story.story_type.value, inline=True)
                embed.add_field(name="Points", value=str(story.story_points), inline=True)
                embed.add_field(name="Assigned", value=story.assigned_to or "Unassigned", inline=True)

                if story.acceptance_criteria:
                    criteria = "\n".join([f"• {c}" for c in story.acceptance_criteria])
                    embed.add_field(name="Acceptance Criteria", value=criteria, inline=False)

                await ctx.reply(embed=embed)

            elif action == "update":
                if not args:
                    await ctx.reply("Usage: `!story update <story_id> <status>`")
                    return

                parts = args.split()
                story_id = parts[0].upper()
                status = parts[1].lower() if len(parts) > 1 else None

                if not status:
                    await ctx.reply("Please specify a status: backlog, sprint, in_progress, in_review, done, blocked")
                    return

                story = await scrum.update_story_status(story_id, status)
                if story:
                    await ctx.reply(f"✅ Story `{story_id}` updated to **{status}**")
                else:
                    await ctx.reply(f"Story `{story_id}` not found.")

            elif action == "assign":
                if not args:
                    await ctx.reply("Usage: `!story assign <story_id> <agent>`")
                    return

                parts = args.split()
                story_id = parts[0].upper()
                agent = parts[1].lower() if len(parts) > 1 else None

                if not agent:
                    await ctx.reply("Please specify an agent: tuning, backtest, risk, strategy, data")
                    return

                story = await scrum.update_story_status(story_id, scrum.stories[story_id].status.value, assigned_to=agent)
                if story:
                    await ctx.reply(f"✅ Story `{story_id}` assigned to **{agent}**")
                else:
                    await ctx.reply(f"Story `{story_id}` not found.")

        @self.bot.command(name="backlog")
        async def backlog(ctx: commands.Context):
            """View the product backlog."""
            scrum = get_scrum_manager()
            view = scrum.get_backlog_view()
            await ctx.reply(view)

        # =====================================================================
        # SPRINT COMMANDS
        # =====================================================================

        @self.bot.command(name="sprint")
        async def sprint(ctx: commands.Context, action: str = None, *, args: str = None):
            """
            Manage sprints.

            Usage:
              !sprint create <name> | <goal>
              !sprint start [sprint_id]
              !sprint add <story_id>
              !sprint end
              !sprint board
            """
            if not action:
                await ctx.reply(
                    "**Sprint Commands:**\n"
                    "`!sprint create <name> | <goal>` - Create new sprint\n"
                    "`!sprint start [sprint_id]` - Start a sprint\n"
                    "`!sprint add <story_id>` - Add story to sprint\n"
                    "`!sprint end` - End current sprint\n"
                    "`!sprint board` - View sprint board\n"
                    "`!sprint retro` - Run retrospective"
                )
                return

            scrum = get_scrum_manager()

            if action == "create":
                if not self._is_owner(ctx.author.id):
                    await ctx.reply("Only the operator can create sprints.")
                    return

                if not args:
                    await ctx.reply("Usage: `!sprint create <name> | <goal>`")
                    return

                parts = [p.strip() for p in args.split("|")]
                name = parts[0]
                goal = parts[1] if len(parts) > 1 else "Sprint goal not specified"

                sprint_obj = await scrum.create_sprint(name=name, goal=goal)

                embed = discord.Embed(
                    title=f"Sprint Created: {sprint_obj.sprint_id}",
                    color=discord.Color.blue()
                )
                embed.add_field(name="Name", value=name, inline=True)
                embed.add_field(name="Goal", value=goal, inline=False)
                embed.add_field(name="Duration", value=f"{sprint_obj.duration_days} days", inline=True)
                embed.set_footer(text="Use !sprint add <story_id> to add stories, then !sprint start")

                await ctx.reply(embed=embed)
                await self.post_to_team_channel(
                    f"📅 **New Sprint Created:** {sprint_obj.sprint_id}\n"
                    f"**{name}**\n"
                    f"Goal: {goal}"
                )

            elif action == "start":
                if not self._is_owner(ctx.author.id):
                    await ctx.reply("Only the operator can start sprints.")
                    return

                sprint_id = args.upper() if args else None

                # If no sprint_id, find latest planning sprint
                if not sprint_id:
                    planning = [s for s in scrum.sprints.values() if s.status == SprintStatus.PLANNING]
                    if planning:
                        sprint_id = planning[-1].sprint_id
                    else:
                        await ctx.reply("No sprint in planning. Create one with `!sprint create`")
                        return

                sprint_obj = await scrum.start_sprint(sprint_id)
                if not sprint_obj:
                    await ctx.reply(f"Sprint `{sprint_id}` not found.")
                    return

                embed = discord.Embed(
                    title=f"🏃 Sprint Started: {sprint_obj.name}",
                    color=discord.Color.green()
                )
                embed.add_field(name="Goal", value=sprint_obj.goal, inline=False)
                embed.add_field(name="Committed", value=f"{sprint_obj.committed_points} points", inline=True)
                embed.add_field(name="Duration", value=f"{sprint_obj.duration_days} days", inline=True)
                embed.add_field(name="Stories", value=str(len(sprint_obj.story_ids)), inline=True)

                await ctx.reply(embed=embed)
                await self.post_to_team_channel(
                    f"🏃 **Sprint Started!**\n\n"
                    f"**{sprint_obj.sprint_id}: {sprint_obj.name}**\n"
                    f"**Goal:** {sprint_obj.goal}\n"
                    f"**Committed:** {sprint_obj.committed_points} points\n\n"
                    f"Let's go team! 🚀"
                )

            elif action == "add":
                if not args:
                    await ctx.reply("Usage: `!sprint add <story_id>`")
                    return

                story_id = args.upper()
                current = scrum.get_current_sprint()

                # If no active sprint, try latest planning sprint
                if not current:
                    planning = [s for s in scrum.sprints.values() if s.status == SprintStatus.PLANNING]
                    if planning:
                        current = planning[-1]
                    else:
                        await ctx.reply("No sprint available. Create one with `!sprint create`")
                        return

                story = await scrum.add_to_sprint(current.sprint_id, story_id)
                if story:
                    await ctx.reply(f"✅ Added `{story_id}` to {current.sprint_id}")
                else:
                    await ctx.reply(f"Story `{story_id}` not found.")

            elif action == "end":
                if not self._is_owner(ctx.author.id):
                    await ctx.reply("Only the operator can end sprints.")
                    return

                current = scrum.get_current_sprint()
                if not current:
                    await ctx.reply("No active sprint to end.")
                    return

                sprint_obj = await scrum.end_sprint(current.sprint_id)
                progress = sprint_obj.get_progress()

                embed = discord.Embed(
                    title=f"Sprint Ended: {sprint_obj.name}",
                    color=discord.Color.gold()
                )
                embed.add_field(name="Completed", value=f"{progress['completed']}/{progress['committed']} points", inline=True)
                embed.add_field(name="Completion", value=f"{progress['percent']}%", inline=True)
                embed.set_footer(text="Run !sprint retro to conduct retrospective")

                await ctx.reply(embed=embed)
                await self.post_to_team_channel(
                    f"🏁 **Sprint Ended:** {sprint_obj.name}\n"
                    f"**Completed:** {progress['completed']}/{progress['committed']} points ({progress['percent']}%)\n\n"
                    f"Time for retrospective! Use `!sprint retro`"
                )

            elif action == "board":
                board = scrum.get_sprint_board()
                if len(board) > 1900:
                    await ctx.reply(board[:1900] + "...")
                else:
                    await ctx.reply(board)

            elif action == "retro":
                if not self._is_owner(ctx.author.id):
                    await ctx.reply("Only the operator can run retrospectives.")
                    return

                # Find sprint in review
                review_sprints = [s for s in scrum.sprints.values() if s.status == SprintStatus.REVIEW]
                if not review_sprints:
                    await ctx.reply("No sprint in review. End a sprint first with `!sprint end`")
                    return

                sprint_obj = review_sprints[-1]

                embed = discord.Embed(
                    title=f"🔄 Sprint Retrospective: {sprint_obj.name}",
                    description="Please provide feedback using:\n`!retro_feedback <type> <feedback>`",
                    color=discord.Color.purple()
                )
                embed.add_field(
                    name="Types",
                    value="• `good` - What went well\n• `improve` - What to improve\n• `action` - Action items",
                    inline=False
                )
                embed.add_field(
                    name="Example",
                    value="`!retro_feedback good Team communication was excellent`",
                    inline=False
                )

                await ctx.reply(embed=embed)

        @self.bot.command(name="retro_feedback")
        async def retro_feedback(ctx: commands.Context, feedback_type: str = None, *, feedback: str = None):
            """Submit retrospective feedback."""
            if not feedback_type or not feedback:
                await ctx.reply("Usage: `!retro_feedback <good|improve|action> <feedback>`")
                return

            scrum = get_scrum_manager()

            # Find sprint in review/retro
            retro_sprints = [
                s for s in scrum.sprints.values()
                if s.status in [SprintStatus.REVIEW, SprintStatus.RETRO]
            ]
            if not retro_sprints:
                await ctx.reply("No sprint in retrospective phase.")
                return

            sprint_obj = retro_sprints[-1]

            if feedback_type.lower() == "good":
                sprint_obj.went_well.append(f"{self.agent_name}: {feedback}")
                await scrum._save_data()
                await ctx.reply("✅ Added to 'What went well'")

            elif feedback_type.lower() == "improve":
                sprint_obj.to_improve.append(f"{self.agent_name}: {feedback}")
                await scrum._save_data()
                await ctx.reply("✅ Added to 'What to improve'")

            elif feedback_type.lower() == "action":
                sprint_obj.action_items.append(f"{self.agent_name}: {feedback}")
                await scrum._save_data()
                await ctx.reply("✅ Added to 'Action items'")

            else:
                await ctx.reply("Invalid type. Use: good, improve, or action")

        @self.bot.command(name="standup")
        async def standup(ctx: commands.Context):
            """Show daily standup summary."""
            scrum = get_scrum_manager()
            standup_view = scrum.get_daily_standup()
            await ctx.reply(standup_view)

        @self.bot.command(name="velocity")
        async def velocity(ctx: commands.Context):
            """Show team velocity report."""
            scrum = get_scrum_manager()
            report = scrum.get_velocity_report()
            await ctx.reply(report)

    def _register_operational_commands(self):
        """Register P0/P1/P2 operational system commands."""

        # =====================================================================
        # EMERGENCY CONTROLS (P0)
        # =====================================================================

        @self.bot.command(name="killswitch")
        async def killswitch(ctx: commands.Context, *, reason: str = "Manual kill switch activation"):
            """Activate emergency kill switch - halts ALL trading (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can activate the kill switch.")
                return

            emergency = get_emergency_system()
            event = await emergency.activate_kill_switch(reason, triggered_by=str(ctx.author))

            embed = discord.Embed(
                title="KILL SWITCH ACTIVATED",
                description=f"All trading has been HALTED.",
                color=discord.Color.red()
            )
            embed.add_field(name="Reason", value=reason, inline=False)
            embed.add_field(name="Event ID", value=event.event_id, inline=True)
            embed.add_field(name="Triggered By", value=str(ctx.author), inline=True)
            embed.set_footer(text="Use !resume_trading to resume after review")

            await ctx.reply(embed=embed)
            await self.post_to_team_channel(
                f"**KILL SWITCH ACTIVATED** by {ctx.author.mention}\n"
                f"Reason: {reason}\n"
                f"**ALL TRADING HALTED**"
            )

        @self.bot.command(name="halt")
        async def halt_trading(ctx: commands.Context, *, reason: str = "Manual halt"):
            """Halt trading (less severe than kill switch) (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can halt trading.")
                return

            emergency = get_emergency_system()
            await emergency.halt_trading(reason, triggered_by=str(ctx.author))
            await ctx.reply(f"Trading halted: {reason}\nUse `!resume_trading` to resume.")

        @self.bot.command(name="resume_trading")
        async def resume_trading(ctx: commands.Context, *, notes: str = ""):
            """Resume trading after halt/kill switch (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can resume trading.")
                return

            emergency = get_emergency_system()
            event = await emergency.resume_trading(str(ctx.author), notes)

            if event:
                await ctx.reply(f"Trading RESUMED.\nNotes: {notes or 'None'}")
                await self.post_to_team_channel(f"**Trading Resumed** by {ctx.author.mention}")
            else:
                await ctx.reply("Trading was already active.")

        @self.bot.command(name="trading_status")
        async def trading_status(ctx: commands.Context):
            """Check trading system status."""
            emergency = get_emergency_system()
            status_display = emergency.get_status_display()
            await ctx.reply(status_display)

        @self.bot.command(name="circuit_breakers")
        async def circuit_breakers(ctx: commands.Context):
            """Show circuit breaker configurations."""
            emergency = get_emergency_system()
            output = ["**Circuit Breaker Configuration:**\n"]
            for cb_type, config in emergency.circuit_breakers.items():
                output.append(
                    f"• **{cb_type.value}**: threshold={config.threshold}, "
                    f"cooldown={config.cooldown_minutes}m, auto_reset={config.auto_reset}"
                )
            await ctx.reply("\n".join(output))

        # =====================================================================
        # MONITORING & ALERTS (P0)
        # =====================================================================

        @self.bot.command(name="dashboard")
        async def dashboard(ctx: commands.Context):
            """Show the monitoring dashboard."""
            monitoring = get_monitoring_system()
            dashboard_view = monitoring.get_dashboard()
            await ctx.reply(dashboard_view)

        @self.bot.command(name="alerts")
        async def alerts(ctx: commands.Context, limit: int = 10):
            """Show active alerts."""
            monitoring = get_monitoring_system()
            alerts_view = monitoring.get_alerts_display(limit)
            await ctx.reply(alerts_view)

        @self.bot.command(name="ack")
        async def acknowledge_alert(ctx: commands.Context, alert_id: str = None):
            """Acknowledge an alert."""
            if not alert_id:
                await ctx.reply("Usage: `!ack <alert_id>`")
                return

            monitoring = get_monitoring_system()
            alert = await monitoring.acknowledge_alert(alert_id.upper(), str(ctx.author))

            if alert:
                await ctx.reply(f"Alert `{alert_id}` acknowledged.")
            else:
                await ctx.reply(f"Alert `{alert_id}` not found.")

        @self.bot.command(name="resolve")
        async def resolve_alert(ctx: commands.Context, alert_id: str = None, *, notes: str = ""):
            """Resolve an alert."""
            if not alert_id:
                await ctx.reply("Usage: `!resolve <alert_id> [notes]`")
                return

            monitoring = get_monitoring_system()
            alert = await monitoring.resolve_alert(alert_id.upper(), notes)

            if alert:
                await ctx.reply(f"Alert `{alert_id}` resolved.")
            else:
                await ctx.reply(f"Alert `{alert_id}` not found.")

        # =====================================================================
        # DECISION LOGGER (P0/P1)
        # =====================================================================

        @self.bot.command(name="decisions")
        async def decisions(ctx: commands.Context, limit: int = 10):
            """Show recent decision log."""
            logger = get_decision_logger()
            log_view = logger.get_decision_log_display(limit)
            await ctx.reply(log_view)

        @self.bot.command(name="decision")
        async def decision_detail(ctx: commands.Context, decision_id: str = None):
            """Show decision details."""
            if not decision_id:
                await ctx.reply("Usage: `!decision <decision_id>`")
                return

            logger = get_decision_logger()
            details = logger.get_decision_details(decision_id.upper())
            await ctx.reply(details)

        @self.bot.command(name="trading_summary")
        async def trading_summary(ctx: commands.Context, days: int = 1):
            """Show trading decision summary."""
            logger = get_decision_logger()
            summary = logger.get_trading_summary(days)
            await ctx.reply(summary)

        # =====================================================================
        # MODEL LIFECYCLE (P1)
        # =====================================================================

        @self.bot.command(name="models")
        async def models(ctx: commands.Context):
            """Show model registry."""
            registry = get_model_registry()
            view = registry.get_registry_display()
            await ctx.reply(view)

        @self.bot.command(name="model")
        async def model_detail(ctx: commands.Context, model_id: str = None, version: str = None):
            """Show model details."""
            if not model_id:
                await ctx.reply("Usage: `!model <model_id> [version]`")
                return

            registry = get_model_registry()
            details = registry.get_model_details(model_id, version)
            await ctx.reply(details)

        @self.bot.command(name="training")
        async def training_history(ctx: commands.Context, model_id: str = None):
            """Show training run history."""
            registry = get_model_registry()
            history = registry.get_training_history(model_id)
            await ctx.reply(history)

        @self.bot.command(name="deploy_model")
        async def deploy_model(ctx: commands.Context, model_id: str = None, version: str = None, *, notes: str = ""):
            """Deploy a model to production (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can deploy models.")
                return

            if not model_id or not version:
                await ctx.reply("Usage: `!deploy_model <model_id> <version> [notes]`")
                return

            registry = get_model_registry()
            model = await registry.deploy_model(model_id, version, str(ctx.author), notes)

            if model:
                await ctx.reply(f"Model `{model_id}` v{version} deployed.")
                await self.post_to_team_channel(
                    f"**Model Deployed:** {model_id} v{version} by {ctx.author.mention}"
                )
            else:
                await ctx.reply(f"Model `{model_id}` v{version} not found.")

        @self.bot.command(name="rollback_model")
        async def rollback_model(ctx: commands.Context, model_id: str = None, to_version: str = None, *, reason: str = ""):
            """Rollback to a previous model version (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can rollback models.")
                return

            if not model_id or not to_version:
                await ctx.reply("Usage: `!rollback_model <model_id> <to_version> [reason]`")
                return

            registry = get_model_registry()
            model = await registry.rollback_model(model_id, to_version, reason)

            if model:
                await ctx.reply(f"Model `{model_id}` rolled back to v{to_version}.")
                await self.post_to_team_channel(
                    f"**Model Rollback:** {model_id} -> v{to_version}\nReason: {reason}"
                )
            else:
                await ctx.reply(f"Rollback failed. Check model ID and version.")

        # =====================================================================
        # DATA QUALITY (P1)
        # =====================================================================

        @self.bot.command(name="data_quality")
        async def data_quality(ctx: commands.Context):
            """Show data quality dashboard."""
            monitor = get_data_quality_monitor()
            dashboard_view = monitor.get_dashboard()
            await ctx.reply(dashboard_view)

        @self.bot.command(name="data_source")
        async def data_source(ctx: commands.Context, source: str = None):
            """Show data quality for a specific source."""
            if not source:
                await ctx.reply(
                    "Usage: `!data_source <source>`\n"
                    "Sources: polymarket_api, coinbase_ws, binance_ws, mysql_candles, taapi_indicators"
                )
                return

            try:
                source_enum = DataSource(source.lower())
            except ValueError:
                await ctx.reply(f"Invalid source: {source}")
                return

            monitor = get_data_quality_monitor()
            details = monitor.get_source_details(source_enum)
            await ctx.reply(details)

        @self.bot.command(name="quality_history")
        async def quality_history(ctx: commands.Context, source: str = None, limit: int = 20):
            """Show data quality check history."""
            source_enum = None
            if source:
                try:
                    source_enum = DataSource(source.lower())
                except ValueError:
                    await ctx.reply(f"Invalid source: {source}")
                    return

            monitor = get_data_quality_monitor()
            history = monitor.get_quality_history(source_enum, limit)
            await ctx.reply(history)

        # =====================================================================
        # SCHEDULER (P2)
        # =====================================================================

        @self.bot.command(name="schedule")
        async def schedule(ctx: commands.Context):
            """Show scheduled tasks."""
            scheduler = get_scheduler()
            view = scheduler.get_schedule_display()
            await ctx.reply(view)

        @self.bot.command(name="task_detail")
        async def task_detail(ctx: commands.Context, task_id: str = None):
            """Show scheduled task details."""
            if not task_id:
                await ctx.reply("Usage: `!task_detail <task_id>`")
                return

            scheduler = get_scheduler()
            details = scheduler.get_task_details(task_id.upper())
            await ctx.reply(details)

        @self.bot.command(name="run_task")
        async def run_task(ctx: commands.Context, task_id: str = None):
            """Run a scheduled task now (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can run tasks manually.")
                return

            if not task_id:
                await ctx.reply("Usage: `!run_task <task_id>`")
                return

            scheduler = get_scheduler()
            await ctx.reply(f"Running task `{task_id}`...")

            execution = await scheduler.run_task(task_id.upper(), force=True)

            if execution:
                await ctx.reply(f"Task `{task_id}` {execution.status}: {execution.result or execution.error_message}")
            else:
                await ctx.reply(f"Task `{task_id}` not found or no handler registered.")

        @self.bot.command(name="pause_task")
        async def pause_task(ctx: commands.Context, task_id: str = None):
            """Pause a scheduled task (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can pause tasks.")
                return

            if not task_id:
                await ctx.reply("Usage: `!pause_task <task_id>`")
                return

            scheduler = get_scheduler()
            task = await scheduler.pause_task(task_id.upper())

            if task:
                await ctx.reply(f"Task `{task_id}` paused.")
            else:
                await ctx.reply(f"Task `{task_id}` not found.")

        @self.bot.command(name="resume_task")
        async def resume_task(ctx: commands.Context, task_id: str = None):
            """Resume a paused task (owner only)."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can resume tasks.")
                return

            if not task_id:
                await ctx.reply("Usage: `!resume_task <task_id>`")
                return

            scheduler = get_scheduler()
            task = await scheduler.resume_task(task_id.upper())

            if task:
                await ctx.reply(f"Task `{task_id}` resumed.")
            else:
                await ctx.reply(f"Task `{task_id}` not found.")

        # =====================================================================
        # TESTING (P2)
        # =====================================================================

        @self.bot.command(name="tests")
        async def tests(ctx: commands.Context):
            """Show test summary."""
            framework = get_testing_framework()
            summary = framework.get_test_summary()
            await ctx.reply(summary)

        @self.bot.command(name="test")
        async def test_detail(ctx: commands.Context, test_id: str = None):
            """Show test details."""
            if not test_id:
                await ctx.reply("Usage: `!test <test_id>`")
                return

            framework = get_testing_framework()
            details = framework.get_test_details(test_id.upper())
            await ctx.reply(details)

        @self.bot.command(name="run_test")
        async def run_test(ctx: commands.Context, test_id: str = None):
            """Run a specific test."""
            if not test_id:
                await ctx.reply("Usage: `!run_test <test_id>`")
                return

            framework = get_testing_framework()
            await ctx.reply(f"Running test `{test_id}`...")

            try:
                result = await framework.run_test(test_id.upper())
                status_emoji = "Pass" if result.status == TestStatus.PASSED else "Fail"
                await ctx.reply(f"Test `{test_id}`: {status_emoji}\n{result.result_message}")
            except ValueError as e:
                await ctx.reply(str(e))

        @self.bot.command(name="run_tests")
        async def run_all_tests(ctx: commands.Context, test_type: str = None):
            """Run all tests or tests of a specific type."""
            if not self._is_owner(ctx.author.id):
                await ctx.reply("Only the operator can run all tests.")
                return

            framework = get_testing_framework()
            await ctx.reply("Running tests...")

            if test_type:
                try:
                    type_enum = TestType(test_type.lower())
                    run = await framework.run_by_type(type_enum)
                except ValueError:
                    await ctx.reply(f"Invalid test type: {test_type}")
                    return
            else:
                run = await framework.run_all()

            results = framework.get_run_results(run.run_id)
            await ctx.reply(results)

        @self.bot.command(name="test_results")
        async def test_results(ctx: commands.Context, run_id: str = None):
            """Show test run results."""
            framework = get_testing_framework()
            results = framework.get_run_results(run_id.upper() if run_id else None)
            await ctx.reply(results)

        # =====================================================================
        # CONTEXT/MEMORY (P2)
        # =====================================================================

        @self.bot.command(name="context")
        async def context(ctx: commands.Context):
            """Show context store summary."""
            store = get_context_store()
            summary = store.get_context_summary()
            await ctx.reply(summary)

        @self.bot.command(name="agent_memory")
        async def agent_memory(ctx: commands.Context, agent: str = None):
            """Show memory summary for an agent."""
            if not agent:
                agent = self.agent_type

            store = get_context_store()
            summary = store.get_agent_memory_summary(agent)
            await ctx.reply(summary)

        @self.bot.command(name="share_context")
        async def share_context(ctx: commands.Context, key: str = None, *, value: str = None):
            """Share context with other agents."""
            if not key or not value:
                await ctx.reply("Usage: `!share_context <key> <value>`")
                return

            store = get_context_store()
            await store.share_context(
                from_agent=self.agent_type,
                key=key,
                value=value,
                summary=f"Shared by {self.agent_name}"
            )
            await ctx.reply(f"Context `{key}` shared with all agents.")

        @self.bot.command(name="sessions")
        async def sessions(ctx: commands.Context):
            """Show active agent sessions."""
            store = get_context_store()
            active = [s for s in store.sessions.values() if s.status == "active"]

            if not active:
                await ctx.reply("No active sessions.")
                return

            output = ["**Active Sessions:**"]
            for session in active:
                output.append(
                    f"• `{session.session_id}` | {session.agent}\n"
                    f"  Task: {session.current_task or 'None'}\n"
                    f"  Actions: {session.actions_taken}"
                )
            await ctx.reply("\n".join(output))

    def _register_interbot_commands(self):
        """Register commands for inter-bot communication."""

        @self.bot.command(name="team")
        async def team_status(ctx: commands.Context):
            """Show registered RALPH bots and their status."""
            online = self._bot_registry.get_online_bots()
            all_bots = list(self._bot_registry.bots.values())

            embed = discord.Embed(
                title="RALPH Team Status",
                description=f"{len(online)}/{len(all_bots)} agents online",
                color=discord.Color.blue()
            )

            for bot in all_bots:
                status = "Online" if bot.is_online else "Offline"
                embed.add_field(
                    name=bot.agent_name,
                    value=f"Status: {status}\nChannel: #{bot.primary_channel}",
                    inline=True
                )

            await ctx.reply(embed=embed)

        @self.bot.command(name="msg")
        async def send_message_to_agent(ctx: commands.Context, target_agent: str = None, *, message: str = None):
            """
            Send a message to another RALPH agent.

            Usage: !msg <agent> <message>
            Example: !msg backtest Please validate the latest parameter changes
            """
            if not target_agent or not message:
                await ctx.reply(
                    "**Send Message to Agent**\n"
                    "Usage: `!msg <agent> <message>`\n\n"
                    "Agents: `tuning`, `backtest`, `risk`, `strategy`, `data`\n\n"
                    "Example: `!msg backtest Please validate the latest parameters`"
                )
                return

            target = target_agent.lower()
            valid_agents = ["tuning", "backtest", "risk", "strategy", "data"]

            if target not in valid_agents:
                await ctx.reply(f"Unknown agent: `{target}`. Valid: {', '.join(valid_agents)}")
                return

            if target == self.agent_type:
                await ctx.reply("You can't message yourself!")
                return

            if not self._communicator:
                await ctx.reply("Communication system not initialized.")
                return

            # Send the task
            inter_msg = await self._communicator.send_task(
                to_agent=target,
                task=message,
                context=f"Request from {self.agent_name}",
                channel_name="ralph_team"
            )

            if inter_msg:
                await ctx.reply(f"Message sent to **{target.title()} Agent** (ID: `{inter_msg.message_id}`)")
            else:
                await ctx.reply(f"Failed to send message. Is **{target.title()} Agent** online?")

        @self.bot.command(name="ask_agent")
        async def ask_agent(ctx: commands.Context, target_agent: str = None, *, question: str = None):
            """
            Ask another RALPH agent a question.

            Usage: !ask_agent <agent> <question>
            """
            if not target_agent or not question:
                await ctx.reply(
                    "Usage: `!ask_agent <agent> <question>`\n"
                    "Example: `!ask_agent risk What's the current drawdown limit?`"
                )
                return

            target = target_agent.lower()
            valid_agents = ["tuning", "backtest", "risk", "strategy", "data"]

            if target not in valid_agents:
                await ctx.reply(f"Unknown agent: `{target}`. Valid: {', '.join(valid_agents)}")
                return

            if not self._communicator:
                await ctx.reply("Communication system not initialized.")
                return

            # Get the target bot
            target_bot = self._bot_registry.get_bot(target)
            if not target_bot:
                await ctx.reply(f"**{target.title()} Agent** not registered yet.")
                return

            # Send the question as a Discord message with @mention
            guild = self.bot.get_guild(self.guild_id)
            if not guild:
                await ctx.reply("Guild not found.")
                return

            team_channel = discord.utils.get(guild.text_channels, name="ralph_team")
            if not team_channel:
                await ctx.reply("Team channel not found.")
                return

            # Send formatted question
            question_content = f"<@{target_bot.bot_user_id}> [QUESTION] {question}"
            await team_channel.send(question_content)
            await ctx.reply(f"Question sent to **{target.title()} Agent**")

        @self.bot.command(name="alert_team")
        async def alert_team(ctx: commands.Context, severity: str = "warning", *, alert: str = None):
            """
            Send an alert to all RALPH agents.

            Usage: !alert_team [severity] <alert message>
            Severity: info, warning, error, critical
            """
            if not alert:
                await ctx.reply(
                    "Usage: `!alert_team [severity] <message>`\n"
                    "Severities: `info`, `warning`, `error`, `critical`\n\n"
                    "Example: `!alert_team warning High API latency detected`"
                )
                return

            if severity not in ["info", "warning", "error", "critical"]:
                # severity is actually part of the message
                alert = f"{severity} {alert}"
                severity = "warning"

            if not self._communicator:
                await ctx.reply("Communication system not initialized.")
                return

            messages = await self._communicator.send_alert(
                to_agents=["*"],  # All agents
                alert=alert,
                severity=severity,
                channel_name="ralph_team"
            )

            if messages:
                await ctx.reply(f"Alert sent to {len(messages)} agents.")
            else:
                await ctx.reply("Failed to send alert.")

        @self.bot.command(name="delegate")
        async def delegate(ctx: commands.Context, target_agent: str = None, *, task: str = None):
            """
            Delegate a task to another agent with full handoff.

            Usage: !delegate <agent> <task>
            This transfers ownership of the task to the target agent.
            """
            if not target_agent or not task:
                await ctx.reply(
                    "**Delegate Task**\n"
                    "Usage: `!delegate <agent> <task>`\n\n"
                    "This hands off work to another agent who takes ownership.\n"
                    "Example: `!delegate backtest Run validation on the new momentum strategy`"
                )
                return

            target = target_agent.lower()
            valid_agents = ["tuning", "backtest", "risk", "strategy", "data"]

            if target not in valid_agents:
                await ctx.reply(f"Unknown agent: `{target}`. Valid: {', '.join(valid_agents)}")
                return

            if not self._communicator:
                await ctx.reply("Communication system not initialized.")
                return

            # Send handoff
            inter_msg = await self._communicator.send_handoff(
                to_agent=target,
                task=task,
                reason=f"Delegated by {self.agent_name}",
                context="",
                channel_name="ralph_team"
            )

            if inter_msg:
                await ctx.reply(
                    f"Task delegated to **{target.title()} Agent**\n"
                    f"Handoff ID: `{inter_msg.message_id}`"
                )
            else:
                await ctx.reply(f"Failed to delegate. Is **{target.title()} Agent** online?")

        @self.bot.command(name="pending")
        async def pending_tasks(ctx: commands.Context):
            """Show pending tasks from other agents."""
            pending = self._bot_registry.get_pending_tasks_for(self.agent_type)

            if not pending:
                await ctx.reply("No pending tasks from other agents.")
                return

            embed = discord.Embed(
                title=f"Pending Tasks for {self.agent_name}",
                color=discord.Color.gold()
            )

            for task in pending[:10]:  # Show max 10
                from_bot = self._bot_registry.get_bot(task.from_agent)
                from_name = from_bot.agent_name if from_bot else task.from_agent

                embed.add_field(
                    name=f"{task.message_id} from {from_name}",
                    value=task.content[:100] + ("..." if len(task.content) > 100 else ""),
                    inline=False
                )

            await ctx.reply(embed=embed)

        # =====================================================================
        # ORCHESTRATION COMMANDS (Cost Optimization)
        # =====================================================================

        @self.bot.command(name="orch_stats")
        async def orchestration_stats(ctx: commands.Context):
            """Show orchestration layer statistics (token savings)."""
            if not self._orchestrator:
                await ctx.reply("Orchestration layer not initialized.")
                return

            report = self._orchestrator.get_stats_report()
            await ctx.reply(report)

        @self.bot.command(name="orch_provider")
        async def orchestration_provider(ctx: commands.Context):
            """Show which LLM provider is being used for orchestration."""
            if not self._orchestrator:
                await ctx.reply("Orchestration layer not initialized.")
                return

            provider = self._orchestrator.provider.value
            model = self._orchestrator.model or "local patterns only"

            embed = discord.Embed(
                title="Orchestration Provider",
                color=discord.Color.blue()
            )
            embed.add_field(name="Provider", value=provider.upper(), inline=True)
            embed.add_field(name="Model", value=model, inline=True)
            embed.add_field(
                name="Purpose",
                value="Handles simple tasks cheaply, reserves Claude Code for complex work",
                inline=False
            )

            await ctx.reply(embed=embed)

        @self.bot.command(name="classify")
        async def classify_task(ctx: commands.Context, *, task: str = None):
            """Classify a task to see how it would be routed."""
            if not task:
                await ctx.reply("Usage: `!classify <task description>`")
                return

            if not self._orchestrator:
                await ctx.reply("Orchestration layer not initialized.")
                return

            result = await self._orchestrator.classify_task(task)

            embed = discord.Embed(
                title="Task Classification",
                color=discord.Color.blue()
            )
            embed.add_field(name="Task", value=task[:200], inline=False)
            embed.add_field(name="Complexity", value=result.complexity.value.upper(), inline=True)
            embed.add_field(name="Confidence", value=f"{result.confidence:.0%}", inline=True)
            embed.add_field(name="Reasoning", value=result.reasoning, inline=False)

            if result.suggested_agent:
                embed.add_field(name="Suggested Agent", value=result.suggested_agent.title(), inline=True)

            if result.can_handle_locally:
                embed.add_field(name="Handling", value="Can be handled locally (no LLM needed)", inline=False)
            elif result.complexity == TaskComplexity.SIMPLE:
                embed.add_field(name="Handling", value="Will use cheap LLM (GPT-4o-mini/Haiku)", inline=False)
            else:
                embed.add_field(name="Handling", value="Will use Claude Code", inline=False)

            await ctx.reply(embed=embed)

    async def _post_task_result(self, channel, result: TaskResult):
        """Post task execution result to Discord."""
        # Determine color based on status
        color = discord.Color.green() if result.status == TaskStatus.COMPLETED else discord.Color.red()

        embed = discord.Embed(
            title=f"Task {result.task_id}: {result.status.value.upper()}",
            color=color,
            timestamp=datetime.utcnow()
        )

        # Truncate output for Discord (max 1024 per field)
        output = result.output[:1000] + "..." if len(result.output) > 1000 else result.output

        if output:
            embed.add_field(name="Output", value=f"```\n{output}\n```", inline=False)

        if result.error:
            error = result.error[:500] + "..." if len(result.error) > 500 else result.error
            embed.add_field(name="Error", value=f"```\n{error}\n```", inline=False)

        embed.add_field(name="Duration", value=f"{result.duration_seconds:.1f}s", inline=True)
        embed.set_footer(text=self.agent_name)

        await channel.send(embed=embed)

    async def execute_task(
        self,
        task: str,
        context: Optional[str] = None,
        notify_channel: bool = True,
        force_claude: bool = False
    ) -> TaskResult:
        """
        Execute a task with tiered LLM orchestration.

        Flow:
        1. Orchestrator classifies the task (cheap/free)
        2. If simple: handled by cheap LLM (GPT-4o-mini/Haiku)
        3. If complex: escalate to Claude Code with summarized context

        Args:
            task: The task description
            context: Additional context from other agents
            notify_channel: Whether to post updates to Discord
            force_claude: Skip orchestration and go straight to Claude Code

        Returns:
            TaskResult with output and status
        """
        task_id = f"{self.agent_type[:3].upper()}-{datetime.utcnow().strftime('%H%M%S')}"

        # Get context from coordinator if available
        if not context and self._coordinator:
            related_agents = []
            for trigger, targets in HANDOFF_RULES.get(self.agent_type, {}).items():
                if "*" in targets:
                    related_agents = ["tuning", "backtest", "risk", "strategy", "data"]
                    break
                related_agents.extend(targets)
            context = self._coordinator.get_context_for(self.agent_type, related_agents)

        # =========================================================
        # TIERED ORCHESTRATION: Try cheap model first
        # =========================================================
        if not force_claude and self._orchestrator:
            try:
                orch_result = await self._orchestrator.process_incoming(
                    task=task,
                    from_agent=self.agent_type,
                    context=context or ""
                )

                # If orchestrator handled it, return immediately (saves Claude tokens!)
                if orch_result.handled:
                    self.logger.info(f"Task handled by orchestrator (saved Claude tokens)")

                    if notify_channel and orch_result.response:
                        await self.post_to_primary_channel(
                            f"Task `{task_id}` handled (orchestrated):\n{orch_result.response[:500]}"
                        )

                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        output=orch_result.response or "Task handled by orchestrator",
                        error=None,
                        duration_seconds=0.0
                    )

                # Use summarized context if provided (reduces Claude tokens)
                if orch_result.summarized_context:
                    self.logger.info(f"Using summarized context (saved ~{len(context or '') - len(orch_result.summarized_context)} chars)")
                    context = orch_result.summarized_context

            except Exception as e:
                self.logger.warning(f"Orchestration failed, falling back to Claude: {e}")

        # =========================================================
        # COMPLEX TASK: Use Claude Code
        # =========================================================
        if not self._executor:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                output="",
                error="Executor not initialized"
            )

        self.running_tasks[task_id] = task[:100]

        if notify_channel:
            await self.post_to_primary_channel(
                f"Starting task `{task_id}` (Claude Code):\n```\n{task[:300]}\n```"
            )

        # Execute with Claude Code
        result = await self._executor.execute(
            agent_name=self.agent_name,
            agent_role=self.agent_role,
            task_prompt=task,
            context=context
        )

        # Update tracking
        del self.running_tasks[task_id]
        self.completed_tasks.append({
            "task_id": result.task_id,
            "status": result.status.value,
            "duration": result.duration_seconds
        })

        # Store output for other agents
        if self._coordinator and result.status == TaskStatus.COMPLETED:
            self._coordinator.store_context(self.agent_type, result.output)

        if notify_channel:
            await self.post_to_primary_channel(
                f"Task `{result.task_id}` {result.status.value}: {result.duration_seconds:.1f}s"
            )

        return result

    async def trigger_handoff(self, target_agent: str, task: str, context: str = ""):
        """
        Trigger an automatic handoff to another agent.
        Used for autonomous workflow progression.
        """
        if self._coordinator:
            await self._coordinator.queue_handoff(
                from_agent=self.agent_type,
                to_agent=target_agent,
                task=task,
                context=context or self._coordinator.context_store.get(self.agent_type, "")
            )

            await self.post_to_team_channel(
                f"**{self.agent_name}** → **{target_agent.title()} Agent**\n"
                f"*Auto-handoff*: {task[:200]}"
            )

    async def _post_to_channel(self, channel_name: str, content: str) -> Optional[discord.Message]:
        """Post a message to a specific channel by name."""
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            self.logger.warning(f"Guild {self.guild_id} not found")
            return None

        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if not channel:
            self.logger.warning(f"Channel #{channel_name} not found")
            return None

        try:
            return await channel.send(content)
        except discord.HTTPException as e:
            self.logger.error(f"Failed to send message to #{channel_name}: {e}")
            return None

    async def post_to_primary_channel(self, content: str) -> Optional[discord.Message]:
        """Post a message to this agent's primary channel."""
        return await self._post_to_channel(self.primary_channel_name, content)

    async def post_to_team_channel(self, content: str) -> Optional[discord.Message]:
        """Post a message to the #ralph_team channel."""
        return await self._post_to_channel("ralph_team", content)

    async def mention_agent(self, agent_role_name: str, message: str):
        """
        Mention another agent in a message.

        Note: This mentions by role name. Agents should have corresponding roles.
        If you want to @mention the bot user directly, use the bot's user ID.
        """
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return

        # Try to find a role matching the agent name
        role = discord.utils.get(guild.roles, name=agent_role_name)
        if role:
            await self.post_to_team_channel(f"{role.mention} {message}")
        else:
            # Fallback: just post with the name
            await self.post_to_team_channel(f"**@{agent_role_name}** {message}")

    async def create_thread(
        self,
        channel_name: str,
        thread_name: str,
        initial_message: str
    ) -> Optional[discord.Thread]:
        """Create a thread in a channel for focused discussion."""
        guild = self.bot.get_guild(self.guild_id)
        if not guild:
            return None

        channel = discord.utils.get(guild.text_channels, name=channel_name)
        if not channel:
            self.logger.warning(f"Channel #{channel_name} not found for thread creation")
            return None

        try:
            # Create message first, then thread from it
            message = await channel.send(f"**Thread: {thread_name}**\n{initial_message}")
            thread = await message.create_thread(name=thread_name)
            return thread
        except discord.HTTPException as e:
            self.logger.error(f"Failed to create thread: {e}")
            return None

    async def run(self):
        """Start the bot with automatic reconnection."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self.logger.info(f"Starting {self.agent_name}...")
                await self.bot.start(self.token)
            except discord.LoginFailure:
                self.logger.critical("Invalid token! Check your .env file.")
                raise
            except discord.HTTPException as e:
                self.logger.error(f"HTTP error: {e}")
                self.reconnect_attempts += 1
                delay = self.reconnect_delay * (2 ** self.reconnect_attempts)  # Exponential backoff
                self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(delay)
            except Exception as e:
                self.logger.exception(f"Unexpected error: {e}")
                self.reconnect_attempts += 1
                delay = self.reconnect_delay * (2 ** self.reconnect_attempts)
                self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                await asyncio.sleep(delay)

        self.logger.critical(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Shutting down.")

    async def shutdown(self):
        """Gracefully shutdown the bot."""
        self.logger.info("Shutting down...")
        await self._post_to_channel("bot-logs", f"**{self.agent_name}** is going offline.")
        await self.bot.close()

    # ==========================================
    # Abstract methods - must be implemented by subclasses
    # ==========================================

    @abstractmethod
    async def on_agent_ready(self):
        """Called when the agent is ready. Override in subclass."""
        pass

    @abstractmethod
    async def on_agent_message(self, message: discord.Message):
        """
        Handle incoming messages specific to this agent.
        Called after command processing.
        Override in subclass.
        """
        pass

    @abstractmethod
    async def get_status(self) -> dict:
        """
        Return agent-specific status information.
        Override in subclass.
        """
        pass

    @abstractmethod
    async def get_commands(self) -> dict:
        """
        Return dict of agent-specific commands: {command_name: description}
        Override in subclass.
        """
        pass
