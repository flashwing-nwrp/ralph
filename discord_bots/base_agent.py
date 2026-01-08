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
from agent_prompts import AGENT_ROLES, HANDOFF_RULES
from vps_deployer import get_deployer, DeploymentStatus


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

        # Get role from prompts
        self.agent_role = AGENT_ROLES.get(agent_type, description)

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

            # Process commands first
            await self.bot.process_commands(message)

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
                    "`!help` - Show this message"
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
            await ctx.reply("Broadcast sent to #ralph-team")

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
        notify_channel: bool = True
    ) -> TaskResult:
        """
        Execute a task using Claude Code.

        Args:
            task: The task description
            context: Additional context from other agents
            notify_channel: Whether to post updates to Discord

        Returns:
            TaskResult with output and status
        """
        if not self._executor:
            return TaskResult(
                task_id="ERROR",
                status=TaskStatus.FAILED,
                output="",
                error="Executor not initialized"
            )

        # Get context from coordinator if available
        if not context and self._coordinator:
            # Get relevant context based on handoff rules
            related_agents = []
            for trigger, targets in HANDOFF_RULES.get(self.agent_type, {}).items():
                if "*" in targets:
                    related_agents = ["tuning", "backtest", "risk", "strategy", "data"]
                    break
                related_agents.extend(targets)
            context = self._coordinator.get_context_for(self.agent_type, related_agents)

        # Track the task
        task_id = f"{self.agent_type[:3].upper()}-{datetime.utcnow().strftime('%H%M%S')}"
        self.running_tasks[task_id] = task[:100]

        if notify_channel:
            await self.post_to_primary_channel(
                f"Starting task `{task_id}`:\n```\n{task[:300]}\n```"
            )

        # Execute
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
        """Post a message to the #ralph-team channel."""
        return await self._post_to_channel("ralph-team", content)

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
