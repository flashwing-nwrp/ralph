"""
Discord Embed Styling for RALPH Agent Ensemble

Professional, clean embeds for showcasing autonomous AI agent collaboration.
Designed to impress executives and demonstrate sophisticated AI orchestration.
"""

import discord
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class AgentStyle(Enum):
    """Visual styling for each agent type."""
    STRATEGY = ("strategy", 0x9B59B6, "chess_pawn", "The Visionary")      # Purple
    TUNING = ("tuning", 0x3498DB, "wrench", "The Perfectionist")          # Blue
    BACKTEST = ("backtest", 0xE67E22, "bar_chart", "The Skeptic")         # Orange
    RISK = ("risk", 0xE74C3C, "shield", "The Guardian")                   # Red
    DATA = ("data", 0x2ECC71, "floppy_disk", "The Librarian")             # Green
    ORCHESTRATOR = ("orchestrator", 0xF1C40F, "robot", "RALPH")           # Gold


# Agent emoji mapping (using Discord-compatible emoji)
AGENT_EMOJI = {
    "strategy": "\u265F",    # Chess pawn
    "tuning": "\u2699",      # Gear
    "backtest": "\U0001F4CA", # Bar chart
    "risk": "\U0001F6E1",    # Shield
    "data": "\U0001F4BE",    # Floppy disk
    "orchestrator": "\U0001F916", # Robot
}

# Status emoji
STATUS_EMOJI = {
    "pending": "\u23F3",     # Hourglass
    "in_progress": "\U0001F504", # Arrows rotating
    "completed": "\u2705",   # Green check
    "failed": "\u274C",      # Red X
    "warning": "\u26A0",     # Warning
}


def get_agent_color(agent_type: str) -> int:
    """Get the color for an agent type."""
    for style in AgentStyle:
        if style.value[0] == agent_type.lower():
            return style.value[1]
    return 0x95A5A6  # Default gray


def get_agent_emoji(agent_type: str) -> str:
    """Get the emoji for an agent type."""
    return AGENT_EMOJI.get(agent_type.lower(), "\U0001F916")


def get_agent_title(agent_type: str) -> str:
    """Get the personality title for an agent."""
    for style in AgentStyle:
        if style.value[0] == agent_type.lower():
            return style.value[3]
    return "Agent"


class RALPHEmbeds:
    """
    Professional Discord embed builder for RALPH system.

    Creates consistent, beautiful embeds that showcase
    the autonomous AI agent collaboration.
    """

    @staticmethod
    def mission_start(
        mission_objective: str,
        initiated_by: str = "Operator"
    ) -> discord.Embed:
        """Create embed for mission initiation."""
        embed = discord.Embed(
            title="\U0001F680 New Mission Initiated",
            description=f"**Objective:**\n{mission_objective}",
            color=0xF1C40F,  # Gold
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="Status",
            value="\U0001F504 Analyzing codebase...",
            inline=True
        )
        embed.add_field(
            name="Initiated By",
            value=initiated_by,
            inline=True
        )
        embed.set_footer(text="RALPH Autonomous Agent Ensemble")
        return embed

    @staticmethod
    def file_discovery(
        files: List[str],
        total_chars: int = 0
    ) -> discord.Embed:
        """Create embed for file discovery phase."""
        # Group files by directory
        file_list = "\n".join([f"`{f}`" for f in files[:15]])

        embed = discord.Embed(
            title="\U0001F4C2 Codebase Analysis",
            description=f"Discovered **{len(files)}** relevant files for analysis",
            color=0x3498DB,  # Blue
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="Files Analyzed",
            value=file_list if file_list else "No files found",
            inline=False
        )
        if total_chars > 0:
            embed.add_field(
                name="Context Size",
                value=f"{total_chars:,} characters",
                inline=True
            )
        embed.set_footer(text="RALPH | File Discovery Phase")
        return embed

    @staticmethod
    def thinking(
        agent_type: str,
        message: str = "Analyzing and planning..."
    ) -> discord.Embed:
        """Create embed for agent thinking/processing."""
        emoji = get_agent_emoji(agent_type)
        color = get_agent_color(agent_type)
        title = get_agent_title(agent_type)

        embed = discord.Embed(
            title=f"{emoji} {agent_type.title()} Agent",
            description=f"*\"{title}\"*\n\n\U0001F504 {message}",
            color=color,
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="Processing...")
        return embed

    @staticmethod
    def task_breakdown(
        tasks: List[Dict[str, str]],
        mission_objective: str
    ) -> discord.Embed:
        """Create embed for mission task breakdown."""
        # Group tasks by agent
        tasks_by_agent: Dict[str, List[str]] = {}
        for task in tasks:
            agent = task.get("agent", "unknown").lower()
            task_desc = task.get("task", "No description")[:100]
            if agent not in tasks_by_agent:
                tasks_by_agent[agent] = []
            tasks_by_agent[agent].append(task_desc)

        embed = discord.Embed(
            title="\U0001F4CB Mission Plan Created",
            description=f"**{len(tasks)}** tasks identified for: *{mission_objective[:100]}*",
            color=0x9B59B6,  # Purple
            timestamp=datetime.utcnow()
        )

        # Add fields for each agent's tasks
        for agent_type, agent_tasks in tasks_by_agent.items():
            emoji = get_agent_emoji(agent_type)
            task_list = "\n".join([f"\u2022 {t}..." for t in agent_tasks[:3]])
            if len(agent_tasks) > 3:
                task_list += f"\n*+{len(agent_tasks) - 3} more*"

            embed.add_field(
                name=f"{emoji} {agent_type.title()} ({len(agent_tasks)})",
                value=task_list or "No tasks",
                inline=False
            )

        embed.set_footer(text="RALPH | Mission Planning Complete")
        return embed

    @staticmethod
    def task_list_detailed(
        tasks: List[Dict[str, str]]
    ) -> List[discord.Embed]:
        """Create detailed task list embeds (one per agent type)."""
        embeds = []

        # Group tasks by agent
        tasks_by_agent: Dict[str, List[str]] = {}
        for task in tasks:
            agent = task.get("agent", "unknown").lower()
            if agent not in tasks_by_agent:
                tasks_by_agent[agent] = []
            tasks_by_agent[agent].append(task.get("task", "No description"))

        for agent_type, agent_tasks in tasks_by_agent.items():
            emoji = get_agent_emoji(agent_type)
            color = get_agent_color(agent_type)
            title = get_agent_title(agent_type)

            embed = discord.Embed(
                title=f"{emoji} {agent_type.title()} Agent Tasks",
                description=f"*\"{title}\"* - {len(agent_tasks)} tasks assigned",
                color=color,
                timestamp=datetime.utcnow()
            )

            for i, task_desc in enumerate(agent_tasks[:5], 1):
                # Truncate long descriptions at word boundary (Discord limit is 1024)
                max_len = 500
                if len(task_desc) > max_len:
                    # Find last space before limit to avoid cutting mid-word
                    truncate_at = task_desc.rfind(' ', 0, max_len)
                    if truncate_at == -1:
                        truncate_at = max_len
                    task_desc = task_desc[:truncate_at] + "..."
                embed.add_field(
                    name=f"Task {i}",
                    value=task_desc,
                    inline=False
                )

            if len(agent_tasks) > 5:
                embed.add_field(
                    name="Additional Tasks",
                    value=f"*+{len(agent_tasks) - 5} more tasks*",
                    inline=False
                )

            embed.set_footer(text=f"RALPH | {agent_type.title()} Agent")
            embeds.append(embed)

        return embeds

    @staticmethod
    def agent_working(
        agent_type: str,
        task_description: str,
        task_id: str = None,
        handoff_from: str = None
    ) -> discord.Embed:
        """Create embed for agent working on a task."""
        emoji = get_agent_emoji(agent_type)
        color = get_agent_color(agent_type)
        title = get_agent_title(agent_type)

        # Include handoff info in title if present
        if handoff_from:
            from_emoji = get_agent_emoji(handoff_from)
            embed_title = f"{from_emoji} → {emoji} {agent_type.title()} Agent Working"
        else:
            embed_title = f"{emoji} {agent_type.title()} Agent Working"

        embed = discord.Embed(
            title=embed_title,
            description=f'*\"{title}\"*',
            color=color,
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="\U0001F504 Current Task",
            value=task_description[:500] if task_description else "Processing...",
            inline=False
        )
        if task_id:
            embed.add_field(
                name="Task ID",
                value=f"`{task_id}`",
                inline=True
            )
        embed.add_field(
            name="Status",
            value="\U0001F7E1 In Progress",
            inline=True
        )
        embed.set_footer(text="RALPH | Task Execution")
        return embed

    @staticmethod
    def agent_complete(
        agent_type: str,
        task_description: str,
        result_summary: str,
        duration_seconds: float = 0,
        task_id: str = None
    ) -> discord.Embed:
        """Create embed for agent task completion with support for longer outputs."""
        # For backwards compatibility, return single embed (truncated)
        embeds = RALPHEmbeds.agent_complete_chunked(
            agent_type, task_description, result_summary, duration_seconds, task_id
        )
        return embeds[0] if embeds else discord.Embed(title="Task Complete")

    @staticmethod
    def agent_complete_chunked(
        agent_type: str,
        task_description: str,
        result_summary: str,
        duration_seconds: float = 0,
        task_id: str = None,
        max_chunk_size: int = 3500
    ) -> list:
        """
        Create list of embeds for agent task completion, chunked for Discord limits.

        Returns multiple embeds for long outputs to avoid truncation.
        """
        color = get_agent_color(agent_type)
        title = get_agent_title(agent_type)

        # Build header with task info
        header = f"**Task:** {task_description[:500]}" if task_description else ""
        if task_id:
            header = f"`{task_id}` | {header}"

        summary = result_summary or "Success"
        embeds = []

        # Calculate how much space we have after header
        header_len = len(header) + 4  # +4 for "\n\n"

        # Split summary into chunks
        remaining = summary
        chunk_num = 0

        while remaining:
            chunk_num += 1
            is_first = chunk_num == 1

            # First chunk has header, subsequent chunks have continuation title
            if is_first:
                available = max_chunk_size - header_len
            else:
                available = max_chunk_size

            if len(remaining) <= available:
                # Last chunk
                chunk = remaining
                remaining = ""
            else:
                # Find a good break point
                chunk = remaining[:available]
                break_points = ["\n\n", "\n", ". ", ", ", " "]
                for bp in break_points:
                    last_break = chunk.rfind(bp)
                    if last_break > available * 0.7:  # At least 70% of available space
                        chunk = chunk[:last_break + len(bp)]
                        break

                remaining = remaining[len(chunk):]

            # Create embed
            if is_first:
                embed = discord.Embed(
                    title=f"✅ {title} Complete",
                    description=f"{header}\n\n{chunk}",
                    color=color,
                    timestamp=datetime.utcnow()
                )
                # Add duration to first embed only
                if duration_seconds > 0:
                    embed.add_field(
                        name="⏱️ Duration",
                        value=f"{duration_seconds:.1f}s",
                        inline=True
                    )
            else:
                embed = discord.Embed(
                    title=f"📄 {title} (continued {chunk_num})",
                    description=chunk,
                    color=color
                )

            embed.set_footer(text=f"RALPH | {agent_type.title()} Agent")
            embeds.append(embed)

        return embeds if embeds else [discord.Embed(title="Task Complete")]

    @staticmethod
    def agent_error(
        agent_type: str,
        task_description: str,
        error_message: str,
        task_id: str = None
    ) -> discord.Embed:
        """Create embed for agent task failure."""
        emoji = get_agent_emoji(agent_type)

        embed = discord.Embed(
            title=f"{emoji} {agent_type.title()} Agent Error",
            color=0xE74C3C,  # Red
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="\u274C Task",
            value=task_description[:500] if task_description else "Task failed",
            inline=False
        )
        embed.add_field(
            name="Error",
            value=f"```\n{error_message[:800]}\n```" if error_message else "Unknown error",
            inline=False
        )
        if task_id:
            embed.add_field(
                name="Task ID",
                value=f"`{task_id}`",
                inline=True
            )
        embed.set_footer(text="RALPH | Error")
        return embed

    @staticmethod
    def handoff(
        from_agent: str,
        to_agent: str,
        task_description: str
    ) -> discord.Embed:
        """Create embed for agent-to-agent handoff."""
        from_emoji = get_agent_emoji(from_agent)
        to_emoji = get_agent_emoji(to_agent)
        to_color = get_agent_color(to_agent)

        embed = discord.Embed(
            title="\U0001F91D Agent Handoff",
            description=f"{from_emoji} **{from_agent.title()}** \u2192 {to_emoji} **{to_agent.title()}**",
            color=to_color,
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="Task",
            value=task_description[:500] if task_description else "Handoff task",
            inline=False
        )
        embed.set_footer(text="RALPH | Autonomous Collaboration")
        return embed

    @staticmethod
    def mission_progress(
        completed: int,
        total: int,
        current_agent: str = None,
        current_task: str = None
    ) -> discord.Embed:
        """Create embed for mission progress update."""
        # Create progress bar
        progress_pct = (completed / total * 100) if total > 0 else 0
        filled = int(progress_pct / 10)
        bar = "\u2588" * filled + "\u2591" * (10 - filled)

        embed = discord.Embed(
            title="\U0001F4CA Mission Progress",
            description=f"**{completed}/{total}** tasks complete ({progress_pct:.0f}%)\n\n`{bar}`",
            color=0xF1C40F,  # Gold
            timestamp=datetime.utcnow()
        )

        if current_agent and current_task:
            emoji = get_agent_emoji(current_agent)
            embed.add_field(
                name=f"{emoji} Currently Active",
                value=f"**{current_agent.title()}**: {current_task[:100]}...",
                inline=False
            )

        embed.set_footer(text="RALPH | Mission In Progress")
        return embed

    @staticmethod
    def mission_complete(
        mission_objective: str,
        total_tasks: int,
        duration_seconds: float,
        tasks_by_agent: Dict[str, int] = None
    ) -> discord.Embed:
        """Create embed for mission completion."""
        embed = discord.Embed(
            title="🏆 Mission Complete",
            description=f"**Objective:** {mission_objective[:200]}",
            color=0x2ECC71,  # Green
            timestamp=datetime.utcnow()
        )

        # Summary stats
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        embed.add_field(
            name="Tasks Completed",
            value=f"✅ **{total_tasks}**",
            inline=True
        )
        embed.add_field(
            name="Total Duration",
            value=f"⏱ **{time_str}**",
            inline=True
        )

        # Breakdown by agent
        if tasks_by_agent:
            breakdown = []
            for agent, count in tasks_by_agent.items():
                emoji = get_agent_emoji(agent)
                breakdown.append(f"{emoji} {agent.title()}: {count}")
            embed.add_field(
                name="Task Distribution",
                value="\n".join(breakdown),
                inline=False
            )

        embed.set_footer(text="RALPH | Mission Successful")
        return embed

    @staticmethod
    def executive_summary(
        mission_id: str,
        mission_objective: str,
        total_tasks: int,
        duration_minutes: float,
        key_findings: List[str],
        work_summary: str,
        suggestions: List[str],
        owner_mention: str = None
    ) -> List[discord.Embed]:
        """
        Create executive summary embeds for mission completion.
        Returns multiple embeds to handle Discord's character limits.
        """
        embeds = []

        # Main summary embed
        main_embed = discord.Embed(
            title=f"📋 Executive Summary: {mission_id}",
            description=f"{'@' + owner_mention + ' ' if owner_mention else ''}**Mission Complete!**\n\n{mission_objective[:300]}",
            color=0xFFD700,  # Gold
            timestamp=datetime.utcnow()
        )

        main_embed.add_field(
            name="📊 Stats",
            value=f"✅ **{total_tasks}** tasks completed\n⏱️ **{duration_minutes:.1f}** minutes total",
            inline=True
        )

        # Work performed summary
        if work_summary:
            main_embed.add_field(
                name="🔨 Work Performed",
                value=work_summary[:1000],
                inline=False
            )

        embeds.append(main_embed)

        # Key findings embed
        if key_findings:
            findings_embed = discord.Embed(
                title="🔍 Key Findings",
                color=0x3498DB,  # Blue
            )
            findings_text = "\n".join([f"• {f[:200]}" for f in key_findings[:8]])
            findings_embed.description = findings_text[:4000]
            embeds.append(findings_embed)

        # Suggestions embed
        if suggestions:
            suggestions_embed = discord.Embed(
                title="💡 Suggested Next Steps",
                description="Based on this mission's results, consider:",
                color=0x9B59B6,  # Purple
            )
            for i, suggestion in enumerate(suggestions[:5], 1):
                suggestions_embed.add_field(
                    name=f"Suggestion {i}",
                    value=suggestion[:500],
                    inline=False
                )
            suggestions_embed.set_footer(text="These are AI-generated suggestions. Use your judgment.")
            embeds.append(suggestions_embed)

        return embeds

    @staticmethod
    def retrospective(
        mission_id: str,
        what_went_well: List[str],
        what_could_improve: List[str],
        learnings: List[str],
        action_items: List[str]
    ) -> discord.Embed:
        """Create retrospective embed for mission review."""
        embed = discord.Embed(
            title=f"🔄 Retrospective: {mission_id}",
            description="Team reflection and continuous improvement",
            color=0xE67E22,  # Orange
            timestamp=datetime.utcnow()
        )

        if what_went_well:
            embed.add_field(
                name="✅ What Went Well",
                value="\n".join([f"• {w[:100]}" for w in what_went_well[:4]]),
                inline=False
            )

        if what_could_improve:
            embed.add_field(
                name="🔧 Areas for Improvement",
                value="\n".join([f"• {w[:100]}" for w in what_could_improve[:4]]),
                inline=False
            )

        if learnings:
            embed.add_field(
                name="📚 Key Learnings",
                value="\n".join([f"• {l[:100]}" for l in learnings[:4]]),
                inline=False
            )

        if action_items:
            embed.add_field(
                name="📝 Action Items",
                value="\n".join([f"• {a[:100]}" for a in action_items[:4]]),
                inline=False
            )

        embed.set_footer(text="Learnings saved to knowledge base")
        return embed

    @staticmethod
    def mission_failed(
        mission_objective: str,
        error_message: str,
        completed_tasks: int = 0,
        total_tasks: int = 0
    ) -> discord.Embed:
        """Create embed for mission failure."""
        embed = discord.Embed(
            title="\u274C Mission Failed",
            description=f"**Objective:** {mission_objective[:200]}",
            color=0xE74C3C,  # Red
            timestamp=datetime.utcnow()
        )

        if total_tasks > 0:
            embed.add_field(
                name="Progress",
                value=f"{completed_tasks}/{total_tasks} tasks completed",
                inline=True
            )

        embed.add_field(
            name="Error",
            value=f"```\n{error_message[:500]}\n```",
            inline=False
        )

        embed.set_footer(text="RALPH | Mission Aborted")
        return embed

    @staticmethod
    def system_status(
        agents_online: List[str],
        agents_offline: List[str] = None,
        active_missions: int = 0
    ) -> discord.Embed:
        """Create embed for system status overview."""
        embed = discord.Embed(
            title="\U0001F916 RALPH System Status",
            description="Autonomous Agent Ensemble",
            color=0xF1C40F,  # Gold
            timestamp=datetime.utcnow()
        )

        # Online agents
        online_list = []
        for agent in agents_online:
            emoji = get_agent_emoji(agent)
            title = get_agent_title(agent)
            online_list.append(f"{emoji} **{agent.title()}** - *{title}*")

        embed.add_field(
            name=f"\U0001F7E2 Online ({len(agents_online)})",
            value="\n".join(online_list) if online_list else "None",
            inline=False
        )

        if agents_offline:
            offline_list = [f"\U0001F534 {a.title()}" for a in agents_offline]
            embed.add_field(
                name=f"\U0001F534 Offline ({len(agents_offline)})",
                value="\n".join(offline_list),
                inline=False
            )

        embed.add_field(
            name="Active Missions",
            value=str(active_missions),
            inline=True
        )

        embed.set_footer(text="RALPH | System Overview")
        return embed

    @staticmethod
    def claude_response(
        agent_type: str,
        response_text: str,
        duration_seconds: float = 0,
        truncated: bool = False
    ) -> discord.Embed:
        """Create embed for Claude Code response."""
        emoji = get_agent_emoji(agent_type)
        color = get_agent_color(agent_type)

        # Truncate response if needed
        if len(response_text) > 1800:
            response_text = response_text[:1800] + "\n\n*... (truncated)*"
            truncated = True

        embed = discord.Embed(
            title=f"{emoji} Claude Analysis",
            description=f"```\n{response_text}\n```",
            color=color,
            timestamp=datetime.utcnow()
        )

        if duration_seconds > 0:
            embed.add_field(
                name="Processing Time",
                value=f"{duration_seconds:.1f}s",
                inline=True
            )

        if truncated:
            embed.add_field(
                name="Note",
                value="Response was truncated for display",
                inline=True
            )

        embed.set_footer(text=f"RALPH | {agent_type.title()} Agent")
        return embed

    @staticmethod
    def experiment_proposal(
        experiment_id: str,
        title: str,
        description: str,
        risk_level: str = "low",
        steps: List[str] = None
    ) -> discord.Embed:
        """Create embed for experiment proposal from Innovation Loop."""
        risk_colors = {
            "low": 0x27AE60,      # Green
            "medium": 0xF39C12,   # Orange
            "high": 0xE74C3C      # Red
        }
        risk_emojis = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴"
        }

        color = risk_colors.get(risk_level, 0x95A5A6)
        risk_emoji = risk_emojis.get(risk_level, "⚪")

        embed = discord.Embed(
            title=f"🔬 Experiment Proposal: {experiment_id}",
            description=f"**{title}**\n\n{description}",
            color=color,
            timestamp=datetime.utcnow()
        )

        embed.add_field(
            name="Risk Level",
            value=f"{risk_emoji} {risk_level.title()}",
            inline=True
        )

        if steps:
            steps_text = "\n".join(f"• {step}" for step in steps[:5])
            embed.add_field(
                name="Implementation Steps",
                value=steps_text,
                inline=False
            )

        embed.add_field(
            name="Actions",
            value=f"Use `!approve_experiment {experiment_id}` to approve\n"
                  f"Use `!reject_experiment {experiment_id} <reason>` to reject",
            inline=False
        )

        embed.set_footer(text="RALPH | Innovation Loop")
        return embed

    @staticmethod
    def innovation_stats(
        cycles: int,
        anomalies: int,
        experiments_proposed: int,
        experiments_approved: int,
        pending: int
    ) -> discord.Embed:
        """Create embed for innovation loop statistics."""
        embed = discord.Embed(
            title="🔬 Innovation Loop Status",
            color=0x9B59B6,  # Purple
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="Cycles Run", value=str(cycles), inline=True)
        embed.add_field(name="Anomalies Detected", value=str(anomalies), inline=True)
        embed.add_field(name="Experiments Proposed", value=str(experiments_proposed), inline=True)
        embed.add_field(name="Experiments Approved", value=str(experiments_approved), inline=True)
        embed.add_field(name="Pending Approval", value=str(pending), inline=True)

        embed.set_footer(text="RALPH | Innovation Loop")
        return embed


# Convenience function for quick embed creation
def quick_embed(
    title: str,
    description: str,
    color: int = 0x95A5A6,
    fields: List[tuple] = None
) -> discord.Embed:
    """Create a quick embed with common formatting."""
    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=datetime.utcnow()
    )
    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)
    embed.set_footer(text="RALPH")
    return embed
