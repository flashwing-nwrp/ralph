"""
Discord Interactive Components for RALPH

Provides dropdown menus, buttons, and interactive views to make
the bot more user-friendly without requiring memorization of commands.

Features:
- Main menu with common actions
- Mission control (pause/resume/abort/add task)
- Approval workflows (improvements, proposals, experiments)
- Improvement management (deploy/test/rollback)
- Help category navigation
- Modals for complex input
- Pagination for lists

Usage:
    from discord_interactions import (
        MainMenuView, send_main_menu,
        MissionControlView, ApprovalView,
        ImprovementActionsView, HelpView
    )

    # In a command:
    await send_main_menu(ctx)
"""

import discord
from discord.ui import View, Select, Button, Modal, TextInput
from discord import SelectOption, ButtonStyle, Interaction
from typing import Optional, List, Callable, Any, Dict
import asyncio
import logging
from datetime import datetime

from discord_embeds import get_agent_emoji, get_agent_color

logger = logging.getLogger("discord_interactions")


# =============================================================================
# MAIN MENU
# =============================================================================

class MainMenuView(View):
    """Main menu with dropdown for common actions."""

    def __init__(self, timeout: float = 180):
        super().__init__(timeout=timeout)

    @discord.ui.select(
        placeholder="Choose an action...",
        options=[
            SelectOption(
                label="Sentiment Analysis",
                description="Get crypto news sentiment",
                emoji="📊",
                value="sentiment"
            ),
            SelectOption(
                label="Market Browser",
                description="Browse Polymarket markets",
                emoji="🏪",
                value="markets"
            ),
            SelectOption(
                label="Research Topic",
                description="Search the web for information",
                emoji="🔍",
                value="research"
            ),
            SelectOption(
                label="View Experiments",
                description="See pending innovation experiments",
                emoji="🧪",
                value="experiments"
            ),
            SelectOption(
                label="Mission Status",
                description="Check current mission progress",
                emoji="🎯",
                value="mission"
            ),
            SelectOption(
                label="System Stats",
                description="View parallel execution stats",
                emoji="📈",
                value="stats"
            ),
        ]
    )
    async def select_action(self, interaction: Interaction, select: Select):
        """Handle main menu selection."""
        action = select.values[0]

        if action == "sentiment":
            view = SentimentSelectView()
            await interaction.response.edit_message(
                content="**Select a cryptocurrency for sentiment analysis:**",
                view=view
            )

        elif action == "markets":
            view = MarketCategoryView()
            await interaction.response.edit_message(
                content="**Select a market category:**",
                view=view
            )

        elif action == "research":
            modal = ResearchModal()
            await interaction.response.send_modal(modal)

        elif action == "experiments":
            await self._show_experiments(interaction)

        elif action == "mission":
            await self._show_mission(interaction)

        elif action == "stats":
            await self._show_stats(interaction)

    async def _show_experiments(self, interaction: Interaction):
        """Show experiments with approval buttons."""
        try:
            from innovation_loop import get_innovation_loop

            loop = get_innovation_loop()
            if not loop:
                await interaction.response.edit_message(
                    content="Innovation loop not running. Enable with `INNOVATION_LOOP_ENABLED=true`",
                    view=None
                )
                return

            pending = [e for e in loop.experiments.values() if e.status.value == "proposed"]

            if not pending:
                await interaction.response.edit_message(
                    content="No pending experiments. The innovation loop proposes experiments every 5 minutes.",
                    view=BackToMenuView()
                )
                return

            # Show first experiment with approve/reject buttons
            exp = pending[0]
            view = ExperimentActionView(exp.id, len(pending))

            embed = discord.Embed(
                title=f"🧪 Experiment: {exp.id[:8]}",
                description=exp.hypothesis,
                color=0x9B59B6
            )
            embed.add_field(name="Type", value=exp.experiment_type, inline=True)
            embed.add_field(name="Pending", value=f"{len(pending)} total", inline=True)

            await interaction.response.edit_message(
                content=None,
                embed=embed,
                view=view
            )

        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error loading experiments: {str(e)[:100]}",
                view=BackToMenuView()
            )

    async def _show_mission(self, interaction: Interaction):
        """Show current mission status."""
        try:
            from mission_manager import get_mission_manager

            manager = get_mission_manager()
            if not manager or not manager.current_mission:
                await interaction.response.edit_message(
                    content="No active mission. Start one with `!mission <objective>`",
                    view=BackToMenuView()
                )
                return

            mission = manager.current_mission
            progress = manager.get_progress_percentage()

            embed = discord.Embed(
                title=f"🎯 Mission: {mission.objective[:50]}...",
                color=0x3498DB
            )

            # Progress bar
            filled = int(progress / 10)
            bar = "█" * filled + "░" * (10 - filled)
            embed.add_field(
                name="Progress",
                value=f"`{bar}` {progress:.0f}%",
                inline=False
            )

            # Task summary
            tasks = mission.tasks
            completed = sum(1 for t in tasks if t.status.value == "completed")
            in_progress = sum(1 for t in tasks if t.status.value == "in_progress")
            pending = sum(1 for t in tasks if t.status.value == "pending")

            embed.add_field(name="✅ Completed", value=str(completed), inline=True)
            embed.add_field(name="🔄 In Progress", value=str(in_progress), inline=True)
            embed.add_field(name="⏳ Pending", value=str(pending), inline=True)

            await interaction.response.edit_message(
                content=None,
                embed=embed,
                view=BackToMenuView()
            )

        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error loading mission: {str(e)[:100]}",
                view=BackToMenuView()
            )

    async def _show_stats(self, interaction: Interaction):
        """Show system statistics."""
        try:
            from autonomous_orchestrator import get_orchestrator

            orch = get_orchestrator()
            if not orch:
                await interaction.response.edit_message(
                    content="Orchestrator not available.",
                    view=BackToMenuView()
                )
                return

            embed = discord.Embed(
                title="📈 System Statistics",
                color=0x2ECC71
            )

            # Parallel execution stats
            if orch.parallel_tracker:
                stats = orch.parallel_tracker.get_stats()
                embed.add_field(
                    name="Parallel Execution",
                    value=f"Active: {stats['active_tasks']}\n"
                          f"Completed: {stats['completed_tasks']}\n"
                          f"Failed: {stats['failed_tasks']}",
                    inline=True
                )

            # Innovation loop stats
            if orch.innovation_loop:
                istats = orch.innovation_loop.stats
                embed.add_field(
                    name="Innovation Loop",
                    value=f"Cycles: {istats.get('cycles_run', 0)}\n"
                          f"Proposals: {istats.get('proposals_made', 0)}\n"
                          f"Approved: {istats.get('experiments_approved', 0)}",
                    inline=True
                )

            # Agent status
            active_agents = len([a for a in orch.agents.values() if a.bot.is_ready()])
            embed.add_field(
                name="Agents",
                value=f"Active: {active_agents}/{len(orch.agents)}",
                inline=True
            )

            await interaction.response.edit_message(
                content=None,
                embed=embed,
                view=BackToMenuView()
            )

        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error loading stats: {str(e)[:100]}",
                view=BackToMenuView()
            )


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

class SentimentSelectView(View):
    """Dropdown for selecting crypto symbol for sentiment."""

    def __init__(self, timeout: float = 120):
        super().__init__(timeout=timeout)

    @discord.ui.select(
        placeholder="Select cryptocurrency...",
        options=[
            SelectOption(label="Bitcoin (BTC)", value="BTC", emoji="🟠"),
            SelectOption(label="Ethereum (ETH)", value="ETH", emoji="🔷"),
            SelectOption(label="Solana (SOL)", value="SOL", emoji="🟣"),
            SelectOption(label="XRP", value="XRP", emoji="⚫"),
            SelectOption(label="Dogecoin (DOGE)", value="DOGE", emoji="🐕"),
            SelectOption(label="Cardano (ADA)", value="ADA", emoji="🔵"),
            SelectOption(label="Avalanche (AVAX)", value="AVAX", emoji="🔺"),
            SelectOption(label="Polygon (MATIC)", value="MATIC", emoji="🟪"),
        ]
    )
    async def select_symbol(self, interaction: Interaction, select: Select):
        """Fetch sentiment for selected symbol."""
        symbol = select.values[0]

        await interaction.response.edit_message(
            content=f"📊 Fetching sentiment for **{symbol}**...",
            view=None
        )

        try:
            from research_tools import get_research_tool

            tool = get_research_tool()
            result = await tool.get_crypto_sentiment(symbol)

            if result.error:
                await interaction.edit_original_response(
                    content=f"Error: {result.error}",
                    view=BackToMenuView()
                )
                return

            # Color based on sentiment
            if result.sentiment_label == "bullish":
                color = 0x2ECC71
                emoji = "🟢"
            elif result.sentiment_label == "bearish":
                color = 0xE74C3C
                emoji = "🔴"
            else:
                color = 0xF39C12
                emoji = "🟡"

            embed = discord.Embed(
                title=f"{emoji} {symbol} Sentiment: {result.sentiment_label.upper()}",
                color=color
            )

            # Sentiment gauge
            score_pct = (result.sentiment_score + 1) / 2 * 100
            gauge = "▓" * int(score_pct / 10) + "░" * (10 - int(score_pct / 10))
            embed.add_field(
                name="Score",
                value=f"`{gauge}` {result.sentiment_score:+.2f}",
                inline=False
            )

            embed.add_field(name="🟢 Bullish", value=str(result.bullish_count), inline=True)
            embed.add_field(name="🔴 Bearish", value=str(result.bearish_count), inline=True)
            embed.add_field(name="⚪ Neutral", value=str(result.neutral_count), inline=True)

            # Top headlines
            if result.headlines:
                headlines_text = ""
                for h in result.headlines[:3]:
                    sent = h.get("sentiment", 0)
                    icon = "🟢" if sent > 0.2 else ("🔴" if sent < -0.2 else "⚪")
                    title = h.get("title", "")[:50]
                    headlines_text += f"{icon} {title}...\n"
                embed.add_field(name="Headlines", value=headlines_text, inline=False)

            embed.set_footer(text=f"Source: {result.source}")

            await interaction.edit_original_response(
                content=None,
                embed=embed,
                view=SentimentActionsView(symbol)
            )

        except Exception as e:
            await interaction.edit_original_response(
                content=f"Error: {str(e)[:100]}",
                view=BackToMenuView()
            )


class SentimentActionsView(View):
    """Actions after viewing sentiment."""

    def __init__(self, current_symbol: str, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.current_symbol = current_symbol

    @discord.ui.button(label="Check Another", style=ButtonStyle.primary, emoji="🔄")
    async def check_another(self, interaction: Interaction, button: Button):
        view = SentimentSelectView()
        await interaction.response.edit_message(
            content="**Select a cryptocurrency:**",
            embed=None,
            view=view
        )

    @discord.ui.button(label="Main Menu", style=ButtonStyle.secondary, emoji="🏠")
    async def main_menu(self, interaction: Interaction, button: Button):
        view = MainMenuView()
        await interaction.response.edit_message(
            content="**RALPH Command Center** - Select an action:",
            embed=None,
            view=view
        )


# =============================================================================
# MARKET BROWSER
# =============================================================================

class MarketCategoryView(View):
    """Select market category to browse."""

    def __init__(self, timeout: float = 120):
        super().__init__(timeout=timeout)

    @discord.ui.select(
        placeholder="Select category...",
        options=[
            SelectOption(label="All Markets", value="all", emoji="🌐"),
            SelectOption(label="Crypto", value="crypto", emoji="₿"),
            SelectOption(label="Politics", value="politics", emoji="🏛️"),
            SelectOption(label="Sports", value="sports", emoji="⚽"),
            SelectOption(label="Entertainment", value="entertainment", emoji="🎬"),
            SelectOption(label="Science", value="science", emoji="🔬"),
        ]
    )
    async def select_category(self, interaction: Interaction, select: Select):
        """Load markets for category."""
        category = select.values[0]

        await interaction.response.edit_message(
            content=f"🔄 Loading {category} markets...",
            view=None
        )

        try:
            from polymarket.service import get_polymarket_service

            service = get_polymarket_service()
            markets = await service.get_active_markets(
                category=None if category == "all" else category,
                limit=10
            )

            if not markets:
                await interaction.edit_original_response(
                    content=f"No markets found in {category}.",
                    view=BackToMenuView()
                )
                return

            # Create market selector
            view = MarketSelectView(markets)

            embed = discord.Embed(
                title=f"🏪 {category.title()} Markets",
                description=f"Found {len(markets)} markets. Select one for details:",
                color=0x3498DB
            )

            await interaction.edit_original_response(
                content=None,
                embed=embed,
                view=view
            )

        except Exception as e:
            await interaction.edit_original_response(
                content=f"Error loading markets: {str(e)[:100]}",
                view=BackToMenuView()
            )


class MarketSelectView(View):
    """Select a specific market from list."""

    def __init__(self, markets: list, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.markets = {m.id: m for m in markets}

        # Build options from markets
        options = []
        for m in markets[:25]:  # Discord limit is 25 options
            label = m.question[:100] if len(m.question) <= 100 else m.question[:97] + "..."
            options.append(SelectOption(
                label=label[:100],
                value=m.id,
                description=f"Volume: ${m.volume:,.0f}" if hasattr(m, 'volume') else None
            ))

        self.select_market.options = options

    @discord.ui.select(placeholder="Select a market...")
    async def select_market(self, interaction: Interaction, select: Select):
        """Show market details."""
        market_id = select.values[0]
        market = self.markets.get(market_id)

        if not market:
            await interaction.response.edit_message(
                content="Market not found.",
                view=BackToMenuView()
            )
            return

        embed = discord.Embed(
            title=f"📊 {market.question[:200]}",
            color=0x9B59B6
        )

        if hasattr(market, 'outcomes') and market.outcomes:
            for outcome in market.outcomes[:5]:
                price = f"${outcome.price:.2f}" if hasattr(outcome, 'price') else "N/A"
                embed.add_field(
                    name=outcome.name[:50],
                    value=price,
                    inline=True
                )

        if hasattr(market, 'volume'):
            embed.add_field(name="Volume", value=f"${market.volume:,.0f}", inline=True)

        if hasattr(market, 'end_date'):
            embed.add_field(name="Closes", value=str(market.end_date)[:10], inline=True)

        await interaction.response.edit_message(
            content=None,
            embed=embed,
            view=MarketActionsView(market_id)
        )


class MarketActionsView(View):
    """Actions for a specific market."""

    def __init__(self, market_id: str, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.market_id = market_id

    @discord.ui.button(label="Browse More", style=ButtonStyle.primary, emoji="🔍")
    async def browse_more(self, interaction: Interaction, button: Button):
        view = MarketCategoryView()
        await interaction.response.edit_message(
            content="**Select a market category:**",
            embed=None,
            view=view
        )

    @discord.ui.button(label="Main Menu", style=ButtonStyle.secondary, emoji="🏠")
    async def main_menu(self, interaction: Interaction, button: Button):
        view = MainMenuView()
        await interaction.response.edit_message(
            content="**RALPH Command Center** - Select an action:",
            embed=None,
            view=view
        )


# =============================================================================
# RESEARCH MODAL
# =============================================================================

class ResearchModal(Modal, title="Research Topic"):
    """Modal for entering research query."""

    topic = TextInput(
        label="What would you like to research?",
        placeholder="e.g., prediction market strategies, crypto trading automation",
        max_length=200
    )

    async def on_submit(self, interaction: Interaction):
        query = self.topic.value

        await interaction.response.send_message(
            f"🔍 Researching: **{query}**...",
            ephemeral=False
        )

        try:
            from research_tools import get_research_tool

            tool = get_research_tool()
            result = await tool.research_topic(query, depth="medium")

            if result.error:
                await interaction.edit_original_response(
                    content=f"Error: {result.error}"
                )
                return

            embed = discord.Embed(
                title=f"🔍 Research: {query[:50]}",
                color=0x3498DB
            )

            if result.results:
                for r in result.results[:5]:
                    embed.add_field(
                        name=r.title[:60],
                        value=f"{r.snippet[:150]}...\n[Link]({r.url})",
                        inline=False
                    )

            embed.set_footer(text=f"Found {len(result.results)} sources")

            await interaction.edit_original_response(
                content=None,
                embed=embed
            )

        except Exception as e:
            await interaction.edit_original_response(
                content=f"Error: {str(e)[:100]}"
            )


# =============================================================================
# EXPERIMENT ACTIONS
# =============================================================================

class ExperimentActionView(View):
    """Approve/reject experiment with buttons."""

    def __init__(self, experiment_id: str, pending_count: int, timeout: float = 300):
        super().__init__(timeout=timeout)
        self.experiment_id = experiment_id
        self.pending_count = pending_count

    @discord.ui.button(label="Approve", style=ButtonStyle.success, emoji="✅")
    async def approve(self, interaction: Interaction, button: Button):
        try:
            from innovation_loop import get_innovation_loop

            loop = get_innovation_loop()
            if loop:
                success = await loop.approve_experiment(self.experiment_id)
                if success:
                    await interaction.response.edit_message(
                        content=f"✅ Experiment `{self.experiment_id[:8]}` approved and queued for execution.",
                        embed=None,
                        view=BackToMenuView()
                    )
                else:
                    await interaction.response.edit_message(
                        content="Failed to approve experiment.",
                        view=BackToMenuView()
                    )
        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error: {str(e)[:100]}",
                view=BackToMenuView()
            )

    @discord.ui.button(label="Reject", style=ButtonStyle.danger, emoji="❌")
    async def reject(self, interaction: Interaction, button: Button):
        try:
            from innovation_loop import get_innovation_loop

            loop = get_innovation_loop()
            if loop:
                success = await loop.reject_experiment(self.experiment_id, "Rejected via menu")
                if success:
                    await interaction.response.edit_message(
                        content=f"❌ Experiment `{self.experiment_id[:8]}` rejected.",
                        embed=None,
                        view=BackToMenuView()
                    )
                else:
                    await interaction.response.edit_message(
                        content="Failed to reject experiment.",
                        view=BackToMenuView()
                    )
        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error: {str(e)[:100]}",
                view=BackToMenuView()
            )

    @discord.ui.button(label="Skip", style=ButtonStyle.secondary, emoji="⏭️")
    async def skip(self, interaction: Interaction, button: Button):
        # Show next experiment
        try:
            from innovation_loop import get_innovation_loop

            loop = get_innovation_loop()
            if loop:
                pending = [e for e in loop.experiments.values()
                          if e.status.value == "proposed" and e.id != self.experiment_id]

                if pending:
                    exp = pending[0]
                    view = ExperimentActionView(exp.id, len(pending))

                    embed = discord.Embed(
                        title=f"🧪 Experiment: {exp.id[:8]}",
                        description=exp.hypothesis,
                        color=0x9B59B6
                    )
                    embed.add_field(name="Type", value=exp.experiment_type, inline=True)
                    embed.add_field(name="Remaining", value=str(len(pending)), inline=True)

                    await interaction.response.edit_message(embed=embed, view=view)
                else:
                    await interaction.response.edit_message(
                        content="No more pending experiments.",
                        embed=None,
                        view=BackToMenuView()
                    )
        except Exception as e:
            await interaction.response.edit_message(
                content=f"Error: {str(e)[:100]}",
                view=BackToMenuView()
            )

    @discord.ui.button(label="Main Menu", style=ButtonStyle.secondary, emoji="🏠")
    async def main_menu(self, interaction: Interaction, button: Button):
        view = MainMenuView()
        await interaction.response.edit_message(
            content="**RALPH Command Center** - Select an action:",
            embed=None,
            view=view
        )


# =============================================================================
# UTILITY VIEWS
# =============================================================================

class BackToMenuView(View):
    """Simple view with just a back to menu button."""

    def __init__(self, timeout: float = 120):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Main Menu", style=ButtonStyle.primary, emoji="🏠")
    async def main_menu(self, interaction: Interaction, button: Button):
        view = MainMenuView()
        await interaction.response.edit_message(
            content="**RALPH Command Center** - Select an action:",
            embed=None,
            view=view
        )


class ConfirmView(View):
    """Generic confirmation dialog."""

    def __init__(self, timeout: float = 60):
        super().__init__(timeout=timeout)
        self.value: Optional[bool] = None

    @discord.ui.button(label="Confirm", style=ButtonStyle.success, emoji="✅")
    async def confirm(self, interaction: Interaction, button: Button):
        self.value = True
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="Cancel", style=ButtonStyle.danger, emoji="❌")
    async def cancel(self, interaction: Interaction, button: Button):
        self.value = False
        self.stop()
        await interaction.response.defer()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def send_main_menu(ctx_or_channel, content: str = None):
    """Send the main menu to a channel or context."""
    view = MainMenuView()
    message_content = content or "**RALPH Command Center** - Select an action:"

    if hasattr(ctx_or_channel, 'send'):
        return await ctx_or_channel.send(message_content, view=view)
    elif hasattr(ctx_or_channel, 'reply'):
        return await ctx_or_channel.reply(message_content, view=view)


async def send_sentiment_menu(ctx_or_channel):
    """Send sentiment symbol selector."""
    view = SentimentSelectView()
    content = "**Select a cryptocurrency for sentiment analysis:**"

    if hasattr(ctx_or_channel, 'send'):
        return await ctx_or_channel.send(content, view=view)
    elif hasattr(ctx_or_channel, 'reply'):
        return await ctx_or_channel.reply(content, view=view)


async def send_market_menu(ctx_or_channel):
    """Send market category selector."""
    view = MarketCategoryView()
    content = "**Select a market category to browse:**"

    if hasattr(ctx_or_channel, 'send'):
        return await ctx_or_channel.send(content, view=view)
    elif hasattr(ctx_or_channel, 'reply'):
        return await ctx_or_channel.reply(content, view=view)


# =============================================================================
# MISSION CONTROL VIEW
# =============================================================================

class MissionControlView(View):
    """
    Interactive controls for mission management.
    Shows contextual buttons based on mission state.

    Buttons: Pause/Resume | Add Task | View Tasks | Abort
    """

    def __init__(
        self,
        mission_id: str,
        mission_manager=None,
        is_paused: bool = False,
        timeout: float = 300.0,
        author_id: Optional[int] = None
    ):
        super().__init__(timeout=timeout)
        self.mission_id = mission_id
        self.mission_manager = mission_manager
        self.author_id = author_id
        self.is_paused = is_paused

        # Toggle pause/resume button based on state
        if is_paused:
            self.pause_resume_btn.label = "Resume"
            self.pause_resume_btn.emoji = "▶️"
            self.pause_resume_btn.style = ButtonStyle.success
        else:
            self.pause_resume_btn.label = "Pause"
            self.pause_resume_btn.emoji = "⏸️"
            self.pause_resume_btn.style = ButtonStyle.primary

    async def interaction_check(self, interaction: Interaction) -> bool:
        if self.author_id and interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the operator can control missions.",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Pause", emoji="⏸️", style=ButtonStyle.primary, row=0)
    async def pause_resume_btn(self, interaction: Interaction, button: Button):
        """Toggle mission pause/resume."""
        if self.is_paused:
            # Resume
            if self.mission_manager:
                await self.mission_manager.resume_mission()
            self.is_paused = False
            button.label = "Pause"
            button.emoji = "⏸️"
            button.style = ButtonStyle.primary
            await interaction.response.edit_message(
                content=f"▶️ Mission **{self.mission_id}** resumed!",
                view=self
            )
        else:
            # Pause
            if self.mission_manager:
                await self.mission_manager.pause_mission()
            self.is_paused = True
            button.label = "Resume"
            button.emoji = "▶️"
            button.style = ButtonStyle.success
            await interaction.response.edit_message(
                content=f"⏸️ Mission **{self.mission_id}** paused.",
                view=self
            )

    @discord.ui.button(label="Add Task", emoji="➕", style=ButtonStyle.secondary, row=0)
    async def add_task_btn(self, interaction: Interaction, button: Button):
        """Open modal to add a new task."""
        modal = TaskModal(self.mission_manager, self.mission_id)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="View Tasks", emoji="📋", style=ButtonStyle.secondary, row=0)
    async def view_tasks_btn(self, interaction: Interaction, button: Button):
        """Show detailed task list."""
        if not self.mission_manager or not self.mission_manager.current_mission:
            await interaction.response.send_message("No active mission.", ephemeral=True)
            return

        mission = self.mission_manager.current_mission

        # Build task summary
        tasks_by_status = {"pending": [], "in_progress": [], "completed": [], "failed": []}
        for task in mission.tasks:
            status = task.status.value if hasattr(task.status, 'value') else task.status
            if status in tasks_by_status:
                tasks_by_status[status].append(task)

        embed = discord.Embed(
            title=f"📋 Tasks: {self.mission_id}",
            color=0xF1C40F,
            timestamp=datetime.utcnow()
        )

        for status, tasks in tasks_by_status.items():
            if tasks:
                emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌"}.get(status, "•")
                task_list = "\n".join([
                    f"{get_agent_emoji(t.assigned_to)} `{t.task_id}` {t.description[:50]}..."
                    for t in tasks[:5]
                ])
                if len(tasks) > 5:
                    task_list += f"\n*+{len(tasks) - 5} more*"
                embed.add_field(
                    name=f"{emoji} {status.title()} ({len(tasks)})",
                    value=task_list,
                    inline=False
                )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @discord.ui.button(label="Abort", emoji="⛔", style=ButtonStyle.danger, row=1)
    async def abort_btn(self, interaction: Interaction, button: Button):
        """Confirm and abort mission."""
        # Show confirmation
        confirm_view = ConfirmView()
        await interaction.response.send_message(
            f"⚠️ Are you sure you want to abort mission **{self.mission_id}**?\nThis cannot be undone.",
            view=confirm_view,
            ephemeral=True
        )

        await confirm_view.wait()
        if confirm_view.value:
            if self.mission_manager:
                await self.mission_manager.abort_mission("Aborted by operator via button")
            # Disable all buttons
            for item in self.children:
                item.disabled = True
            try:
                await interaction.message.edit(
                    content=f"⛔ Mission **{self.mission_id}** aborted.",
                    view=self
                )
            except:
                pass


# =============================================================================
# TASK MODAL
# =============================================================================

class TaskModal(Modal, title="Add Task to Mission"):
    """Modal for adding a task to current mission."""

    description = TextInput(
        label="Task Description",
        style=discord.TextStyle.paragraph,
        placeholder="What needs to be done?",
        required=True,
        max_length=500
    )

    agent = TextInput(
        label="Assign to Agent",
        style=discord.TextStyle.short,
        placeholder="strategy, tuning, backtest, risk, or data",
        required=True,
        max_length=20
    )

    priority = TextInput(
        label="Priority (high/medium/low)",
        style=discord.TextStyle.short,
        placeholder="medium",
        required=False,
        max_length=10,
        default="medium"
    )

    def __init__(self, mission_manager=None, mission_id: str = None):
        super().__init__()
        self.mission_manager = mission_manager
        self.mission_id = mission_id

    async def on_submit(self, interaction: Interaction):
        description = self.description.value
        agent = self.agent.value.lower().strip()
        priority = self.priority.value.lower() if self.priority.value else "medium"

        valid_agents = ["strategy", "tuning", "backtest", "risk", "data"]
        if agent not in valid_agents:
            await interaction.response.send_message(
                f"❌ Invalid agent. Choose from: {', '.join(valid_agents)}",
                ephemeral=True
            )
            return

        if self.mission_manager:
            try:
                task = await self.mission_manager.add_task_to_mission(
                    description=description,
                    assigned_to=agent,
                    priority=priority
                )
                if task:
                    emoji = get_agent_emoji(agent)
                    await interaction.response.send_message(
                        f"✅ Task `{task.task_id}` added to {emoji} **{agent.title()}** Agent",
                        ephemeral=False
                    )
                else:
                    await interaction.response.send_message(
                        "❌ Failed to add task. No active mission?",
                        ephemeral=True
                    )
            except Exception as e:
                await interaction.response.send_message(
                    f"❌ Error: {str(e)}",
                    ephemeral=True
                )
        else:
            await interaction.response.send_message(
                f"✅ Task created for **{agent.title()}**:\n{description[:200]}",
                ephemeral=True
            )


# =============================================================================
# APPROVAL VIEW
# =============================================================================

class ApprovalView(View):
    """
    Generic approval view for proposals, experiments, improvements, backlog items.

    Buttons: Approve | Reject | Details
    """

    def __init__(
        self,
        item_id: str,
        item_type: str,  # "proposal", "experiment", "improvement", "backlog"
        approve_callback: Optional[Callable] = None,
        reject_callback: Optional[Callable] = None,
        details_callback: Optional[Callable] = None,
        timeout: float = 600.0,
        author_id: Optional[int] = None
    ):
        super().__init__(timeout=timeout)
        self.item_id = item_id
        self.item_type = item_type
        self.approve_callback = approve_callback
        self.reject_callback = reject_callback
        self.details_callback = details_callback
        self.author_id = author_id
        self.result: Optional[str] = None  # "approved", "rejected", None

        # Hide details button if no callback
        if not details_callback:
            self.remove_item(self.details_btn)

    async def interaction_check(self, interaction: Interaction) -> bool:
        if self.author_id and interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the operator can approve/reject.",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Approve", emoji="✅", style=ButtonStyle.success)
    async def approve_btn(self, interaction: Interaction, button: Button):
        self.result = "approved"
        self.stop()

        # Disable all buttons
        for item in self.children:
            item.disabled = True

        if self.approve_callback:
            await self.approve_callback(interaction, self.item_id)
        else:
            await interaction.response.edit_message(
                content=f"✅ {self.item_type.title()} **{self.item_id}** approved!",
                view=self
            )

    @discord.ui.button(label="Reject", emoji="❌", style=ButtonStyle.danger)
    async def reject_btn(self, interaction: Interaction, button: Button):
        """Open modal for rejection reason."""
        modal = RejectReasonModal(
            item_id=self.item_id,
            item_type=self.item_type,
            callback=self._handle_rejection
        )
        await interaction.response.send_modal(modal)

    async def _handle_rejection(self, interaction: Interaction, reason: str):
        self.result = "rejected"
        self.stop()

        for item in self.children:
            item.disabled = True

        if self.reject_callback:
            await self.reject_callback(interaction, self.item_id, reason)
        else:
            await interaction.followup.send(
                f"❌ {self.item_type.title()} **{self.item_id}** rejected: {reason}",
                ephemeral=False
            )

    @discord.ui.button(label="Details", emoji="🔍", style=ButtonStyle.secondary)
    async def details_btn(self, interaction: Interaction, button: Button):
        if self.details_callback:
            await self.details_callback(interaction, self.item_id)
        else:
            await interaction.response.send_message(
                f"No details available for {self.item_id}",
                ephemeral=True
            )


class RejectReasonModal(Modal, title="Rejection Reason"):
    """Modal for providing rejection reason."""

    reason = TextInput(
        label="Reason for Rejection",
        style=discord.TextStyle.paragraph,
        placeholder="Explain why this is being rejected...",
        required=True,
        max_length=500
    )

    def __init__(
        self,
        item_id: str,
        item_type: str,
        callback: Optional[Callable] = None
    ):
        super().__init__()
        self.item_id = item_id
        self.item_type = item_type
        self._callback = callback

    async def on_submit(self, interaction: Interaction):
        reason = self.reason.value

        if self._callback:
            await self._callback(interaction, reason)
        else:
            await interaction.response.send_message(
                f"❌ {self.item_type.title()} **{self.item_id}** rejected: {reason}",
                ephemeral=True
            )


# =============================================================================
# IMPROVEMENT ACTIONS VIEW
# =============================================================================

class ImprovementActionsView(View):
    """
    Actions for approved improvements.

    Buttons: Deploy | Test | Rollback | View Code
    """

    def __init__(
        self,
        improvement_id: str,
        improvement_manager=None,
        is_deployed: bool = False,
        timeout: float = 600.0,
        author_id: Optional[int] = None
    ):
        super().__init__(timeout=timeout)
        self.improvement_id = improvement_id
        self.improvement_manager = improvement_manager
        self.author_id = author_id
        self.is_deployed = is_deployed

        # Adjust buttons based on state
        if is_deployed:
            self.deploy_btn.disabled = True
            self.rollback_btn.disabled = False
        else:
            self.rollback_btn.disabled = True

    async def interaction_check(self, interaction: Interaction) -> bool:
        if self.author_id and interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the operator can manage improvements.",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Deploy", emoji="🚀", style=ButtonStyle.success)
    async def deploy_btn(self, interaction: Interaction, button: Button):
        """Deploy the improvement."""
        await interaction.response.defer()

        try:
            if self.improvement_manager:
                result = await self.improvement_manager.deploy(self.improvement_id)
                if result.get("success"):
                    self.is_deployed = True
                    button.disabled = True
                    self.rollback_btn.disabled = False
                    await interaction.followup.send(
                        f"🚀 Improvement **{self.improvement_id}** deployed successfully!"
                    )
                else:
                    await interaction.followup.send(
                        f"❌ Deployment failed: {result.get('error', 'Unknown error')}",
                        ephemeral=True
                    )
            else:
                await interaction.followup.send("Improvement manager not available.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)

    @discord.ui.button(label="Test", emoji="🧪", style=ButtonStyle.primary)
    async def test_btn(self, interaction: Interaction, button: Button):
        """Run sandbox tests."""
        await interaction.response.defer()

        try:
            if self.improvement_manager:
                result = await self.improvement_manager.test_in_sandbox(self.improvement_id)
                status = "✅ Passed" if result.get("success") else f"❌ Failed: {result.get('error', 'Unknown')}"
                await interaction.followup.send(
                    f"🧪 Test result for **{self.improvement_id}**: {status}",
                    ephemeral=True
                )
            else:
                await interaction.followup.send("Improvement manager not available.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)

    @discord.ui.button(label="Rollback", emoji="⏪", style=ButtonStyle.danger, disabled=True)
    async def rollback_btn(self, interaction: Interaction, button: Button):
        """Rollback the improvement."""
        confirm_view = ConfirmView()
        await interaction.response.send_message(
            f"⚠️ Rollback **{self.improvement_id}**? This will revert all changes.",
            view=confirm_view,
            ephemeral=True
        )

        await confirm_view.wait()
        if confirm_view.value:
            try:
                if self.improvement_manager:
                    result = await self.improvement_manager.rollback(self.improvement_id)
                    if result.get("success"):
                        self.is_deployed = False
                        button.disabled = True
                        self.deploy_btn.disabled = False
                        await interaction.followup.send(
                            f"⏪ Improvement **{self.improvement_id}** rolled back.",
                            ephemeral=True
                        )
            except Exception as e:
                await interaction.followup.send(f"❌ Error: {str(e)}", ephemeral=True)

    @discord.ui.button(label="View Code", emoji="📝", style=ButtonStyle.secondary)
    async def view_code_btn(self, interaction: Interaction, button: Button):
        """Show code changes."""
        if not self.improvement_manager:
            await interaction.response.send_message("Improvement manager not available.", ephemeral=True)
            return

        improvement = self.improvement_manager.get_improvement(self.improvement_id)
        if not improvement:
            await interaction.response.send_message("Improvement not found.", ephemeral=True)
            return

        changes = improvement.get("code_changes", [])
        if not changes:
            await interaction.response.send_message("No code changes recorded.", ephemeral=True)
            return

        embed = discord.Embed(
            title=f"📝 Code Changes: {self.improvement_id}",
            color=0x3498DB,
            timestamp=datetime.utcnow()
        )

        for i, change in enumerate(changes[:5], 1):
            file_path = change.get("file_path", "Unknown")
            desc = change.get("description", "No description")[:200]
            embed.add_field(
                name=f"Change {i}: `{file_path.split('/')[-1]}`",
                value=desc,
                inline=False
            )

        if len(changes) > 5:
            embed.set_footer(text=f"+{len(changes) - 5} more changes")

        await interaction.response.send_message(embed=embed, ephemeral=True)


# =============================================================================
# HELP SYSTEM WITH CATEGORY SELECT
# =============================================================================

class HelpCategorySelect(Select):
    """Select menu for help categories."""

    CATEGORIES = {
        "mission": ("Mission Commands", "🚀", [
            "`!mission <objective>` - Start new mission",
            "`!mission_status` - View current status",
            "`!pause_mission` / `!resume_mission` - Control flow",
            "`!add_task <agent> <desc>` - Add task dynamically",
            "`!abort_mission` - Cancel mission"
        ]),
        "agents": ("Agent Commands", "🤖", [
            "`!do <task>` - Execute a task",
            "`!handoff <agent> <task>` - Pass work to another agent",
            "`!status` - View agent status",
            "`!tokens` - View Claude Code sessions"
        ]),
        "research": ("Research Commands", "🔍", [
            "`!research <topic>` - Web research",
            "`!arxiv <topic>` - Search papers",
            "`!github_search <topic>` - Find code",
            "`!best_practices <topic>` - Find patterns"
        ]),
        "improvements": ("Self-Improvement", "💡", [
            "`!improvements` - List improvements",
            "`!approve_improvement <id>` - Approve",
            "`!deploy_improvement <id>` - Deploy",
            "`!rollback_improvement <id>` - Revert"
        ]),
        "operational": ("Operational", "⚙️", [
            "`!deploy <target>` - Deploy to VPS",
            "`!vps` - VPS status",
            "`!logs <lines>` - View logs",
            "`!restart <component>` - Restart service"
        ]),
        "backlog": ("Backlog & Agile", "📝", [
            "`!team_backlog` - View observation backlog",
            "`!approve_backlog <id>` - Approve item",
            "`!sprint` - View current sprint",
            "`!retro_feedback` - Submit retrospective"
        ])
    }

    def __init__(self, help_callback: Optional[Callable] = None):
        options = [
            SelectOption(
                label=data[0],
                value=key,
                emoji=data[1]
            )
            for key, data in self.CATEGORIES.items()
        ]

        super().__init__(
            placeholder="Choose a help category...",
            options=options
        )
        self._help_callback = help_callback

    async def callback(self, interaction: Interaction):
        category = self.values[0]
        title, emoji, commands = self.CATEGORIES[category]

        embed = discord.Embed(
            title=f"{emoji} {title}",
            description="\n".join(commands),
            color=0x3498DB,
            timestamp=datetime.utcnow()
        )
        embed.set_footer(text="RALPH | Help System")

        if self._help_callback:
            await self._help_callback(interaction, category, embed)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)


class HelpView(View):
    """Interactive help menu with category selection."""

    def __init__(self, help_callback: Optional[Callable] = None, timeout: float = 120.0):
        super().__init__(timeout=timeout)
        self.add_item(HelpCategorySelect(help_callback))


# =============================================================================
# AGENT SELECT MENU
# =============================================================================

class AgentSelectView(View):
    """Select menu for choosing an agent type."""

    def __init__(
        self,
        callback: Optional[Callable] = None,
        include_all: bool = False,
        timeout: float = 60.0
    ):
        super().__init__(timeout=timeout)
        self._callback = callback
        self.include_all = include_all

        options = [
            SelectOption(
                label="Strategy Agent",
                value="strategy",
                description="The Visionary - Planning and coordination",
                emoji="♟️"
            ),
            SelectOption(
                label="Tuning Agent",
                value="tuning",
                description="The Perfectionist - Parameter optimization",
                emoji="⚙️"
            ),
            SelectOption(
                label="Backtest Agent",
                value="backtest",
                description="The Skeptic - Testing and validation",
                emoji="📊"
            ),
            SelectOption(
                label="Risk Agent",
                value="risk",
                description="The Guardian - Risk assessment",
                emoji="🛡️"
            ),
            SelectOption(
                label="Data Agent",
                value="data",
                description="The Librarian - Data management",
                emoji="💾"
            ),
        ]

        if include_all:
            options.insert(0, SelectOption(
                label="All Agents",
                value="all",
                description="Assign to all agents",
                emoji="🤖"
            ))

        self.select = Select(
            placeholder="Select an agent...",
            options=options
        )
        self.select.callback = self._handle_selection
        self.add_item(self.select)

    async def _handle_selection(self, interaction: Interaction):
        if self._callback:
            await self._callback(interaction, self.select.values[0])
        else:
            await interaction.response.send_message(
                f"Selected: **{self.select.values[0].title()}** Agent",
                ephemeral=True
            )


# =============================================================================
# PAGINATION VIEW
# =============================================================================

class PaginatedView(View):
    """
    Generic pagination for list displays.
    """

    def __init__(
        self,
        pages: List[discord.Embed],
        author_id: Optional[int] = None,
        timeout: float = 120.0
    ):
        super().__init__(timeout=timeout)
        self.pages = pages
        self.current_page = 0
        self.author_id = author_id

        # Disable buttons appropriately
        self._update_buttons()

    def _update_buttons(self):
        self.first_btn.disabled = self.current_page == 0
        self.prev_btn.disabled = self.current_page == 0
        self.next_btn.disabled = self.current_page >= len(self.pages) - 1
        self.last_btn.disabled = self.current_page >= len(self.pages) - 1
        self.page_indicator.label = f"{self.current_page + 1}/{len(self.pages)}"

    async def interaction_check(self, interaction: Interaction) -> bool:
        if self.author_id and interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only the person who ran this command can navigate.",
                ephemeral=True
            )
            return False
        return True

    @discord.ui.button(emoji="⏮️", style=ButtonStyle.secondary)
    async def first_btn(self, interaction: Interaction, button: Button):
        self.current_page = 0
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[0], view=self)

    @discord.ui.button(emoji="◀️", style=ButtonStyle.secondary)
    async def prev_btn(self, interaction: Interaction, button: Button):
        self.current_page = max(0, self.current_page - 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[self.current_page], view=self)

    @discord.ui.button(label="1/1", style=ButtonStyle.secondary, disabled=True)
    async def page_indicator(self, interaction: Interaction, button: Button):
        pass  # Just a display

    @discord.ui.button(emoji="▶️", style=ButtonStyle.secondary)
    async def next_btn(self, interaction: Interaction, button: Button):
        self.current_page = min(len(self.pages) - 1, self.current_page + 1)
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[self.current_page], view=self)

    @discord.ui.button(emoji="⏭️", style=ButtonStyle.secondary)
    async def last_btn(self, interaction: Interaction, button: Button):
        self.current_page = len(self.pages) - 1
        self._update_buttons()
        await interaction.response.edit_message(embed=self.pages[self.current_page], view=self)


# =============================================================================
# QUICK ACTION VIEW
# =============================================================================

class QuickActionsView(View):
    """
    Quick action buttons for common operations.
    Shown on main status embeds.
    """

    def __init__(
        self,
        on_new_mission: Optional[Callable] = None,
        on_view_status: Optional[Callable] = None,
        on_view_tokens: Optional[Callable] = None,
        timeout: float = 300.0
    ):
        super().__init__(timeout=timeout)
        self.on_new_mission = on_new_mission
        self.on_view_status = on_view_status
        self.on_view_tokens = on_view_tokens

    @discord.ui.button(label="New Mission", emoji="🚀", style=ButtonStyle.success)
    async def new_mission_btn(self, interaction: Interaction, button: Button):
        if self.on_new_mission:
            await self.on_new_mission(interaction)
        else:
            modal = MissionModal()
            await interaction.response.send_modal(modal)

    @discord.ui.button(label="Status", emoji="📊", style=ButtonStyle.primary)
    async def status_btn(self, interaction: Interaction, button: Button):
        if self.on_view_status:
            await self.on_view_status(interaction)
        else:
            await interaction.response.send_message("No status handler configured.", ephemeral=True)

    @discord.ui.button(label="Tokens", emoji="💰", style=ButtonStyle.secondary)
    async def tokens_btn(self, interaction: Interaction, button: Button):
        if self.on_view_tokens:
            await self.on_view_tokens(interaction)
        else:
            await interaction.response.send_message("No token handler configured.", ephemeral=True)

    @discord.ui.button(label="Help", emoji="❓", style=ButtonStyle.secondary)
    async def help_btn(self, interaction: Interaction, button: Button):
        view = HelpView()
        await interaction.response.send_message(
            "**RALPH Help** - Select a category:",
            view=view,
            ephemeral=True
        )


class MissionModal(Modal, title="New Mission"):
    """Modal for creating a new mission."""

    objective = TextInput(
        label="Mission Objective",
        style=discord.TextStyle.paragraph,
        placeholder="Describe what you want to accomplish...",
        required=True,
        max_length=1000
    )

    priority = TextInput(
        label="Priority (high/medium/low)",
        style=discord.TextStyle.short,
        placeholder="medium",
        required=False,
        max_length=10,
        default="medium"
    )

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__()
        self._callback = callback

    async def on_submit(self, interaction: Interaction):
        objective = self.objective.value
        priority = self.priority.value.lower() if self.priority.value else "medium"

        if self._callback:
            await self._callback(interaction, objective, priority)
        else:
            await interaction.response.send_message(
                f"🚀 Mission created!\n**Objective:** {objective[:200]}...\n**Priority:** {priority}",
                ephemeral=False
            )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_approval_embed_with_view(
    item_id: str,
    item_type: str,
    title: str,
    description: str,
    fields: List[tuple] = None,
    color: int = 0x3498DB,
    approve_callback: Callable = None,
    reject_callback: Callable = None,
    author_id: int = None
) -> tuple:
    """
    Create an approval embed with attached view.

    Returns: (embed, view)
    """
    embed = discord.Embed(
        title=f"📋 {item_type.title()}: {item_id}",
        description=f"**{title}**\n\n{description}",
        color=color,
        timestamp=datetime.utcnow()
    )

    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)

    embed.set_footer(text=f"RALPH | {item_type.title()} awaiting approval")

    view = ApprovalView(
        item_id=item_id,
        item_type=item_type,
        approve_callback=approve_callback,
        reject_callback=reject_callback,
        author_id=author_id
    )

    return embed, view


def create_mission_status_with_controls(
    mission,
    mission_manager,
    author_id: int = None
) -> tuple:
    """
    Create mission status embed with interactive controls.

    Returns: (embed, view)
    """
    is_paused = mission.status.value == "paused" if hasattr(mission.status, 'value') else mission.status == "paused"

    completed = len([t for t in mission.tasks if (t.status.value if hasattr(t.status, 'value') else t.status) == "completed"])
    total = len(mission.tasks)
    progress_pct = (completed / total * 100) if total > 0 else 0
    filled = int(progress_pct / 10)
    bar = "█" * filled + "░" * (10 - filled)

    status_emoji = "⏸️" if is_paused else "🔄"
    status_text = "Paused" if is_paused else "In Progress"

    embed = discord.Embed(
        title=f"🚀 Mission: {mission.mission_id}",
        description=f"**Objective:** {mission.objective[:300]}",
        color=0xF39C12 if is_paused else 0x3498DB,
        timestamp=datetime.utcnow()
    )

    embed.add_field(
        name="Progress",
        value=f"`{bar}` {completed}/{total} ({progress_pct:.0f}%)",
        inline=False
    )

    embed.add_field(
        name="Status",
        value=f"{status_emoji} {status_text}",
        inline=True
    )

    # Current task
    in_progress = [t for t in mission.tasks if (t.status.value if hasattr(t.status, 'value') else t.status) == "in_progress"]
    if in_progress:
        current = in_progress[0]
        agent_emoji = get_agent_emoji(current.assigned_to)
        embed.add_field(
            name="Current",
            value=f"{agent_emoji} {current.description[:50]}...",
            inline=True
        )

    embed.set_footer(text="RALPH | Use buttons below to control mission")

    view = MissionControlView(
        mission_id=mission.mission_id,
        mission_manager=mission_manager,
        is_paused=is_paused,
        author_id=author_id
    )

    return embed, view


async def send_help_menu(ctx_or_channel):
    """Send the interactive help menu."""
    view = HelpView()
    content = "**RALPH Help** - Select a category:"

    if hasattr(ctx_or_channel, 'send'):
        return await ctx_or_channel.send(content, view=view)
    elif hasattr(ctx_or_channel, 'reply'):
        return await ctx_or_channel.reply(content, view=view)
