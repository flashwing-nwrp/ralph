"""
RALPH Strategy Agent

Responsible for:
- Trading strategy design
- Signal generation logic
- Feature engineering
- Strategy iteration and improvement
"""

import discord
from discord.ext import commands
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgentBot


class StrategyAgent(BaseAgentBot):
    """Strategy Agent for logic and feature design."""

    def __init__(self):
        super().__init__(
            agent_name="Strategy Agent",
            token_env_var="STRATEGY_AGENT_TOKEN",
            primary_channel_name="strategy",
            agent_type="strategy",
            description="Logic and features agent - designs trading strategies, implements signal generation, manages feature engineering."
        )

        # Strategy registry
        self.strategies = {}
        self.features = []
        self.proposals = []

        # Register agent-specific commands
        self._register_strategy_commands()

    def _register_strategy_commands(self):
        """Register strategy-specific commands."""

        @self.bot.command(name="propose_strategy")
        async def propose_strategy(ctx: commands.Context, strategy_name: str = None, *, description: str = None):
            """Propose a new strategy or modification."""
            if not strategy_name:
                await ctx.reply(
                    "Usage: `!propose_strategy <strategy_name> <description>`\n"
                    "Example: `!propose_strategy momentum_v3 Add RSI divergence filter to reduce false signals`"
                )
                return

            proposal_id = f"PROP-{len(self.proposals) + 1:04d}"
            self.proposals.append({
                "id": proposal_id,
                "name": strategy_name,
                "description": description or "No description provided",
                "status": "pending",
                "timestamp": datetime.utcnow().isoformat()
            })

            embed = discord.Embed(
                title="Strategy Proposal",
                color=discord.Color.blue()
            )
            embed.add_field(name="Proposal ID", value=f"`{proposal_id}`", inline=True)
            embed.add_field(name="Strategy", value=f"`{strategy_name}`", inline=True)
            embed.add_field(name="Description", value=description or "N/A", inline=False)
            embed.add_field(
                name="Next Steps",
                value=(
                    "1. @Data Agent - Prepare required features\n"
                    "2. @Backtest Agent - Run simulation\n"
                    "3. @Risk Agent - Audit results\n"
                    "4. @Tuning Agent - Optimize parameters"
                ),
                inline=False
            )

            await ctx.reply(embed=embed)

            # Notify the team
            await self._post_to_channel(
                "ralph_team",
                f"**Strategy Agent** submitted proposal `{proposal_id}`: **{strategy_name}**\n"
                f"Description: {description or 'N/A'}\n\n"
                f"Requesting review from all agents."
            )

        @self.bot.command(name="feature")
        async def feature(ctx: commands.Context, action: str = None, feature_name: str = None):
            """Manage trading features."""
            if action not in ["add", "list", "remove"]:
                await ctx.reply(
                    "Usage: `!feature <add|list|remove> [feature_name]`\n"
                    "Example: `!feature add rsi_14`"
                )
                return

            if action == "list":
                if not self.features:
                    await ctx.reply("No features registered yet. Use `!feature add <name>` to add one.")
                    return

                embed = discord.Embed(
                    title="Registered Features",
                    description="\n".join([f"- `{f}`" for f in self.features]),
                    color=discord.Color.blue()
                )
                await ctx.reply(embed=embed)
                return

            if not feature_name:
                await ctx.reply(f"Please specify a feature name for `{action}`")
                return

            if action == "add":
                if feature_name not in self.features:
                    self.features.append(feature_name)
                    await ctx.reply(f"✅ Feature `{feature_name}` added to registry")

                    # Notify data agent
                    await self._post_to_channel(
                        "data",
                        f"**Strategy Agent** registered new feature: `{feature_name}`\n"
                        f"@Data Agent - Please prepare this feature for the pipeline."
                    )
                else:
                    await ctx.reply(f"Feature `{feature_name}` already exists")

            elif action == "remove":
                if feature_name in self.features:
                    self.features.remove(feature_name)
                    await ctx.reply(f"✅ Feature `{feature_name}` removed from registry")
                else:
                    await ctx.reply(f"Feature `{feature_name}` not found")

        @self.bot.command(name="signals")
        async def signals(ctx: commands.Context, strategy: str = None):
            """Show signal generation logic for a strategy."""
            if not strategy:
                await ctx.reply("Usage: `!signals <strategy_name>`")
                return

            # Simulated signal logic display
            embed = discord.Embed(
                title=f"Signal Logic: {strategy}",
                color=discord.Color.green()
            )
            embed.add_field(
                name="Entry Conditions",
                value="```\nIF momentum > threshold AND\n   volume > avg_volume * 1.5 AND\n   NOT in_drawdown\nTHEN signal = LONG\n```",
                inline=False
            )
            embed.add_field(
                name="Exit Conditions",
                value="```\nIF profit > target OR\n   loss > stop_loss OR\n   holding_time > max_hold\nTHEN signal = CLOSE\n```",
                inline=False
            )
            embed.add_field(
                name="Features Used",
                value="`momentum`, `volume`, `drawdown_flag`, `profit`, `holding_time`",
                inline=False
            )
            await ctx.reply(embed=embed)

        @self.bot.command(name="strategies")
        async def strategies(ctx: commands.Context):
            """List all strategy proposals."""
            if not self.proposals:
                await ctx.reply("No strategies proposed yet. Use `!propose <name> <description>`")
                return

            embed = discord.Embed(
                title="Strategy Proposals",
                color=discord.Color.blue()
            )
            for prop in self.proposals[-10:]:
                status_emoji = "⏳" if prop["status"] == "pending" else "✅" if prop["status"] == "approved" else "❌"
                embed.add_field(
                    name=f"{status_emoji} {prop['id']}: {prop['name']}",
                    value=prop["description"][:100] + "..." if len(prop["description"]) > 100 else prop["description"],
                    inline=False
                )
            await ctx.reply(embed=embed)

    async def on_agent_ready(self):
        """Called when Strategy Agent is ready."""
        self.logger.info("Strategy Agent initialized - ready for strategy design")

    async def on_agent_message(self, message: discord.Message):
        """Handle strategy-related messages."""
        content_lower = message.content.lower()

        # Respond if mentioned
        if "strategy agent" in content_lower or self.bot.user.mentioned_in(message):
            if any(word in content_lower for word in ["idea", "approach", "how", "design"]):
                await message.reply(
                    "I can help design strategies! Use:\n"
                    "- `!propose <name> <description>` - Submit new strategy\n"
                    "- `!feature add <name>` - Register a feature\n"
                    "- `!signals <strategy>` - View signal logic"
                )

    async def get_status(self) -> dict:
        """Return strategy-specific status."""
        pending = len([p for p in self.proposals if p["status"] == "pending"])
        return {
            "Proposals (Pending)": pending,
            "Features Registered": len(self.features),
            "Mode": "Ready"
        }

    async def get_commands(self) -> dict:
        """Return strategy-specific commands."""
        return {
            "propose": "Submit a strategy proposal",
            "feature": "Manage trading features",
            "signals": "View signal generation logic",
            "strategies": "List all proposals"
        }


# Allow running this agent standalone
if __name__ == "__main__":
    import asyncio
    agent = StrategyAgent()
    asyncio.run(agent.run())
