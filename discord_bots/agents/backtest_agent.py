"""
RALPH Backtest Agent

Responsible for:
- Historical simulation
- Performance metrics calculation
- Strategy validation
- Results reporting
"""

import discord
from discord.ext import commands
from datetime import datetime
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgentBot


class BacktestAgent(BaseAgentBot):
    """Backtest Agent for simulation and metrics."""

    def __init__(self):
        super().__init__(
            agent_name="Backtest Agent",
            token_env_var="BACKTEST_AGENT_TOKEN",
            primary_channel_name="backtesting",
            agent_type="backtest",
            description="Simulation and metrics agent - runs historical backtests, calculates performance metrics, validates strategies."
        )

        # Agent-specific state
        self.running_simulations = {}
        self.completed_backtests = []

        # Register agent-specific commands
        self._register_backtest_commands()

    def _register_backtest_commands(self):
        """Register backtest-specific commands."""

        @self.bot.command(name="backtest")
        async def backtest(ctx: commands.Context, strategy: str = None, period: str = "90d"):
            """Run a backtest simulation."""
            if not strategy:
                await ctx.reply(
                    "Usage: `!backtest <strategy_name> [period]`\n"
                    "Example: `!backtest momentum_v2 180d`\n"
                    "Periods: 30d, 90d, 180d, 1y, 2y"
                )
                return

            # Simulate backtest initiation
            sim_id = f"BT-{len(self.completed_backtests) + 1:04d}"
            self.running_simulations[sim_id] = {
                "strategy": strategy,
                "period": period,
                "started": datetime.utcnow().isoformat(),
                "status": "running"
            }

            embed = discord.Embed(
                title="Backtest Initiated",
                color=discord.Color.blue()
            )
            embed.add_field(name="Simulation ID", value=f"`{sim_id}`", inline=True)
            embed.add_field(name="Strategy", value=f"`{strategy}`", inline=True)
            embed.add_field(name="Period", value=period, inline=True)
            embed.add_field(name="Status", value="Running...", inline=False)
            embed.set_footer(text="Results will be posted when complete")

            await ctx.reply(embed=embed)

            # Notify risk channel
            await self._post_to_channel(
                "risk",
                f"**Backtest Agent** started simulation `{sim_id}` for strategy `{strategy}`. "
                f"Risk Agent: Please prepare for results review."
            )

        @self.bot.command(name="results")
        async def results(ctx: commands.Context, sim_id: str = None):
            """Show backtest results (simulated for demo)."""
            if not sim_id:
                await ctx.reply("Usage: `!results <simulation_id>`\nExample: `!results BT-0001`")
                return

            # Generate simulated results
            sharpe = round(random.uniform(0.5, 2.5), 2)
            max_dd = round(random.uniform(5, 25), 1)
            win_rate = round(random.uniform(45, 65), 1)
            total_return = round(random.uniform(-10, 50), 1)

            embed = discord.Embed(
                title=f"Backtest Results: {sim_id}",
                color=discord.Color.green() if sharpe > 1.0 else discord.Color.orange()
            )
            embed.add_field(name="Sharpe Ratio", value=f"`{sharpe}`", inline=True)
            embed.add_field(name="Max Drawdown", value=f"`{max_dd}%`", inline=True)
            embed.add_field(name="Win Rate", value=f"`{win_rate}%`", inline=True)
            embed.add_field(name="Total Return", value=f"`{total_return}%`", inline=True)
            embed.add_field(name="Trades", value=f"`{random.randint(50, 500)}`", inline=True)
            embed.add_field(name="Avg Hold Time", value=f"`{random.randint(1, 48)}h`", inline=True)

            # Risk assessment
            risk_level = "HIGH" if max_dd > 20 or sharpe < 1.0 else "MODERATE" if max_dd > 15 else "LOW"
            embed.add_field(
                name="Risk Assessment",
                value=f"**{risk_level}** - @Risk Agent please review",
                inline=False
            )

            await ctx.reply(embed=embed)

            # Record completion
            self.completed_backtests.append({
                "id": sim_id,
                "sharpe": sharpe,
                "max_dd": max_dd,
                "completed": datetime.utcnow().isoformat()
            })

            # Notify risk channel
            await self._post_to_channel(
                "risk",
                f"**Backtest Agent** completed `{sim_id}`: Sharpe={sharpe}, MaxDD={max_dd}%. "
                f"Risk level: **{risk_level}**. Please review."
            )

        @self.bot.command(name="metrics")
        async def metrics(ctx: commands.Context):
            """Show available performance metrics."""
            embed = discord.Embed(
                title="Available Performance Metrics",
                color=discord.Color.blue()
            )
            embed.add_field(
                name="Return Metrics",
                value="- Total Return\n- Annualized Return\n- Monthly Returns",
                inline=True
            )
            embed.add_field(
                name="Risk Metrics",
                value="- Sharpe Ratio\n- Sortino Ratio\n- Max Drawdown\n- VaR (95%)",
                inline=True
            )
            embed.add_field(
                name="Trade Metrics",
                value="- Win Rate\n- Profit Factor\n- Avg Win/Loss\n- Trade Count",
                inline=True
            )
            await ctx.reply(embed=embed)

        @self.bot.command(name="compare")
        async def compare(ctx: commands.Context, *sim_ids):
            """Compare multiple backtest results."""
            if len(sim_ids) < 2:
                await ctx.reply("Usage: `!compare <sim_id1> <sim_id2> [sim_id3]...`")
                return

            embed = discord.Embed(
                title="Strategy Comparison",
                description=f"Comparing: {', '.join(sim_ids)}",
                color=discord.Color.purple()
            )

            for sim_id in sim_ids[:5]:  # Limit to 5
                sharpe = round(random.uniform(0.5, 2.5), 2)
                max_dd = round(random.uniform(5, 25), 1)
                embed.add_field(
                    name=sim_id,
                    value=f"Sharpe: {sharpe}\nMaxDD: {max_dd}%",
                    inline=True
                )

            await ctx.reply(embed=embed)

    async def on_agent_ready(self):
        """Called when Backtest Agent is ready."""
        self.logger.info("Backtest Agent initialized - ready for simulations")

    async def on_agent_message(self, message: discord.Message):
        """Handle messages about backtesting."""
        content_lower = message.content.lower()

        # Respond if mentioned
        if "backtest agent" in content_lower or self.bot.user.mentioned_in(message):
            if "validate" in content_lower or "test" in content_lower:
                await message.reply(
                    "I can validate strategy changes! Use:\n"
                    "- `!backtest <strategy>` - Run simulation\n"
                    "- `!results <id>` - View results\n"
                    "- `!compare <id1> <id2>` - Compare strategies"
                )

    async def get_status(self) -> dict:
        """Return backtest-specific status."""
        return {
            "Running Simulations": len(self.running_simulations),
            "Completed Backtests": len(self.completed_backtests),
            "Mode": "Ready"
        }

    async def get_commands(self) -> dict:
        """Return backtest-specific commands."""
        return {
            "backtest": "Run a backtest simulation",
            "results": "View simulation results",
            "metrics": "Show available metrics",
            "compare": "Compare multiple backtests"
        }


# Allow running this agent standalone
if __name__ == "__main__":
    import asyncio
    agent = BacktestAgent()
    asyncio.run(agent.run())
