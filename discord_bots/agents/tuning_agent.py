"""
RALPH Tuning Agent

Responsible for:
- Hyperparameter optimization
- Learning rate scheduling
- Model configuration
- Parameter sensitivity analysis
"""

import discord
from discord.ext import commands

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgentBot


class TuningAgent(BaseAgentBot):
    """Tuning Agent for parameter optimization."""

    def __init__(self):
        super().__init__(
            agent_name="Tuning Agent",
            token_env_var="TUNING_AGENT_TOKEN",
            primary_channel_name="tuning",
            agent_type="tuning",
            description="Parameter optimization agent - handles hyperparameter tuning, learning rate scheduling, and model configuration."
        )

        # Agent-specific state
        self.current_experiment = None
        self.parameter_history = []

        # Register agent-specific commands
        self._register_tuning_commands()

    def _register_tuning_commands(self):
        """Register tuning-specific commands."""

        @self.bot.command(name="tune")
        async def tune(ctx: commands.Context, param: str = None, value: str = None):
            """Propose or update a parameter value."""
            if not param:
                await ctx.reply("Usage: `!tune <parameter> <value>`\nExample: `!tune learning_rate 0.001`")
                return

            if not value:
                await ctx.reply(f"Current value for `{param}`: (not set)\nUse `!tune {param} <value>` to set.")
                return

            # Log the tuning action
            self.parameter_history.append({"param": param, "value": value})
            self.logger.info(f"Parameter tuned: {param} = {value}")

            embed = discord.Embed(
                title="Parameter Update Proposal",
                color=discord.Color.purple()
            )
            embed.add_field(name="Parameter", value=f"`{param}`", inline=True)
            embed.add_field(name="New Value", value=f"`{value}`", inline=True)
            embed.add_field(
                name="Next Steps",
                value="@Backtest Agent - Please validate this parameter change with a simulation.",
                inline=False
            )
            await ctx.reply(embed=embed)

            # Notify backtest channel
            await self._post_to_channel(
                "backtesting",
                f"**Tuning Agent** requests validation for parameter change: `{param}` = `{value}`"
            )

        @self.bot.command(name="sweep")
        async def sweep(ctx: commands.Context, param: str = None):
            """Propose a parameter sweep experiment."""
            if not param:
                await ctx.reply("Usage: `!sweep <parameter>`\nExample: `!sweep learning_rate`")
                return

            embed = discord.Embed(
                title="Parameter Sweep Proposal",
                color=discord.Color.gold(),
                description=f"Proposing grid search for `{param}`"
            )
            embed.add_field(
                name="Sweep Configuration",
                value="```\nRange: [0.0001, 0.001, 0.01, 0.1]\nMetric: Sharpe Ratio\nCross-validation: 5-fold\n```",
                inline=False
            )
            embed.add_field(
                name="Estimated Runtime",
                value="Awaiting Backtest Agent estimate",
                inline=True
            )
            await ctx.reply(embed=embed)

        @self.bot.command(name="params")
        async def params(ctx: commands.Context):
            """Show recent parameter history."""
            if not self.parameter_history:
                await ctx.reply("No parameter changes recorded yet.")
                return

            history_text = "\n".join([
                f"- `{p['param']}` = `{p['value']}`"
                for p in self.parameter_history[-10:]
            ])
            embed = discord.Embed(
                title="Recent Parameter Changes",
                description=history_text,
                color=discord.Color.purple()
            )
            await ctx.reply(embed=embed)

    async def on_agent_ready(self):
        """Called when Tuning Agent is ready."""
        self.logger.info("Tuning Agent initialized - ready for parameter optimization")

    async def on_agent_message(self, message: discord.Message):
        """Handle messages mentioning tuning or parameters."""
        content_lower = message.content.lower()

        # Respond to questions about parameters
        if any(word in content_lower for word in ["learning rate", "lr", "hyperparameter", "tune"]):
            # Check if this is in our channel or mentions us
            if message.channel.name in ["tuning", "ralph-team"]:
                if "?" in message.content and self.bot.user not in message.mentions:
                    # Don't auto-respond unless directly mentioned or it's a clear question for us
                    pass

        # Respond if mentioned by name
        if "tuning agent" in content_lower or self.bot.user.mentioned_in(message):
            if "help" in content_lower:
                await message.reply(
                    "I can help with parameter optimization! Commands:\n"
                    "- `!tune <param> <value>` - Propose parameter change\n"
                    "- `!sweep <param>` - Propose parameter sweep\n"
                    "- `!params` - View recent changes"
                )

    async def get_status(self) -> dict:
        """Return tuning-specific status."""
        return {
            "Current Experiment": self.current_experiment or "None",
            "Parameters Tuned": len(self.parameter_history),
            "Mode": "Ready"
        }

    async def get_commands(self) -> dict:
        """Return tuning-specific commands."""
        return {
            "tune": "Propose a parameter value change",
            "sweep": "Propose a parameter sweep experiment",
            "params": "Show recent parameter history"
        }


# Allow running this agent standalone
if __name__ == "__main__":
    import asyncio
    agent = TuningAgent()
    asyncio.run(agent.run())
