"""
RALPH Risk Agent

Responsible for:
- Safety auditing
- Risk exposure monitoring
- Position limit validation
- Safety constraint enforcement
"""

import discord
from discord.ext import commands
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgentBot


class RiskAgent(BaseAgentBot):
    """Risk Agent for safety auditing."""

    def __init__(self):
        super().__init__(
            agent_name="Risk Agent",
            token_env_var="RISK_AGENT_TOKEN",
            primary_channel_name="risk",
            agent_type="risk",
            description="Safety auditing agent - monitors risk exposure, validates position limits, enforces safety constraints."
        )

        # Risk thresholds (configurable)
        self.thresholds = {
            "max_drawdown": 25.0,  # percent
            "max_position_size": 10.0,  # percent of portfolio
            "min_sharpe": 0.8,
            "max_leverage": 2.0,
            "max_correlation": 0.7,
            "var_95_limit": 5.0  # percent
        }

        # Alert history
        self.alerts = []
        self.audits = []

        # Register agent-specific commands
        self._register_risk_commands()

    def _register_risk_commands(self):
        """Register risk-specific commands."""

        @self.bot.command(name="audit")
        async def audit(ctx: commands.Context, target: str = None):
            """Run a risk audit on a strategy or simulation."""
            if not target:
                await ctx.reply(
                    "Usage: `!audit <target>`\n"
                    "Example: `!audit BT-0001` or `!audit momentum_strategy`"
                )
                return

            # Simulate audit
            audit_id = f"AUDIT-{len(self.audits) + 1:04d}"

            embed = discord.Embed(
                title=f"Risk Audit: {target}",
                color=discord.Color.orange()
            )

            # Check against thresholds (simulated)
            import random
            checks = [
                ("Max Drawdown", f"{random.uniform(10, 30):.1f}%", self.thresholds["max_drawdown"]),
                ("Position Size", f"{random.uniform(5, 15):.1f}%", self.thresholds["max_position_size"]),
                ("Sharpe Ratio", f"{random.uniform(0.5, 2.0):.2f}", self.thresholds["min_sharpe"]),
                ("Leverage", f"{random.uniform(1, 3):.1f}x", self.thresholds["max_leverage"]),
                ("VaR (95%)", f"{random.uniform(2, 8):.1f}%", self.thresholds["var_95_limit"]),
            ]

            passed = 0
            warnings = 0
            failed = 0

            for check_name, value, threshold in checks:
                val = float(value.replace('%', '').replace('x', ''))
                if check_name == "Sharpe Ratio":
                    status = "PASS" if val >= threshold else "FAIL"
                else:
                    status = "PASS" if val <= threshold else "FAIL"

                if status == "PASS":
                    passed += 1
                    emoji = "✅"
                else:
                    failed += 1
                    emoji = "❌"

                embed.add_field(
                    name=f"{emoji} {check_name}",
                    value=f"Value: `{value}`\nLimit: `{threshold}`",
                    inline=True
                )

            # Overall verdict
            if failed == 0:
                verdict = "✅ APPROVED"
                color = discord.Color.green()
            elif failed <= 1:
                verdict = "⚠️ CONDITIONAL"
                color = discord.Color.orange()
            else:
                verdict = "❌ REJECTED"
                color = discord.Color.red()

            embed.color = color
            embed.add_field(
                name="Verdict",
                value=f"**{verdict}**\nPassed: {passed} | Failed: {failed}",
                inline=False
            )

            self.audits.append({
                "id": audit_id,
                "target": target,
                "verdict": verdict,
                "timestamp": datetime.utcnow().isoformat()
            })

            await ctx.reply(embed=embed)

            # If rejected, alert the team
            if "REJECTED" in verdict:
                await self._post_to_channel(
                    "ralph_team",
                    f"**RISK ALERT**: Audit `{audit_id}` for `{target}` was **REJECTED**. "
                    f"Strategy Agent and Tuning Agent: Please review and adjust parameters."
                )

        @self.bot.command(name="limits")
        async def limits(ctx: commands.Context):
            """Show current risk limits."""
            embed = discord.Embed(
                title="Current Risk Limits",
                color=discord.Color.blue()
            )
            for name, value in self.thresholds.items():
                formatted_name = name.replace("_", " ").title()
                if "percent" in name or "drawdown" in name or "size" in name or "var" in name:
                    embed.add_field(name=formatted_name, value=f"`{value}%`", inline=True)
                elif "leverage" in name:
                    embed.add_field(name=formatted_name, value=f"`{value}x`", inline=True)
                else:
                    embed.add_field(name=formatted_name, value=f"`{value}`", inline=True)

            await ctx.reply(embed=embed)

        @self.bot.command(name="setlimit")
        async def setlimit(ctx: commands.Context, limit_name: str = None, value: str = None):
            """Update a risk limit."""
            if not limit_name or not value:
                await ctx.reply(
                    "Usage: `!setlimit <limit_name> <value>`\n"
                    f"Available limits: {', '.join(self.thresholds.keys())}"
                )
                return

            if limit_name not in self.thresholds:
                await ctx.reply(f"Unknown limit: `{limit_name}`")
                return

            try:
                new_value = float(value)
                old_value = self.thresholds[limit_name]
                self.thresholds[limit_name] = new_value

                embed = discord.Embed(
                    title="Risk Limit Updated",
                    color=discord.Color.green()
                )
                embed.add_field(name="Limit", value=f"`{limit_name}`", inline=True)
                embed.add_field(name="Old Value", value=f"`{old_value}`", inline=True)
                embed.add_field(name="New Value", value=f"`{new_value}`", inline=True)

                await ctx.reply(embed=embed)
                self.logger.info(f"Risk limit updated: {limit_name} = {new_value}")

            except ValueError:
                await ctx.reply(f"Invalid value: `{value}` (must be a number)")

        @self.bot.command(name="alert")
        async def alert(ctx: commands.Context, level: str = "WARNING", *, message: str = None):
            """Post a risk alert."""
            if not message:
                await ctx.reply("Usage: `!alert [WARNING|CRITICAL] <message>`")
                return

            level = level.upper()
            if level not in ["WARNING", "CRITICAL"]:
                level = "WARNING"

            color = discord.Color.red() if level == "CRITICAL" else discord.Color.orange()
            emoji = "🚨" if level == "CRITICAL" else "⚠️"

            embed = discord.Embed(
                title=f"{emoji} Risk Alert: {level}",
                description=message,
                color=color,
                timestamp=datetime.utcnow()
            )

            self.alerts.append({
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })

            await ctx.reply(embed=embed)

            # Post to team channel if critical
            if level == "CRITICAL":
                await self._post_to_channel(
                    "ralph_team",
                    f"🚨 **CRITICAL RISK ALERT** 🚨\n{message}\n\nAll agents: Please acknowledge."
                )

    async def on_agent_ready(self):
        """Called when Risk Agent is ready."""
        self.logger.info("Risk Agent initialized - monitoring for safety")

    async def on_agent_message(self, message: discord.Message):
        """Handle risk-related messages."""
        content_lower = message.content.lower()

        # Auto-respond to risk keywords
        if "risk agent" in content_lower or self.bot.user.mentioned_in(message):
            if "review" in content_lower or "audit" in content_lower:
                await message.reply(
                    "I'll review that! Use `!audit <target>` for a full risk assessment."
                )

        # Flag dangerous keywords
        danger_keywords = ["yolo", "all in", "max leverage", "ignore risk"]
        if any(kw in content_lower for kw in danger_keywords):
            if message.channel.name in ["strategy", "ralph_team", "tuning"]:
                await message.reply(
                    "⚠️ **Risk Agent Notice**: Detected potentially risky language. "
                    "Please ensure all strategies comply with risk limits (`!limits`)."
                )

    async def get_status(self) -> dict:
        """Return risk-specific status."""
        return {
            "Active Alerts": len([a for a in self.alerts if a["level"] == "CRITICAL"]),
            "Audits Completed": len(self.audits),
            "Mode": "Monitoring"
        }

    async def get_commands(self) -> dict:
        """Return risk-specific commands."""
        return {
            "audit": "Run risk audit on target",
            "limits": "Show current risk limits",
            "setlimit": "Update a risk limit",
            "alert": "Post a risk alert"
        }


# Allow running this agent standalone
if __name__ == "__main__":
    import asyncio
    agent = RiskAgent()
    asyncio.run(agent.run())
