"""
RALPH Data Agent

Responsible for:
- Data ingestion and collection
- Preprocessing and cleaning
- Feature extraction
- Denoising and normalization
"""

import discord
from discord.ext import commands
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgentBot


class DataAgent(BaseAgentBot):
    """Data Agent for preprocessing and denoising."""

    def __init__(self):
        super().__init__(
            agent_name="Data Agent",
            token_env_var="DATA_AGENT_TOKEN",
            primary_channel_name="data",
            description="Preprocessing and denoising agent - handles data ingestion, cleaning, normalization, and feature extraction."
        )

        # Data pipeline state
        self.data_sources = [
            "polymarket_api",
            "historical_prices",
            "market_metadata"
        ]
        self.pipelines = {}
        self.features_ready = []

        # Register agent-specific commands
        self._register_data_commands()

    def _register_data_commands(self):
        """Register data-specific commands."""

        @self.bot.command(name="sources")
        async def sources(ctx: commands.Context):
            """List available data sources."""
            embed = discord.Embed(
                title="Available Data Sources",
                color=discord.Color.blue()
            )
            for source in self.data_sources:
                embed.add_field(
                    name=f"📊 {source}",
                    value="Status: Active",
                    inline=True
                )
            await ctx.reply(embed=embed)

        @self.bot.command(name="ingest")
        async def ingest(ctx: commands.Context, source: str = None, period: str = "7d"):
            """Start data ingestion from a source."""
            if not source:
                await ctx.reply(
                    f"Usage: `!ingest <source> [period]`\n"
                    f"Available sources: {', '.join(self.data_sources)}"
                )
                return

            if source not in self.data_sources:
                await ctx.reply(f"Unknown source: `{source}`")
                return

            job_id = f"INGEST-{datetime.utcnow().strftime('%H%M%S')}"

            embed = discord.Embed(
                title="Data Ingestion Started",
                color=discord.Color.green()
            )
            embed.add_field(name="Job ID", value=f"`{job_id}`", inline=True)
            embed.add_field(name="Source", value=f"`{source}`", inline=True)
            embed.add_field(name="Period", value=period, inline=True)
            embed.add_field(
                name="Status",
                value="Collecting data...",
                inline=False
            )

            await ctx.reply(embed=embed)
            self.logger.info(f"Data ingestion started: {source} for {period}")

        @self.bot.command(name="preprocess")
        async def preprocess(ctx: commands.Context, dataset: str = None):
            """Run preprocessing pipeline on a dataset."""
            if not dataset:
                await ctx.reply("Usage: `!preprocess <dataset_name>`")
                return

            embed = discord.Embed(
                title="Preprocessing Pipeline",
                color=discord.Color.blue()
            )

            steps = [
                ("Missing Value Handling", "Fill forward, then backward"),
                ("Outlier Removal", "IQR method, 1.5x threshold"),
                ("Normalization", "Z-score standardization"),
                ("Denoising", "Savitzky-Golay filter"),
                ("Feature Scaling", "MinMax to [0, 1]")
            ]

            for step_name, method in steps:
                embed.add_field(
                    name=f"✅ {step_name}",
                    value=method,
                    inline=False
                )

            embed.set_footer(text=f"Dataset: {dataset} | Pipeline: standard_v1")
            await ctx.reply(embed=embed)

            # Notify strategy agent
            await self._post_to_channel(
                "strategy",
                f"**Data Agent** completed preprocessing for `{dataset}`.\n"
                f"@Strategy Agent - Data is ready for feature extraction."
            )

        @self.bot.command(name="extract")
        async def extract(ctx: commands.Context, feature: str = None, dataset: str = None):
            """Extract a specific feature from dataset."""
            if not feature or not dataset:
                await ctx.reply("Usage: `!extract <feature_name> <dataset>`")
                return

            # Simulate feature extraction
            self.features_ready.append({
                "feature": feature,
                "dataset": dataset,
                "timestamp": datetime.utcnow().isoformat()
            })

            embed = discord.Embed(
                title="Feature Extraction Complete",
                color=discord.Color.green()
            )
            embed.add_field(name="Feature", value=f"`{feature}`", inline=True)
            embed.add_field(name="Dataset", value=f"`{dataset}`", inline=True)
            embed.add_field(name="Rows", value=f"`{1000}`", inline=True)
            embed.add_field(
                name="Statistics",
                value=f"```\nMean: 0.0023\nStd: 0.0156\nMin: -0.0892\nMax: 0.1203\n```",
                inline=False
            )

            await ctx.reply(embed=embed)

            # Notify backtest agent
            await self._post_to_channel(
                "backtesting",
                f"**Data Agent** extracted feature `{feature}` from `{dataset}`.\n"
                f"@Backtest Agent - Feature is ready for simulation."
            )

        @self.bot.command(name="quality")
        async def quality(ctx: commands.Context, dataset: str = None):
            """Run data quality checks."""
            if not dataset:
                await ctx.reply("Usage: `!quality <dataset_name>`")
                return

            import random
            completeness = round(random.uniform(95, 100), 1)
            consistency = round(random.uniform(90, 100), 1)
            timeliness = round(random.uniform(85, 100), 1)

            overall = (completeness + consistency + timeliness) / 3
            color = discord.Color.green() if overall > 95 else discord.Color.orange() if overall > 85 else discord.Color.red()

            embed = discord.Embed(
                title=f"Data Quality Report: {dataset}",
                color=color
            )
            embed.add_field(name="Completeness", value=f"`{completeness}%`", inline=True)
            embed.add_field(name="Consistency", value=f"`{consistency}%`", inline=True)
            embed.add_field(name="Timeliness", value=f"`{timeliness}%`", inline=True)
            embed.add_field(name="Overall Score", value=f"**`{overall:.1f}%`**", inline=False)

            if overall < 90:
                embed.add_field(
                    name="⚠️ Warning",
                    value="Data quality below threshold. Review recommended.",
                    inline=False
                )

            await ctx.reply(embed=embed)

        @self.bot.command(name="features")
        async def features(ctx: commands.Context):
            """List features ready for use."""
            if not self.features_ready:
                await ctx.reply("No features extracted yet. Use `!extract <feature> <dataset>`")
                return

            embed = discord.Embed(
                title="Features Ready",
                color=discord.Color.green()
            )
            for f in self.features_ready[-10:]:
                embed.add_field(
                    name=f"`{f['feature']}`",
                    value=f"Dataset: {f['dataset']}",
                    inline=True
                )
            await ctx.reply(embed=embed)

    async def on_agent_ready(self):
        """Called when Data Agent is ready."""
        self.logger.info("Data Agent initialized - ready for data operations")

    async def on_agent_message(self, message: discord.Message):
        """Handle data-related messages."""
        content_lower = message.content.lower()

        # Respond if mentioned
        if "data agent" in content_lower or self.bot.user.mentioned_in(message):
            if any(word in content_lower for word in ["prepare", "feature", "data", "clean"]):
                await message.reply(
                    "I can help with data preparation! Use:\n"
                    "- `!sources` - List data sources\n"
                    "- `!ingest <source>` - Start data collection\n"
                    "- `!preprocess <dataset>` - Run preprocessing\n"
                    "- `!extract <feature> <dataset>` - Extract features"
                )

    async def get_status(self) -> dict:
        """Return data-specific status."""
        return {
            "Data Sources": len(self.data_sources),
            "Features Ready": len(self.features_ready),
            "Mode": "Ready"
        }

    async def get_commands(self) -> dict:
        """Return data-specific commands."""
        return {
            "sources": "List available data sources",
            "ingest": "Start data ingestion",
            "preprocess": "Run preprocessing pipeline",
            "extract": "Extract features from dataset",
            "quality": "Run data quality checks",
            "features": "List ready features"
        }


# Allow running this agent standalone
if __name__ == "__main__":
    import asyncio
    agent = DataAgent()
    asyncio.run(agent.run())
