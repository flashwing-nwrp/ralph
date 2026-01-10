"""
Polymarket Service for RALPH Agent Ensemble

Provides a unified interface to Polymarket data for agent market awareness.
Bridges the RALPH agent system with the Polymarket AI Bot trading infrastructure.

Features:
- Lazy loading - only fetches data when requested
- Intelligent caching with configurable TTLs
- API health monitoring
- Arbitrage detection
- Market analysis helpers

Usage:
    service = await PolymarketService.create()
    markets = await service.get_markets(category="crypto")
    market = await service.get_market("0x...")
    arbs = await service.scan_arbitrage()
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import (
    Market, MarketOutcome, MarketStatus, OutcomeResult,
    OrderBook, OrderBookLevel, ArbitrageOpportunity, MarketAnalysis
)
from .cache import MarketDataCache, get_market_cache

logger = logging.getLogger("polymarket_service")

# Polymarket AI Bot integration
POLYMARKET_PROJECT_DIR = os.getenv("POLYMARKET_PROJECT_DIR", r"E:\Polymarket AI Bot")


class PolymarketService:
    """
    Unified Polymarket data service for RALPH agents.

    Provides market awareness without direct trading capabilities.
    Trading execution is handled by the Polymarket AI Bot infrastructure.
    """

    def __init__(self):
        """Initialize service (use create() for async initialization)."""
        self.cache = get_market_cache()
        self._initialized = False
        self._polymarket_available = False
        self._config: Optional[Dict] = None
        self._client = None  # CLOB client reference

        # API health tracking
        self._api_status = {
            "gamma": {"healthy": None, "last_check": None, "error": None},
            "clob": {"healthy": None, "last_check": None, "error": None},
            "subgraph": {"healthy": None, "last_check": None, "error": None}
        }

    @classmethod
    async def create(cls) -> "PolymarketService":
        """
        Create and initialize the service.

        Returns:
            Initialized PolymarketService instance
        """
        service = cls()
        await service.initialize()
        return service

    async def initialize(self):
        """Initialize the service by loading Polymarket AI Bot modules."""
        if self._initialized:
            return

        try:
            # Add Polymarket project to path
            poly_path = Path(POLYMARKET_PROJECT_DIR)
            if poly_path.exists():
                if str(poly_path) not in sys.path:
                    sys.path.insert(0, str(poly_path))

                # Try to import key modules
                try:
                    import yaml
                    config_path = poly_path / "config.yaml"
                    if config_path.exists():
                        with open(config_path, "r") as f:
                            self._config = yaml.safe_load(f)
                        logger.info("Loaded Polymarket AI Bot config")
                except Exception as e:
                    logger.warning(f"Could not load Polymarket config: {e}")

                self._polymarket_available = True
                logger.info(f"Polymarket AI Bot integration available at {poly_path}")
            else:
                logger.warning(f"Polymarket project not found at {poly_path}")
                self._polymarket_available = False

        except Exception as e:
            logger.error(f"Failed to initialize Polymarket service: {e}")
            self._polymarket_available = False

        self._initialized = True

    @property
    def is_available(self) -> bool:
        """Check if Polymarket integration is available."""
        return self._polymarket_available

    # =========================================================================
    # Market Data
    # =========================================================================

    async def get_markets(
        self,
        category: str = None,
        crypto_only: bool = False,
        active_only: bool = True,
        limit: int = 100
    ) -> List[Market]:
        """
        Get list of markets.

        Args:
            category: Filter by category (e.g., "crypto", "politics")
            crypto_only: Only return crypto-related markets
            active_only: Only return active (tradeable) markets
            limit: Maximum markets to return

        Returns:
            List of Market objects
        """
        cache_key = f"all:{category or 'all'}:{crypto_only}:{active_only}"
        cached = await self.cache.get_markets_list(cache_key)
        if cached:
            return cached[:limit]

        markets = []

        if self._polymarket_available:
            try:
                # Use Polymarket AI Bot utilities
                from src.utils import get_all_markets_from_clob, get_crypto_markets_from_clob

                if crypto_only:
                    raw_markets = get_crypto_markets_from_clob(self._config, limit=limit)
                else:
                    raw_markets = get_all_markets_from_clob(
                        self._config,
                        crypto_only=False,
                        limit=limit
                    )

                for m in raw_markets:
                    market = self._parse_market(m)
                    if market:
                        if active_only and not market.active:
                            continue
                        if category and category.lower() not in market.category.lower():
                            continue
                        markets.append(market)

            except Exception as e:
                logger.error(f"Error fetching markets: {e}")

        # Cache results
        if markets:
            await self.cache.set_markets_list(markets, cache_key)

        return markets[:limit]

    async def get_market(self, condition_id: str) -> Optional[Market]:
        """
        Get a specific market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            Market object or None if not found
        """
        # Check cache
        cached = await self.cache.get_market(condition_id)
        if cached:
            return cached

        if not self._polymarket_available:
            return None

        try:
            from src.utils import get_market_details_by_condition

            raw = get_market_details_by_condition(condition_id, self._config)
            if raw:
                market = self._parse_market(raw)
                if market:
                    await self.cache.set_market(condition_id, market)
                    return market

        except Exception as e:
            logger.error(f"Error fetching market {condition_id}: {e}")

        return None

    async def get_prices(self, token_ids: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            Dict mapping token_id to price
        """
        prices = {}

        # Check cache first
        uncached = []
        for token_id in token_ids:
            cached = await self.cache.get_price(token_id)
            if cached is not None:
                prices[token_id] = cached
            else:
                uncached.append(token_id)

        if not uncached or not self._polymarket_available:
            return prices

        # Fetch uncached prices
        try:
            from src.live_trading.client import TradingClient

            if self._client is None:
                self._client = TradingClient.from_config(self._config)

            for token_id in uncached:
                try:
                    price = self._client.get_midpoint(token_id)
                    if price:
                        prices[token_id] = price
                        await self.cache.set_price(token_id, price)
                except Exception:
                    pass  # Skip failed price fetches

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")

        return prices

    async def get_orderbook(self, token_id: str) -> Optional[OrderBook]:
        """
        Get order book for a token.

        Args:
            token_id: Token ID

        Returns:
            OrderBook object or None
        """
        cached = await self.cache.get_orderbook(token_id)
        if cached:
            return cached

        if not self._polymarket_available:
            return None

        try:
            from src.live_trading.client import TradingClient

            if self._client is None:
                self._client = TradingClient.from_config(self._config)

            raw = self._client.get_order_book(token_id)
            if raw:
                orderbook = self._parse_orderbook(token_id, raw)
                await self.cache.set_orderbook(token_id, orderbook)
                return orderbook

        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")

        return None

    # =========================================================================
    # Arbitrage Detection
    # =========================================================================

    async def scan_arbitrage(
        self,
        min_profit: float = 0.005,
        category: str = None
    ) -> List[ArbitrageOpportunity]:
        """
        Scan for arbitrage opportunities (YES + NO < threshold).

        Args:
            min_profit: Minimum profit potential (default 0.5%)
            category: Optional category filter

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        markets = await self.get_markets(category=category, active_only=True)

        for market in markets:
            if market.has_arbitrage:
                profit = 1.0 - market.combined_price
                if profit >= min_profit:
                    opportunities.append(ArbitrageOpportunity(
                        market=market,
                        yes_price=market.yes_price,
                        no_price=market.no_price,
                        combined_price=market.combined_price,
                        profit_potential=profit
                    ))

        # Sort by profit potential descending
        opportunities.sort(key=lambda x: x.profit_potential, reverse=True)

        return opportunities

    # =========================================================================
    # API Health
    # =========================================================================

    async def check_api_health(self) -> Dict[str, Dict]:
        """
        Check health of Polymarket APIs.

        Returns:
            Dict with health status for each API
        """
        if not self._polymarket_available:
            return {
                api: {"healthy": False, "error": "Polymarket integration not available"}
                for api in self._api_status
            }

        # Check each API
        await asyncio.gather(
            self._check_gamma_health(),
            self._check_clob_health(),
            self._check_subgraph_health(),
            return_exceptions=True
        )

        return self._api_status

    async def _check_gamma_health(self):
        """Check Gamma API health."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://gamma-api.polymarket.com/markets?limit=1",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    self._api_status["gamma"] = {
                        "healthy": resp.status == 200,
                        "last_check": datetime.utcnow().isoformat(),
                        "error": None if resp.status == 200 else f"HTTP {resp.status}"
                    }
        except Exception as e:
            self._api_status["gamma"] = {
                "healthy": False,
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)[:100]
            }

    async def _check_clob_health(self):
        """Check CLOB API health."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://clob.polymarket.com/markets?limit=1",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    self._api_status["clob"] = {
                        "healthy": resp.status == 200,
                        "last_check": datetime.utcnow().isoformat(),
                        "error": None if resp.status == 200 else f"HTTP {resp.status}"
                    }
        except Exception as e:
            self._api_status["clob"] = {
                "healthy": False,
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)[:100]
            }

    async def _check_subgraph_health(self):
        """Check PNL Subgraph health."""
        try:
            import aiohttp
            # Simple query to check subgraph availability
            query = '{"query": "{ _meta { block { number } } }"}'
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.goldsky.com/api/public/project_cl6mb8i2e0004kz0887e1x5dt/subgraphs/polymarket-pnl/prod/gn",
                    data=query,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    self._api_status["subgraph"] = {
                        "healthy": resp.status == 200,
                        "last_check": datetime.utcnow().isoformat(),
                        "error": None if resp.status == 200 else f"HTTP {resp.status}"
                    }
        except Exception as e:
            self._api_status["subgraph"] = {
                "healthy": False,
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e)[:100]
            }

    def get_health_report(self) -> str:
        """Get formatted health report."""
        lines = ["**Polymarket API Status:**"]

        for api, status in self._api_status.items():
            if status["healthy"] is None:
                emoji = "❓"
                state = "Not checked"
            elif status["healthy"]:
                emoji = "✅"
                state = "Healthy"
            else:
                emoji = "❌"
                state = f"Error: {status.get('error', 'Unknown')}"

            lines.append(f"- {emoji} **{api.upper()}**: {state}")

        return "\n".join(lines)

    # =========================================================================
    # Parsing Helpers
    # =========================================================================

    def _parse_market(self, raw: Dict) -> Optional[Market]:
        """Parse raw API response into Market object."""
        try:
            # Parse outcomes
            outcomes = []
            tokens = raw.get("tokens", []) or raw.get("clobTokenIds", [])

            for i, token in enumerate(tokens):
                if isinstance(token, dict):
                    token_id = token.get("token_id", token.get("tokenId", ""))
                    outcome_name = token.get("outcome", "Yes" if i == 0 else "No")
                    price = float(token.get("price", 0) or 0)
                else:
                    token_id = str(token)
                    outcome_name = "Yes" if i == 0 else "No"
                    price = 0.0

                outcomes.append(MarketOutcome(
                    token_id=token_id,
                    outcome=outcome_name,
                    price=price
                ))

            # Parse status
            active = raw.get("active", True)
            closed = raw.get("closed", False)
            resolved = raw.get("resolved", False)

            if resolved:
                status = MarketStatus.RESOLVED
            elif closed:
                status = MarketStatus.CLOSED
            elif active:
                status = MarketStatus.ACTIVE
            else:
                status = MarketStatus.UNKNOWN

            # Parse end date
            end_date = None
            end_str = raw.get("end_date_iso") or raw.get("endDate")
            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            return Market(
                condition_id=raw.get("condition_id", raw.get("conditionId", "")),
                question_id=raw.get("question_id", raw.get("questionId", "")),
                slug=raw.get("slug", raw.get("market_slug", "")),
                question=raw.get("question", raw.get("title", "")),
                description=raw.get("description", ""),
                category=raw.get("category", raw.get("market_type", "")),
                tags=raw.get("tags", []) or [],
                end_date=end_date,
                status=status,
                active=active,
                closed=closed,
                resolved=resolved,
                outcomes=outcomes,
                volume_24h=float(raw.get("volume24hr", 0) or 0),
                total_volume=float(raw.get("volume", raw.get("volumeNum", 0)) or 0),
                liquidity=float(raw.get("liquidity", 0) or 0),
                fetched_at=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error parsing market: {e}")
            return None

    def _parse_orderbook(self, token_id: str, raw: Dict) -> OrderBook:
        """Parse raw order book response."""
        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in raw.get("asks", [])
        ]

        return OrderBook(
            token_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=datetime.utcnow()
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_stats = self.cache.get_stats()
        return {
            "available": self._polymarket_available,
            "initialized": self._initialized,
            "cache": cache_stats,
            "api_health": self._api_status
        }

    def get_stats_report(self) -> str:
        """Get formatted statistics report."""
        lines = [
            "**Polymarket Service Stats:**",
            f"- Available: {'Yes' if self._polymarket_available else 'No'}",
            f"- Initialized: {'Yes' if self._initialized else 'No'}",
            "",
            self.cache.get_stats_report(),
            "",
            self.get_health_report()
        ]
        return "\n".join(lines)


# Singleton instance
_service: Optional[PolymarketService] = None


async def get_polymarket_service() -> PolymarketService:
    """Get or create the global Polymarket service."""
    global _service
    if _service is None:
        _service = await PolymarketService.create()
    return _service


def set_polymarket_service(service: PolymarketService):
    """Set the global Polymarket service."""
    global _service
    _service = service
