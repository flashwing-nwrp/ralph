"""
Polymarket Integration Package for RALPH Agent Ensemble

Provides market awareness capabilities for RALPH agents by bridging
to the Polymarket AI Bot infrastructure.

Usage:
    from polymarket import get_polymarket_service

    service = await get_polymarket_service()
    markets = await service.get_markets(crypto_only=True)
    arbs = await service.scan_arbitrage()

Key Components:
- PolymarketService: Main service class with caching and API integration
- Market: Data model for prediction markets
- MarketDataCache: Intelligent caching with TTL support
- ArbitrageOpportunity: Detected arbitrage opportunities
"""

from .service import (
    PolymarketService,
    get_polymarket_service,
    set_polymarket_service,
)

from .models import (
    Market,
    MarketOutcome,
    MarketStatus,
    OutcomeResult,
    OrderBook,
    OrderBookLevel,
    ArbitrageOpportunity,
    MarketAnalysis,
)

from .cache import (
    MarketDataCache,
    CacheEntry,
    CacheTTL,
    get_market_cache,
    set_market_cache,
)

__all__ = [
    # Service
    "PolymarketService",
    "get_polymarket_service",
    "set_polymarket_service",

    # Models
    "Market",
    "MarketOutcome",
    "MarketStatus",
    "OutcomeResult",
    "OrderBook",
    "OrderBookLevel",
    "ArbitrageOpportunity",
    "MarketAnalysis",

    # Cache
    "MarketDataCache",
    "CacheEntry",
    "CacheTTL",
    "get_market_cache",
    "set_market_cache",
]
