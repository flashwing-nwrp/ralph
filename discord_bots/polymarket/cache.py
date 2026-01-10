"""
Market Data Cache for RALPH Polymarket Integration

Provides intelligent caching with configurable TTLs to reduce API calls
and improve response times while maintaining data freshness.

Cache Strategy:
- Market metadata: 1 hour (rarely changes)
- Current prices: 10 seconds (real-time trading needs)
- Order books: 30 seconds (balance freshness vs API load)
- Resolutions: Forever (immutable once resolved)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, TypeVar, Generic
from enum import Enum

logger = logging.getLogger("polymarket_cache")

T = TypeVar('T')


class CacheTTL(Enum):
    """Standard cache time-to-live values."""
    REALTIME = 10       # 10 seconds for prices
    SHORT = 30          # 30 seconds for order books
    MEDIUM = 300        # 5 minutes for active market lists
    LONG = 3600         # 1 hour for market metadata
    FOREVER = -1        # Never expires (resolutions)


@dataclass
class CacheEntry(Generic[T]):
    """A single cached item with metadata."""
    value: T
    cached_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: float = 300  # Default 5 minutes
    hits: int = 0

    @property
    def age_seconds(self) -> float:
        """Seconds since cached."""
        return (datetime.utcnow() - self.cached_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds < 0:  # FOREVER
            return False
        return self.age_seconds > self.ttl_seconds

    @property
    def expires_in(self) -> float:
        """Seconds until expiration, or -1 if never."""
        if self.ttl_seconds < 0:
            return -1
        return max(0, self.ttl_seconds - self.age_seconds)


class MarketDataCache:
    """
    Intelligent cache for Polymarket data.

    Features:
    - Per-type TTL configuration
    - Thread-safe operations via asyncio.Lock
    - Automatic cleanup of expired entries
    - Statistics tracking
    - Memory-bounded with LRU eviction
    """

    def __init__(
        self,
        max_entries: int = None,
        ttl_realtime: float = None,
        ttl_short: float = None,
        ttl_medium: float = None,
        ttl_long: float = None
    ):
        """
        Initialize cache with configurable TTLs.

        Args:
            max_entries: Maximum cache entries (default 10000)
            ttl_realtime: Seconds for price data (default 10)
            ttl_short: Seconds for order books (default 30)
            ttl_medium: Seconds for market lists (default 300)
            ttl_long: Seconds for metadata (default 3600)
        """
        self.max_entries = max_entries or int(os.getenv("POLYMARKET_CACHE_MAX", "10000"))

        # Configure TTLs from env or parameters
        self.ttls = {
            "realtime": ttl_realtime or float(os.getenv("POLYMARKET_CACHE_TTL_REALTIME", "10")),
            "short": ttl_short or float(os.getenv("POLYMARKET_CACHE_TTL_SHORT", "30")),
            "medium": ttl_medium or float(os.getenv("POLYMARKET_CACHE_TTL_MEDIUM", "300")),
            "long": ttl_long or float(os.getenv("POLYMARKET_CACHE_TTL_LONG", "3600")),
            "forever": -1
        }

        # Storage
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

        logger.info(f"MarketDataCache initialized: max={self.max_entries}, "
                   f"ttls={{realtime={self.ttls['realtime']}s, short={self.ttls['short']}s, "
                   f"medium={self.ttls['medium']}s, long={self.ttls['long']}s}}")

    def _make_key(self, category: str, *parts) -> str:
        """Create a cache key from category and parts."""
        return f"{category}:{':'.join(str(p) for p in parts)}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self.stats["expirations"] += 1
                self.stats["misses"] += 1
                return None

            entry.hits += 1
            self.stats["hits"] += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_type: str = "medium"
    ):
        """
        Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_type: TTL category ("realtime", "short", "medium", "long", "forever")
        """
        ttl = self.ttls.get(ttl_type, self.ttls["medium"])

        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_entries:
                await self._evict_oldest()

            self._cache[key] = CacheEntry(
                value=value,
                ttl_seconds=ttl
            )

    async def _evict_oldest(self):
        """Evict the oldest entries (LRU-ish based on cached_at)."""
        if not self._cache:
            return

        # Find oldest 10% of entries
        entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at
        )
        evict_count = max(1, len(entries) // 10)

        for key, _ in entries[:evict_count]:
            del self._cache[key]
            self.stats["evictions"] += 1

        logger.debug(f"Evicted {evict_count} old cache entries")

    async def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self, pattern: str = None):
        """
        Clear cache entries.

        Args:
            pattern: Optional key prefix to match (clears all if None)
        """
        async with self._lock:
            if pattern:
                keys_to_delete = [k for k in self._cache if k.startswith(pattern)]
                for key in keys_to_delete:
                    del self._cache[key]
                logger.info(f"Cleared {len(keys_to_delete)} cache entries matching '{pattern}'")
            else:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all {count} cache entries")

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self.stats["expirations"] += 1

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    # =========================================================================
    # Convenience methods for specific data types
    # =========================================================================

    async def get_market(self, condition_id: str) -> Optional[Any]:
        """Get cached market data."""
        return await self.get(self._make_key("market", condition_id))

    async def set_market(self, condition_id: str, market: Any):
        """Cache market data (long TTL)."""
        await self.set(self._make_key("market", condition_id), market, "long")

    async def get_price(self, token_id: str) -> Optional[float]:
        """Get cached price."""
        return await self.get(self._make_key("price", token_id))

    async def set_price(self, token_id: str, price: float):
        """Cache price (realtime TTL)."""
        await self.set(self._make_key("price", token_id), price, "realtime")

    async def get_orderbook(self, token_id: str) -> Optional[Any]:
        """Get cached order book."""
        return await self.get(self._make_key("orderbook", token_id))

    async def set_orderbook(self, token_id: str, orderbook: Any):
        """Cache order book (short TTL)."""
        await self.set(self._make_key("orderbook", token_id), orderbook, "short")

    async def get_markets_list(self, category: str = "all") -> Optional[Any]:
        """Get cached markets list."""
        return await self.get(self._make_key("markets", category))

    async def set_markets_list(self, markets: Any, category: str = "all"):
        """Cache markets list (medium TTL)."""
        await self.set(self._make_key("markets", category), markets, "medium")

    async def get_resolution(self, condition_id: str) -> Optional[str]:
        """Get cached resolution (never expires)."""
        return await self.get(self._make_key("resolution", condition_id))

    async def set_resolution(self, condition_id: str, resolution: str):
        """Cache resolution (forever TTL - immutable)."""
        await self.set(self._make_key("resolution", condition_id), resolution, "forever")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self.stats,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def get_stats_report(self) -> str:
        """Get formatted statistics report."""
        stats = self.get_stats()
        return f"""**Polymarket Cache Stats:**
- Entries: {stats['entries']}/{stats['max_entries']}
- Hit rate: {stats['hit_rate']:.1%}
- Hits: {stats['hits']}
- Misses: {stats['misses']}
- Evictions: {stats['evictions']}
- Expirations: {stats['expirations']}"""


# Singleton instance
_cache: Optional[MarketDataCache] = None


def get_market_cache() -> MarketDataCache:
    """Get or create the global market data cache."""
    global _cache
    if _cache is None:
        _cache = MarketDataCache()
    return _cache


def set_market_cache(cache: MarketDataCache):
    """Set the global market data cache."""
    global _cache
    _cache = cache
