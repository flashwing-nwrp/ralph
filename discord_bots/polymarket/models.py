"""
Polymarket Data Models for RALPH Agent Ensemble

Dataclasses representing Polymarket market data, prices, and order books.
These models provide a clean interface for agents to work with market data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class MarketStatus(Enum):
    """Market lifecycle status."""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class OutcomeResult(Enum):
    """Resolution outcome."""
    YES = "yes"
    NO = "no"
    PENDING = "pending"
    UNKNOWN = "unknown"


@dataclass
class MarketOutcome:
    """A single outcome (YES/NO) within a market."""
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float = 0.0

    # Order book summary
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    midpoint: float = 0.0

    # Volume
    volume_24h: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "outcome": self.outcome,
            "price": self.price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "midpoint": self.midpoint,
            "volume_24h": self.volume_24h
        }


@dataclass
class Market:
    """
    A Polymarket prediction market.

    Contains both metadata and current pricing information.
    """
    # Identifiers
    condition_id: str
    question_id: str = ""
    slug: str = ""

    # Metadata
    question: str = ""
    description: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)

    # Timing
    end_date: Optional[datetime] = None
    created_at: Optional[datetime] = None

    # Status
    status: MarketStatus = MarketStatus.UNKNOWN
    active: bool = True
    closed: bool = False
    resolved: bool = False
    resolution: Optional[OutcomeResult] = None

    # Outcomes (YES/NO tokens)
    outcomes: List[MarketOutcome] = field(default_factory=list)

    # Volume and liquidity
    volume_24h: float = 0.0
    total_volume: float = 0.0
    liquidity: float = 0.0

    # Computed properties
    yes_price: float = 0.0
    no_price: float = 0.0

    # Cache metadata
    fetched_at: Optional[datetime] = None

    def __post_init__(self):
        """Compute derived values after initialization."""
        for outcome in self.outcomes:
            if outcome.outcome.lower() == "yes":
                self.yes_price = outcome.price
            elif outcome.outcome.lower() == "no":
                self.no_price = outcome.price

    @property
    def implied_probability(self) -> float:
        """Get implied probability from YES price."""
        return self.yes_price

    @property
    def combined_price(self) -> float:
        """YES + NO price (should be ~1.0, <1.0 indicates arbitrage)."""
        return self.yes_price + self.no_price

    @property
    def has_arbitrage(self) -> bool:
        """Check if YES+NO < 0.995 (potential arb opportunity)."""
        return self.combined_price < 0.995

    @property
    def time_until_close(self) -> Optional[float]:
        """Seconds until market closes, or None if no end date."""
        if not self.end_date:
            return None
        return (self.end_date - datetime.utcnow()).total_seconds()

    @property
    def is_crypto(self) -> bool:
        """Check if this is a crypto-related market."""
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol"]
        text = f"{self.question} {self.description} {' '.join(self.tags)}".lower()
        return any(kw in text for kw in crypto_keywords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "slug": self.slug,
            "question": self.question,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "active": self.active,
            "closed": self.closed,
            "resolved": self.resolved,
            "resolution": self.resolution.value if self.resolution else None,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "volume_24h": self.volume_24h,
            "total_volume": self.total_volume,
            "liquidity": self.liquidity,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "combined_price": self.combined_price,
            "has_arbitrage": self.has_arbitrage,
            "is_crypto": self.is_crypto,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None
        }

    def summary(self) -> str:
        """Get a brief summary for display."""
        status_emoji = "✅" if self.active else "🔒" if self.closed else "📊"
        arb_note = " ⚠️ ARB" if self.has_arbitrage else ""
        return (
            f"{status_emoji} **{self.question[:80]}**\n"
            f"YES: ${self.yes_price:.3f} | NO: ${self.no_price:.3f}{arb_note}\n"
            f"24h Vol: ${self.volume_24h:,.0f}"
        )


@dataclass
class OrderBookLevel:
    """A single price level in the order book."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book for a market outcome."""
    token_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def best_bid(self) -> float:
        """Best (highest) bid price."""
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Best (lowest) ask price."""
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0.0

    @property
    def midpoint(self) -> float:
        """Midpoint price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    def depth_at_price(self, price: float, side: str = "bid") -> float:
        """Get total size available at or better than price."""
        levels = self.bids if side == "bid" else self.asks
        total = 0.0
        for level in levels:
            if side == "bid" and level.price >= price:
                total += level.size
            elif side == "ask" and level.price <= price:
                total += level.size
        return total


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    market: Market
    yes_price: float
    no_price: float
    combined_price: float
    profit_potential: float  # 1.0 - combined_price
    detected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def profit_pct(self) -> float:
        """Profit as percentage."""
        return self.profit_potential * 100

    def summary(self) -> str:
        return (
            f"**ARB: {self.market.question[:60]}**\n"
            f"YES: ${self.yes_price:.4f} + NO: ${self.no_price:.4f} = ${self.combined_price:.4f}\n"
            f"Potential profit: {self.profit_pct:.2f}%"
        )


@dataclass
class MarketAnalysis:
    """Analysis result for a market."""
    market: Market
    sentiment: str = "neutral"  # bullish, bearish, neutral
    confidence: float = 0.0  # 0.0 to 1.0
    signals: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market.condition_id,
            "question": self.market.question,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "signals": self.signals,
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
            "analyzed_at": self.analyzed_at.isoformat()
        }
