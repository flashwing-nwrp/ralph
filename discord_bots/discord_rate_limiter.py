"""
Discord Rate Limiter for RALPH Agent Ensemble

This module provides rate-limited Discord embed posting to prevent
flooding the Discord API during parallel task execution.

Key Features:
- Priority queue for embeds (errors > completions > progress)
- Configurable rate limiting with burst support
- Background processor task
- Graceful queue draining on shutdown
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any, List
from enum import IntEnum

logger = logging.getLogger("discord_rate_limiter")


class EmbedPriority(IntEnum):
    """Priority levels for Discord embeds."""
    CRITICAL = 4     # System alerts, emergency stops
    ERROR = 3        # Task failures, exceptions
    COMPLETION = 2   # Task completions
    WORKING = 1      # Task started notifications
    PROGRESS = 0     # Progress updates, status


@dataclass(order=True)
class QueuedEmbed:
    """
    An embed waiting to be sent.

    Uses dataclass ordering for priority queue.
    Priority is negated so higher priority = lower number = sent first.
    """
    priority: int = field(compare=True)
    queued_at: float = field(compare=True, default_factory=lambda: datetime.utcnow().timestamp())
    embed: Any = field(compare=False, default=None)  # discord.Embed
    channel_id: int = field(compare=False, default=0)
    content: str = field(compare=False, default="")  # Optional text content


class DiscordEmbedQueue:
    """
    Rate-limited Discord embed queue.

    Prevents flooding Discord with embeds during parallel execution.
    Uses a priority queue with configurable rate limiting.

    Rate Limiting Strategy:
    - Base rate: Minimum seconds between sends
    - Burst: Allow N sends in M seconds, then throttle
    - Priority: Higher priority embeds sent first

    Thread Safety:
    - Queue operations are thread-safe via asyncio.PriorityQueue
    - Rate limit state protected by asyncio.Lock
    """

    def __init__(
        self,
        rate_limit: float = None,
        burst_limit: int = None,
        burst_window: float = None
    ):
        """
        Initialize the embed queue.

        Args:
            rate_limit: Minimum seconds between sends (default: 0.5)
            burst_limit: Max embeds in burst window (default: 10)
            burst_window: Seconds for burst window (default: 5.0)
        """
        self.rate_limit = rate_limit or float(os.getenv("EMBED_RATE_LIMIT", "0.5"))
        self.burst_limit = burst_limit or int(os.getenv("EMBED_BURST_LIMIT", "10"))
        self.burst_window = burst_window or float(os.getenv("EMBED_BURST_WINDOW", "5.0"))

        # Queue
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Rate limiting state
        self._recent_sends: List[float] = []  # Timestamps of recent sends
        self._rate_lock = asyncio.Lock()

        # Control
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Callback for sending embeds
        self._send_callback: Optional[Callable] = None

        # Statistics
        self.stats = {
            "embeds_queued": 0,
            "embeds_sent": 0,
            "embeds_dropped": 0,
            "rate_limited_count": 0,
            "queue_peak_size": 0
        }

        logger.info(f"DiscordEmbedQueue initialized: rate={self.rate_limit}s, "
                   f"burst={self.burst_limit}/{self.burst_window}s")

    def set_send_callback(self, callback: Callable):
        """
        Set the callback for sending embeds.

        The callback should have signature:
        async def send(embed, channel_id, content=None) -> bool

        Args:
            callback: Async function to send embeds
        """
        self._send_callback = callback

    async def enqueue(
        self,
        embed: Any,
        channel_id: int,
        priority: EmbedPriority = EmbedPriority.COMPLETION,
        content: str = ""
    ):
        """
        Add an embed to the queue.

        Args:
            embed: discord.Embed to send
            channel_id: Discord channel ID
            priority: Embed priority level
            content: Optional text content to send with embed
        """
        # Negate priority for PriorityQueue (lower number = higher priority)
        item = QueuedEmbed(
            priority=-int(priority),
            embed=embed,
            channel_id=channel_id,
            content=content
        )

        await self._queue.put(item)
        self.stats["embeds_queued"] += 1

        # Track peak queue size
        current_size = self._queue.qsize()
        self.stats["queue_peak_size"] = max(
            self.stats["queue_peak_size"],
            current_size
        )

        logger.debug(f"Queued embed with priority {priority.name}, "
                    f"queue size: {current_size}")

    async def enqueue_error(self, embed: Any, channel_id: int, content: str = ""):
        """Convenience method for error embeds."""
        await self.enqueue(embed, channel_id, EmbedPriority.ERROR, content)

    async def enqueue_completion(self, embed: Any, channel_id: int, content: str = ""):
        """Convenience method for completion embeds."""
        await self.enqueue(embed, channel_id, EmbedPriority.COMPLETION, content)

    async def enqueue_working(self, embed: Any, channel_id: int, content: str = ""):
        """Convenience method for working/started embeds."""
        await self.enqueue(embed, channel_id, EmbedPriority.WORKING, content)

    async def start(self):
        """Start the queue processor."""
        if self._running:
            logger.warning("EmbedQueue already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("EmbedQueue processor started")

    async def stop(self, drain: bool = True, drain_timeout: float = 10.0):
        """
        Stop the queue processor.

        Args:
            drain: If True, process remaining queue items before stopping
            drain_timeout: Maximum seconds to wait for drain
        """
        if not self._running:
            return

        self._running = False

        if drain and not self._queue.empty():
            logger.info(f"Draining {self._queue.qsize()} remaining embeds...")
            try:
                await asyncio.wait_for(
                    self._drain_queue(),
                    timeout=drain_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Drain timeout - some embeds may be lost")

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("EmbedQueue processor stopped")

    async def _drain_queue(self):
        """Process all remaining items in queue."""
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                await self._send_embed(item)
                await asyncio.sleep(self.rate_limit)
            except asyncio.QueueEmpty:
                break

    async def _can_send(self) -> bool:
        """
        Check if we can send within rate limits.

        Returns:
            True if send is allowed, False if rate limited
        """
        async with self._rate_lock:
            now = datetime.utcnow().timestamp()

            # Clean old entries outside burst window
            self._recent_sends = [
                ts for ts in self._recent_sends
                if now - ts < self.burst_window
            ]

            return len(self._recent_sends) < self.burst_limit

    async def _record_send(self):
        """Record a send for rate limiting."""
        async with self._rate_lock:
            self._recent_sends.append(datetime.utcnow().timestamp())

    async def _process_queue(self):
        """
        Background processor for the embed queue.

        Runs continuously, pulling items from queue and sending
        with rate limiting.
        """
        logger.info("EmbedQueue processor loop started")

        while self._running:
            try:
                # Wait for item with timeout (allows checking _running flag)
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Rate limiting - wait until we can send
                while not await self._can_send():
                    self.stats["rate_limited_count"] += 1
                    await asyncio.sleep(self.rate_limit)

                    # Check if we should stop
                    if not self._running:
                        # Put item back for drain
                        await self._queue.put(item)
                        return

                # Send embed
                await self._send_embed(item)
                await self._record_send()

                # Small delay between sends
                await asyncio.sleep(self.rate_limit)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in embed queue processor: {e}")
                await asyncio.sleep(1)

        logger.info("EmbedQueue processor loop ended")

    async def _send_embed(self, item: QueuedEmbed) -> bool:
        """
        Send a single embed.

        Args:
            item: QueuedEmbed to send

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._send_callback:
            logger.warning("No send callback set - dropping embed")
            self.stats["embeds_dropped"] += 1
            return False

        try:
            await self._send_callback(item.embed, item.channel_id, item.content)
            self.stats["embeds_sent"] += 1
            logger.debug(f"Sent embed to channel {item.channel_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send embed: {e}")
            self.stats["embeds_dropped"] += 1
            return False

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            **self.stats,
            "queue_size": self._queue.qsize(),
            "is_running": self._running
        }

    def get_stats_report(self) -> str:
        """Get formatted statistics report."""
        stats = self.get_stats()
        return f"""**Discord Embed Queue Stats:**
- Queued: {stats['embeds_queued']}
- Sent: {stats['embeds_sent']}
- Dropped: {stats['embeds_dropped']}
- Rate limited: {stats['rate_limited_count']} times
- Current queue: {stats['queue_size']}
- Peak queue: {stats['queue_peak_size']}
- Running: {stats['is_running']}"""


# Singleton instance
_embed_queue: Optional[DiscordEmbedQueue] = None


def get_embed_queue() -> DiscordEmbedQueue:
    """Get or create the global embed queue."""
    global _embed_queue
    if _embed_queue is None:
        _embed_queue = DiscordEmbedQueue()
    return _embed_queue


def set_embed_queue(queue: DiscordEmbedQueue):
    """Set the global embed queue."""
    global _embed_queue
    _embed_queue = queue


async def initialize_embed_queue(send_callback: Callable) -> DiscordEmbedQueue:
    """
    Initialize and start the global embed queue.

    Args:
        send_callback: Async function to send embeds

    Returns:
        The initialized and started DiscordEmbedQueue
    """
    queue = get_embed_queue()
    queue.set_send_callback(send_callback)
    await queue.start()
    return queue
