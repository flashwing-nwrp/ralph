"""
Token Optimization System for RALPH Agent Ensemble

This module provides comprehensive token optimization to reduce Claude API costs
for non-user-facing operations (internal agent work, handoffs, automation).

Key Strategies:
1. Prompt Compression - Minimal prompts for internal work, verbose for user-facing
2. Output Caching - Semantic similarity matching for repeated tasks
3. Session Continuity - Reuse conversation context via --continue flag
4. Context Windowing - Smart truncation preserving key information

Estimated Savings: 60-80% token reduction for internal operations.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import OrderedDict

logger = logging.getLogger("token_optimizer")

# =============================================================================
# COMPACT PROMPT CONFIGURATION
# =============================================================================

# Use compact prompts by default (set to False to use verbose)
USE_COMPACT_PROMPTS = os.environ.get("USE_COMPACT_PROMPTS", "true").lower() == "true"

# Import compact prompts
try:
    from agent_prompts_compact import AGENT_ROLES_COMPACT, EXPERIMENTATION_MINDSET_COMPACT
    COMPACT_PROMPTS_AVAILABLE = True
except ImportError:
    COMPACT_PROMPTS_AVAILABLE = False
    logger.warning("Compact prompts not available - using verbose prompts")


# =============================================================================
# MINIMAL PROMPT TEMPLATES (For Internal Work)
# =============================================================================

# These replace the 500+ line verbose prompts for non-user-facing work
MINIMAL_PROMPTS = {
    "tuning": """You are Tuning Agent. Optimize parameters.
Task: {task}
{context}
Output format: [PARAM] name: old→new (reason) or [TASK: agent] description""",

    "backtest": """You are Backtest Agent. Run simulations and validate.
Task: {task}
{context}
Output format: Metrics table, verdict (PROCEED/REVIEW/REJECT), or [TASK: agent] description""",

    "risk": """You are Risk Agent. Audit for safety. VETO power if unsafe.
Limits: MaxDD 25%, MaxPos 10%, MinSharpe 0.8, MaxLeverage 2x
Task: {task}
{context}
Output format: APPROVED/CONDITIONAL/REJECTED with reasons, or [TASK: agent] description""",

    "strategy": """You are Strategy Agent. Design trading logic and plan missions.
Task: {task}
{context}
Output tasks as: [TASK: data|tuning|backtest|risk|strategy] specific description with file refs""",

    "data": """You are Data Agent. Handle data pipelines and features.
Task: {task}
{context}
Output format: Dataset stats, quality metrics, or [TASK: agent] description""",
}

# Instructions appended to all internal prompts (much shorter than EXPERIMENTATION_MINDSET)
MINIMAL_INSTRUCTIONS = """
Steps: 1) Explore files with Glob/Read 2) Check if ALREADY_DONE 3) Execute 4) Output tasks/handoffs
Valid agents: tuning, backtest, risk, data, strategy"""


# =============================================================================
# OUTPUT CACHE
# =============================================================================

@dataclass
class CachedOutput:
    """Cached task output with metadata."""
    task_hash: str
    task_text: str
    output: str
    agent_type: str
    created_at: datetime
    hit_count: int = 0
    tokens_saved: int = 0


class OutputCache:
    """
    LRU cache for task outputs with semantic similarity matching.

    Caches task outputs to avoid re-running identical or similar tasks.
    Uses hash-based exact matching and keyword similarity for near-matches.
    """

    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self._cache: OrderedDict[str, CachedOutput] = OrderedDict()
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> [task_hashes]

        # Stats
        self.hits = 0
        self.misses = 0
        self.tokens_saved_total = 0

    def _hash_task(self, task: str, agent_type: str) -> str:
        """Create hash for exact matching."""
        normalized = task.lower().strip()
        # Remove timestamps and dynamic values
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', normalized)
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', normalized)
        content = f"{agent_type}:{normalized}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _extract_keywords(self, task: str) -> set:
        """Extract significant keywords for similarity matching."""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'to', 'for', 'in', 'on', 'at', 'and', 'or', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b\w+\b', task.lower())
        return {w for w in words if len(w) > 3 and w not in stop_words}

    def _similarity_score(self, keywords1: set, keywords2: set) -> float:
        """Jaccard similarity between keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        return intersection / union if union > 0 else 0.0

    def get(self, task: str, agent_type: str, similarity_threshold: float = 0.8) -> Optional[CachedOutput]:
        """
        Get cached output for a task.

        First tries exact match, then falls back to similarity matching.
        """
        # Clean expired entries
        self._clean_expired()

        task_hash = self._hash_task(task, agent_type)

        # Exact match
        if task_hash in self._cache:
            entry = self._cache[task_hash]
            entry.hit_count += 1
            # Move to end (LRU)
            self._cache.move_to_end(task_hash)
            self.hits += 1
            logger.info(f"Cache HIT (exact): {task[:50]}...")
            return entry

        # Similarity match
        task_keywords = self._extract_keywords(task)
        best_match = None
        best_score = 0.0

        for cached in self._cache.values():
            if cached.agent_type != agent_type:
                continue
            cached_keywords = self._extract_keywords(cached.task_text)
            score = self._similarity_score(task_keywords, cached_keywords)
            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = cached

        if best_match:
            best_match.hit_count += 1
            self.hits += 1
            logger.info(f"Cache HIT (similarity {best_score:.2f}): {task[:50]}...")
            return best_match

        self.misses += 1
        return None

    def put(self, task: str, agent_type: str, output: str, estimated_tokens: int = 0):
        """Store task output in cache."""
        task_hash = self._hash_task(task, agent_type)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        entry = CachedOutput(
            task_hash=task_hash,
            task_text=task,
            output=output,
            agent_type=agent_type,
            created_at=datetime.utcnow(),
            tokens_saved=estimated_tokens
        )

        self._cache[task_hash] = entry

        # Update keyword index
        keywords = self._extract_keywords(task)
        for kw in keywords:
            if kw not in self._keyword_index:
                self._keyword_index[kw] = []
            if task_hash not in self._keyword_index[kw]:
                self._keyword_index[kw].append(task_hash)

        logger.debug(f"Cached output for: {task[:50]}...")

    def _clean_expired(self):
        """Remove expired entries."""
        now = datetime.utcnow()
        expired = [
            k for k, v in self._cache.items()
            if now - v.created_at > self.ttl
        ]
        for key in expired:
            del self._cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
            "tokens_saved_total": self.tokens_saved_total
        }

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._keyword_index.clear()
        logger.info("Output cache cleared")


# =============================================================================
# SESSION MANAGER (For --continue support)
# =============================================================================

@dataclass
class AgentSession:
    """Tracks an active Claude Code session for an agent."""
    agent_type: str
    conversation_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    task_count: int = 0
    tokens_accumulated: int = 0
    is_primed: bool = False  # True after first task with full context


class SessionManager:
    """
    Manages Claude Code sessions for agents.

    Enables session continuity via --continue flag to avoid
    rebuilding context on every invocation.
    """

    # Session expires after this many minutes of inactivity
    SESSION_TIMEOUT_MINUTES = 30

    # Max tasks per session before forcing refresh (prevents context bloat)
    MAX_TASKS_PER_SESSION = 10

    def __init__(self):
        self._sessions: Dict[str, AgentSession] = {}

    def get_session(self, agent_type: str) -> Optional[AgentSession]:
        """Get active session for an agent, if valid."""
        session = self._sessions.get(agent_type)
        if not session:
            return None

        # Check if expired
        elapsed = datetime.utcnow() - session.last_used
        if elapsed > timedelta(minutes=self.SESSION_TIMEOUT_MINUTES):
            logger.info(f"Session expired for {agent_type} (inactive {elapsed})")
            del self._sessions[agent_type]
            return None

        # Check if task limit reached
        if session.task_count >= self.MAX_TASKS_PER_SESSION:
            logger.info(f"Session task limit reached for {agent_type}")
            del self._sessions[agent_type]
            return None

        return session

    def create_session(self, agent_type: str, conversation_id: str = None) -> AgentSession:
        """Create or refresh a session for an agent."""
        session = AgentSession(
            agent_type=agent_type,
            conversation_id=conversation_id
        )
        self._sessions[agent_type] = session
        logger.info(f"Created new session for {agent_type}")
        return session

    def update_session(self, agent_type: str, conversation_id: str = None, tokens_used: int = 0, mark_primed: bool = False):
        """Update session after a task execution."""
        session = self._sessions.get(agent_type)
        if session:
            session.last_used = datetime.utcnow()
            session.task_count += 1
            session.tokens_accumulated += tokens_used
            if conversation_id:
                session.conversation_id = conversation_id
            if mark_primed:
                session.is_primed = True

    def is_session_primed(self, agent_type: str) -> bool:
        """Check if the agent's session has been primed with full context."""
        session = self._sessions.get(agent_type)
        return session.is_primed if session else False

    def needs_priming(self, agent_type: str) -> bool:
        """Check if the agent needs a priming prompt (first task or session expired)."""
        session = self.get_session(agent_type)
        if not session:
            return True  # No session = needs priming
        return not session.is_primed

    def end_session(self, agent_type: str):
        """Explicitly end a session."""
        if agent_type in self._sessions:
            del self._sessions[agent_type]
            logger.info(f"Ended session for {agent_type}")

    def end_all_sessions(self):
        """End all active sessions."""
        self._sessions.clear()
        logger.info("All sessions ended")

    def get_active_sessions(self) -> List[str]:
        """Get list of agents with active sessions."""
        return list(self._sessions.keys())


# =============================================================================
# CONTEXT COMPRESSOR
# =============================================================================

class ContextCompressor:
    """
    Compresses context while preserving key information.

    Uses multiple strategies:
    1. Remove redundant whitespace
    2. Truncate verbose sections
    3. Extract and preserve key data points
    4. Remove markdown formatting for internal use
    """

    # Maximum context length for internal operations
    MAX_INTERNAL_CONTEXT = 2000

    # Patterns to extract and preserve
    PRESERVE_PATTERNS = [
        (r'\[TASK:\s*\w+\].*?(?=\[TASK:|$)', 'tasks'),  # Task definitions
        (r'\[HANDOFF:\s*\w+\].*?(?=\[HANDOFF:|$)', 'handoffs'),  # Handoffs
        (r'(error|warning|failed|exception)[:\s]+[^\n]+', 'errors'),  # Errors
        (r'(sharpe|sortino|drawdown|returns?)[:\s]*[\d.]+%?', 'metrics'),  # Metrics
        (r'(approved|rejected|proceed|review)[:\s]*[^\n]*', 'verdicts'),  # Verdicts
    ]

    def compress(self, context: str, preserve_all_tasks: bool = True) -> str:
        """
        Compress context for internal agent use.

        Args:
            context: Original context string
            preserve_all_tasks: If True, keeps all [TASK:] and [HANDOFF:] intact

        Returns:
            Compressed context string
        """
        if len(context) <= self.MAX_INTERNAL_CONTEXT:
            return context

        # Extract key information
        preserved = []

        for pattern, label in self.PRESERVE_PATTERNS:
            matches = re.findall(pattern, context, re.IGNORECASE | re.DOTALL)
            if matches:
                preserved.extend(matches[:5])  # Limit per category

        # Remove markdown formatting
        text = re.sub(r'^#+\s*', '', context, flags=re.MULTILINE)  # Headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'```[\w]*\n?', '', text)  # Code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code

        # Collapse whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Truncate
        if len(text) > self.MAX_INTERNAL_CONTEXT:
            # Keep first and last portions
            half = self.MAX_INTERNAL_CONTEXT // 2
            text = text[:half] + "\n...[truncated]...\n" + text[-half:]

        # Prepend preserved items
        if preserved:
            preserved_text = "\n".join(preserved[:10])
            return f"KEY INFO:\n{preserved_text}\n\nCONTEXT:\n{text}"

        return text

    def compress_output(self, output: str, max_length: int = 1500) -> str:
        """
        Compress task output for storage/handoff.

        Preserves actionable items (tasks, handoffs, verdicts).
        """
        if len(output) <= max_length:
            return output

        # Extract all tasks and handoffs first
        tasks = re.findall(r'\[TASK:\s*\w+\][^\[]*', output, re.IGNORECASE)
        handoffs = re.findall(r'\[HANDOFF:\s*\w+\][^\[]*', output, re.IGNORECASE)

        # Build compressed output
        parts = []

        if tasks:
            parts.append("TASKS:\n" + "\n".join(tasks))
        if handoffs:
            parts.append("HANDOFFS:\n" + "\n".join(handoffs))

        # Add summary of remaining content
        remaining_budget = max_length - sum(len(p) for p in parts) - 100
        if remaining_budget > 200:
            # Get first portion of output (usually has summary)
            summary = output[:remaining_budget]
            # Remove any incomplete sentence at the end
            last_period = summary.rfind('.')
            if last_period > remaining_budget // 2:
                summary = summary[:last_period + 1]
            parts.append(f"SUMMARY:\n{summary}")

        return "\n\n".join(parts)


# =============================================================================
# MAIN TOKEN OPTIMIZER
# =============================================================================

class TokenOptimizer:
    """
    Main token optimization coordinator.

    Integrates all optimization strategies:
    - Minimal prompts for internal work
    - Output caching
    - Session continuity
    - Context compression
    """

    def __init__(self, cache_size: int = 100, cache_ttl_hours: int = 24):
        self.cache = OutputCache(max_size=cache_size, ttl_hours=cache_ttl_hours)
        self.sessions = SessionManager()
        self.compressor = ContextCompressor()

        # Stats
        self.prompts_optimized = 0
        self.tokens_saved_estimate = 0

        # A/B Testing tracking
        self._last_variant = "unknown"
        self.variant_counts = {"compact": 0, "verbose": 0, "minimal": 0}
        self.using_compact = USE_COMPACT_PROMPTS and COMPACT_PROMPTS_AVAILABLE

    def build_optimized_prompt(
        self,
        agent_type: str,
        task: str,
        context: str = "",
        is_user_facing: bool = False,
        full_role: str = ""
    ) -> Tuple[str, int, bool]:
        """
        Build an optimized prompt for a task using session priming strategy.

        Session Priming Strategy:
        - First task in session: Use FULL verbose prompt (establishes agent identity)
        - Subsequent tasks: Use MINIMAL prompt (Claude retains context from priming)

        Args:
            agent_type: Type of agent (tuning, backtest, etc.)
            task: The task to perform
            context: Additional context
            is_user_facing: If True, use verbose prompt for quality
            full_role: Full verbose role (used for priming and user-facing)

        Returns:
            Tuple of (optimized_prompt, estimated_tokens_saved, needs_priming)
            needs_priming: True if this was a priming prompt (caller should mark session as primed)
        """
        # Always use full prompt for user-facing work
        if is_user_facing and full_role:
            prompt = f"{full_role}\n\nTASK: {task}"
            if context:
                prompt += f"\n\nCONTEXT:\n{context}"
            return prompt, 0, False

        # Check if session needs priming (first task or expired session)
        needs_priming = self.sessions.needs_priming(agent_type)

        if needs_priming:
            # PRIMING PROMPT: Establish agent identity
            # Use compact prompts by default for significant token savings
            if USE_COMPACT_PROMPTS and COMPACT_PROMPTS_AVAILABLE and agent_type in AGENT_ROLES_COMPACT:
                compact_role = AGENT_ROLES_COMPACT[agent_type]
                prompt = f"{compact_role}\n{EXPERIMENTATION_MINDSET_COMPACT}\n\nTASK: {task}"
                variant = "compact"
                # Estimate savings vs verbose (compact is ~70% smaller)
                verbose_tokens = len(full_role.split()) * 1.3 if full_role else 800
                compact_tokens = len(prompt.split()) * 1.3
                tokens_saved = max(0, int(verbose_tokens - compact_tokens))
                logger.info(f"Session priming for {agent_type} - using COMPACT prompt (saving ~{tokens_saved} tokens)")
            elif full_role:
                # Fallback to verbose prompt
                prompt = f"{full_role}\n\nTASK: {task}"
                variant = "verbose"
                tokens_saved = 0
                logger.info(f"Session priming for {agent_type} - using verbose prompt")
            else:
                # No role available, use minimal
                template = MINIMAL_PROMPTS.get(agent_type, MINIMAL_PROMPTS["strategy"])
                prompt = template.format(task=task, context="")
                variant = "minimal"
                tokens_saved = 0

            if context:
                compressed_context = self.compressor.compress(context)
                prompt += f"\n\nCONTEXT:\n{compressed_context}"
                # Add context compression savings
                tokens_saved += max(0, len(context) - len(compressed_context)) // 4

            # Track variant for A/B testing
            self._last_variant = variant
            self.variant_counts[variant] = self.variant_counts.get(variant, 0) + 1
            return prompt, tokens_saved, True

        # CONTINUATION PROMPT: Use minimal prompt (session is primed)
        template = MINIMAL_PROMPTS.get(agent_type, MINIMAL_PROMPTS["strategy"])

        # Compress context
        compressed_context = ""
        if context:
            compressed_context = self.compressor.compress(context)
            compressed_context = f"Context: {compressed_context}"

        prompt = template.format(task=task, context=compressed_context)
        prompt += MINIMAL_INSTRUCTIONS

        # Estimate savings (verbose prompts are ~2000-3000 tokens)
        estimated_verbose_tokens = 2500
        estimated_minimal_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        tokens_saved = max(0, int(estimated_verbose_tokens - estimated_minimal_tokens))

        self.prompts_optimized += 1
        self.tokens_saved_estimate += tokens_saved

        logger.debug(f"Using minimal prompt for {agent_type} (session primed, saving ~{tokens_saved} tokens)")
        return prompt, tokens_saved, False

    def check_cache(self, task: str, agent_type: str) -> Optional[str]:
        """
        Check if task output is cached.

        Returns cached output if found, None otherwise.
        """
        cached = self.cache.get(task, agent_type)
        if cached:
            self.tokens_saved_estimate += cached.tokens_saved
            return cached.output
        return None

    def cache_output(self, task: str, agent_type: str, output: str, tokens_used: int = 0):
        """Cache task output for future reuse."""
        # Compress output before caching
        compressed = self.compressor.compress_output(output)
        self.cache.put(task, agent_type, compressed, estimated_tokens=tokens_used)
        self.cache.tokens_saved_total += tokens_used

    def get_session_args(self, agent_type: str) -> List[str]:
        """
        Get CLI arguments for session continuity.

        Returns args like ['--continue', '--conversation-id', 'abc123']
        if there's an active session.
        """
        session = self.sessions.get_session(agent_type)
        if session and session.conversation_id:
            return ["--continue", "--conversation-id", session.conversation_id]
        return []

    def update_session(self, agent_type: str, conversation_id: str = None, tokens_used: int = 0, mark_primed: bool = False):
        """Update or create session after task execution."""
        session = self.sessions.get_session(agent_type)
        if session:
            self.sessions.update_session(agent_type, conversation_id, tokens_used, mark_primed)
        else:
            # Create new session
            self.sessions.create_session(agent_type, conversation_id)
            if mark_primed:
                self.sessions.update_session(agent_type, mark_primed=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()

        # Get primed sessions info
        primed_sessions = []
        for agent_type in self.sessions.get_active_sessions():
            if self.sessions.is_session_primed(agent_type):
                primed_sessions.append(agent_type)

        return {
            "prompts_optimized": self.prompts_optimized,
            "tokens_saved_estimate": self.tokens_saved_estimate,
            "cost_saved_estimate": f"${self.tokens_saved_estimate * 0.000015:.4f}",
            "cache": cache_stats,
            "active_sessions": self.sessions.get_active_sessions(),
            "primed_sessions": primed_sessions,
            "prompt_mode": "compact" if self.using_compact else "verbose",
            "variant_counts": self.variant_counts.copy()
        }

    def get_stats_report(self) -> str:
        """Get formatted stats report."""
        stats = self.get_stats()
        cache = stats["cache"]
        active = stats['active_sessions']
        primed = stats['primed_sessions']

        # Format sessions with priming status
        if active:
            session_info = []
            for agent in active:
                status = "primed" if agent in primed else "unprimed"
                session_info.append(f"{agent} ({status})")
            sessions_str = ', '.join(session_info)
        else:
            sessions_str = 'None'

        # Format variant counts
        variants = stats.get('variant_counts', {})
        variant_str = ', '.join(f"{k}:{v}" for k, v in variants.items() if v > 0) or "none yet"

        return f"""**Token Optimization Stats:**
- Prompt Mode: **{stats.get('prompt_mode', 'unknown').upper()}**
- Prompts optimized: {stats['prompts_optimized']}
- Est. tokens saved: {stats['tokens_saved_estimate']:,}
- Est. cost saved: {stats['cost_saved_estimate']}
- Cache size: {cache['size']}/{cache['max_size']}
- Cache hit rate: {cache['hit_rate']}
- Sessions: {sessions_str}
- Priming variants used: {variant_str}"""

    def reset_stats(self):
        """Reset optimization statistics."""
        self.prompts_optimized = 0
        self.tokens_saved_estimate = 0
        self.cache.hits = 0
        self.cache.misses = 0
        self.cache.tokens_saved_total = 0


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_token_optimizer: Optional[TokenOptimizer] = None


def get_token_optimizer() -> TokenOptimizer:
    """Get or create the global token optimizer instance."""
    global _token_optimizer
    if _token_optimizer is None:
        _token_optimizer = TokenOptimizer()
    return _token_optimizer


def set_token_optimizer(optimizer: TokenOptimizer):
    """Set the global token optimizer instance."""
    global _token_optimizer
    _token_optimizer = optimizer
