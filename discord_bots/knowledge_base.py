"""
Knowledge Base for RALPH Agent Ensemble

Stores and retrieves learnings from mission completions using lightweight
vector similarity search. Designed to provide relevant context without
overwhelming Claude Code's context window.

Key Features:
- TF-IDF based similarity (no external API needed)
- Automatic consolidation to prevent unlimited growth
- Top-K retrieval for context-efficient responses
- Agent-specific knowledge filtering
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math

logger = logging.getLogger("knowledge_base")


class TextVectorizer:
    """
    Lightweight TF-IDF vectorizer for semantic similarity.
    No external dependencies - pure Python implementation.
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9_]+\b', text)
        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'this', 'that', 'these', 'those', 'it', 'its'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents."""
        self.doc_count = len(documents)
        doc_freq = Counter()

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        # Calculate IDF
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1

    def transform(self, text: str) -> Dict[str, float]:
        """Transform text to TF-IDF vector (sparse dict)."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1

        vector = {}
        for token, count in tf.items():
            if token in self.vocabulary:
                tf_val = count / total
                idf_val = self.idf.get(token, 1.0)
                vector[token] = tf_val * idf_val

        return vector

    def similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Dot product
        dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) | set(vec2))

        # Magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)


class LearningEntry:
    """A single learning/insight from a completed mission."""

    def __init__(
        self,
        learning_id: str,
        mission_id: str,
        agent_type: str,
        content: str,
        category: str = "general",
        created_at: str = None,
        importance: float = 1.0,
        usage_count: int = 0
    ):
        self.learning_id = learning_id
        self.mission_id = mission_id
        self.agent_type = agent_type
        self.content = content
        self.category = category
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.importance = importance
        self.usage_count = usage_count
        self._vector: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_id": self.learning_id,
            "mission_id": self.mission_id,
            "agent_type": self.agent_type,
            "content": self.content,
            "category": self.category,
            "created_at": self.created_at,
            "importance": self.importance,
            "usage_count": self.usage_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningEntry":
        return cls(
            learning_id=data.get("learning_id", ""),
            mission_id=data.get("mission_id", ""),
            agent_type=data.get("agent_type", "general"),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            created_at=data.get("created_at"),
            importance=data.get("importance", 1.0),
            usage_count=data.get("usage_count", 0)
        )


class KnowledgeBase:
    """
    Manages learnings with efficient retrieval and consolidation.

    Design Principles:
    - Keep context small: Return only most relevant learnings
    - Recency bias: Recent learnings weighted higher
    - Usage tracking: Frequently accessed learnings are more important
    - Consolidation: Merge similar learnings to prevent bloat
    """

    MAX_LEARNINGS = 500  # Maximum entries before forced consolidation
    MAX_CONTEXT_CHARS = 2000  # Max chars to inject into agent context
    MAX_RESULTS = 5  # Max learnings per query

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.kb_dir = self.project_dir / "knowledge_base"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.learnings_file = self.kb_dir / "learnings.json"
        self.index_file = self.kb_dir / "index.json"

        self.learnings: List[LearningEntry] = []
        self.vectorizer = TextVectorizer()

        self._load()

    def _load(self):
        """Load learnings from disk."""
        if self.learnings_file.exists():
            try:
                with open(self.learnings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.learnings = [LearningEntry.from_dict(d) for d in data]
                    logger.info(f"Loaded {len(self.learnings)} learnings from knowledge base")
            except Exception as e:
                logger.error(f"Failed to load learnings: {e}")
                self.learnings = []

        # Rebuild vectorizer
        self._rebuild_index()

    def _save(self):
        """Save learnings to disk."""
        try:
            with open(self.learnings_file, "w", encoding="utf-8") as f:
                json.dump([l.to_dict() for l in self.learnings], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learnings: {e}")

    def _rebuild_index(self):
        """Rebuild the TF-IDF index from all learnings."""
        if not self.learnings:
            return

        documents = [l.content for l in self.learnings]
        self.vectorizer.fit(documents)

        # Pre-compute vectors
        for learning in self.learnings:
            learning._vector = self.vectorizer.transform(learning.content)

    def add_learning(
        self,
        content: str,
        mission_id: str,
        agent_type: str = "general",
        category: str = "general",
        importance: float = 1.0
    ) -> str:
        """
        Add a new learning to the knowledge base.

        Returns:
            learning_id
        """
        # Generate ID
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        learning_id = f"LRN-{timestamp}-{len(self.learnings):04d}"

        entry = LearningEntry(
            learning_id=learning_id,
            mission_id=mission_id,
            agent_type=agent_type,
            content=content,
            category=category,
            importance=importance
        )

        self.learnings.append(entry)

        # Rebuild index to include new learning
        self._rebuild_index()
        self._save()

        # Check if consolidation needed
        if len(self.learnings) > self.MAX_LEARNINGS:
            self._consolidate()

        logger.info(f"Added learning {learning_id}: {content[:50]}...")
        return learning_id

    def add_learnings_from_mission(
        self,
        mission_id: str,
        summary_data: Dict[str, Any]
    ):
        """
        Extract and add learnings from a mission summary.

        Args:
            mission_id: The completed mission ID
            summary_data: Dict with key_findings, suggestions, work_summary
        """
        # Add key findings
        for finding in summary_data.get("key_findings", []):
            if isinstance(finding, str) and len(finding) > 20:
                self.add_learning(
                    content=finding,
                    mission_id=mission_id,
                    category="finding",
                    importance=1.2
                )

        # Add suggestions as learnings
        for suggestion in summary_data.get("suggestions", []):
            if isinstance(suggestion, str) and len(suggestion) > 20:
                self.add_learning(
                    content=suggestion,
                    mission_id=mission_id,
                    category="suggestion",
                    importance=1.0
                )

        # Add work summary as a high-importance learning
        work_summary = summary_data.get("work_summary", "")
        if work_summary and len(work_summary) > 30:
            self.add_learning(
                content=work_summary,
                mission_id=mission_id,
                category="summary",
                importance=1.5
            )

    def search(
        self,
        query: str,
        agent_type: str = None,
        category: str = None,
        max_results: int = None,
        min_similarity: float = 0.1
    ) -> List[Tuple[LearningEntry, float]]:
        """
        Search for relevant learnings.

        Args:
            query: Search query
            agent_type: Filter by agent type (optional)
            category: Filter by category (optional)
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (LearningEntry, similarity_score) tuples
        """
        if not self.learnings:
            return []

        max_results = max_results or self.MAX_RESULTS
        query_vector = self.vectorizer.transform(query)

        results = []
        for learning in self.learnings:
            # Apply filters
            if agent_type and learning.agent_type != agent_type:
                continue
            if category and learning.category != category:
                continue

            # Calculate similarity
            similarity = self.vectorizer.similarity(query_vector, learning._vector)

            # Apply recency boost (learnings from last 7 days get up to 20% boost)
            try:
                created = datetime.fromisoformat(learning.created_at)
                days_old = (datetime.utcnow() - created).days
                recency_boost = max(0, 1.0 - (days_old / 30) * 0.2)
                similarity *= recency_boost
            except:
                pass

            # Apply importance weight
            similarity *= learning.importance

            if similarity >= min_similarity:
                results.append((learning, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        # Track usage for top results
        for learning, _ in results[:max_results]:
            learning.usage_count += 1

        self._save()  # Save updated usage counts

        return results[:max_results]

    def get_context_for_agent(
        self,
        agent_type: str,
        task_description: str
    ) -> str:
        """
        Get relevant learnings formatted for agent context injection.

        This is the main method to call when building agent prompts.
        Returns a concise string that fits within MAX_CONTEXT_CHARS.

        Args:
            agent_type: The agent type requesting context
            task_description: The current task to find relevant learnings for

        Returns:
            Formatted string of relevant learnings
        """
        # Search for relevant learnings
        results = self.search(
            query=task_description,
            max_results=self.MAX_RESULTS,
            min_similarity=0.15
        )

        if not results:
            return ""

        # Format learnings for context
        lines = ["### Relevant Learnings from Previous Missions:"]
        char_count = len(lines[0])

        for learning, score in results:
            # Format entry
            entry = f"- [{learning.category.upper()}] {learning.content}"

            # Check if we'd exceed limit
            if char_count + len(entry) + 1 > self.MAX_CONTEXT_CHARS:
                break

            lines.append(entry)
            char_count += len(entry) + 1

        if len(lines) == 1:
            return ""  # No learnings fit

        return "\n".join(lines)

    def _consolidate(self):
        """
        Consolidate learnings to reduce bloat.

        Strategy:
        - Keep high-importance learnings
        - Keep recently used learnings
        - Merge similar learnings
        - Remove very old, unused learnings
        """
        logger.info(f"Consolidating knowledge base ({len(self.learnings)} learnings)")

        # Score each learning
        scored = []
        now = datetime.utcnow()

        for learning in self.learnings:
            score = learning.importance

            # Boost for usage
            score += min(learning.usage_count * 0.1, 1.0)

            # Penalty for age
            try:
                created = datetime.fromisoformat(learning.created_at)
                days_old = (now - created).days
                score -= min(days_old / 90, 0.5)  # Max 0.5 penalty
            except:
                pass

            scored.append((learning, score))

        # Sort by score and keep top entries
        scored.sort(key=lambda x: x[1], reverse=True)
        target_count = int(self.MAX_LEARNINGS * 0.7)  # Keep 70%

        self.learnings = [l for l, _ in scored[:target_count]]

        # Rebuild index and save
        self._rebuild_index()
        self._save()

        logger.info(f"Consolidated to {len(self.learnings)} learnings")

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.learnings:
            return {"total": 0}

        by_category = Counter(l.category for l in self.learnings)
        by_agent = Counter(l.agent_type for l in self.learnings)

        return {
            "total": len(self.learnings),
            "by_category": dict(by_category),
            "by_agent": dict(by_agent),
            "most_used": sorted(
                [(l.content[:50], l.usage_count) for l in self.learnings],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def export_for_prompt(self, max_chars: int = 1500) -> str:
        """
        Export top learnings for inclusion in agent system prompts.

        This provides a static set of most important learnings.
        """
        # Get most important learnings
        sorted_learnings = sorted(
            self.learnings,
            key=lambda l: l.importance * (1 + l.usage_count * 0.1),
            reverse=True
        )

        lines = ["## Learnings from Previous Work:"]
        char_count = len(lines[0])

        for learning in sorted_learnings[:10]:
            entry = f"- {learning.content}"
            if char_count + len(entry) + 1 > max_chars:
                break
            lines.append(entry)
            char_count += len(entry) + 1

        if len(lines) == 1:
            return ""

        return "\n".join(lines)


# Singleton instance
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the knowledge base singleton."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
