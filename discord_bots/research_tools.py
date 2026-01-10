"""
Research Tools - Web Research and Knowledge Gathering for RALPH Agents

Enables agents to:
1. Search the web for best practices and solutions
2. Find relevant research papers and articles
3. Discover cutting-edge techniques and advancements
4. Gather knowledge from documentation and tutorials

Uses multiple sources:
- Web search (DuckDuckGo, Google)
- arXiv for research papers
- GitHub for code examples
- Documentation sites
"""

import asyncio
import aiohttp
import json
import re
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from urllib.parse import quote_plus
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0


@dataclass
class ResearchFinding:
    """A research finding with analysis."""
    query: str
    results: List[SearchResult]
    summary: str
    key_insights: List[str]
    recommended_actions: List[str]
    timestamp: str


@dataclass
class SentimentResult:
    """Crypto sentiment analysis result."""
    symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # bullish, bearish, neutral
    bullish_count: int
    bearish_count: int
    neutral_count: int
    headlines: List[Dict[str, Any]]
    source: str
    error: Optional[str] = None


class ResearchTools:
    """Web research capabilities for agents."""

    def __init__(self):
        self.cache: Dict[str, ResearchFinding] = {}
        self.cache_ttl = 3600  # 1 hour

    async def search_web(
        self,
        query: str,
        num_results: int = 10,
        focus: str = None
    ) -> List[SearchResult]:
        """
        Search the web for information.

        Args:
            query: Search query
            num_results: Number of results to return
            focus: Optional focus area (python, ml, trading, etc.)

        Returns:
            List of SearchResult objects
        """
        results = []

        # Add focus to query if specified
        if focus:
            query = f"{query} {focus}"

        # Try DuckDuckGo HTML search
        try:
            ddg_results = await self._search_duckduckgo(query, num_results)
            results.extend(ddg_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        return results[:num_results]

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo HTML."""
        results = []
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        results = self._parse_ddg_html(html, num_results)
        except Exception as e:
            logger.warning(f"DuckDuckGo request failed: {e}")

        return results

    def _parse_ddg_html(self, html: str, num_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # Simple regex parsing for results
        link_pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'

        links = re.findall(link_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(links[:num_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(SearchResult(
                title=title.strip(),
                url=url,
                snippet=snippet.strip(),
                source="duckduckgo"
            ))

        return results

    async def search_arxiv(
        self,
        query: str,
        num_results: int = 5,
        category: str = None
    ) -> List[SearchResult]:
        """
        Search arXiv for research papers.

        Args:
            query: Search query
            num_results: Number of results
            category: Optional arXiv category (cs.AI, cs.LG, q-fin, etc.)

        Returns:
            List of SearchResult objects
        """
        results = []

        search_query = quote_plus(query)
        if category:
            search_query = f"cat:{category}+AND+all:{search_query}"

        url = f"http://export.arxiv.org/api/query?search_query={search_query}&max_results={num_results}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        xml = await response.text()
                        results = self._parse_arxiv_xml(xml)
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

        return results

    def _parse_arxiv_xml(self, xml: str) -> List[SearchResult]:
        """Parse arXiv API XML response."""
        results = []

        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)

        for entry in entries:
            title_match = re.search(r'<title>([^<]+)</title>', entry)
            summary_match = re.search(r'<summary>([^<]+)</summary>', entry)
            link_match = re.search(r'<id>([^<]+)</id>', entry)

            if title_match and link_match:
                results.append(SearchResult(
                    title=title_match.group(1).strip().replace('\n', ' '),
                    url=link_match.group(1).strip(),
                    snippet=summary_match.group(1).strip()[:300] if summary_match else "",
                    source="arxiv"
                ))

        return results

    async def search_github(
        self,
        query: str,
        language: str = "python",
        num_results: int = 5
    ) -> List[SearchResult]:
        """Search GitHub for code examples and repositories."""
        results = []

        url = f"https://api.github.com/search/repositories?q={quote_plus(query)}+language:{language}&sort=stars&per_page={num_results}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "RALPH-Agent"
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        for repo in data.get("items", [])[:num_results]:
                            results.append(SearchResult(
                                title=repo.get("full_name", ""),
                                url=repo.get("html_url", ""),
                                snippet=repo.get("description", "") or "",
                                source="github",
                                relevance_score=repo.get("stargazers_count", 0)
                            ))
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")

        return results

    async def research_topic(
        self,
        topic: str,
        context: str = "",
        depth: str = "standard"
    ) -> ResearchFinding:
        """
        Comprehensive research on a topic.

        Args:
            topic: Main topic to research
            context: Additional context
            depth: Research depth - "quick", "standard", "thorough"

        Returns:
            ResearchFinding with aggregated results
        """
        all_results = []
        queries = self._build_research_queries(topic, context)

        if depth == "quick":
            num_web, num_arxiv, num_github = 5, 2, 2
        elif depth == "thorough":
            num_web, num_arxiv, num_github = 15, 10, 5
        else:
            num_web, num_arxiv, num_github = 10, 5, 3

        tasks = []
        for query in queries[:3]:
            tasks.append(self.search_web(query, num_web))
            tasks.append(self.search_arxiv(query, num_arxiv))
            tasks.append(self.search_github(query, "python", num_github))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_results.extend(result)

        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        return ResearchFinding(
            query=topic,
            results=unique_results,
            summary=self._generate_summary(topic, unique_results),
            key_insights=self._extract_insights(unique_results),
            recommended_actions=self._generate_recommendations(topic, unique_results),
            timestamp=datetime.now().isoformat()
        )

    def _build_research_queries(self, topic: str, context: str) -> List[str]:
        """Build multiple query variations."""
        queries = [topic]
        if context:
            queries.append(f"{topic} {context}")
        for mod in ["best practices", "implementation"][:2]:
            queries.append(f"{topic} {mod}")
        return queries

    def _generate_summary(self, topic: str, results: List[SearchResult]) -> str:
        """Generate a summary of findings."""
        if not results:
            return f"No results found for '{topic}'"
        sources = set(r.source for r in results)
        return f"Found {len(results)} results across {', '.join(sources)}."

    def _extract_insights(self, results: List[SearchResult]) -> List[str]:
        """Extract key insights."""
        insights = []
        arxiv_results = [r for r in results if r.source == "arxiv"]
        github_results = [r for r in results if r.source == "github"]

        if arxiv_results:
            insights.append(f"Found {len(arxiv_results)} research papers")
        if github_results:
            top = max(github_results, key=lambda x: x.relevance_score, default=None)
            if top:
                insights.append(f"Top repo: {top.title} ({int(top.relevance_score)} stars)")
        return insights

    def _generate_recommendations(self, topic: str, results: List[SearchResult]) -> List[str]:
        """Generate recommended actions."""
        if not results:
            return [f"Try broader search terms for '{topic}'"]
        return [
            f"Review top {min(3, len(results))} results",
            "Consider academic approaches from papers",
            "Examine GitHub implementations"
        ]

    async def search_best_practices(self, topic: str, technology: str = "python") -> ResearchFinding:
        """Search for best practices."""
        return await self.research_topic(f"{topic} best practices {technology}", depth="standard")

    async def search_cutting_edge(self, topic: str, field: str = "machine learning") -> ResearchFinding:
        """Search for cutting-edge advancements."""
        return await self.research_topic(f"{topic} latest advances 2024 2025", context=field, depth="thorough")

    async def get_crypto_sentiment(self, symbol: str) -> SentimentResult:
        """
        Get crypto sentiment from CryptoPanic API.

        Args:
            symbol: Crypto symbol (BTC, ETH, etc.)

        Returns:
            SentimentResult with sentiment analysis
        """
        api_key = os.getenv("CRYPTOPANIC_API_KEY")
        if not api_key:
            return SentimentResult(
                symbol=symbol,
                sentiment_score=0,
                sentiment_label="neutral",
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                headlines=[],
                source="CryptoPanic",
                error="CRYPTOPANIC_API_KEY not configured"
            )

        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={symbol}&filter=hot"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        return SentimentResult(
                            symbol=symbol,
                            sentiment_score=0,
                            sentiment_label="neutral",
                            bullish_count=0,
                            bearish_count=0,
                            neutral_count=0,
                            headlines=[],
                            source="CryptoPanic",
                            error=f"API returned status {response.status}"
                        )

                    data = await response.json()
                    results = data.get("results", [])

                    # Analyze sentiment from posts
                    bullish = 0
                    bearish = 0
                    neutral = 0
                    headlines = []

                    for post in results[:20]:  # Analyze top 20
                        votes = post.get("votes", {})
                        pos = votes.get("positive", 0)
                        neg = votes.get("negative", 0)

                        # Determine sentiment for this post
                        if pos > neg:
                            bullish += 1
                            sent_score = min(1, (pos - neg) / max(pos + neg, 1))
                        elif neg > pos:
                            bearish += 1
                            sent_score = max(-1, (pos - neg) / max(pos + neg, 1))
                        else:
                            neutral += 1
                            sent_score = 0

                        headlines.append({
                            "title": post.get("title", ""),
                            "url": post.get("url", ""),
                            "sentiment": sent_score,
                            "source": post.get("source", {}).get("title", "Unknown")
                        })

                    total = bullish + bearish + neutral
                    if total > 0:
                        # Calculate overall sentiment score
                        score = (bullish - bearish) / total
                        if score > 0.2:
                            label = "bullish"
                        elif score < -0.2:
                            label = "bearish"
                        else:
                            label = "neutral"
                    else:
                        score = 0
                        label = "neutral"

                    return SentimentResult(
                        symbol=symbol,
                        sentiment_score=score,
                        sentiment_label=label,
                        bullish_count=bullish,
                        bearish_count=bearish,
                        neutral_count=neutral,
                        headlines=headlines,
                        source="CryptoPanic"
                    )

        except Exception as e:
            logger.error(f"CryptoPanic API error: {e}")
            return SentimentResult(
                symbol=symbol,
                sentiment_score=0,
                sentiment_label="neutral",
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                headlines=[],
                source="CryptoPanic",
                error=str(e)
            )

    async def search_news(self, topic: str, num_results: int = 10) -> ResearchFinding:
        """
        Search for news on a topic using web search.

        Args:
            topic: Topic to search for
            num_results: Number of results to return

        Returns:
            ResearchFinding with news results
        """
        query = f"{topic} news"
        results = await self.search_web(query, num_results)

        return ResearchFinding(
            query=topic,
            results=results,
            summary=f"Found {len(results)} news items for '{topic}'",
            key_insights=[],
            recommended_actions=["Review top headlines for sentiment"],
            timestamp=datetime.now().isoformat()
        )


# Singleton
_research_tools: Optional[ResearchTools] = None


def get_research_tools() -> ResearchTools:
    """Get singleton research tools instance."""
    global _research_tools
    if _research_tools is None:
        _research_tools = ResearchTools()
    return _research_tools
