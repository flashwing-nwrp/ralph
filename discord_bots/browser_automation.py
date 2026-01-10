"""
Browser Automation for RALPH Agents

Provides browser capabilities using Playwright for:
- Web scraping with JavaScript rendering
- Screenshot capture
- Form interaction
- Data extraction from dynamic sites

Usage:
    from browser_automation import BrowserTool

    browser = BrowserTool()
    result = await browser.fetch_and_extract(
        url="https://polymarket.com/markets",
        task="Extract all active crypto markets"
    )
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger("browser_automation")

# Try to import Playwright
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = None  # Type stub for when Playwright not installed
    Page = None
    logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")


@dataclass
class BrowserResult:
    """Result from a browser operation."""
    success: bool
    content: str
    screenshot_path: Optional[str] = None
    error: Optional[str] = None
    url: str = ""
    title: str = ""
    extracted_data: Optional[Dict[str, Any]] = None


class BrowserTool:
    """
    Browser automation tool for RALPH agents.

    Allows agents to:
    - Navigate to URLs and extract content
    - Take screenshots for visual analysis
    - Interact with JavaScript-rendered pages
    - Extract structured data from web pages
    """

    def __init__(
        self,
        headless: bool = True,
        screenshot_dir: str = None,
        timeout: int = 30000  # 30 seconds
    ):
        self.headless = headless
        self.screenshot_dir = Path(screenshot_dir or os.getenv("RALPH_SCREENSHOT_DIR", "./screenshots"))
        self.timeout = timeout
        self._browser: Optional[Browser] = None
        self._playwright = None

        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def _ensure_browser(self) -> Browser:
        """Ensure browser is initialized."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium")

        if not self._browser:
            self._playwright = await async_playwright().start()
            # Use stealth settings to avoid headless detection
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',  # Avoid detection
                    '--disable-infobars',
                    '--window-size=1920,1080',
                ]
            )
            logger.info("Browser initialized (stealth mode)")

        return self._browser

    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def fetch_page(
        self,
        url: str,
        wait_for: str = "networkidle",
        take_screenshot: bool = False
    ) -> BrowserResult:
        """
        Fetch a page and return its content.

        Args:
            url: URL to fetch
            wait_for: Wait condition ('load', 'domcontentloaded', 'networkidle')
            take_screenshot: Whether to take a screenshot

        Returns:
            BrowserResult with page content
        """
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()

            # Navigate with timeout
            await page.goto(url, wait_until=wait_for, timeout=self.timeout)

            # Get page info
            title = await page.title()
            content = await page.content()

            # Extract text content (cleaner than HTML)
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, noscript');
                    scripts.forEach(s => s.remove());
                    return document.body.innerText;
                }
            """)

            screenshot_path = None
            if take_screenshot:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                screenshot_path = str(self.screenshot_dir / f"screenshot_{timestamp}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Screenshot saved: {screenshot_path}")

            await page.close()

            return BrowserResult(
                success=True,
                content=text_content,
                screenshot_path=screenshot_path,
                url=url,
                title=title
            )

        except Exception as e:
            logger.error(f"Browser fetch error: {e}")
            return BrowserResult(
                success=False,
                content="",
                error=str(e),
                url=url
            )

    async def extract_data(
        self,
        url: str,
        selectors: Dict[str, str],
        wait_for_selector: str = None
    ) -> BrowserResult:
        """
        Extract structured data from a page using CSS selectors.

        Args:
            url: URL to fetch
            selectors: Dict of {field_name: css_selector}
            wait_for_selector: Optional selector to wait for before extracting

        Returns:
            BrowserResult with extracted_data dict
        """
        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()

            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=self.timeout)

            # Extract data using selectors
            extracted = {}
            for field, selector in selectors.items():
                try:
                    elements = await page.query_selector_all(selector)
                    if len(elements) == 1:
                        extracted[field] = await elements[0].inner_text()
                    else:
                        extracted[field] = [await el.inner_text() for el in elements]
                except Exception as e:
                    extracted[field] = f"Error: {e}"

            title = await page.title()
            await page.close()

            return BrowserResult(
                success=True,
                content=str(extracted),
                url=url,
                title=title,
                extracted_data=extracted
            )

        except Exception as e:
            logger.error(f"Data extraction error: {e}")
            return BrowserResult(
                success=False,
                content="",
                error=str(e),
                url=url
            )

    async def fetch_polymarket_markets(
        self,
        category: str = "crypto"
    ) -> BrowserResult:
        """
        Fetch current Polymarket markets.

        Args:
            category: Market category ('crypto', 'politics', 'sports', etc.)

        Returns:
            BrowserResult with market data
        """
        url = f"https://polymarket.com/markets?_c={category}"

        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()

            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Wait for markets to load
            await page.wait_for_selector('[data-testid="market-card"]', timeout=10000)

            # Extract market data
            markets = await page.evaluate("""
                () => {
                    const cards = document.querySelectorAll('[data-testid="market-card"]');
                    return Array.from(cards).slice(0, 20).map(card => {
                        const title = card.querySelector('h3')?.innerText || '';
                        const price = card.querySelector('[data-testid="price"]')?.innerText || '';
                        const volume = card.querySelector('[data-testid="volume"]')?.innerText || '';
                        return { title, price, volume };
                    });
                }
            """)

            await page.close()

            return BrowserResult(
                success=True,
                content=f"Found {len(markets)} markets",
                url=url,
                title="Polymarket Markets",
                extracted_data={"markets": markets}
            )

        except Exception as e:
            logger.error(f"Polymarket fetch error: {e}")
            return BrowserResult(
                success=False,
                content="",
                error=str(e),
                url=url
            )

    async def search_web(
        self,
        query: str,
        num_results: int = 5
    ) -> BrowserResult:
        """
        Search the web using DuckDuckGo (no API key needed).

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            BrowserResult with search results
        """
        url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"

        try:
            browser = await self._ensure_browser()
            page = await browser.new_page()

            await page.goto(url, wait_until="networkidle", timeout=self.timeout)

            # Wait for results
            await asyncio.sleep(2)  # DuckDuckGo loads dynamically

            # Extract search results
            results = await page.evaluate(f"""
                () => {{
                    const results = document.querySelectorAll('[data-testid="result"]');
                    return Array.from(results).slice(0, {num_results}).map(r => {{
                        const title = r.querySelector('h2')?.innerText || '';
                        const snippet = r.querySelector('[data-result="snippet"]')?.innerText || '';
                        const link = r.querySelector('a')?.href || '';
                        return {{ title, snippet, link }};
                    }});
                }}
            """)

            await page.close()

            return BrowserResult(
                success=True,
                content=f"Found {len(results)} results for: {query}",
                url=url,
                title=f"Search: {query}",
                extracted_data={"query": query, "results": results}
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return BrowserResult(
                success=False,
                content="",
                error=str(e),
                url=url
            )


# Singleton instance
_browser_tool: Optional[BrowserTool] = None


def get_browser_tool() -> BrowserTool:
    """Get or create the browser tool instance."""
    global _browser_tool
    if _browser_tool is None:
        _browser_tool = BrowserTool()
    return _browser_tool


async def cleanup_browser():
    """Cleanup browser on shutdown."""
    global _browser_tool
    if _browser_tool:
        await _browser_tool.close()
        _browser_tool = None
