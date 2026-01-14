"""Web search for gathering context before code generation.

Supports multiple search backends:
- DuckDuckGo Search (primary) - MIT licensed, free, no API key
- SerpAPI (optional) - requires SERPAPI_API_KEY, more comprehensive
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import quote_plus, unquote

import httpx

# Import ddgs library (MIT licensed) - formerly duckduckgo-search
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old package name
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False

from atlas.core.config import Config, get_config

logger = logging.getLogger(__name__)

# SerpAPI endpoint (optional paid service)
SERPAPI_URL = "https://serpapi.com/search"


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class WebSearchResults:
    """Results from a web search."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str | None = None

    @property
    def combined_content(self) -> str:
        """Get all results combined into context string."""
        if not self.results:
            return ""

        parts = []
        for r in self.results:
            parts.append(f"### {r.title}\nSource: {r.url}\n{r.snippet}")

        return "\n\n".join(parts)


class WebSearchClient:
    """Client for web searches to gather coding context.

    Uses SerpAPI (if API key available) or DuckDuckGo HTML scraping as fallback.
    SerpAPI provides more comprehensive and reliable results.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the web search client."""
        self.config = config or get_config()
        self._cache: dict[str, WebSearchResults] = {}

        # Check for SerpAPI key (from config or environment)
        self.serpapi_key = self.config.serpapi_api_key or os.getenv("SERPAPI_API_KEY", "")

        # Allowlist of trusted domains for code-related searches
        # This filter can be disabled by passing filter_domains=False to search()
        self.trusted_domains = [
            # Q&A and community
            "stackoverflow.com",
            "stackexchange.com",
            "github.com",
            "gist.github.com",
            # Python docs
            "docs.python.org",
            "python.org",
            "pypi.org",
            "readthedocs.io",
            # JavaScript/TypeScript ecosystem
            "developer.mozilla.org",
            "reactjs.org",
            "react.dev",
            "nextjs.org",
            "vuejs.org",
            "angular.io",
            "svelte.dev",
            "tanstack.com",
            "trpc.io",
            "prisma.io",
            "typescriptlang.org",
            "nodejs.org",
            "npmjs.com",
            "deno.land",
            "bun.sh",
            # Python frameworks
            "docs.djangoproject.com",
            "fastapi.tiangolo.com",
            "flask.palletsprojects.com",
            "pydantic.dev",
            "sqlalchemy.org",
            # Data science / ML
            "numpy.org",
            "pandas.pydata.org",
            "pytorch.org",
            "tensorflow.org",
            "scikit-learn.org",
            "huggingface.co",
            # Cloud providers
            "learn.microsoft.com",
            "cloud.google.com",
            "aws.amazon.com",
            "docs.aws.amazon.com",
            "vercel.com",
            "netlify.com",
            # Rust/Go/Other languages
            "doc.rust-lang.org",
            "docs.rs",
            "go.dev",
            "pkg.go.dev",
            "kotlinlang.org",
            "docs.swift.org",
            # Tutorials and blogs
            "medium.com",
            "dev.to",
            "realpython.com",
            "geeksforgeeks.org",
            "tutorialspoint.com",
            "w3schools.com",
            "freecodecamp.org",
            "towardsdatascience.com",
            "hackernoon.com",
            "css-tricks.com",
            "smashingmagazine.com",
            # Python-specific tutorials
            "superfastpython.com",
            "pythontutorial.net",
            "pytutorial.com",
            "zetcode.com",
            "programiz.com",
            "digitalocean.com",
            "linuxize.com",
            # Tech blogs
            "bomberbot.com",
            "runebook.dev",
            "readmedium.com",
            # Python community
            "discuss.python.org",
            "mail.python.org",
            "nedbatchelder.com",
            "blog.python.org",
            # More tech sites
            "baeldung.com",
            "guru99.com",
            "javatpoint.com",
            "pythonguides.com",
            # UI component libraries
            "ui.shadcn.com",
            "tailwindcss.com",
            "chakra-ui.com",
            "mui.com",
            "ant.design",
            "mantine.dev",
        ]

    async def search(
        self,
        query: str,
        max_results: int = 5,
        site_filter: str | None = None,
        filter_domains: bool = False,  # Disabled by default - trust DuckDuckGo ranking
    ) -> WebSearchResults:
        """Perform a web search.

        Uses SerpAPI if available, otherwise falls back to DuckDuckGo scraping.

        Args:
            query: Search query
            max_results: Maximum number of results
            site_filter: Optional site to restrict search to (e.g., "stackoverflow.com")
            filter_domains: If True, filter results to trusted domains only.
                           Set to False to allow any domain (useful for niche frameworks).

        Returns:
            WebSearchResults with search results
        """
        # Check cache
        cache_key = f"{query}:{site_filter}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build search query
        search_query = query
        if site_filter:
            search_query = f"site:{site_filter} {query}"

        try:
            # Try DuckDuckGo Search library first (MIT licensed, free)
            if DDGS_AVAILABLE:
                results = await self._search_duckduckgo(search_query, max_results)
            # Fallback to SerpAPI if available and DDG fails
            elif self.serpapi_key:
                results = await self._search_serpapi(search_query, max_results)
            else:
                # Last resort: manual HTML scraping
                results = await self._search_duckduckgo_html(search_query, max_results)

            # Filter to trusted domains (unless disabled or site filter specified)
            if filter_domains and not site_filter:
                filtered_results = [
                    r for r in results.results
                    if any(domain in r.url for domain in self.trusted_domains)
                ]
                results.results = filtered_results[:max_results]
            else:
                results.results = results.results[:max_results]

            # Cache results
            self._cache[cache_key] = results
            return results

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return WebSearchResults(query=query, error=str(e))

    async def _search_serpapi(
        self,
        query: str,
        max_results: int,
    ) -> WebSearchResults:
        """Search using SerpAPI (Google results).

        SerpAPI provides clean, structured results from Google search.
        """
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "engine": "google",
            "num": max_results * 2,  # Get more to filter
            "hl": "en",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                SERPAPI_URL,
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        results = []

        # Parse organic results
        for item in data.get("organic_results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="serpapi",
            ))

        # Also check answer box if available
        if "answer_box" in data:
            answer = data["answer_box"]
            snippet = answer.get("answer", answer.get("snippet", ""))
            if snippet:
                results.insert(0, SearchResult(
                    title=answer.get("title", "Featured Answer"),
                    url=answer.get("link", ""),
                    snippet=snippet,
                    source="serpapi_featured",
                ))

        return WebSearchResults(query=query, results=results)

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> WebSearchResults:
        """Search using duckduckgo-search library (MIT licensed, free).

        Uses the DDGS class from the duckduckgo-search package.
        """
        results = []

        try:
            # Run synchronous DDGS in thread pool to not block async
            loop = asyncio.get_running_loop()
            ddg_results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results * 2))
            )

            for item in ddg_results[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                    source="duckduckgo",
                ))

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            # Try HTML fallback
            return await self._search_duckduckgo_html(query, max_results)

        return WebSearchResults(query=query, results=results)

    async def _search_duckduckgo_html(
        self,
        query: str,
        max_results: int,
    ) -> WebSearchResults:
        """Fallback: Search using DuckDuckGo HTML scraping (less reliable)."""

        encoded_query = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            html = response.text

        # Parse results from HTML
        results = self._parse_duckduckgo_html(html, max_results)

        return WebSearchResults(query=query, results=results)

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # Find result blocks
        # DuckDuckGo HTML uses class="result" for each result
        result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*)*)</a>'

        # Find all links
        links = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(links[:max_results]):
            snippet = ""
            if i < len(snippets):
                # Clean HTML from snippet
                snippet = re.sub(r'<[^>]+>', '', snippets[i])
                snippet = snippet.strip()

            # Decode URL if needed
            if url.startswith("//duckduckgo.com/l/?uddg="):
                # Extract actual URL from redirect
                url_match = re.search(r'uddg=([^&]+)', url)
                if url_match:
                    from urllib.parse import unquote
                    url = unquote(url_match.group(1))

            results.append(SearchResult(
                title=title.strip(),
                url=url,
                snippet=snippet,
                source="duckduckgo",
            ))

        return results

    async def fetch_page_content(
        self,
        url: str,
        max_chars: int = 5000,
    ) -> str:
        """Fetch and extract main content from a web page.

        Args:
            url: URL to fetch
            max_chars: Maximum characters to return

        Returns:
            Extracted text content
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, headers=headers, timeout=15.0)
                response.raise_for_status()
                html = response.text

            # Extract text content - simple HTML stripping
            # Remove script and style tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text[:max_chars]

        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return ""

    async def search_for_code_context(
        self,
        issue_description: str,
        libraries: list[str] | None = None,
    ) -> WebSearchResults:
        """Search specifically for code-related context.

        Performs multiple targeted searches and optionally fetches page content
        for comprehensive research results.

        Args:
            issue_description: The issue to search for
            libraries: Optional list of libraries to focus on

        Returns:
            Combined search results with rich content
        """
        all_results = []
        searches = []

        # Create targeted search queries
        base_query = issue_description.lower()

        # 1. Best practices search
        searches.append(self.search(
            f"{base_query} best practices",
            max_results=3,
        ))

        # 2. Stack Overflow - Solutions
        searches.append(self.search(
            f"{base_query} solution example",
            max_results=3,
            site_filter="stackoverflow.com",
        ))

        # 3. Official documentation
        if libraries:
            for lib in libraries[:2]:
                searches.append(self.search(
                    f"{lib} documentation {base_query}",
                    max_results=2,
                ))

        # 4. GitHub examples
        searches.append(self.search(
            f"{base_query} implementation",
            max_results=2,
            site_filter="github.com",
        ))

        # 5. Real Python / tutorials
        searches.append(self.search(
            f"python {base_query} tutorial",
            max_results=2,
        ))

        # 6. CRITICAL: Python-specific best practices patterns
        # These searches target common antipatterns and their fixes
        python_best_practice_queries = [
            "python time.monotonic vs time.time which to use",
            "python asyncio.Lock vs threading.Lock difference",
            "python asyncio.sleep vs time.sleep blocking event loop",
            "python async with lock context manager",
        ]
        for query in python_best_practice_queries:
            searches.append(self.search(query, max_results=2))

        # 7. Official Python documentation (more reliable than Stack Overflow)
        searches.append(self.search(
            "python time module monotonic clock",
            max_results=2,
            site_filter="docs.python.org",
        ))
        searches.append(self.search(
            "python asyncio synchronization Lock",
            max_results=2,
            site_filter="docs.python.org",
        ))

        # 8. Real Python tutorials (high quality, scrapeable)
        searches.append(self.search(
            f"python {base_query}",
            max_results=2,
            site_filter="realpython.com",
        ))
        searches.append(self.search(
            "python asyncio tutorial",
            max_results=2,
            site_filter="superfastpython.com",
        ))

        # Execute all searches in parallel
        results = await asyncio.gather(*searches, return_exceptions=True)

        for result in results:
            if isinstance(result, WebSearchResults) and result.results:
                all_results.extend(result.results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        # Fetch full page content for ALL top results to get comprehensive context
        # DuckDuckGo/ddgs snippets are often too short (100-300 chars)
        # This is critical for discovering best practices
        enhanced_results = []
        fetch_tasks = []

        for r in unique_results[:12]:  # Increased from 8 to 12
            # ALWAYS fetch page content regardless of snippet length
            # This ensures we get rich context like the HTML scraping approach
            # Skip Stack Overflow (blocks scraping) - keep original snippet
            if r.url and "stackoverflow.com" not in r.url:
                fetch_tasks.append((r, self.fetch_page_content(r.url, max_chars=8000)))
            else:
                enhanced_results.append(r)

        # Fetch content in parallel
        if fetch_tasks:
            contents = await asyncio.gather(
                *[task for _, task in fetch_tasks],
                return_exceptions=True
            )
            for (r, _), content in zip(fetch_tasks, contents):
                if isinstance(content, str) and content:
                    # Use up to 6000 chars of fetched content for rich context
                    r.snippet = content[:6000]
                enhanced_results.append(r)

        return WebSearchResults(
            query=issue_description,
            results=enhanced_results,
        )

    async def search_multiple(
        self,
        queries: list[str],
        max_results_per_query: int = 3,
    ) -> dict[str, WebSearchResults]:
        """Search multiple queries in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Max results per query

        Returns:
            Dict mapping queries to their results
        """
        tasks = [
            self.search(query, max_results_per_query)
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            query: result if not isinstance(result, Exception)
                   else WebSearchResults(query=query, error=str(result))
            for query, result in zip(queries, results)
        }

    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()
