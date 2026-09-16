import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing or empty.")
    return TavilyClient(api_key=api_key)


# --- Standard Tools (Lightweight, Token-Efficient) ---

@tool
def web_search(query: str) -> str:
    """Standard web search for fast, concise information. Returns 3 source snippets."""
    try:
        print(f"[Standard Search] Querying: {query.strip()}")
        tavily = _get_tavily_client()
        results = tavily.search(query=query.strip(), max_results=3)
        res_list = results.get("results", [])
        out = []
        for r in res_list:
            title = r.get("title", "N/A").strip()
            url = r.get("url", "").strip()
            snippet = r.get("content", "")[:180].strip()
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
        return "\n----\n".join(out) if out else "No search results found. Do not retry."
    except Exception as e:
        return f"Search error: {e}. Do not retry."


@tool
def scrape_urls(urls: list[str]) -> str:
    """Standard web scraper. Extracts up to 600 characters from the first valid URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    for url in urls:
        clean_url = str(url).strip()
        if not clean_url.startswith(("http://", "https://")):
            continue
        try:
            resp = requests.get(clean_url, timeout=6, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            clean_text = " ".join(soup.stripped_strings)
            if len(clean_text) > 150:
                return f"Source URL: {clean_url}\n\n{clean_text[:600]}"
        except Exception:
            continue
    return "Could not extract content from candidate URLs. Do not retry."


# --- Deep Research Tools (Comprehensive Intelligence) ---

@tool
def deep_web_search(query: str) -> str:
    """Deep web search with advanced crawling and multi-source AI synthesis."""
    try:
        print(f"[Deep Search] Crawling Tavily Advanced for: {query.strip()}")
        tavily = _get_tavily_client()
        results = tavily.search(
            query=query.strip(),
            search_depth="advanced",
            include_answer=True,
            max_results=5,
        )
        out = []
        if results.get("answer"):
            out.append(f"### EXECUTIVE SYNTHESIS:\n{results['answer']}\n")

        res_list = results.get("results", [])
        out.append("### DISCOVERED SOURCES:")
        for r in res_list:
            title = r.get("title", "N/A").strip()
            url = r.get("url", "").strip()
            snippet = r.get("content", "")[:350].strip()
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
        return "\n----\n".join(out) if out else "No deep search results found. Do not retry."
    except Exception as e:
        return f"Deep search error: {e}. Do not retry."


@tool
def deep_scrape_urls(urls: list[str]) -> str:
    """Deep web scraper. Extracts up to 1,200 characters each from the top 2 authoritative URLs."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    scraped_data = []
    for url in urls[:2]:
        clean_url = str(url).strip()
        if not clean_url.startswith(("http://", "https://")):
            continue
        try:
            resp = requests.get(clean_url, timeout=8, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                tag.decompose()
            clean_text = " ".join(soup.stripped_strings)
            if len(clean_text) > 150:
                scraped_data.append(f"Source URL: {clean_url}\nContent:\n{clean_text[:1200]}")
        except Exception:
            continue
    return "\n\n=====\n\n".join(scraped_data) if scraped_data else "Could not extract content. Do not retry."
