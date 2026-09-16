import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


def _get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")
    return TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        tavily = _get_tavily_client()
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results.get("results", []):
            out.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Snippet: {r.get('content', '')[:300]}\n"
            )
        return "\n----\n".join(out) if out else "No relevant search results found."
    except Exception as e:
        return f"Error executing web search: {e}"


@tool
def scrape_urls(urls: list[str]) -> str:
    """Attempt to scrape candidate URLs sequentially until one returns clean text content."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    for url in urls:
        clean_url = url.strip()
        if not clean_url.startswith(("http://", "https://")):
            continue
        try:
            resp = requests.get(clean_url, timeout=6, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            clean_text = " ".join(soup.stripped_strings)
            if len(clean_text) > 200:
                return f"Source URL: {clean_url}\n\n{clean_text[:1500]}"
        except Exception:
            continue
    return "Could not extract content from the provided URLs."
