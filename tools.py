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


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    try:
        tavily = _get_tavily_client()
        # Limit to 3 results and 180 chars to conserve Groq TPM quota
        results = tavily.search(query=query.strip(), max_results=3)
        out = []
        for r in results.get("results", []):
            title = r.get("title", "N/A").strip()
            url = r.get("url", "").strip()
            snippet = r.get("content", "")[:180].strip()
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
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
            # Capped at 800 chars to avoid TPM spikes
            if len(clean_text) > 150:
                return f"Source URL: {clean_url}\n\n{clean_text[:800]}"
        except Exception:
            continue
    return "Could not extract content from the provided URLs."
