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
        print(f"[Search Tool] Querying Tavily for: {query.strip()}")
        tavily = _get_tavily_client()
        results = tavily.search(query=query.strip(), max_results=3)
        res_list = results.get("results", [])
        print(f"[Search Tool] Received {len(res_list)} results.")
        out = []
        for r in res_list:
            title = r.get("title", "N/A").strip()
            url = r.get("url", "").strip()
            snippet = r.get("content", "")[:180].strip()
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
        return "\n----\n".join(out) if out else "No search results found. Do not retry calling this tool."
    except Exception as e:
        print(f"[Search Tool] Error during search: {e}")
        return f"Search failed with error: {e}. Do not retry calling this tool."


@tool
def scrape_urls(urls: list[str]) -> str:
    """Attempt to scrape candidate URLs sequentially until one returns clean text content."""
    print(f"[Scrape Tool] Scraping candidate URLs: {urls}")
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
                print(f"[Scrape Tool] Successfully scraped {clean_url}")
                return f"Source URL: {clean_url}\n\n{clean_text[:800]}"
        except Exception as e:
            print(f"[Scrape Tool] Failed {clean_url}: {e}")
            continue
    return "Could not extract content from the candidate URLs. Do not retry calling this tool."
