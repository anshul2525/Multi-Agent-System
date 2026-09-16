import re
import time

from groq import RateLimitError

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain


def _extract_content(output) -> str:
    """Helper function to cleanly extract text content from strings or AIMessage objects."""
    if hasattr(output, "content"):
        return str(output.content)
    return str(output)


def _wait_seconds_from_error(exc: Exception, default: float = 15.0) -> float:
    """Parse Groq's '...Please try again in 12.98s...' message for the wait time."""
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        return float(match.group(1))
    return default


def _invoke_with_retry(fn, *args, max_retries: int = 8, **kwargs):
    """Call fn(*args, **kwargs), automatically waiting and retrying on Groq
    rate-limit (429) errors using Groq's reported wait time with safety padding."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            parsed_wait = _wait_seconds_from_error(e)
            wait = parsed_wait + 3.0
            print(
                f"Rate limited by Groq (attempt {attempt}/{max_retries}). "
                f"Waiting {wait:.1f}s for token bucket replenishment..."
            )
            time.sleep(wait)


def run_research_pipeline(topic: str) -> dict:
    state = {}
    config = {"recursion_limit": 15}

    # Step 1: Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = _invoke_with_retry(
        search_agent.invoke,
        {
            "messages": [
                (
                    "user",
                    f"Call web_search once to find information about: {topic}. Then output the findings.",
                )
            ]
        },
        config=config,
    )
    state["search_results"] = search_result["messages"][-1].content
    print("\nSearch Results:\n", state["search_results"])

    # Cooldown pause: allows rolling TPM window to replenish
    print("\nPacing cooldown (6s)...")
    time.sleep(6)

    # Step 2: Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    search_context_trimmed = state["search_results"][:800]

    reader_agent = build_reader_agent()
    reader_result = _invoke_with_retry(
        reader_agent.invoke,
        {
            "messages": [
                (
                    "user",
                    f"From these search results, call scrape_urls once for 2 URLs and summarize findings:\n\n{search_context_trimmed}",
                )
            ]
        },
        config=config,
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\nScraped Content:\n", state["scraped_content"])

    # Cooldown pause
    print("\nPacing cooldown (6s)...")
    time.sleep(6)

    # Step 3: Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results'][:800]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content'][:1000]}"
    )

    writer_output = _invoke_with_retry(
        writer_chain.invoke,
        {
            "topic": topic,
            "research": research_combined,
        },
    )
    state["report"] = _extract_content(writer_output)
    print("\nFinal Report:\n", state["report"])

    # Cooldown pause
    print("\nPacing cooldown (4s)...")
    time.sleep(4)

    # Step 4: Critic Report
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    critic_output = _invoke_with_retry(
        critic_chain.invoke,
        {
            "report": state["report"][:1500],
        },
    )
    state["feedback"] = _extract_content(critic_output)
    print("\nCritic Report:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic_input = input("\nEnter a research topic: ")
    run_research_pipeline(topic_input)
