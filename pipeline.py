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


def _invoke_with_retry(fn, *args, max_retries: int = 6, **kwargs):
    """Call fn(*args, **kwargs), automatically waiting and retrying on Groq
    rate-limit (429) errors, using the wait time Groq reports plus backoff buffer."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            parsed_wait = _wait_seconds_from_error(e)
            # Add progressive buffer (min 5s, scaling with attempt count)
            # to let rolling 60-second token windows drain properly.
            wait = max(parsed_wait + 3.0, attempt * 5.0)
            print(
                f"Rate limited by Groq (attempt {attempt}/{max_retries}). "
                f"Waiting {wait:.1f}s before retrying..."
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
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]},
        config=config,
    )
    state["search_results"] = search_result["messages"][-1].content
    print("\nSearch Results:\n", state["search_results"])

    # Pacing pause to prevent burst TPM exhaustion
    time.sleep(3)

    # Step 2: Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    # Limit search results context to 1,200 chars to avoid spiking Step 2 input tokens
    search_context_trimmed = state["search_results"][:1200]

    reader_agent = build_reader_agent()
    reader_result = _invoke_with_retry(
        reader_agent.invoke,
        {
            "messages": [
                (
                    "user",
                    f"From the search results below, pick the 2 to 3 most relevant URLs and scrape them:\n\n{search_context_trimmed}",
                )
            ]
        },
        config=config,
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\nScraped Content:\n", state["scraped_content"])

    # Pacing pause
    time.sleep(3)

    # Step 3: Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results'][:1500]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content'][:2000]}"
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

    # Pacing pause
    time.sleep(2)

    # Step 4: Critic Report
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    critic_output = _invoke_with_retry(
        critic_chain.invoke,
        {
            "report": state["report"],
        },
    )
    state["feedback"] = _extract_content(critic_output)
    print("\nCritic Report:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic_input = input("\nEnter a research topic: ")
    run_research_pipeline(topic_input)
