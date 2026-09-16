import re
import time

from groq import RateLimitError

from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain_deep,
    critic_chain_standard,
    writer_chain_deep,
    writer_chain_standard,
)


def _extract_content(output) -> str:
    if hasattr(output, "content"):
        return str(output.content)
    return str(output)


def _wait_seconds_from_error(exc: Exception, default: float = 15.0) -> float:
    match = re.search(r"try again in ([\d.]+)s", str(exc))
    if match:
        return float(match.group(1))
    return default


def _invoke_with_retry(fn, *args, max_retries: int = 8, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = _wait_seconds_from_error(e) + 3.0
            print(
                f"Rate limited by Groq (attempt {attempt}/{max_retries}). "
                f"Waiting {wait:.1f}s for token bucket replenishment..."
            )
            time.sleep(wait)


def run_research_pipeline(topic: str, deep_mode: bool = False) -> dict:
    state = {}
    config = {"recursion_limit": 15}
    mode_tag = "🔬 Deep Mode" if deep_mode else "⚡ Standard Mode"

    print("\n" + "=" * 60)
    print(f"Starting Pipeline [{mode_tag}] for topic: {topic}")
    print("=" * 60)

    # Step 1: Search Agent
    search_agent = build_search_agent(deep=deep_mode)
    prompt_search = (
        f"Perform deep web research on: {topic}."
        if deep_mode
        else f"Find information on: {topic}. Output findings with URLs."
    )
    search_result = _invoke_with_retry(
        search_agent.invoke,
        {"messages": [("user", prompt_search)]},
        config=config,
    )
    state["search_results"] = search_result["messages"][-1].content
    print("\n[Step 1 Complete] Search results captured.")

    # Inter-step cooldown (15s for Deep, 3s for Standard)
    cooldown = 15 if deep_mode else 3
    print(f"Token bucket cooldown ({cooldown}s)...")
    time.sleep(cooldown)

    # Step 2: Reader Agent
    reader_agent = build_reader_agent(deep=deep_mode)
    char_limit = 1500 if deep_mode else 700
    search_snippet = state["search_results"][:char_limit]

    reader_result = _invoke_with_retry(
        reader_agent.invoke,
        {
            "messages": [
                (
                    "user",
                    f"From these search findings, pick candidate URLs, scrape them, and summarize:\n\n{search_snippet}",
                )
            ]
        },
        config=config,
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    print("\n[Step 2 Complete] Scraping and source synthesis finished.")

    # Inter-step cooldown
    print(f"Token bucket cooldown ({cooldown}s)...")
    time.sleep(cooldown)

    # Step 3: Writer Chain
    writer = writer_chain_deep if deep_mode else writer_chain_standard
    research_text = (
        f"SEARCH INTELLIGENCE:\n{state['search_results'][:1600]}\n\n"
        f"SOURCE CONTENT:\n{state['scraped_content'][:2000]}"
        if deep_mode
        else f"SEARCH:\n{state['search_results'][:700]}\n\nCONTENT:\n{state['scraped_content'][:700]}"
    )

    writer_output = _invoke_with_retry(
        writer.invoke,
        {
            "topic": topic,
            "research": research_text,
        },
    )
    state["report"] = _extract_content(writer_output)
    print("\n[Step 3 Complete] Report drafted.")

    # Inter-step cooldown before critic
    critic_cooldown = 10 if deep_mode else 2
    print(f"Token bucket cooldown ({critic_cooldown}s)...")
    time.sleep(critic_cooldown)

    # Step 4: Critic Chain
    critic = critic_chain_deep if deep_mode else critic_chain_standard
    report_snippet = state["report"][:2200] if deep_mode else state["report"][:1200]

    critic_output = _invoke_with_retry(
        critic.invoke,
        {"report": report_snippet},
    )
    state["feedback"] = _extract_content(critic_output)
    print("\n[Step 4 Complete] Peer critique finished.")

    return state
