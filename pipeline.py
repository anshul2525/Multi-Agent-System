from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}
    config = {"recursion_limit": 15}

    # Step 1: Search Agent
    print("\n" + "=" * 50)
    print("Step 1 - Search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]},
        config=config
    )
    state["search_results"] = search_result['messages'][-1].content
    print("\nSearch Results:\n", state["search_results"])

    # Step 2: Reader Agent
    print("\n" + "=" * 50)
    print("Step 2 - Reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke(
        {"messages": [("user",
            f"From the search results below, pick the 2 to 3 most relevant URLs and scrape them:\n\n{state['search_results']}"
        )]},
        config=config
    )
    state["scraped_content"] = reader_result['messages'][-1].content
    print("\nScraped Content:\n", state["scraped_content"])

    # Step 3: Writer Chain
    print("\n" + "=" * 50)
    print("Step 3 - Writer is drafting the report...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    print("\nFinal Report:\n", state["report"])

    # Step 4: Critic Report
    print("\n" + "=" * 50)
    print("Step 4 - Critic is reviewing the report...")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
    print("\nCritic Report:\n", state["feedback"])

    return state

if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)