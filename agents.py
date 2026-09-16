import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import scrape_urls, web_search

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

llm = ChatGroq(
    model=model_name,
    temperature=0,
    api_key=groq_api_key,
    max_retries=0,
)

# 1st agent: Search Agent (with strict single-turn termination)
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a dedicated search assistant. Follow this EXACT 2-step protocol:\n"
            "Step 1: Call web_search exactly once using the user topic.\n"
            "Step 2: Once web_search returns results, DO NOT invoke web_search or any other tool again under any circumstances.\n"
            "Immediately write a final text response containing the titles, URLs, and snippets, and STOP."
        ),
    )

# 2nd agent: Reader Agent (with strict single-turn termination)
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_urls],
        system_prompt=(
            "You are a research reader assistant. Follow this EXACT 2-step protocol:\n"
            "Step 1: Call scrape_urls exactly once with 2 candidate URLs from the search results.\n"
            "Step 2: Once scraped content is returned, DO NOT invoke scrape_urls or any other tool again under any circumstances.\n"
            "Immediately write a final 2-paragraph summary of the scraped content and STOP."
        ),
    )

# 3rd component: Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert technical writer. You must generate complete, concise, and structured reports without stopping mid-sentence.",
    ),
    (
        "human",
        """Write a concise research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Requirements:
- Total length: around 350-450 words.
- Structure:
  1. Introduction (1 concise paragraph)
  2. Key Findings (3 focused bullet points)
  3. Conclusion (1 short wrap-up paragraph)
  4. Sources (list of discovered URLs)

Ensure you write the full report all the way to the Sources section without cutting off.""",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# 4th component: Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific.",
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
