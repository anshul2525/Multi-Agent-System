import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import scrape_urls, web_search

load_dotenv()

# .strip() prevents newline or whitespace breakage from secrets/environment
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

# Active Groq model (openai/gpt-oss-120b or openai/gpt-oss-20b)
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

llm = ChatGroq(
    model=model_name,
    temperature=0,
    api_key=groq_api_key,
    max_retries=0,  # pipeline.py manages rate-limit retries explicitly
)

# 1st agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a search assistant. Call web_search once to find 2-3 reliable resources. "
            "Output the findings with concise titles and full URLs."
        ),
    )

# 2nd agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_urls],
        system_prompt=(
            "You are a research reader. Call scrape_urls once with 2 candidate URLs from the search results. "
            "Once scraped content is returned, do not call any more tools. Summarize key points in 2 paragraphs and finish."
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
