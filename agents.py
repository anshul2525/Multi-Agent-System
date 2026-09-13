import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import scrape_urls, web_search

load_dotenv()

# app.py already copies st.secrets into os.environ before importing this
# module (see app.py), so a plain os.getenv() call here is enough and
# works both locally (.env) and on Streamlit Cloud (st.secrets).
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key,
)

# 1st agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="You are a search assistant. Call web_search once to find relevant resources and output the findings with full URLs."
    )

# 2nd agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_urls],
        system_prompt=(
            "You are a research reader. Call scrape_urls exactly once using a list of 2-3 candidate URLs from the search results. "
            "Once scraped content is returned, do not invoke any more tools. Immediately summarize the findings and complete your turn."
        )
    )

# 3rd component: Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical writer. You must generate complete, concise, and structured reports without stopping mid-sentence."),
    ("human", """Write a concise research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Requirements:
- Total length: around 450-550 words.
- Structure:
  1. Introduction (1 concise paragraph)
  2. Key Findings (exactly 3 focused points, use 3-4 sentences per point)
  3. Conclusion (1 short wrap-up paragraph)
  4. Sources (list of discovered URLs)

Ensure you write the full report all the way to the Sources section without cutting off."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# 4th component: Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

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
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
