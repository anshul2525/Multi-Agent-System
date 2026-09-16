import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import deep_scrape_urls, deep_web_search, scrape_urls, web_search

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

llm = ChatGroq(
    model=model_name,
    temperature=0.1,
    api_key=groq_api_key,
    max_retries=0,
)

# --- Agent Builders (Standard & Deep) ---

def build_search_agent(deep: bool = False):
    tool = deep_web_search if deep else web_search
    tool_name = "deep_web_search" if deep else "web_search"
    return create_agent(
        model=llm,
        tools=[tool],
        system_prompt=(
            f"You are a search assistant. Follow this EXACT 2-step protocol:\n"
            f"1. Call {tool_name} exactly once with the search topic.\n"
            f"2. Once {tool_name} returns, DO NOT call any tool again under any circumstances.\n"
            f"Immediately write your final response listing the discovered sources and STOP."
        ),
    )


def build_reader_agent(deep: bool = False):
    tool = deep_scrape_urls if deep else scrape_urls
    tool_name = "deep_scrape_urls" if deep else "scrape_urls"
    return create_agent(
        model=llm,
        tools=[tool],
        system_prompt=(
            f"You are a research reader. Follow this EXACT 2-step protocol:\n"
            f"1. Call {tool_name} exactly once with candidate URLs from search results.\n"
            f"2. Once scraped text is returned, DO NOT call any tool again under any circumstances.\n"
            f"Immediately provide your summary of the findings and STOP."
        ),
    )


# --- Writer Chains ---

# Standard Writer (350-450 words)
writer_prompt_standard = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical writer. Produce concise, clear summaries without trailing off."),
    ("human", """Write a concise research summary on: {topic}

Research Material:
{research}

Format:
1. Overview (1 concise paragraph)
2. Key Findings (3 bullet points)
3. Conclusion (1 short paragraph)
4. Sources (list of discovered URLs)"""),
])
writer_chain_standard = writer_prompt_standard | llm | StrOutputParser()

# Deep Writer (750-1000+ words)
writer_prompt_deep = ChatPromptTemplate.from_messages([
    ("system", "You are an elite research scientist. Author thorough, exhaustive, multi-section dossiers."),
    ("human", """Write an authoritative, in-depth research dossier on: {topic}

Research Material:
{research}

Requirements:
- Length: Extensive (around 750 to 1,000 words).
- Sections:
  # Executive Summary & Core Challenge
  # Key Technical Breakthroughs & Architecture
  # Empirical Evidence, Benchmarks & Trade-offs
  # Strategic Impact & 2-5 Year Outlook
  # Authoritative Sources (Numbered with notes)

Ensure comprehensive technical depth throughout without cutting off."""),
])
writer_chain_deep = writer_prompt_deep | llm | StrOutputParser()


# --- Critic Chains ---

critic_prompt_standard = ChatPromptTemplate.from_messages([
    ("system", "You are a constructive research critic."),
    ("human", """Evaluate this brief summary strictly:
{report}

Format:
Score: X/10
Strengths:
- ...
Areas to Improve:
- ...
One line verdict: ..."""),
])
critic_chain_standard = critic_prompt_standard | llm | StrOutputParser()

critic_prompt_deep = ChatPromptTemplate.from_messages([
    ("system", "You are a senior peer-review journal editor. Conduct an exhaustive critical evaluation."),
    ("human", """Conduct a rigorous peer evaluation of this research dossier:
{report}

Format:
### 📊 Overall Score: X/10
### 🔬 Technical Depth & Rigor
- (Assess mechanisms, data points, and technical soundness)
### 💡 Core Strengths
- ...
### ⚠️ Potential Gaps or Biases
- ...
### 🎯 Final Strategic Verdict
..."""),
])
critic_chain_deep = critic_prompt_deep | llm | StrOutputParser()
