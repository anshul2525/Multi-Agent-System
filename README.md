<div align="center">

# 🔎 Multi-Agent Research System

**An autonomous AI research pipeline powered by LangChain, Groq, Tavily, and Streamlit.**  
*Performs live web intelligence gathering, deep DOM scraping, authoritative dossier synthesis, and peer review.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-agent-system-mfx4gvtzf3mht52nthyo4q.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2%2B-1C3C3C.svg?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq Fast Inference](https://img.shields.io/badge/Groq-LPU_Accelerated-f55036.svg)](https://groq.com/)
[![Tavily Search](https://img.shields.io/badge/Tavily-AI_Search-4F46E5.svg)](https://tavily.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

### 🌐 [**👉 Click Here to Launch the Live Demo 👈**](https://multi-agent-system-mfx4gvtzf3mht52nthyo4q.streamlit.app/)

---

</div>

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Dual-Tier Research Pipeline](#-dual-tier-research-pipeline)
  - [⚡ Standard Mode](#-standard-mode)
  - [🔬 Deep Mode](#-deep-mode)
- [Core Tool Engine (`tools.py`)](#-core-tool-engine-toolspy)
- [Resilience & Token-Bucket Pacing](#-resilience--token-bucket-pacing)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the App](#running-the-app)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [License](#-license)

---

## 🌟 Overview

The **Multi-Agent Research System** bridges the gap between raw web search and high-rigor academic reporting. Rather than relying on single-turn LLM generation that risks hallucination and context cutoff, this application utilizes an **autonomous multi-agent assembly line**:

1. **Search Agent**: Queries Tavily with strict single-execution protocols to collect live web citations.
2. **Reader Agent**: Inspects candidate URLs, cleans and strips DOM clutter, and extracts verified primary source text.
3. **Writer Chain**: Synthesizes verified facts into an authoritative, structured report.
4. **Critic Chain**: Emulates a senior peer-review editor to evaluate technical depth, detect bias, and deliver a benchmark score.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    User([User Query]) --> ModeSelect{Selected Mode}

    subgraph "Mode Configuration"
        ModeSelect -->|Standard Mode| StdTools["⚡ Standard Tools\n(web_search, scrape_urls)"]
        ModeSelect -->|Deep Mode| DeepTools["🔬 Deep Tools\n(deep_web_search, deep_scrape_urls)"]
    end

    subgraph "Autonomous Pipeline"
        StdTools & DeepTools --> SearchAgent["1. Search Agent\n(Tavily Live Intelligence)"]
        SearchAgent -->|Discovered Sources| Cooldown1["⏳ Inter-step Cooldown"]
        Cooldown1 --> ReaderAgent["2. Reader Agent\n(BS4 DOM Scraping & Extraction)"]
        ReaderAgent -->|Synthesized Evidence| Cooldown2["⏳ Inter-step Cooldown"]
        Cooldown2 --> WriterChain["3. Technical Writer Chain\n(Summary or Exhaustive Dossier)"]
        WriterChain -->|Draft Dossier| Cooldown3["⏳ Inter-step Cooldown"]
        Cooldown3 --> CriticChain["4. Critic Review Chain\n(Peer-Review & Rubric Scoring)"]
    end

    subgraph "Dashboard & Exports"
        CriticChain --> StreamlitUI["Streamlit Dashboard"]
        StreamlitUI --> Exp1["Markdown Summary (.md)"]
        StreamlitUI --> Exp2["Full Dossier (.md)"]
        StreamlitUI --> Exp3["Formatted PDF (.pdf)"]
    end
