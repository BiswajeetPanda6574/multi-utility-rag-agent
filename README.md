# Multi-Utility RAG Agent (LangGraph & Streamlit)

An advanced, stateful conversational AI agent built with LangGraph and Streamlit. This application emphasizes strict state management, memory persistence, and modular architecture to orchestrate complex LLM workflows. The agent dynamically routes queries between general conversation, tool execution, and document retrieval.

## ✅ Core Architecture & Completed Features
* **Stateful Orchestration:** Built on LangGraph for robust agentic routing, deterministic controls, and clean separation of backend logic from frontend interfaces.
* **Human-in-the-Loop (HITL):** Graph execution strategically pauses for user UI approval before executing sensitive external tool calls.
* **Retrieval-Augmented Generation (RAG):** Ingests PDFs, chunks text, and retrieves context using FAISS and local HuggingFace embeddings (`all-MiniLM-L6-v2`).
* **Session Persistence:** Utilizes SQLite memory checkpointers to maintain multi-threaded chat history across different browser sessions.
* **Agentic Tool Calling:** Capable of executing mathematical logic, executing code, and scraping the web based on LLM decision-making (powered by Groq Llama-3).

## 🚀 Tech Stack
* **Frontend:** Streamlit
* **Backend:** LangGraph, LangChain, SQLite, Python
* **AI/ML:** Groq API (Llama 3.3 / Llama 3), FAISS, Sentence Transformers (HuggingFace)

## 🚧 Known Limitations & Development Roadmap
* **Cloud Web Search Rate-Limiting:** The current implementation uses the free `duckduckgo-search` library. While this works flawlessly in local development, DuckDuckGo aggressively blocks/rate-limits queries originating from Streamlit Community Cloud IPs (resulting in a `DDGSException`). **Roadmap:** Migrate to an official API (e.g., Tavily Search or OpenWeatherMap) for stable cloud execution.
* **Model Context Protocol (MCP) Integration:** Building a standardized MCP client to allow the LangGraph backend to securely interface with external local filesystem servers and enterprise tools over standard I/O.
* **Multi-Agent Subgraphs:** Architecting a "Supervisor Node" to delegate specialized tasks to independent sub-agents (e.g., separating a dedicated Research Agent from a Quantitative/Math Agent).
* **Dynamic Chat State UI:** Implementing background LLM calls to map raw SQLite thread UUIDs into human-readable chat titles based on the user's initial prompt context.

## 💻 Local Setup
To run this project locally and bypass cloud IP blocks:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your `GROQ_API_KEY`.
4. Run the app: `python -m streamlit run app.py`

## 🚧 In Progress & Upcoming Features

* **Official Search API Migration (Hotfix in progress):** Swapping the free `duckduckgo-search` scraper for an enterprise-grade API (like Tavily Search) to resolve Streamlit Cloud IP rate-limiting and ensure stable, unblocked web execution.
* **Model Context Protocol (MCP) Integration:** Implementing a standardized MCP client to allow the LangGraph backend to securely interface with local filesystem servers and external enterprise tools over standard I/O.
* **Multi-Agent Subgraph Architecture:** Evolving the single-agent graph into a "Supervisor Node" architecture that intelligently delegates specialized tasks to independent, domain-specific sub-agents (e.g., separating a dedicated Research Agent from a Math Agent).
* **Dynamic Chat State UI:** Adding background LLM calls to automatically map raw SQLite thread UUIDs into clean, human-readable chat titles in the sidebar based on the context of the user's first prompt.
