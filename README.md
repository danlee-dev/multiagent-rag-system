# Multi-Agent RAG System (Local Version)

## Overview
This project is a refactored local Python version of a Multi-Agent RAG (Retrieval-Augmented Generation) system originally developed in Colab. It is designed for food product development teams, featuring a persona-driven workflow, mock databases, and a modular multi-agent architecture based on LangChain and LangGraph.

---

## System Architecture
The Multi-Agent RAG system orchestrates a set of specialized agents, each responsible for a distinct stage in the information retrieval and answer generation pipeline. Agents communicate via real-time feedback channels, enabling iterative refinement and robust multi-source retrieval.

### Core Agents
- **PlanningAgent**: Analyzes the user query, decomposes it if necessary, and creates a query plan with required sub-queries and database selection.
- **RetrieverAgentXWithFeedback**: Focuses on graph database retrieval, performs keyword optimization, and collaborates with Retriever Y via feedback loops.
- **RetrieverAgentYWithFeedback**: Handles multi-source retrieval (vector DB, RDB, web search), receives hints from X, and provides feedback for further graph search.
- **CriticAgent1**: Evaluates the sufficiency of retrieved information and suggests improvements if needed.
- **ContextIntegratorAgent**: Integrates and organizes all search results into a coherent context for answer generation.
- **CriticAgent2**: Assesses the quality and reliability of the integrated context.
- **ReportGeneratorAgent**: Generates the final report or answer for the user.
- **SimpleAnswererAgent**: Handles simple queries directly using the vector database.

### Real-Time Feedback Channel
Agents X and Y communicate through an asynchronous feedback channel, exchanging hints and feedback to iteratively improve search results. This mechanism is key to the system's robustness and adaptability.

---

## Workflow Diagram
Below are two visualizations of the RAG system workflow:

### 1. LangGraph Workflow (Mermaid)
![LangGraph Workflow](static/langgraph-workflow-visualization(mermaid).png)

### 2. Feedback Channel & Agent Interaction (Graphviz)
![RAG System Feedback Channel](static/rag-system-workflow-diagram(graphviz).png)

---

## Folder Structure
```
.
├── main.py              # Main entry point
├── agents.py            # Agent class implementations
├── models.py            # Data models (Pydantic)
├── mock_databases.py    # Mock database implementations
├── utils.py             # Helper functions
├── requirements.txt     # Python dependencies
├── static/              # Workflow diagrams and images
├── test_multi-agent-rag.ipynb # Original Colab notebook
└── README.md            # Project documentation
```

---

## How to Run
1. **Python 3.9+ required**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your OpenAI API key:
   - Create a `.env` file with:
     ```
     OPENAI_API_KEY=sk-...
     ```
   - Or set the environment variable directly.
4. Run the main script:
   ```bash
   python main.py
   ```

---

## Key Files
- `main.py` : Entry point for the full workflow and sample queries
- `models.py` : Data models for agents, messages, and DB records
- `agents.py` : All core agent class implementations
- `mock_databases.py` : Mock DBs for local testing
- `utils.py` : Helper functions and message formatting
- `static/` : Workflow diagrams (PNG)

---

## Notes
- This project uses mock databases for demonstration; no real DB or external API is required.
- Built with LangChain, LangGraph, and OpenAI API.
- For more details, see the original Colab notebook or the code comments.
