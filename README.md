# AI Career Assistant — RAG Portfolio Chatbot

Interactive AI assistant answering questions about my projects, technical experience, and professional background.

Built using Retrieval-Augmented Generation (RAG) over GitHub repositories and portfolio information.

## Live Demo

Try the application:

👉 https://huggingface.co/spaces/JoannaW/AI_career_chatbot

The chatbot retrieves information from portfolio projects and generates grounded responses using a RAG pipeline.

---

## Features

- Interactive portfolio chatbot
- Retrieval-Augmented Generation (RAG)
- GitHub repository ingestion
- Repository-based retrieval
- LLM reranking
- Automated evaluation pipeline
- Synthetic test set generation

---

## Architecture

Question 
   ↓ 
Retriever (Top 20) 
   ↓ 
Reranker 
   ↓ 
Top 8 Context 
   ↓ 
LLM 
   ↓ 
Answer

---

## Knowledge Sources

The assistant combines two information sources:

- **Portfolio projects (RAG knowledge base)** built from GitHub repositories
- **Professional profile information** extracted from LinkedIn

Design principles:

- LinkedIn information is treated as the source of truth for career history and professional background.
- Repository content is treated as the source of truth for project implementation details.
- Project-related questions are answered using retrieved repository context whenever available.
- The assistant avoids generating unsupported information.

---

## Project Structure

```text
app.py                      # chatbot application
retriever.py                # retrieval pipeline
ingest.py                   # vector database creation

career_chat_1.ipynb         # baseline RAG implementation
career_chat_2.ipynb         # optimization and evaluation

creator_test_data.ipynb     # synthetic evaluation dataset
prepare_github_data.ipynb   # repository ingestion

linkedin.pdf                # professional profile information
project_summaries.json      # generated project summaries
tests.jsonl                 # RAG evaluation test set (queries + expected answers)
```

The project includes both a baseline implementation (`career_chat_1`) and an optimized version (`career_chat_2`) used for retrieval experiments and evaluation.

---

## Optimization Summary

✅ Context enrichment (chunk summaries + project summaries)

✅ LLM reranking

✅ Chunk optimization

✅ Retrieval depth tuning

❌ Query rewriting removed after evaluation

The largest improvements came from enriching repository content with structured summaries and applying LLM-based reranking.

---
## Evaluation Methodology & Synthetic Dataset

To rigorously evaluate the RAG pipeline, a synthetic evaluation dataset was generated using the pipeline implemented in `creator_test_data.ipynb` (leveraging `gpt-4.1-mini`). 

The evaluation dataset (`tests.jsonl`) consists of non-trivial, grounded question-answer pairs divided into three distinct reasoning categories:
1. **Atomic Facts & Local Reasoning:** Questions requiring precise information extraction from specific code chunks.
2. **Project-Level Reasoning:** Comprehensive questions built by analyzing combined chunk summaries across an entire repository.
3. **Cross-Project Evaluation:** Complex queries starting with *"Across projects..."* that require retrieving and comparing implementation strategies, metrics, or technologies between at least two different repositories.

Trivial or overly generic questions (e.g., *"What is the purpose of this repository?"*) were automatically filtered out during the pipeline cleanup phase to ensure high-quality benchmarking.

---

## Final Results

| Metric | Final |
|---|---:|
| MRR | 0.7285 |
| nDCG | 0.7264 |
| Coverage | 80.52% |
| Accuracy | 4.68 |
| Completeness | 4.46 |
| Relevance | 4.95 |

---

## Repository Usage & Local Development

This repository features an environment-aware hybrid architecture. It dynamically detects whether it is running in a local environment or deployed to the cloud, adjusting its database configuration automatically. 

### Reviewing the Code & Experiments Locally
To explore the implementation, experiments, and RAG evaluation pipeline on your machine:

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt```
2. **Run the chatbot interface locally:**
   ```bash
   python app.py```
3. **Explore the Notebooks & Experiments:**
   - Open `career_chat_1.ipynb` to review the baseline RAG exploration.
   - Open `career_chat_2.ipynb` to see the optimization experiments and metrics calculation.
   - Open `creator_test_data.ipynb` to analyze how the synthetic test dataset was generated.
---

## Deployment & Production Notes (Hugging Face Spaces)

The application code is fully production-optimized for cloud deployment. It utilizes conditional execution blocks to handle different environment states natively:

- **Local Development:** The system utilizes Chroma's `PersistentClient` to save the vector database (`vector_db/`) to the local disk, preventing redundant OpenAI embedding API costs and allowing offline repository ingestion.
- **Production (Hugging Face Spaces):** Due to strict stateless container architectures and read-only file system permissions (`readonly database error`), the application seamlessly switches to Chroma's `EphemeralClient()`. 
- **In-Memory Shared Runtime:** Upon container startup, `app.py` triggers the indexing pipeline (`ingest.py`), dynamically building the vector store in RAM. Using strategic local imports, the retrieval process (`retriever.py`) instantly shares this in-memory collection without disk access dependencies.

---

## Notes

- **Git Tracking:** Cloned repositories (`repos/`) and persistent local vector databases (`vector_db/`) are excluded from version control via `.gitignore` to maintain a clean repository structure.
- **Updating Knowledge Base:** To update the chatbot's knowledge base on Hugging Face, simply push new projects or modifications into the repository directory and trigger a **Factory Rebuild** in your Hugging Face Space settings. The container will automatically re-ingest all active repositories into memory on startup.