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
```

The project includes both a baseline implementation (`career_chat_1`) and an optimized version (`career_chat_2`) used for retrieval experiments and evaluation.

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

## Optimization Summary

✅ Context enrichment (chunk summaries + project summaries)

✅ LLM reranking

✅ Chunk optimization

✅ Retrieval depth tuning

❌ Query rewriting removed after evaluation

The largest improvements came from enriching repository content with structured summaries and applying LLM-based reranking.

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

---

## Notes

Vector databases and cloned repositories are generated locally and are not included in the repository.

The knowledge base can be updated by re-ingesting GitHub repositories and rebuilding embeddings after adding new projects.