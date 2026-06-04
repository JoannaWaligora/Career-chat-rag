# AI Career Assistant — RAG Portfolio Chatbot

Interactive AI assistant answering questions about my projects, technical experience, and professional background.

Built using Retrieval-Augmented Generation (RAG) over GitHub repositories and portfolio information.

## Live Demo

The chatbot retrieves repository content, reranks relevant chunks, and generates grounded answers using GPT-4.1-mini.

Try the application:

👉 https://huggingface.co/spaces/JoannaW/AI_career_chatbot

---

## Features

* Interactive portfolio chatbot
* Retrieval-Augmented Generation (RAG)
* GitHub repository ingestion
* Repository-based retrieval
* LLM reranking
* Automated evaluation pipeline
* Synthetic test set generation

---

## Tech Stack

* Python
* OpenAI API
* ChromaDB
* Hugging Face Spaces
* Gradio
* GitHub API
* RAG (Retrieval-Augmented Generation)
* LLM Reranking
* Synthetic Dataset Generation

---

## Architecture

```text
Question
    │
    ▼
Retriever (Top 20)
    │
    ▼
LLM Reranker
    │
    ▼
Top 8 Context Chunks
    │
    ▼
LLM
    │
    ▼
Answer
```

---

## Knowledge Sources

The assistant combines two information sources:

* **Portfolio projects (RAG knowledge base)** built from GitHub repositories
* **Professional profile information** extracted from LinkedIn

Design principles:

* LinkedIn information is treated as the source of truth for career history and professional background.
* Repository content is treated as the source of truth for project implementation details.
* Project-related questions are answered using retrieved repository context whenever available.
* The assistant avoids generating unsupported information.

---

## Project Structure

```text
app.py                      # chatbot application
retriever.py                # retrieval pipeline
ingest.py                   # vector database creation

notebooks/
├── career_chat_1.ipynb         # baseline RAG implementation
├── career_chat_2.ipynb         # optimization and evaluation
├── creator_test_data.ipynb     # synthetic evaluation dataset generation
└── prepare_github_data.ipynb   # repository ingestion and preprocessing

data/
├── project_summaries.json      # generated project summaries
└── tests.jsonl                 # RAG evaluation test set
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

1. **Atomic Facts & Local Reasoning**
   Questions requiring precise information extraction from specific code chunks.

2. **Project-Level Reasoning**
   Comprehensive questions built by analyzing combined chunk summaries across an entire repository.

3. **Cross-Project Evaluation**
   Questions requiring retrieval and comparison of implementation strategies, technologies, or results across multiple repositories.

Trivial and generic questions were automatically filtered out to create a more challenging evaluation benchmark.

---

## Final Results

The optimized pipeline was evaluated after expanding the knowledge base from approximately **155 to 250 indexed chunks** by adding an additional GitHub repository.

| Metric       |  Final |
| ------------ | -----: |
| MRR          | 0.7286 |
| nDCG         | 0.7314 |
| Coverage     | 81.24% |
| Accuracy     |   4.67 |
| Completeness |   4.48 |
| Relevance    |   4.90 |

The results demonstrate strong retrieval performance while maintaining high answer quality despite the larger retrieval space.

---

## Scalability Validation

After the initial optimization experiments, an additional GitHub repository (`career-chat-rag`) was added to the knowledge base, increasing the number of indexed chunks from approximately **155 to 250** (+61%).

The evaluation was then rerun to verify whether the retrieval improvements remained effective as the knowledge base grew.

| Metric       | Initial Repository Set | Expanded Repository Set |
| ------------ | ---------------------: | ----------------------: |
| MRR          |                 0.7285 |                  0.7286 |
| nDCG         |                 0.7264 |                  0.7314 |
| Coverage     |                 80.52% |                  81.24% |
| Accuracy     |                   4.68 |                    4.67 |
| Completeness |                   4.46 |                    4.48 |
| Relevance    |                   4.95 |                    4.90 |

Key observations:

* The number of indexed chunks increased by approximately 61%.
* MRR remained virtually unchanged, indicating that highly relevant chunks continued to be ranked near the top.
* nDCG improved slightly, suggesting better overall ranking quality.
* Coverage increased despite the larger search space.
* Answer quality remained stable across all evaluation dimensions.
* No significant retrieval degradation was observed after repository expansion.

These results suggest that the combination of context enrichment, LLM reranking, and retrieval optimization generalizes well to moderate growth in repository content.

---

## Repository Usage & Local Development

This repository features an environment-aware architecture that automatically adapts to local and cloud execution environments.

### Reviewing the Code & Experiments Locally

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the chatbot interface locally:

```bash
python app.py
```

3. Explore the notebooks:

* Open `notebooks/career_chat_1.ipynb` to review the baseline RAG implementation.
* Open `notebooks/career_chat_2.ipynb` to explore optimization experiments and evaluation.
* Open `notebooks/creator_test_data.ipynb` to see how the synthetic benchmark dataset was generated.
* Open `notebooks/prepare_github_data.ipynb` to inspect repository ingestion and preprocessing.

---

## Deployment & Production Notes (Hugging Face Spaces)

The application is designed to work in both local and cloud environments.

### Local Development

The system uses ChromaDB's `PersistentClient` to store embeddings locally, preventing unnecessary embedding regeneration and reducing OpenAI API costs during development.

### Hugging Face Spaces Deployment

Because Hugging Face Spaces runs in a stateless environment with restricted filesystem access, the application automatically switches to ChromaDB's `EphemeralClient()`.

### In-Memory Runtime

When the application starts:

1. `app.py` triggers the ingestion pipeline.
2. `ingest.py` builds the vector database in memory.
3. `retriever.py` accesses the same in-memory collection for retrieval.

This architecture avoids filesystem dependency issues while keeping deployment simple.

---

## Notes

## Notes

* The LinkedIn source document used during development is not included in the public repository for privacy reasons.
* Cloned repositories (`repos/`) and local vector databases (`vector_db/`) are excluded from version control through `.gitignore`.
* To update the chatbot knowledge base on Hugging Face Spaces, add or modify repositories and trigger a Factory Rebuild.
* The project includes both baseline and optimized RAG implementations to make retrieval improvements measurable and reproducible.