from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
import json



MODEL = "gpt-4.1-mini"
openai = OpenAI()
DB_NAME = "vector_db_2"
collection_name = "docs"
embedding_model = "text-embedding-3-small"
load_dotenv(override=True)
RETRIEVAL_K = 20
FINAL_K = 8

with open("project_summaries.json", "r", encoding="utf-8") as f:
    project_summaries_cache = json.load(f)

chroma = PersistentClient(path=DB_NAME)
collection = None

def get_collection():
    global collection

    if collection is None:
        collection = chroma.get_or_create_collection(collection_name)

    return collection


class Result(BaseModel):
    page_content: str
    metadata: dict

def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You must rank all chunks by relevance.

STRICT RULES:
- Use ONLY indices from 1 to N
- Do NOT invent indices
- Do NOT skip any
- Do NOT repeat any
- Return exactly N indices

Return JSON:
{"order": [1,2,3,...]}
"""

    user_prompt = f"Question:\n{question}\n\nChunks:\n"

    for i, chunk in enumerate(chunks, start=1):
        user_prompt += f"\n[{i}] {chunk.page_content[:300]}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"}
    )

    reply = response.choices[0].message.content

    try:
        order = json.loads(reply)["order"]
    except:
        return chunks  # fallback

    # 🔒 zabezpieczenie
    seen = set()
    ranked = []

    for i in order:
        if isinstance(i, int) and 1 <= i <= len(chunks) and i not in seen:
            ranked.append(chunks[i - 1])
            seen.add(i)

    # 🔥 fallback – uzupełnij brakujące
    for idx in range(len(chunks)):
        if (idx + 1) not in seen:
            ranked.append(chunks[idx])

    return ranked

def format_project_summaries(cache):
    return "\n\n".join(
        f"""
PROJECT: {p.get("project_name", "")}
DESCRIPTION: {p.get("short_description", "")}
GOAL: {p.get("main_goal", "")}
TECH STACK: {", ".join(p.get("tech_stack", []))}
MODELS: {", ".join(p.get("models_used", []))}
""".strip()
        for p in cache.values()
    )


def make_rag_messages(question, history, chunks):

    project_summaries = format_project_summaries(project_summaries_cache)  # TWOJA FUNKCJA

    context = "\n\n".join(
        f"[PROJECT: {c.metadata.get('project','UNKNOWN')}]\n"
        f"{c.page_content}"
        for c in chunks
    )

    system_prompt = rag_system_prompt.format(
        project_summaries=project_summaries,
        context=context
    )

    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": question}
    ]


rag_system_prompt = """
You answer questions about GitHub projects.

RULES:
- Use ONLY provided context
- Never invent projects
- Group information by project
- If project is missing in context → it does not exist
- All projects in context belong to Joanna Waligóra (her GitHub portfolio)

Each chunk contains:
- project name
- source file

Project summaries contain:
- project description
- goal
- tech stack
- models used

Use summaries for:
- project overview
- listing projects
- technology questions

Use chunks for:
- implementation details
- architecture
- code behavior

PROJECT SUMMARIES:
{project_summaries}

CONTEXT:
{context}
"""

def fetch_context_unranked(question): 

    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = get_collection().query(query_embeddings=[query], n_results=RETRIEVAL_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append(Result(page_content=doc,metadata=meta or {}))
        
    return chunks

def fetch_context(original_question):
    chunks = fetch_context_unranked(original_question)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]

def answer_question(question: str, history: list[dict] = None):
    if history is None:
        history = []

    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)

    return response.choices[0].message.content, chunks
