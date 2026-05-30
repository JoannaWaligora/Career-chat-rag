from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel
import json
import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL = "gpt-4.1-mini"
openai = OpenAI()
DB_NAME = "vector_db_2"
collection_name = "docs"
embedding_model = "text-embedding-3-small"
load_dotenv(override=True)

def load_ipynb(file_path):
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            texts.append("MARKDOWN:\n" + "".join(cell.get("source", [])))
        elif cell.get("cell_type") == "code":
            texts.append("CODE:\n" + "".join(cell.get("source", [])))

    return "\n".join(texts)

def extract_project_name(path: str) -> str:
    if not path:
        return "UNKNOWN_PROJECT"

    # normalizacja separatorów (działa dla Windows i Unix)
    path = path.replace("\\", "/")

    match = re.search(r"repos/([^/]+)/", path)

    if match:
        return match.group(1)

    return "UNKNOWN_PROJECT"


def load_files():
    documents = []

    repo_path = "repos"

    for root, _, files in os.walk(repo_path):
        if ".git" in root:
            continue

        print("Checking:", root)

        for file in files:
            print("Found:", file)

            if file.endswith((".md", ".ipynb")):
                file_path = os.path.join(root, file)

                try:
                    if file.endswith(".ipynb"):
                        content = load_ipynb(file_path)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                    project_name = extract_project_name(file_path)

                    documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "project": project_name
                            }
                        )
                    )

                except Exception as e:
                    print("ERROR:", file_path, e)

    print(f"Loaded {len(documents)} documents")
    return documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str
    summary: str
    original_text: str

    def as_result(self, document):
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata=document.metadata
        )


def make_prompt(chunk_text):
    return f"""
You are an expert AI assistant summarizing technical code and markdown.

Your task:
- Create a short headline
- Create a short summary

IMPORTANT RULES:
- Use ONLY the provided text
- Keep summary concise (2-3 sentences)
- Return ONLY valid JSON

{{
  "headline": "short title",
  "summary": "short summary"
}}

TEXT:
{chunk_text}
"""

def process_chunk(chunk, retries=3):
    for i in range(retries):
        try:
            prompt = make_prompt(chunk.page_content)

            response = completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )

            content = response.choices[0].message.content

            # 🔥 bez regexów
            data = json.loads(content)

            # 🔥 manualne złożenie obiektu (bez original_text z LLM)
            parsed = Chunk(
                headline=data["headline"],
                summary=data["summary"],
                original_text=chunk.page_content
            )

            return parsed.as_result(chunk)

        except Exception as e:
            print(f"Retry {i+1}: {e}")

    # fallback
    return Result(
        page_content=f"""TITLE: FALLBACK

SUMMARY: Failed to parse structured chunk, using raw content

CONTENT:
{chunk.page_content[:1000]}""",
        metadata=chunk.metadata
    )


def process_all_chunks(chunks):
    return [process_chunk(c) for c in chunks]


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

def group_by_project(results):
    grouped = {}

    for r in results:
        project = r.metadata.get("project", "UNKNOWN_PROJECT")

        if project not in grouped:
            grouped[project] = []

        grouped[project].append(r)

    return grouped

def generate_project_summary(project_name, context):

    prompt = f"""
You are analyzing an AI project.

Project name:
{project_name}

Your task:
Create structured JSON summary.

IMPORTANT:
If machine learning or deep learning models appear in the project,
include ALL detected models in "models_used".

Return ONLY valid JSON:

{{
    "project_name": "",
    "short_description": "",
    "main_goal": "",
    "tech_stack": [],
    "models_used": []
}}

PROJECT CONTENT:
{context}
"""

    response = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

def build_project_summaries_cache(chunks):

    grouped = group_by_project(chunks)

    cache = {}

    for project_name, items in grouped.items():

        context = "\n".join(
            c.page_content for c in items
        )

        summary = generate_project_summary(project_name, context)

        cache[project_name] = summary

    return cache

def main(build_summaries=True):
    documents = load_files()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    chunks_result = process_all_chunks(chunks)

    create_embeddings(chunks_result)

    if build_summaries:
        project_summaries_cache = build_project_summaries_cache(chunks_result)

        with open("project_summaries.json", "w", encoding="utf-8") as f:
            json.dump(project_summaries_cache, f, ensure_ascii=False, indent=2)

        print("Project summaries saved")
    else:
        print("Skipping summaries generation")

if __name__ == "__main__":
    main()



