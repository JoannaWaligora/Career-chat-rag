from openai import OpenAI
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel
import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv(override=True)
MODEL = "gpt-4.1-mini"
openai = OpenAI()
collection_name = "docs"
embedding_model = "text-embedding-3-small"

ROOT = Path(__file__).resolve().parent
REPOS_DIR = ROOT / "repos"
DB_DIR = ROOT / "vector_db_2"
DATA_DIR = ROOT / "data"

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

def extract_project_name(path: Path) -> str:
    parts = path.parts

    if "repos" in parts:
        idx = parts.index("repos")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "UNKNOWN_PROJECT"


def load_files():
    documents = []

    for file_path in REPOS_DIR.rglob("*"):

        if ".git" in file_path.parts:
            continue

        if file_path.suffix not in [".md", ".ipynb"]:
            continue

        print("Loading:", file_path)

        try:
            if file_path.suffix == ".ipynb":
                content = load_ipynb(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            project_name = extract_project_name(file_path)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
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

            data = json.loads(content)

            parsed = Chunk(
                headline=data["headline"],
                summary=data["summary"],
                original_text=chunk.page_content
            )

            return parsed.as_result(chunk)

        except Exception as e:
            print(f"Retry {i+1}: {e}")

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
    texts = [chunk.page_content for chunk in chunks]    
    if not texts:
        print("No texts to embed!")
        return

    global chroma

    if "SPACE_ID" in os.environ:
        from chromadb import EphemeralClient
        print("Production detected (Hugging Face). Launching Chroma IN-MEMORY...")
        chroma = EphemeralClient()    
    else:
        from chromadb import PersistentClient
        print("Local environment detected. Launching Chroma PERSISTENT (Disk)...")
        chroma = PersistentClient(path=str(DB_DIR))  
    
    emb = openai.embeddings.create(model=embedding_model, input=texts).data    
    vectors = [e.embedding for e in emb]    
    
    if "SPACE_ID" in os.environ:
        collection = chroma.get_or_create_collection(collection_name)
    else:
        if collection_name in [c.name for c in chroma.list_collections()]:
            chroma.delete_collection(collection_name)
        collection = chroma.get_or_create_collection(collection_name)   

    ids = [str(i) for i in range(len(chunks))]    
    metas = [chunk.metadata for chunk in chunks]    
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)   
 
    print(f"Vectorstore created IN MEMORY with {collection.count()} documents")
    
    return chroma

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

        with open(DATA_DIR / "project_summaries.json", "w", encoding="utf-8") as f:
            json.dump(project_summaries_cache, f, ensure_ascii=False, indent=2)

        print("Project summaries saved")
    else:
        print("Skipping summaries generation")

if __name__ == "__main__":
    main()



