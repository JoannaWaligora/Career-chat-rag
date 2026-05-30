from dotenv import load_dotenv
from chromadb import PersistentClient
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from retriever import answer_question

load_dotenv(override=True)
MODEL = "gpt-4.1-mini"
DB_NAME = "vector_db_2"
chroma = PersistentClient(path=DB_NAME)

def ensure_vector_db():
    chroma = PersistentClient(path=DB_NAME)

    try:
        col = chroma.get_collection("docs")
        if col.count() > 0:
            return
    except:
        pass

    print("Building DB...")
    from ingest import main
    main(build_summaries=False)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email:str, name:str="Name not provided", notes:str="not provided")-> dict:
    """Store user contact details and notify via push"""
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

def record_unknown_question(question: str)-> dict:
    """Store unknown user question and notify via push"""
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

def answer_with_rag(question: str, history: list[dict] = None):
    if history is None:
        history = []

    answer, chunks = answer_question(question, history)

    return {
        "answer": answer,
        "context_snippets": [c.page_content[:300] for c in chunks],
        "sources": [c.metadata.get("source", "unknown") for c in chunks]
    }

answer_with_rag_json = {
    "name": "answer_with_rag",
    "description": "Answer user questions using RAG over Joanna Waligóra projects. Returns answer and retrieved context snippets and sources.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question"
            },
            "history": {
                "type": "array",
                "description": "Chat history in OpenAI format (role/content dicts)",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json},
        {"type": "function", "function": answer_with_rag_json }]

class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Joanna Waligóra"
        reader = PdfReader("linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
    
    def handle_tool_calls(self, tool_calls):
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)
            except:
                arguments = {}

            print(f"Tool called: {tool_name}", flush=True)

            tool = globals().get(tool_name)

            if tool is None:
                result = {"error": f"Tool {tool_name} not found"}
            else:
                try:
                    result = tool(**arguments)
                except Exception as e:
                    result = {"error": str(e)}

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        return results
 
    def main_system_prompt(self):
        prompt = f"""
You are Joanna Waligóra's portfolio assistant.

{self.linkedin}

RULES:
- LinkedIn = career truth
- RAG = GitHub project truth
- For project questions ALWAYS use RAG first
- Never invent projects or skills
- If information is missing → say "I don't know"

Use RAG for:
- projects
- tech stack
- models
- implementation details
- portfolio questions

Be concise and factual.
"""
        return prompt
        
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.main_system_prompt()}]

        for h in history:
            messages.append({
                "role": "assistant" if h["role"] == "assistant" else "user",
                "content": h["content"]
            })

        messages.append({
            "role": "user",
            "content": message
        })

        for _ in range(2):

            response = self.openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            msg = response.choices[0].message

            # TOOL CALLING
            if msg.tool_calls:

                messages.append(msg)

                tool_results = self.handle_tool_calls(msg.tool_calls)

                messages.extend(tool_results)

            else:
                return msg.content or "No response generated."

        return "Tool loop limit reached."

if __name__ == "__main__":
    me = Me()

    demo = gr.ChatInterface(
        fn=me.chat,

        title="Joanna Waligóra — AI Career Assistant",

        description="""
Ask me about:
- AI / ML projects
- career background and education
- CNN / LSTM models
- Python stack
- portfolio projects

This assistant uses Retrieval-Augmented Generation (RAG)
over Joanna Waligóra's GitHub portfolio and career profile.
""",

        examples=[
            "What projects has Joanna built?",
            "What AI technologies does Joanna use?",
            "Tell me about Joanna's experience",
            "Which projects use CNN models?",
            "Tell me about the LIDAR classification project",
            "What is Joanna's background?"
        ]
    )

    demo.launch()