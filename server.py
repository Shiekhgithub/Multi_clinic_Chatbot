"""
server.py
---------
FastAPI REST API for the Healthcare RAG system.
Exposes /chat and /Upload_File endpoints for web and mobile clients.

Run:
    uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vector_store import load_all_stores, stores_exist, get_embeddings
from agent import build_agent

load_dotenv()

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
agent_executor = None
stores = None
uploaded_collection = None


def _rebuild_agent():
    """Rebuild agent with current tools (including uploaded docs if any)."""
    global agent_executor, stores, uploaded_collection

    provider = os.getenv("LLM_PROVIDER", "groq")
    k = int(os.getenv("K_DOCS", "5"))

    if stores is None:
        return

    agent_executor = build_agent(stores, provider=provider, k=k)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vector stores and build agent on startup."""
    global agent_executor, stores

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    if not stores_exist(persist_dir):
        print("⚠️  Vector stores not found. Please run `python ingest.py` first.")
    else:
        stores = load_all_stores(persist_dir)
        _rebuild_agent()
        print("✅ Healthcare RAG agent ready")

    yield


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Healthcare RAG API",
    description="REST API for the multi-clinic Healthcare RAG chatbot",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    Assistant: str


class UploadResponse(BaseModel):
    filenames: List[str]
    count: int
    status: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "agent_ready": agent_executor is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the Healthcare RAG agent."""
    if agent_executor is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please run `python ingest.py` first.",
        )

    try:
        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": request.question}]},
            {"configurable": {"thread_id": "default"}},
        )

        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message.content, list):
                answer = last_message.content[0].get("text", "")
            else:
                answer = last_message.content
        else:
            answer = "No response generated."

        return ChatResponse(Assistant=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/Upload_File", response_model=UploadResponse)
async def upload_file(files: List[UploadFile] = File(...)):
    """Upload PDF, TXT, or MD documents for RAG processing."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    global uploaded_collection

    allowed_extensions = {".pdf", ".txt", ".md"}
    filenames = []
    all_docs: List[Document] = []

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load document content
            if ext == ".pdf":
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                except ImportError:
                    # Fallback: read as text
                    text = content.decode("utf-8", errors="ignore")
                    docs = [Document(page_content=text, metadata={"source": file.filename})]
            else:
                text = content.decode("utf-8", errors="ignore")
                docs = [Document(page_content=text, metadata={"source": file.filename})]

            all_docs.extend(docs)
            filenames.append(file.filename or "unknown")
        finally:
            os.unlink(tmp_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    if chunks:
        embeddings = get_embeddings()
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        store_path = os.path.join(persist_dir, "uploaded_docs")

        if uploaded_collection is None:
            uploaded_collection = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="uploaded_docs",
                persist_directory=store_path,
            )
        else:
            uploaded_collection.add_documents(chunks)

    return UploadResponse(
        filenames=filenames,
        count=len(filenames),
        status="Processed successfully",
    )
