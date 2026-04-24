import os
from langchain_ollama import OllamaLLM
from langchain_classic.agents import initialize_agent, Tool
from ollama import web_search
import requests
import json
from dotenv import load_dotenv
import warnings
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List
from database import DataBaseController
from rag_system import _get_embed_model
from rag_system import _get_embed_model

# Suppress LangChain deprecation warnings about LangGraph and Chain.run
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*LangGraph.*")
    warnings.filterwarnings("ignore", message=".*Chain.run.*")

# Load environment variables
load_dotenv()


# Note: local documents have been removed. The LLM will decide whether to call
# `WebSearch` (provided below) when it needs external information.

# Function for web search (using the API from trial.py)

class Orchestrator:
    def __init__(self):
        tools = [
            Tool(name="WebSearch", func=self.web_search, description="Search real-time web data for current information; returns summary plus SOURCES block."),
            Tool(name="RAGSearch", func=self.rag_search, description="Search the local RAG vector DB (Postgres+pgvector) for relevant chunks; input should be a short search query.")
        ]
        llm = OllamaLLM(model="llama3", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.agent = initialize_agent(
            tools,
            llm,
            verbose=True
        )

        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._db = DataBaseController()  # Initialize the database controller

    def web_search(self, query):
        url = "https://api.langsearch.com/v1/web-search"
        payload = json.dumps({
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": 5
        })
        headers = {
            'Authorization': os.getenv('LANGSEARCH_API_KEY'),
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()

        if response.status_code != 200:
            return f"Web search failed with status {response.status_code}: {data.get('error', 'No error message')}"
        elif len(response.content) == 0:
            return "Web search returned no content."
        
        pages = data["data"]["webPages"]["value"]
        for i, page in enumerate(pages, start=1):
            title = page.get("name") or page.get("title") or f"web_result_{i}"
            url = page.get("url") or page.get("link") or ""
            summary = page.get("summary")
            for ch in chunk_text(summary, chunk_size=500, overlap=50):
                self.db.insert_vector(id=None, title=title, source=url, chunk=ch, vector=None)


    def rag_search(self, query: str, k: int = 5, table: str = 'embeddings') -> str:
        """Run a vector similarity search in Postgres (pgvector) and return concise results."""
        model = self._embed_model
        qvec = model.encode([query], show_progress_bar=False)[0].tolist()
        rows = self._db.query_similar_vectors(qvec, top_k=k)
        parts = []
        for row in rows:
            title, source, chunk, dist = row
            parts.append(f"{title} ({source})\n{chunk}\nDIST: {dist}")
        if not parts:
            return "No RAG results found."
        return "\n\n---\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple deterministic chunker: splits text into overlapping chunks."""
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        start = end - overlap
    return chunks


def ingest_webpages(query: str, index_path: str = "faiss_index.idx", meta_path: str = "faiss_meta.pkl",
                    count: int = 5, chunk_size: int = 500, overlap: int = 50):
    """Search the web, split results into chunks, embed them, and store into a FAISS index.

    Returns metadata about ingested documents (number of vectors, saved paths).
    """
    pages = self.raw_web_search(query, count=count)
    if not pages:
        return {"status": "no_results", "ingested": 0}

    chunks = []
    metadatas = []
    for pi, page in enumerate(pages):
        title = page.get("name") or page.get("title") or f"web_result_{pi}"
        url = page.get("url") or page.get("link") or ""
        content = page.get("summary") or page.get("snippet") or ""
        # fallback to title if no content
        if not content:
            content = title
        page_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(page_chunks):
            chunks.append(ch)
            metadatas.append({
                "source": url,
                "title": title,
                "page_index": pi,
                "chunk_index": ci,
            })

    if not chunks:
        return {"status": "no_chunks", "ingested": 0}

    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False)

    # Build FAISS index (flat L2 index)
    import numpy as np
    vecs = np.array(embeddings).astype('float32')
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadatas, f)

    return {"status": "ok", "ingested": len(chunks), "index_path": index_path, "meta_path": meta_path}

def ingest_webpages_to_postgres(query: str, table: str = 'embeddings', count: int = 5, chunk_size: int = 500, overlap: int = 50):
    """Search the web, chunk, embed, and ingest into Postgres (pgvector).

    Returns metadata about ingestion.
    """
    pages = raw_web_search(query, count=count)
    if not pages:
        return {"status": "no_results", "ingested": 0}

    chunks = []
    metadatas = []
    for pi, page in enumerate(pages):
        title = page.get("name") or page.get("title") or f"web_result_{pi}"
        url = page.get("url") or page.get("link") or ""
        content = page.get("summary") or page.get("snippet") or ""
        if not content:
            content = title
        page_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        for ci, ch in enumerate(page_chunks):
            chunks.append(ch)
            metadatas.append({"title": title, "source": url, "page_index": pi, "chunk_index": ci})

    if not chunks:
        return {"status": "no_chunks", "ingested": 0}

    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    import numpy as np
    vecs = np.array(embeddings).astype('float32')
    dim = vecs.shape[1]

    _create_table_if_needed(table, dim)

    conn = _pg_connect()
    cur = conn.cursor()
    for ch, meta, vec in zip(chunks, metadatas, vecs):
        cur.execute(
            f"INSERT INTO {table} (title, source, chunk, embedding) VALUES (%s, %s, %s, %s);",
            (meta.get('title'), meta.get('source'), ch, vec.tolist())
        )
    conn.commit()
    cur.close()
    conn.close()

    return {"status": "ok", "ingested": len(chunks), "table": table}




# For citations, we can modify to return sources
def agentic_rag_query(query):
    # Instruct the agent to prefer its own knowledge and only call WebSearch when
    # unsure or when up-to-date info is required. If it uses WebSearch, include
    # the SOURCES block from the tool output in the final answer.
    instruction = (
        "You are a knowledgeable assistant. Prefer answering from your internal knowledge. "
        "If you need fresh or recent information, first try the 'RAGSearch' tool which queries the local vector DB. "
        "When calling 'RAGSearch', modify the user's query to a concise search phrase (e.g., shorten or add keywords) and pass that as the tool input. "
        "Call the 'WebSearch' tool only if RAGSearch does not return sufficient evidence or if live web retrieval is required. "
        "When you use WebSearch, cite the SOURCES section.\n\n"
    )
    response = agent.invoke(instruction + query, handle_parsing_errors=True)
    return response

if __name__ == "__main__":
    query = "What is the latest RAG system?"
    answer = agentic_rag_query(query)
    print(answer)
    print("When you are done, write: Final Answer: <answer>. Do NOT output Action: none or any tool name unless calling a registered tool.")