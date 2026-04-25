import os
from langchain_ollama import OllamaLLM
from langchain_classic.agents import initialize_agent, Tool
import requests
import json
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List
from database import DataBaseController
from langchain_core.prompts import ChatPromptTemplate

# Suppress LangChain deprecation warnings about LangGraph and Chain.run
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    warnings.filterwarnings("ignore", message=".*LangGraph.*")
    warnings.filterwarnings("ignore", message=".*Chain.run.*")

load_dotenv()

def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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

class Orchestrator:
    def __init__(self):
        tools = [
            Tool(name="WebSearch", func=self.web_search, description="Search real-time web data for current information and inserts the results into the local RAG vector DB."),
            Tool(name="WebSearchTavily", func=self.web_search_tavily, description="Search real-time web data using Tavily and insert the results into the local RAG vector DB."),
            Tool(name="RAGSearch", func=self.rag_search, description="Search the local RAG vector DB (Postgres+pgvector) for relevant chunks; input should be a short search query.")
        ]
        llm = OllamaLLM(model="llama3", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a retrieval-augmented assistant.\n"
            "You MUST use tools when needed before answering.\n\n"
            "Rules:\n"
            "1. If the question requires factual or specific information, ALWAYS use RAGSearch first.\n"
            "2. If RAGSearch is insufficient, you may use WebSearch.\n"
            "3. Never guess facts without tool results.\n"
            "4. After receiving tool results, read them carefully and synthesize a final answer.\n"
            "5. Keep answers concise and factual."),
            ("human", "{input}")
        ])
        self.agent = initialize_agent(
            tools,
            llm,
            verbose=True,
            agent_kwargs={"prompt": prompt}
        )

        self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._db = DataBaseController()  # Initialize the database controller
    
    def agentic_rag_query(self, query: str) -> str:
        """Main entry point: runs the agent with the given query."""
        response = self.agent.run(query)
        return response

    def _ingest_source(self, title: str, source: str, text: str) -> int:
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        if not chunks:
            return 0
        vectors = self._embed_model.encode(chunks, show_progress_bar=False).tolist()
        self._db.insert_or_replace_source_embeddings(
            title=title,
            source=source,
            chunks=chunks,
            vectors=vectors,
        )
        return len(chunks)

    # temporary disable due to 502 gateway from langsearch
    def web_search(self, query):
        url = "https://api.langsearch.com/v1/web-search"
        headers = {
            'Authorization': os.getenv("LANGSEARCH_API_KEY"),
            'Content-Type': 'application/json'
        }
        payload = json.dumps({
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": 5
        })
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            return f"Web search failed: {e}"

        try:
            data = response.json()
        except json.JSONDecodeError:
            return "Failed to parse JSON from web search API"

        if 'error' in data and data['error']:
            return f"Web search failed with error: {data.get('error', 'No error message')}"
        pages = data["data"]["webPages"]["value"]
        ingested_pages = 0
        ingested_chunks = 0
        for i, page in enumerate(pages, start=1):
            title = page.get("name") or page.get("title") or f"web_result_{i}"
            url = page.get("url") or page.get("link") or ""
            summary = page.get("summary")
            added = self._ingest_source(title=title, source=url, text=summary)
            if added > 0:
                ingested_pages += 1
                ingested_chunks += added
        return f"Web search successful: ingested {ingested_pages} pages and {ingested_chunks} chunks into RAG DB."

    def web_search_tavily(self, query):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Tavily web search failed: missing TAVILY_API_KEY."

        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Tavily web search failed: {e}"

        try:
            data = response.json()
        except json.JSONDecodeError:
            return "Failed to parse JSON from Tavily API"

        if data.get("error"):
            return f"Tavily web search failed with error: {data.get('error')}"

        pages = data.get("results", [])
        if not pages:
            return "Tavily web search returned no results."

        ingested_pages = 0
        ingested_chunks = 0
        for i, page in enumerate(pages, start=1):
            title = page.get("title") or f"tavily_result_{i}"
            source = page.get("url") or ""
            summary = page.get("content") or page.get("raw_content") or ""
            added = self._ingest_source(title=title, source=source, text=summary)
            if added > 0:
                ingested_pages += 1
                ingested_chunks += added

        return f"Tavily web search successful: ingested {ingested_pages} pages and {ingested_chunks} chunks into RAG DB."


    def rag_search(self, query: str, k: int = 5) -> str:
        """Run a vector similarity search in Postgres (pgvector) and return concise results."""
        model = self._embed_model
        qvec = model.encode([query], show_progress_bar=False)[0].tolist()
        rows = self._db.query_similar_vectors(qvec, top_k=k)
        parts = []
        for row in rows:
            title, source, chunk = row
            parts.append(f"{title} ({source})\n{chunk}\n")
        if not parts:
            return "No RAG results found."
        return "\n\n---\n\n".join(parts)
