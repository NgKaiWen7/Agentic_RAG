# Agentic RAG with Docker

This project sets up an Agentic RAG system using Docker, with a Streamlit frontend and a local Ollama LLM backend.

## Prerequisites

- Docker and Docker Compose installed

## Setup

1. Clone or navigate to the project directory.

2. Set your API keys in `.env`:
   ```
   LANGSEARCH_API_KEY=your_langsearch_key
   TAVILY_API_KEY=your_talivy_key
   ```

3. Build and run the services:
   ```bash
   docker-compose up --build
   ```

   This will:
   - Start the Ollama service
   - Pull the llama3 model: ```docker exec -it ollama ollama pull llama3```
   - Build and start the Streamlit app

4. Access the Streamlit app at http://localhost:8501

## Services

- **ollama**: Runs the Ollama LLM server on port 11434
- **pull_model**: Pulls the llama2 model (runs once)
- **app**: Streamlit application on port 8501

## Stopping

```bash
docker-compose down
```

## Notes

- The first run may take time to download the llama2 model
- Ensure sufficient RAM for the LLM (llama2 requires ~4GB)
