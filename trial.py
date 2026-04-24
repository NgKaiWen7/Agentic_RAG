import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

url = "https://api.langsearch.com/v1/web-search"

payload = json.dumps({
  "query": "what is agentic RAG and how to use it?",
  "freshness": "noLimit",
  "summary": True,
  "count": 10
})
headers = {
  'Authorization': os.getenv('LANGSEARCH_API_KEY'),
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json()["data"]["webPages"]["value"][1]["summary"])