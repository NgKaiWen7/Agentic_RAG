import unittest
import os
from orchrastrator import Orchestrator

class TestAgenticRAG(unittest.TestCase):
    def rag_search(self):
        orchestrator = Orchestrator()
        query = "What is badminton?"
        result = orchestrator.rag_search(query)
        print("Agentic RAG Result:\n", result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def web_search(self):
        # Test the RAG search function directly (without agent) to ensure it returns results.
        orchestrator = Orchestrator()
        rag_out = orchestrator.web_search_tavily('badminton latest news')
        print('RAG search output:\n', rag_out)
        self.assertIsInstance(rag_out, str)

    def agentic_query(self):
        # Test the full agentic RAG query flow with a sample query.
        orchestrator = Orchestrator()
        response = orchestrator.agentic_rag_query("What are the latest news on badminton?")
        print("Agentic RAG Query Response:\n", response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()