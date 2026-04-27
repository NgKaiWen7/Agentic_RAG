import unittest
from orchrastrator import Orchestrator

class TestAgenticRAG(unittest.TestCase):
    def rag_search(self):
        orchestrator = Orchestrator()
        query = "what happened between iran and us recently"
        result = orchestrator.rag_search(query)
        print("Agentic RAG Result:\n", result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def web_search(self):
        # Test the RAG search function directly (without agent) to ensure it returns results.
        orchestrator = Orchestrator()
        rag_out = orchestrator.web_search_tavily('what happened between iran and usa recently')
        print('RAG search output:\n', rag_out)
        self.assertIsInstance(rag_out, str)

    def test_agentic_query(self):
        orchestrator = Orchestrator()
    
        test_queries = [
            "what happened between iran and us recently",
            "what is the latest technology in maistorage",
            "what is newton first law"
        ]
    
        for query in test_queries:
            response = orchestrator.agentic_rag_query(query)
            print(f"\nQuery: {query}\nResponse: {response}")
    
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()
