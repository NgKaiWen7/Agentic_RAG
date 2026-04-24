import unittest
import os
from orchrastrator import web_search, agentic_rag_query, ingest_webpages_to_postgres, rag_search

try:
    import psycopg2
except Exception:
    psycopg2 = None

class TestAgenticRAG(unittest.TestCase):
    #def test_agentic_rag(self):
    #    query = "What is badminton?"
    #    result = agentic_rag_query(query)
    #    print("Agentic RAG Result:\n", result['output'])
    #    self.assertIsInstance(result["output"], str)
    #    self.assertGreater(len(result), 0)
    
    def test_rag_search(self):
        # Test the RAG search function directly (without agent) to ensure it returns results.
        if not os.getenv('LANGSEARCH_API_KEY'):
            self.skipTest('LANGSEARCH_API_KEY not set; skipping RAG search test')
        rag_out = rag_search('badminton latest news', k=3)
        print('RAG search output:\n', rag_out)
        self.assertIsInstance(rag_out, str)

    #def test_ingest_and_rag(self):
    #    # Ingest a small set of web results into Postgres (pgvector) and query the RAG DB.
    #    if not os.getenv('LANGSEARCH_API_KEY'):
    #        self.skipTest('LANGSEARCH_API_KEY not set; skipping ingestion test')
    #    if psycopg2 is None:
    #        self.skipTest('psycopg2 not installed; skipping ingestion test')
    #    # Try connecting to Postgres
    #    pg_host = os.getenv('POSTGRES_HOST', 'localhost')
    #    pg_port = int(os.getenv('POSTGRES_PORT', 5432))
    #    pg_db = os.getenv('POSTGRES_DB', 'ragdb')
    #    pg_user = os.getenv('POSTGRES_USER', 'postgres')
    #    pg_pass = os.getenv('POSTGRES_PASSWORD', 'postgres')
    #    try:
    #        conn = psycopg2.connect(host=pg_host, port=pg_port, dbname=pg_db, user=pg_user, password=pg_pass)
    #        conn.close()
    #    except Exception:
    #        self.skipTest('Postgres not available; skipping ingestion test')
    #    ingest_res = ingest_webpages_to_postgres('latest news badminton', count=2)
    #    print('Ingest result:', ingest_res)
    #    self.assertIn('status', ingest_res)
    #    # Now run a RAG query (shortened query should be used by agent when needed)
    #    rag_out = rag_search('badminton latest news', k=3)
    #    print('RAG search output:\n', rag_out)
    #    self.assertIsInstance(rag_out, str)

if __name__ == '__main__':
    unittest.main()