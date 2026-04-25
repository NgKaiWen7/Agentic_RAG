import psycopg2
from pgvector.psycopg2 import register_vector
import dotenv
import os
from typing import List

dotenv.load_dotenv()  # Load environment variables from .env file

class DataBaseController:
    def __init__(self, dim: int = 384):
        self.sources_table = "sources"
        self.embeddings_table = "embeddings"
        self.dim = dim
        with self._raw_pg_connect() as conn:
            self._ensure_vector_extension(conn)
            register_vector(conn)
            self._create_tables_if_needed(conn, dim)

    def _raw_pg_connect(self):
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = int(os.getenv('POSTGRES_PORT', 5432))
        dbname = os.getenv('POSTGRES_DB', 'ragdb')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
        return conn

    def _ensure_vector_extension(self, conn):
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def _create_table_if_needed(self, conn, table: str, dim: int):
        with self._raw_pg_connect().cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    chunk TEXT,
                    embedding vector({dim})
                );
                """
            )

    def insert_vector(self, title, source, chunk, vector):
        with self._raw_pg_connect() as conn:
            cur = conn.cursor()
            cur.execute(f"INSERT INTO {self.vector_table} (title, source, chunk, embedding) VALUES (%s, %s, %s, %s)", (title, source, chunk, vector))
            conn.commit()

    def query_similar_vectors(self, query_vector, top_k=5):
        with self._raw_pg_connect() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT title, source, chunk FROM {self.vector_table} ORDER BY embedding <-> %s LIMIT %s;", (query_vector, top_k))
            results = cur.fetchall()
            return results
