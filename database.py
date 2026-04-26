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
        with self._pg_connect() as conn:
            self._ensure_vector_extension(conn)
            register_vector(conn)
            self._create_tables_if_needed(conn, dim)

    def _pg_connect(self):
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

    def _create_tables_if_needed(self, conn, dim: int):
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.sources_table} (
                id SERIAL PRIMARY KEY,
                title TEXT,
                source_url TEXT UNIQUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                id SERIAL PRIMARY KEY,
                source_id INTEGER NOT NULL REFERENCES {self.sources_table}(id) ON DELETE CASCADE,
                chunk TEXT NOT NULL,
                embedding vector({dim}) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.embeddings_table}_source_id
            ON {self.embeddings_table} (source_id);
            """
        )

    def insert_or_replace_source_embeddings(self, title: str, source: str, chunks: List[str], vectors: List[List[float]]):
        if not chunks or not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        source_url = source.strip() if source and source.strip() else None

        with self._pg_connect() as conn:
            cur = conn.cursor()
            if source_url is not None:
                cur.execute(
                    f"""
                    INSERT INTO {self.sources_table} (title, source_url, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (source_url)
                    DO UPDATE SET title = EXCLUDED.title, updated_at = NOW()
                    RETURNING id;
                    """,
                    (title, source_url),
                )
            else:
                cur.execute(
                    f"INSERT INTO {self.sources_table} (title, source_url, updated_at) VALUES (%s, %s, NOW()) RETURNING id;",
                    (title, None),
                )
            source_id = cur.fetchone()[0]

            cur.execute(f"DELETE FROM {self.embeddings_table} WHERE source_id = %s;", (source_id,))

            rows = [(source_id, chunk, vector) for chunk, vector in zip(chunks, vectors)]
            cur.executemany(
                f"INSERT INTO {self.embeddings_table} (source_id, chunk, embedding) VALUES (%s, %s, %s);",
                rows,
            )

    def query_similar_vectors(self, query_vector, top_k=5):
        vector_param = str(query_vector) if isinstance(query_vector, list) else query_vector
        with self._pg_connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT s.title, s.source_url, e.chunk
                FROM {self.embeddings_table} e
                JOIN {self.sources_table} s ON s.id = e.source_id
                ORDER BY e.embedding <-> %s::vector
                LIMIT %s;
                """,
                (vector_param, top_k),
            )
            results = cur.fetchall()
            return results
