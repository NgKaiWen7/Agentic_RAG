import psycopg2
from pgvector.psycopg2 import register_vector
import dotenv
import os

from streamlit import connection

dotenv.load_dotenv()  # Load environment variables from .env file

class DataBaseController:
    def __init__(self):
        with self._pg_connect() as conn:
            register_vector(conn)
            self._create_table_if_needed(conn, 'vectors', 1536)  # Assuming 1536-dim vectors

    def _pg_connect(self):
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = int(os.getenv('POSTGRES_PORT', 5432))
        dbname = os.getenv('POSTGRES_DB', 'ragdb')
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
        return conn

    def _create_table_if_needed(self, conn, table: str, dim: int):
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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

    def insert_vector(self, id, title, source, chunk, vector):
        with self._pg_connect() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO vectors (id, title, source, chunk, embedding) VALUES (%s, %s, %s, %s, %s)", (id, title, source, chunk, vector))
            conn.commit()

    def query_similar_vectors(self, query_vector, top_k=5):
        with self._pg_connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM vectors ORDER BY vector <-> %s LIMIT %s", (query_vector, top_k))
            results = cur.fetchall()
            return results