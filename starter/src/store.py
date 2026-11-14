"""Store and search vectors using PostgreSQL + pgvector."""

from typing import Any
import math
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings and validate embedding dimensions."""
        if not database_url:
            raise ValueError("database_url must not be empty")

        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")

        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        conn = psycopg.connect(self.database_url)
        register_vector(conn)
        return conn

    def create_schema(self) -> None:
        """Create pgvector extension and chunk table."""
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_strategy
                    ON chunks (strategy)
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks (document_id)
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_source
                    ON chunks (source)
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_page
                    ON chunks (page)
                    """
                )

    def count_chunks(self, strategy: str) -> int:
        """Return the number of indexed chunks for one strategy."""
        if not strategy:
            raise ValueError("strategy must not be empty")
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE strategy = %s", (strategy,))
                row = cur.fetchone()
        return int(row[0])

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        document_ids: list[str] | None = None,
    ) -> None:
        """Insert/update chunks and remove obsolete chunks for affected documents."""
        if not strategy:
            raise ValueError("strategy must not be empty")

        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}"
            )

        if not chunks and not document_ids:
            return

        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Vector {index} has dimension {len(vector)}, "
                    f"expected {self.dimensions}"
                )
            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError(f"Vector {index} contains non-finite values")

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk_id found in input chunks")

        affected_documents = sorted(set(document_ids or ()) | {c.document_id for c in chunks})

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Remove chunks that belonged to re-ingested documents but
                # are no longer present in the new chunk set.
                cur.execute(
                    """
                    DELETE FROM chunks
                    WHERE strategy = %s
                      AND document_id = ANY(%s)
                      AND NOT (chunk_id = ANY(%s))
                    """,
                    (
                        strategy,
                        affected_documents,
                        chunk_ids,
                    ),
                )

                rows = [
                    (
                        strategy,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        psycopg.types.json.Jsonb(chunk.metadata),
                        vector,
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]

                cur.executemany(
                    """
                    INSERT INTO chunks (
                        strategy,
                        chunk_id,
                        document_id,
                        source,
                        page,
                        text,
                        metadata,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy, chunk_id)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search by cosine similarity with optional metadata filtering."""
        if not strategy:
            raise ValueError("strategy must not be empty")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, "
                f"expected {self.dimensions}"
            )

        if not all(math.isfinite(float(value)) for value in query_vector):
            raise ValueError("Query vector contains non-finite values")
        query_vector_np = np.array(query_vector, dtype=np.float32)

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        filters = filters or {}

        allowed_filters = {
            "document_id",
            "source",
            "page",
        }

        unknown_filters = set(filters) - allowed_filters

        if unknown_filters:
            raise ValueError(
                f"Unsupported filters: {sorted(unknown_filters)}"
            )
        for key, value in filters.items():
            if value is None or value == "":
                raise ValueError(f"Filter {key!r} must have a value")

        conditions = ["strategy = %s"]
        params: list[Any] = [strategy]

        if "document_id" in filters:
            conditions.append("document_id = %s")
            params.append(filters["document_id"])

        if "source" in filters:
            conditions.append("source = %s")
            params.append(filters["source"])

        if "page" in filters:
            conditions.append("page = %s")
            params.append(filters["page"])

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                1 - (embedding <=> %s) AS score
            FROM chunks
            WHERE {where_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        query_params = [
            query_vector_np,
            *params,
            query_vector_np,
            top_k,
        ]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, query_params)

                rows = cur.fetchall()

        results: list[dict[str, Any]] = []

        for row in rows:
            (
                chunk_id,
                document_id,
                source,
                page,
                text,
                metadata,
                score,
            ) = row

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(score),
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "source": source,
                        "page": page,
                    },
                }
            )

        return results
