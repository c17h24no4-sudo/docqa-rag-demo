"""Хранилище документов и кусков с поиском по близости векторов.

PostgresStore — рабочий вариант (PostgreSQL + расширение pgvector).
MemoryStore — тот же интерфейс в памяти, для тестов и запуска без базы.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from .embeddings import cosine


@dataclass
class Hit:
    document: str
    position: int
    text: str
    score: float


@dataclass
class DocInfo:
    id: int
    name: str
    chunks: int


class Store(Protocol):
    def add_document(self, name: str, chunks: List[str], vectors: List[List[float]]) -> DocInfo: ...
    def list_documents(self) -> List[DocInfo]: ...
    def delete_document(self, doc_id: int) -> bool: ...
    def search(self, vector: List[float], k: int) -> List[Hit]: ...


class MemoryStore:
    def __init__(self) -> None:
        self._docs: Dict[int, dict] = {}
        self._next = 1

    def add_document(self, name, chunks, vectors) -> DocInfo:
        doc_id = self._next
        self._next += 1
        self._docs[doc_id] = {"name": name, "chunks": list(zip(chunks, vectors))}
        return DocInfo(doc_id, name, len(chunks))

    def list_documents(self) -> List[DocInfo]:
        return [DocInfo(i, d["name"], len(d["chunks"])) for i, d in self._docs.items()]

    def delete_document(self, doc_id: int) -> bool:
        return self._docs.pop(doc_id, None) is not None

    def search(self, vector, k) -> List[Hit]:
        hits: List[Hit] = []
        for d in self._docs.values():
            for pos, (text, vec) in enumerate(d["chunks"]):
                hits.append(Hit(d["name"], pos, text, cosine(vector, vec)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits[:k] if h.score > 0]


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
"""


class PostgresStore:
    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg

        self._psycopg = psycopg
        self._dsn = dsn
        self._dim = dim
        with self._conn() as conn:
            conn.execute(SCHEMA.format(dim=dim))

    def _conn(self):
        return self._psycopg.connect(self._dsn, autocommit=True)

    @staticmethod
    def _vec(v: List[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

    def add_document(self, name, chunks, vectors) -> DocInfo:
        with self._conn() as conn:
            doc_id = conn.execute(
                "INSERT INTO documents(name) VALUES (%s) RETURNING id", (name,)
            ).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks(document_id, position, text, embedding) VALUES (%s,%s,%s,%s::vector)",
                    [(doc_id, i, t, self._vec(v)) for i, (t, v) in enumerate(zip(chunks, vectors))],
                )
        return DocInfo(doc_id, name, len(chunks))

    def list_documents(self) -> List[DocInfo]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.id, d.name, COUNT(c.id) FROM documents d "
                "LEFT JOIN chunks c ON c.document_id = d.id GROUP BY d.id ORDER BY d.id"
            ).fetchall()
        return [DocInfo(r[0], r[1], r[2]) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            return cur.rowcount > 0

    def search(self, vector, k) -> List[Hit]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT d.name, c.position, c.text, 1 - (c.embedding <=> %s::vector) AS score "
                "FROM chunks c JOIN documents d ON d.id = c.document_id "
                "ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (self._vec(vector), self._vec(vector), k),
            ).fetchall()
        return [Hit(r[0], r[1], r[2], float(r[3])) for r in rows if r[3] and r[3] > 0]
