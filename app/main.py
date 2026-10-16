"""FastAPI-сервис: загрузка документов и вопросы по ним (RAG)."""
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .chunking import split_text
from .config import Settings
from .embeddings import HashingEmbedder
from .llm import make_answerer
from .loaders import UnsupportedFile, extract_text
from .store import MemoryStore, PostgresStore


class Question(BaseModel):
    question: str
    top_k: Optional[int] = None


class Source(BaseModel):
    document: str
    fragment: int
    score: float
    text: str


class Answer(BaseModel):
    answer: str
    mode: str
    sources: List[Source]


def create_app(settings: Optional[Settings] = None, store=None, embedder=None, answerer=None) -> FastAPI:
    settings = settings or Settings.from_env()
    embedder = embedder or HashingEmbedder(settings.embed_dim)
    if store is None:
        store = PostgresStore(settings.database_url, embedder.dim) if settings.database_url else MemoryStore()
    answerer = answerer or make_answerer(settings.anthropic_api_key, settings.claude_model)

    app = FastAPI(title="docqa — вопросы по документам", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "storage": type(store).__name__, "answer_mode": answerer.mode}

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        try:
            text = extract_text(file.filename or "file", data)
        except UnsupportedFile as exc:
            raise HTTPException(status_code=415, detail=str(exc))
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=422, detail="В файле не найден текст")
        doc = store.add_document(file.filename or "file", chunks, embedder.embed(chunks))
        return {"id": doc.id, "name": doc.name, "chunks": doc.chunks}

    @app.get("/documents")
    def documents():
        return [d.__dict__ for d in store.list_documents()]

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete(doc_id: int):
        if not store.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Документ не найден")

    @app.post("/ask", response_model=Answer)
    def ask(q: Question):
        if not q.question.strip():
            raise HTTPException(status_code=422, detail="Пустой вопрос")
        vector = embedder.embed([q.question])[0]
        hits = [h for h in store.search(vector, q.top_k or settings.top_k) if h.score >= settings.min_score]
        return Answer(
            answer=answerer.answer(q.question, hits),
            mode=answerer.mode,
            sources=[Source(document=h.document, fragment=h.position + 1, score=round(h.score, 3), text=h.text) for h in hits],
        )

    return app


app = create_app() if __name__ != "__main__" else None
