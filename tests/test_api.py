import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.llm import ClaudeAnswerer, ExtractiveAnswerer
from app.main import create_app
from app.store import MemoryStore

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"


def settings():
    return Settings(None, 1024, None, "claude-sonnet-5-5", 4, 350, 60)


@pytest.fixture()
def client():
    app = create_app(settings(), store=MemoryStore(), embedder=HashingEmbedder(1024), answerer=ExtractiveAnswerer())
    return TestClient(app)


def upload(client, path):
    return client.post("/documents", files={"file": (path.name, path.read_bytes())})


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["answer_mode"] == "extractive"


def test_upload_and_ask_with_source(client):
    assert upload(client, SAMPLES / "dogovor-ustnyj-obrazec.md").status_code == 201
    upload(client, SAMPLES / "instrukciya-montazh.md")
    r = client.post("/ask", json={"question": "В какой срок оплачивается остаток?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "должны быть источники"
    assert body["sources"][0]["document"] == "dogovor-ustnyj-obrazec.md"
    assert "5 рабочих дней" in body["answer"]


def test_unrelated_question_has_no_answer(client):
    upload(client, SAMPLES / "dogovor-ustnyj-obrazec.md")
    r = client.post("/ask", json={"question": "zzz qqq xxx"}).json()
    assert r["sources"] == []
    assert "нет" in r["answer"].lower()


def test_unsupported_file(client):
    r = client.post("/documents", files={"file": ("a.exe", b"data")})
    assert r.status_code == 415


def test_empty_question_and_empty_file(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/documents", files={"file": ("a.txt", b"   ")}).status_code == 422


def test_list_and_delete(client):
    doc = upload(client, SAMPLES / "instrukciya-montazh.md").json()
    assert client.get("/documents").json()[0]["id"] == doc["id"]
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_claude_answerer_builds_request():
    """Проверяем запрос к Claude без сети: подменяем HTTP-клиент."""
    import httpx

    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-api-key"]
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "Через 5 рабочих дней [1]"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    from app.store import Hit

    a = ClaudeAnswerer("test-key", "claude-sonnet-5-5", client)
    out = a.answer("Когда оплата?", [Hit("d.md", 0, "Остаток в течение 5 рабочих дней", 0.9)])
    assert out == "Через 5 рабочих дней [1]"
    assert seen["key"] == "test-key" and "api.anthropic.com" in seen["url"]
    assert "Остаток в течение 5 рабочих дней" in seen["body"]
