from app.embeddings import HashingEmbedder, cosine
from app.store import MemoryStore


def test_similar_question_is_closer_than_unrelated():
    e = HashingEmbedder(384)
    doc, q, other = e.embed(["Срок оплаты по договору — 7 дней", "Какой срок оплаты договора?", "Монтаж сцены в 8 утра"])
    assert cosine(doc, q) > cosine(doc, other)


def test_store_search_returns_best_chunk_first():
    e = HashingEmbedder(384)
    store = MemoryStore()
    chunks = ["Оплата остатка в течение 5 рабочих дней", "Монтаж сцены начинается в 8 утра"]
    store.add_document("a.md", chunks, e.embed(chunks))
    hits = store.search(e.embed(["Когда оплачивать остаток?"])[0], 2)
    assert hits and "Оплата" in hits[0].text


def test_delete_document():
    e = HashingEmbedder(384)
    store = MemoryStore()
    info = store.add_document("a.md", ["текст"], e.embed(["текст"]))
    assert store.delete_document(info.id) is True
    assert store.delete_document(info.id) is False
    assert store.list_documents() == []
