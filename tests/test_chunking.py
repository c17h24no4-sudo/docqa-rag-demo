from app.chunking import split_text


def test_empty_text_gives_no_chunks():
    assert split_text("   \n  ") == []


def test_short_text_is_one_chunk():
    assert split_text("Короткий текст.") == ["Короткий текст."]


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(f"Абзац номер {i}. " + "слово " * 30 for i in range(10))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 + 40 + 1 for c in chunks)
    # нахлёст: конец предыдущего куска повторяется в начале следующего
    assert chunks[0][-20:].strip() in chunks[1]


def test_very_long_paragraph_is_split():
    chunks = split_text("а" * 2000, size=500, overlap=50)
    assert len(chunks) >= 4


def test_bad_params():
    import pytest

    with pytest.raises(ValueError):
        split_text("текст", size=100, overlap=100)


def test_overlap_does_not_cut_words():
    text = "\n\n".join(f"Предложение номер {i} про договорённости и сроки оплаты услуг." for i in range(30))
    chunks = split_text(text, size=180, overlap=50)
    for c in chunks[1:]:
        assert c.split()[0] in {"Предложение", "номер", "про", "договорённости", "и", "сроки", "оплаты", "услуг."} or c.split()[0].isdigit()
