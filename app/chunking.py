"""Разбиение текста на куски с нахлёстом, чтобы смысл не рвался на границах."""
import re
from typing import List


def split_text(text: str, size: int = 700, overlap: int = 120) -> List[str]:
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:  # длинный абзац режем по предложениям, затем по символам
            parts = _split_long(para, size)
        else:
            parts = [para]
        for part in parts:
            if current and len(current) + len(part) + 1 > size:
                chunks.append(current)
                current = (_tail(current, overlap) + " " + part).strip() if overlap else part
            else:
                current = (current + "\n" + part).strip() if current else part
    if current:
        chunks.append(current)
    return chunks


def _split_long(para: str, size: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?…])\s+", para)
    out: List[str] = []
    buf = ""
    for s in sentences:
        while len(s) > size:
            out.append(s[:size])
            s = s[size:]
        if buf and len(buf) + len(s) + 1 > size:
            out.append(buf)
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        out.append(buf)
    return out


def _tail(text: str, size: int) -> str:
    """Последние ~size символов, начиная с границы слова (без обрезанных слов)."""
    tail = text[-size:]
    if len(text) > size and not text[-size - 1].isspace():
        _, _, tail = tail.partition(" ")
    return tail.strip()
