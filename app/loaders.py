"""Чтение текста из загруженных файлов: .txt, .md, .pdf, .docx."""
import io
from typing import Optional


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise UnsupportedFile("Поддерживаются файлы .txt, .md, .pdf, .docx")
