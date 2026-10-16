"""Эмбеддинги: превращаем текст в вектор, чтобы искать по смыслу.

HashingEmbedder — лёгкий детерминированный вариант без скачивания моделей: мешок слов
с хэш-трюком. Подходит для демо и тестов, ищет по общим словам (с простой нормализацией
окончаний для русского языка). FastEmbedEmbedder — настоящая многоязычная модель
(опционально: pip install fastembed), ищет по смыслу.
"""
import hashlib
import math
import re
from typing import List, Protocol

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]: ...


def _stem(word: str) -> str:
    """Грубая нормализация русских окончаний: 'договора', 'договору' -> 'договор'."""
    word = word.lower().replace("ё", "е")
    return word[:5] if len(word) > 5 else word


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for w in _WORD.findall(text):
            if len(w) < 2:
                continue
            h = int(hashlib.md5(_stem(w).encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedEmbedder:
    """Многоязычная модель через fastembed (ONNX, работает без GPU)."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        from fastembed import TextEmbedding  # импорт здесь: пакет необязательный

        self._model = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._model.embed(["проверка"]))))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def make_embedder(dim: int, use_fastembed: bool) -> Embedder:
    if use_fastembed:
        return FastEmbedEmbedder()
    return HashingEmbedder(dim)


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
