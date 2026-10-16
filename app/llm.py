"""Ответ на вопрос по найденным кускам.

ClaudeAnswerer отправляет куски в Claude и просит отвечать только по ним.
ExtractiveAnswerer — запасной режим без ключа: возвращает самые подходящие места
из документов (демо работает и без нейросети).
"""
from typing import List, Optional

import httpx

from .store import Hit

SYSTEM = (
    "Ты отвечаешь на вопрос только по приведённым фрагментам документов. "
    "Если ответа во фрагментах нет, так и скажи: «В документах этого нет». "
    "Ничего не выдумывай. Отвечай по-русски, кратко, в конце укажи номера фрагментов [1], [2]."
)


def build_prompt(question: str, hits: List[Hit]) -> str:
    parts = [f"[{i}] ({h.document}, фрагмент {h.position + 1}):\n{h.text}" for i, h in enumerate(hits, 1)]
    return "Фрагменты:\n\n" + "\n\n".join(parts) + f"\n\nВопрос: {question}"


class ExtractiveAnswerer:
    mode = "extractive"

    def answer(self, question: str, hits: List[Hit]) -> str:
        if not hits:
            return "В документах этого нет."
        best = hits[0]
        return f"Самое подходящее место: {best.text.strip()} [1]"


class ClaudeAnswerer:
    mode = "claude"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, client: Optional[httpx.Client] = None) -> None:
        self._key = api_key
        self._model = model
        self._client = client or httpx.Client(timeout=60)

    def answer(self, question: str, hits: List[Hit]) -> str:
        if not hits:
            return "В документах этого нет."
        resp = self._client.post(
            self.URL,
            headers={
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": 600,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": build_prompt(question, hits)}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def make_answerer(api_key: Optional[str], model: str):
    return ClaudeAnswerer(api_key, model) if api_key else ExtractiveAnswerer()
