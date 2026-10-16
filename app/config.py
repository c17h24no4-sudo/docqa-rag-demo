"""Настройки сервиса из переменных окружения."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    database_url: Optional[str]
    embed_dim: int
    anthropic_api_key: Optional[str]
    claude_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    min_score: float = 0.12

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            database_url=os.getenv("DATABASE_URL") or None,
            embed_dim=int(os.getenv("EMBED_DIM", "1024")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            top_k=int(os.getenv("TOP_K", "4")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "350")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "60")),
            min_score=float(os.getenv("MIN_SCORE", "0.12")),
        )
