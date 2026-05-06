from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MessageInput(BaseModel):
    conversation_id: str | None = None
    user_id: str | None = None
    content: str = Field(min_length=1)


class Citation(BaseModel):
    chunk_id: str
    source_path: str
    snippet: str
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation]


class HistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime


class IngestResponse(BaseModel):
    docs_path: str
    docs_path_exists: bool
    scanned_files: int
    updated_files: int
    skipped_unchanged_files: int
    failed_files: int
    indexed_chunks: int
    total_chunks_in_db: int
    failures: list[str]
