from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Telefonia RAG"
    api_prefix: str = ""
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    database_url: str = Field(
        default="postgresql+psycopg://rag:rag@db:5432/rag_db",
        alias="DATABASE_URL",
    )

    docs_path: Path = Field(default=Path("docs/Wiki"), alias="DOCS_PATH")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_pool: int = Field(default=40, alias="RETRIEVAL_CANDIDATE_POOL")
    history_window: int = Field(default=6, alias="HISTORY_WINDOW")

    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    enable_reranker: bool = Field(default=True, alias="ENABLE_RERANKER")
    reranker_model: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", alias="RERANKER_MODEL"
    )

    ingest_rebuild: bool = Field(default=False, alias="INGEST_REBUILD")

    @property
    def cors_list(self) -> list[str]:
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
