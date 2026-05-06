import logging
import re
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder
from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.db.models import Document, DocumentChunk
from src.helpers.utils import (
    chunk_text,
    file_sha256,
    lexical_overlap_score,
    rank_fusion,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    source_path: str
    content: str
    score: float


class RAGService:
    def __init__(self) -> None:
        self.embedder: TextEmbedding | None = None
        self.reranker: CrossEncoder | None = None
        self.last_ingest_summary: dict[str, object] = {
            "docs_path": str(settings.docs_path),
            "docs_path_exists": settings.docs_path.exists(),
            "scanned_files": 0,
            "updated_files": 0,
            "skipped_unchanged_files": 0,
            "failed_files": 0,
            "indexed_chunks": 0,
            "total_chunks_in_db": 0,
            "failures": [],
        }

    def _get_embedder(self) -> TextEmbedding:
        if self.embedder is None:
            self.embedder = TextEmbedding(model_name=settings.embedding_model)
        return self.embedder

    def _get_reranker(self) -> CrossEncoder:
        if self.reranker is None:
            self.reranker = CrossEncoder(settings.reranker_model)
        return self.reranker

    def _extract_mhtml_text(self, file_path: Path) -> tuple[str, str]:
        raw_mhtml = file_path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw_mhtml)
        html_body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    html_body = payload.decode(charset, errors="ignore")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                html_body = payload.decode(charset, errors="ignore")

        if not html_body:
            raise ValueError(f"No HTML payload found in {file_path}")

        soup = BeautifulSoup(html_body, "lxml")
        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else file_path.stem
        )
        for tag in soup(["script", "style", "noscript", "svg", "img"]):
            tag.decompose()

        for selector in (
            "#mw-navigation",
            "#footer",
            "#catlinks",
            "#siteSub",
            "#jump-to-nav",
            ".noprint",
        ):
            for node in soup.select(selector):
                node.decompose()

        content_node = (
            soup.select_one("#mw-content-text")
            or soup.select_one("#bodyContent")
            or soup.select_one("article")
            or soup.body
            or soup
        )
        text = content_node.get_text(separator=" ", strip=True)
        text = unescape(text).replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return title, text

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = list(self._get_embedder().embed(texts))
        return [v.tolist() for v in vectors]

    def ingest_docs(
        self, db: Session, root_path: Path, rebuild: bool = False
    ) -> dict[str, object]:
        files = sorted(root_path.rglob("*.mhtml"))
        scanned_files = len(files)
        docs_path_exists = root_path.exists()
        updated_files = 0
        skipped_unchanged_files = 0
        failed_files = 0
        indexed_chunks = 0
        failures: list[str] = []

        for file_path in files:
            try:
                raw = file_path.read_bytes()
                sha = file_sha256(raw)
                rel_path = str(file_path)

                existing = db.scalar(select(Document).where(Document.path == rel_path))
                if existing and existing.sha256 == sha and not rebuild:
                    skipped_unchanged_files += 1
                    continue

                title, text = self._extract_mhtml_text(file_path)
                chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                if not chunks:
                    continue
                embeddings = self._embed(chunks)

                if existing:
                    db.execute(
                        delete(DocumentChunk).where(
                            DocumentChunk.document_id == existing.id
                        )
                    )
                    existing.title = title
                    existing.sha256 = sha
                    document = existing
                else:
                    document = Document(path=rel_path, title=title, sha256=sha)
                    db.add(document)
                    db.flush()

                chunk_rows = [
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=i,
                        content=chunk,
                        source_path=rel_path,
                        embedding=embedding,
                    )
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
                db.add_all(chunk_rows)
                db.commit()

                updated_files += 1
                indexed_chunks += len(chunk_rows)
            except Exception as exc:
                db.rollback()
                failed_files += 1
                failures.append(f"{file_path}: {type(exc).__name__}: {exc}")

        total_chunks_in_db = db.query(DocumentChunk).count()
        summary: dict[str, object] = {
            "docs_path": str(root_path),
            "docs_path_exists": docs_path_exists,
            "scanned_files": scanned_files,
            "updated_files": updated_files,
            "skipped_unchanged_files": skipped_unchanged_files,
            "failed_files": failed_files,
            "indexed_chunks": indexed_chunks,
            "total_chunks_in_db": int(total_chunks_in_db),
            "failures": failures[:20],
        }
        self.last_ingest_summary = summary
        logger.info(
            (
                "Ingestion finished: path=%s exists=%s scanned=%s updated=%s "
                "skipped=%s failed=%s indexed_chunks=%s total_chunks=%s"
            ),
            summary["docs_path"],
            summary["docs_path_exists"],
            summary["scanned_files"],
            summary["updated_files"],
            summary["skipped_unchanged_files"],
            summary["failed_files"],
            summary["indexed_chunks"],
            summary["total_chunks_in_db"],
        )
        return summary

    def has_chunks(self, db: Session) -> bool:
        return db.scalar(select(DocumentChunk.id).limit(1)) is not None

    def total_chunks(self, db: Session) -> int:
        return int(db.query(DocumentChunk).count())

    def _semantic_candidates(self, db: Session, query_vector: list[float], limit: int):
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.source_path,
                DocumentChunk.content,
                DocumentChunk.embedding.cosine_distance(query_vector).label("distance"),
            )
            .order_by("distance")
            .limit(limit)
        )
        return db.execute(stmt).all()

    def _lexical_candidates(self, db: Session, query: str, limit: int):
        stmt = text(
            """
            SELECT
                id,
                source_path,
                content,
                ts_rank_cd(
                    to_tsvector('spanish', content),
                    plainto_tsquery('spanish', :q)
                ) AS lexical_rank
            FROM document_chunks
            WHERE to_tsvector('spanish', content) @@ plainto_tsquery('spanish', :q)
            ORDER BY lexical_rank DESC
            LIMIT :limit
            """
        )
        return db.execute(stmt, {"q": query, "limit": limit}).all()

    def retrieve(
        self, db: Session, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        candidate_pool = max(settings.retrieval_candidate_pool, k * 5)
        query_vector = self._embed([query])[0]

        semantic_rows = self._semantic_candidates(db, query_vector, candidate_pool)
        semantic_ids = [row.id for row in semantic_rows]
        lexical_rows = self._lexical_candidates(db, query, candidate_pool)
        lexical_ids = [row.id for row in lexical_rows]
        if not lexical_ids:
            query_terms = set(re.findall(r"\w+", query.lower()))
            lexical_scored = sorted(
                (
                    (row.id, lexical_overlap_score(query_terms, row.content))
                    for row in semantic_rows
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            lexical_ids = [chunk_id for chunk_id, score in lexical_scored if score > 0]

        fused = rank_fusion(semantic_ids, lexical_ids)
        fused_ids = [
            x[0] for x in sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ]
        candidate_ids = fused_ids[:candidate_pool]
        candidate_set = set(candidate_ids)

        by_id = {
            row.id: RetrievedChunk(
                chunk_id=row.id,
                source_path=row.source_path,
                content=row.content,
                score=float(fused.get(row.id, 0.0)),
            )
            for row in semantic_rows
            if row.id in candidate_set
        }
        for row in lexical_rows:
            if row.id in candidate_set and row.id not in by_id:
                by_id[row.id] = RetrievedChunk(
                    chunk_id=row.id,
                    source_path=row.source_path,
                    content=row.content,
                    score=float(fused.get(row.id, 0.0)),
                )

        candidates = [by_id[i] for i in candidate_ids if i in by_id]
        if settings.enable_reranker and candidates:
            reranker_scores = self._get_reranker().predict(
                [(query, c.content) for c in candidates]
            )
            for idx, score in enumerate(reranker_scores):
                candidates[idx].score = float(score)
            candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]

    def build_answer(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return (
                "No encontré evidencia suficiente en la base documental local para responder "
                "con precisión."
            )
        bullets = []
        for idx, chunk in enumerate(chunks, start=1):
            snippet = chunk.content[:280].strip()
            bullets.append(f"{idx}. {snippet} (fuente: {chunk.source_path})")
        return "Respuesta basada en evidencia recuperada:\n" + "\n".join(bullets)
