from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.db.session import get_db
from src.schemas.message import ChatResponse, HistoryItem, IngestResponse, MessageInput
from src.services.chat import ChatService
from src.services.rag import RAGService

router = APIRouter()
settings = get_settings()
rag_service = RAGService()
chat_service = ChatService(rag_service=rag_service)


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, str | bool | int]:
    return {
        "status": "ok",
        "indexed": rag_service.has_chunks(db),
        "docs_path_exists": settings.docs_path.exists(),
        "total_chunks_in_db": rag_service.total_chunks(db),
        "scanned_files_last_run": int(rag_service.last_ingest_summary["scanned_files"]),
    }


@router.post("/ingest", response_model=IngestResponse)
def ingest(rebuild: bool = False, db: Session = Depends(get_db)) -> IngestResponse:
    summary = rag_service.ingest_docs(
        db, settings.docs_path, rebuild=rebuild or settings.ingest_rebuild
    )
    return IngestResponse(**summary)


@router.post("/chat", response_model=ChatResponse)
def chat(data: MessageInput, db: Session = Depends(get_db)) -> ChatResponse:
    if not rag_service.has_chunks(db):
        rag_service.ingest_docs(
            db, settings.docs_path, rebuild=settings.ingest_rebuild
        )

    try:
        conversation_id, answer, citations = chat_service.chat(
            db=db,
            content=data.content,
            conversation_id=data.conversation_id,
            user_id=data.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        citations=citations,
    )


@router.get("/history/{conversation_id}", response_model=list[HistoryItem])
def get_history(
    conversation_id: str, db: Session = Depends(get_db)
) -> list[HistoryItem]:
    messages = chat_service.history(db, conversation_id)
    return [
        HistoryItem(role=m.role, content=m.content, created_at=m.created_at)
        for m in messages
    ]
