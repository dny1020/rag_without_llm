from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.db.models import Conversation, Message
from src.schemas.message import Citation
from src.services.rag import RAGService

settings = get_settings()


class ChatService:
    def __init__(self, rag_service: RAGService | None = None) -> None:
        self.rag_service = rag_service or RAGService()

    def _get_or_create_conversation(
        self, db: Session, conversation_id: str | None, user_id: str | None
    ) -> Conversation:
        if conversation_id:
            conversation = db.scalar(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            if not conversation:
                raise ValueError("Conversation not found")
            return conversation

        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    def _recent_history(self, db: Session, conversation_id: str) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(settings.history_window)
        )
        return list(reversed(list(db.scalars(stmt).all())))

    def chat(
        self,
        db: Session,
        content: str,
        conversation_id: str | None,
        user_id: str | None,
    ) -> tuple[str, str, list[Citation]]:
        conversation = self._get_or_create_conversation(db, conversation_id, user_id)

        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=content,
        )
        db.add(user_msg)
        db.commit()

        history = self._recent_history(db, conversation.id)
        contextual_query = " ".join(
            [m.content for m in history if m.role == "user"] + [content]
        ).strip()

        retrieved = self.rag_service.retrieve(db, contextual_query)
        answer = self.rag_service.build_answer(retrieved)

        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
        )
        db.add(assistant_msg)
        db.commit()

        citations = [
            Citation(
                chunk_id=x.chunk_id,
                source_path=x.source_path,
                snippet=x.content[:280],
                score=x.score,
            )
            for x in retrieved
        ]
        return conversation.id, answer, citations

    def history(self, db: Session, conversation_id: str) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        return list(db.scalars(stmt).all())
