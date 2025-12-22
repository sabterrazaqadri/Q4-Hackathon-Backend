"""
Chat business logic services.
Based on the data-model.md specification and user stories.
"""
from typing import List
import uuid
from datetime import datetime

from ..chat.models import UserQuery, AIResponse, RetrievedContext
from ..rag.services import RAGService
from ..services.session_service import SessionService


class ChatService:
    """
    Service class to handle chat business logic.
    """

    def __init__(self, session_service: SessionService = None):
        self.session_service = session_service or SessionService()

    async def process_query(
        self,
        user_query: UserQuery,
        rag_service: RAGService,
        temperature: float = 0.7,
        session_id: str = None
    ) -> AIResponse:
        """
        Process a user query through the RAG system and generate a response.
        If a session_id is provided, maintains conversational context.
        """
        # If session_id is provided, update the session
        if session_id:
            await self.session_service.update_session_interaction(session_id)

            # Add the user's query to the conversation history
            user_message = {
                "role": "user",
                "content": user_query.content,
                "timestamp": user_query.timestamp.isoformat() if user_query.timestamp else None
            }
            await self.session_service.add_message_to_conversation(session_id, user_message)

        # Use the RAG service to retrieve relevant context
        retrieved_contexts: List[RetrievedContext] = await rag_service.retrieve_context(
            user_query.content,
            user_query.selected_text
        )

        # Generate response based on the retrieved context
        response_content = await rag_service.generate_response(
            user_query.content,
            retrieved_contexts,
            temperature
        )

        # Create and return the AI response
        ai_response = AIResponse(
            id=str(uuid.uuid4()),
            content=response_content,
            query_id=user_query.id,
            retrieved_context_ids=[ctx.id for ctx in retrieved_contexts],
            timestamp=datetime.now(),
            confidence_score=0.85,  # Placeholder - would come from actual model
            source_documents=list(set([ctx.source_document for ctx in retrieved_contexts]))
        )

        # If session_id is provided, add the AI response to the conversation history
        if session_id:
            assistant_message = {
                "role": "assistant",
                "content": ai_response.content,
                "timestamp": ai_response.timestamp.isoformat()
            }
            await self.session_service.add_message_to_conversation(session_id, assistant_message)

        return ai_response


# Dependency for FastAPI
def get_chat_service(session_service=SessionService()):
    return ChatService(session_service)