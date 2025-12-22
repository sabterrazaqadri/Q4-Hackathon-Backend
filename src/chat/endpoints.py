"""
ChatKit-compatible API endpoints for the RAG system.
Based on the OpenAPI contract in contracts/openapi.yaml.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import uuid
from datetime import datetime

from ..chat.models import UserQuery, AIResponse, RetrievedContext, ChatSession
from ..chat.services import ChatService
from ..rag.services import RAGService
from ..core.config import settings
from ..chat.services import get_chat_service
from ..services.session_service import SessionService, get_session_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions")
async def create_chat_completion(
    request: dict,  # Using dict to match OpenAI's format
    chat_service: ChatService = Depends(get_chat_service),
    rag_service: RAGService = Depends(),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Process a user query through the RAG system and return a response
    based on the textbook content.
    Supports session-based conversations to maintain context.
    """
    try:
        # Extract messages from the request (OpenAI format)
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")

        # Get the last user message as the query
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content")
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Extract optional parameters
        selected_text = request.get("selected_text", None)
        session_id = request.get("session_id", None)
        user_id = request.get("user_id", None)
        temperature = request.get("temperature", 0.7)

        # If no session_id is provided, create a new session
        if not session_id:
            session = await session_service.create_session(user_id)
            session_id = session.id
        else:
            # Validate that the session exists
            existing_session = await session_service.get_session(session_id)
            if not existing_session:
                # If session doesn't exist, create a new one but keep the ID
                session = await session_service.create_session(user_id)

        # Create UserQuery model
        user_query = UserQuery(
            id=str(uuid.uuid4()),
            content=user_message,
            selected_text=selected_text,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id
        )

        # Process the query through the RAG service with session context
        response = await chat_service.process_query(
            user_query,
            rag_service,
            temperature,
            session_id
        )

        # Format response to match OpenAI API format
        # Note: The AIResponse model doesn't match OpenAI's format exactly,
        # so we'll return a dict that conforms to the OpenAI schema
        return {
            "id": f"chatcmpl-{response.id}",
            "object": "chat.completion",
            "created": int(response.timestamp.timestamp()),
            "model": settings.GEMINI_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Placeholder - would need actual token count
                "completion_tokens": 0,  # Placeholder - would need actual token count
                "total_tokens": 0  # Placeholder - would need actual token count
            },
            "retrieved_context": []  # Placeholder - would add actual context if needed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")


@router.post("/validate", response_model=dict)
async def validate_query(
    request: dict,
    rag_service: RAGService = Depends()
):
    """
    Check if a query can be answered using the available textbook content.
    """
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        selected_text = request.get("selected_text", None)

        # Validate the query against available content
        is_valid, confidence, relevant_sources = await rag_service.validate_query(query, selected_text)

        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "relevant_sources": relevant_sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating query: {str(e)}")