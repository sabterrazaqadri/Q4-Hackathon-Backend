"""
Chat-related data models using Pydantic.
Based on the data-model.md specification.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class UserQuery(BaseModel):
    """
    A text-based question or statement submitted through the ChatKit interface,
    which may reference general book content or specific selected text.
    """
    id: str
    content: str = Field(..., min_length=1, max_length=1000)
    selected_text: Optional[str] = Field(None, min_length=1, max_length=5000)
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class RetrievedContext(BaseModel):
    """
    Relevant segments of the Physical AI & Humanoid Robotics textbook that are 
    retrieved by the RAG system to inform the response.
    """
    id: str
    content: str = Field(..., min_length=10, max_length=5000)
    source_document: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    embedding_id: str


class AIResponse(BaseModel):
    """
    The generated text response that addresses the user's query based solely 
    on the retrieved textbook content.
    """
    id: str
    content: str = Field(..., min_length=10, max_length=10000)
    query_id: str
    retrieved_context_ids: List[str]
    timestamp: datetime
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    source_documents: List[str]


class ChatSession(BaseModel):
    """
    A sequence of related queries and responses that maintains conversational 
    context while remaining grounded in the textbook content.
    """
    id: str
    created_at: datetime
    last_interaction: datetime
    user_id: Optional[str] = None
    is_active: bool
    query_count: int = Field(..., ge=0)