"""
OpenAI Agent integration for the RAG system.
"""
import asyncio
from typing import Dict, Any, List

from ..chat.models import UserQuery, RetrievedContext
from ..rag.services import RAGService


class RAGAgent:
    """
    Agent class to handle interactions with OpenAI's agent system for RAG.
    """
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    async def process_with_agent(self, user_query: UserQuery) -> Dict[str, Any]:
        """
        Process a user query using an OpenAI agent with RAG capabilities.
        """
        # Retrieve relevant context
        contexts = await self.rag_service.retrieve_context(
            user_query.content, 
            user_query.selected_text
        )
        
        # This would integrate with OpenAI's agent system
        # For now, we'll simulate the agent behavior by using the standard RAG flow
        response_content = await self.rag_service.generate_response(
            user_query.content,
            contexts
        )
        
        return {
            "response": response_content,
            "retrieved_contexts": [ctx.dict() for ctx in contexts],
            "query_id": user_query.id
        }
    
    async def validate_with_agent(self, query: str, selected_text: str = None) -> Dict[str, Any]:
        """
        Validate a query using the agent system.
        """
        is_valid, confidence, sources = await self.rag_service.validate_query(
            query, 
            selected_text
        )
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "relevant_sources": sources
        }