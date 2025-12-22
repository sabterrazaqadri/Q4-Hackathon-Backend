"""
Service for providing tools to the RAG agent.
Currently provides retrieval functionality as a tool.
"""

from typing import Dict, Any, List
from src.services.retrieval_service import RetrievalService
from src.config.settings import settings


class AgentToolsService:
    """
    Service that provides tools for the RAG agent.
    Currently includes retrieval functionality.
    """
    
    def __init__(self):
        self.retrieval_service = RetrievalService()
    
    async def retrieval_tool(self, query: str, selected_text: str = None) -> Dict[str, Any]:
        """
        Retrieval tool for the RAG agent.
        
        Args:
            query: The search query
            selected_text: Optional selected text to provide additional context
            
        Returns:
            Dictionary with retrieval results including documents and sources
        """
        try:
            # Determine the actual query to use
            actual_query = query
            if selected_text:
                # Combine the question with selected text for more context
                actual_query = f"{query} Context: {selected_text}"
            
            # Retrieve relevant chunks using the retrieval service
            result = await self.retrieval_service.retrieve_relevant_chunks(
                query=actual_query,
                top_k=settings.default_top_k
            )
            
            return {
                "documents": result.get("documents", []),
                "sources": result.get("sources", []),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "documents": [],
                "sources": [],
                "metadata": {}
            }