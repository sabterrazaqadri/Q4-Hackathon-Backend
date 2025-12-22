import json
from typing import Dict, Any, List
from src.services.retrieval_service import RetrievalService
from src.config.settings import settings
from src.services.base_service import BaseService


class AgentToolsService(BaseService):
    """
    Service providing tools for the RAG agent, primarily the retrieval tool
    """
    
    def __init__(self):
        super().__init__()
        self.retrieval_service = RetrievalService()

    async def retrieval_tool(self, query: str, selected_text: str = None) -> Dict[str, Any]:
        """
        Tool that the agent can call to retrieve relevant context from the knowledge base
        """
        try:
            self.logger.info(f"Agent calling retrieval tool with query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
            
            # Use the retrieval service to get relevant context
            if selected_text:
                # If selected text is provided, use the method that incorporates both query and selected text
                result = await self.retrieval_service.retrieve_context_with_selected_text(
                    query=query,
                    selected_text=selected_text
                )
            else:
                # Otherwise, use the standard retrieval method
                result = await self.retrieval_service.retrieve_context(
                    query=query
                )
                
            self.logger.info(f"Retrieval tool returned {len(result['documents'])} documents")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in retrieval tool: {str(e)}")
            # Return an error response that the agent can understand
            return {
                "documents": [],
                "relevance_scores": [],
                "sources": [],
                "error": str(e)
            }