"""
Validation service for the ChatKit RAG integration.
Implements response accuracy verification and grounding checks.
"""
from typing import List, Tuple
import logging

from ..chat.models import RetrievedContext
from ..rag.services import RAGService
from ..models.validation_models import ValidationRequest, ValidationResponse


logger = logging.getLogger(__name__)


class ValidationService:
    """
    Service class to handle validation business logic.
    Ensures responses are grounded in textbook content without hallucinations.
    """
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
    
    async def validate_query_response(
        self, 
        query: str, 
        response: str, 
        retrieved_contexts: List[RetrievedContext]
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate that a response is properly grounded in the retrieved contexts.
        Returns: (is_accurate, confidence_score, supporting_sources)
        """
        # Check if response content is supported by retrieved contexts
        accuracy_score = self._calculate_response_accuracy(response, retrieved_contexts)
        
        # Determine if the response is sufficiently grounded
        is_accurate = accuracy_score >= 0.7  # Threshold for accuracy
        
        # Get the sources that support the response
        supporting_sources = self._get_supporting_sources(response, retrieved_contexts)
        
        logger.info(f"Response validation: accuracy={accuracy_score}, is_accurate={is_accurate}")
        
        return is_accurate, accuracy_score, supporting_sources
    
    def _calculate_response_accuracy(self, response: str, contexts: List[RetrievedContext]) -> float:
        """
        Calculate how well the response is supported by the retrieved contexts.
        This is a simplified implementation - in production, use more sophisticated NLP techniques.
        """
        if not contexts:
            return 0.0
        
        response_lower = response.lower()
        total_support_score = 0.0
        
        for context in contexts:
            context_lower = context.content.lower()
            
            # Count how many context phrases appear in the response
            context_words = set(context_lower.split())
            response_words = set(response_lower.split())
            
            # Calculate overlap between context and response
            intersection = context_words.intersection(response_words)
            if context_words:  # Avoid division by zero
                overlap_score = len(intersection) / len(context_words)
                total_support_score += overlap_score * context.similarity_score  # Weight by context relevance
        
        # Normalize the score based on number of contexts
        avg_support_score = total_support_score / len(contexts) if contexts else 0.0
        
        # Ensure the score is between 0 and 1
        return min(1.0, avg_support_score)
    
    def _get_supporting_sources(self, response: str, contexts: List[RetrievedContext]) -> List[str]:
        """
        Identify which sources support the given response.
        """
        response_lower = response.lower()
        supporting_sources = set()
        
        for context in contexts:
            context_lower = context.content.lower()
            
            # Check if there's significant overlap between response and context
            context_words = set(context_lower.split())
            response_words = set(response_lower.split())
            
            intersection = context_words.intersection(response_words)
            
            # If there's enough overlap, consider this source as supporting
            if len(intersection) > 0:
                supporting_sources.add(context.source_document)
        
        return list(supporting_sources)
    
    async def validate_query_content(self, query: str, selected_text: str = None) -> ValidationResponse:
        """
        Validate if a query can be answered using the available textbook content.
        """
        # Use the RAG service to validate the query
        is_valid, confidence, relevant_sources = await self.rag_service.validate_query(
            query, 
            selected_text
        )
        
        return ValidationResponse(
            is_valid=is_valid,
            confidence=confidence,
            relevant_sources=relevant_sources
        )


# Dependency for FastAPI
def get_validation_service(rag_service: RAGService):
    return ValidationService(rag_service)