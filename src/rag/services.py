"""
RAG (Retrieval Augmented Generation) business logic services.
Based on the data-model.md specification and user stories.
"""
from typing import List, Tuple
import uuid
from datetime import datetime

from ..chat.models import UserQuery, RetrievedContext, AIResponse
from ..core.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
import google.generativeai as genai


class RAGService:
    """
    Service class to handle RAG business logic.
    """

    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,  # Use REST API to avoid gRPC timeout issues
            timeout=30  # Set 30 second timeout
        )

        # Initialize Cohere client for embeddings only
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)

        # Initialize Gemini client for text generation
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # Set the collection name for textbook content
        self.collection_name = settings.TEXTBOOK_COLLECTION_NAME
    
    async def retrieve_context(self, query: str, selected_text: str = None) -> List[RetrievedContext]:
        """
        Retrieve relevant context from the textbook based on the query.
        """
        # Prepare the query text (use selected text if provided, otherwise use the query)
        search_text = selected_text if selected_text else query

        # Generate embedding for the search text using Cohere
        embedding_response = self.cohere_client.embed(
            texts=[search_text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = embedding_response.embeddings[0]

        # Search in Qdrant for similar content
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=settings.SEARCH_LIMIT,
            score_threshold=settings.MIN_SIMILARITY_SCORE
        )

        # Convert search results to RetrievedContext models
        retrieved_contexts = []
        for result in search_results.points:  # Access the points attribute
            payload = result.payload
            retrieved_context = RetrievedContext(
                id=str(result.id),
                content=payload.get("content", ""),
                source_document=payload.get("source_document", ""),
                page_number=payload.get("page_number"),
                section_title=payload.get("section_title"),
                similarity_score=result.score,
                embedding_id=str(result.id)
            )
            retrieved_contexts.append(retrieved_context)

        return retrieved_contexts
    
    async def generate_response(
        self,
        query: str,
        retrieved_contexts: List[RetrievedContext],
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response based on the query and retrieved contexts.
        """
        # Check if we have any retrieved contexts
        if not retrieved_contexts or len(retrieved_contexts) == 0:
            return "I don't have enough information from the textbook to answer this question. Please try asking about topics covered in the Physical AI & Humanoid Robotics textbook."

        # Prepare the context content with source information
        context_parts = []
        for i, ctx in enumerate(retrieved_contexts, 1):
            source_info = f"[Source: {ctx.source_document}"
            if ctx.section_title:
                source_info += f", Section: {ctx.section_title}"
            if ctx.page_number:
                source_info += f", Page: {ctx.page_number}"
            source_info += "]"

            context_parts.append(f"Document {i} {source_info}:\n{ctx.content}")

        context_content = "\n\n".join(context_parts)

        # Construct a strict prompt to ground the response in textbook content
        system_instructions = """You are an AI assistant for the Physical AI & Humanoid Robotics textbook.

CRITICAL RULES:
1. You MUST answer ONLY using information from the provided textbook documents below
2. Do NOT use your general knowledge or training data
3. Do NOT make up, infer, or hallucinate any information
4. If the documents don't contain enough information to answer the question, say: "I don't have enough information from the textbook to answer this question completely."
5. Always cite which document(s) you're referencing in your answer (e.g., "According to Document 1...")
6. Stay strictly within the scope of the provided textbook content"""

        # Combine system instructions with context and query
        full_prompt = f"""{system_instructions}

Textbook Documents:

{context_content}

Question: {query}

Remember: Answer ONLY based on the information in the documents above. If the answer is not in the documents, say so clearly."""

        # Generate response using Gemini API with strict RAG mode
        try:
            # Configure generation settings
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=settings.MAX_RESPONSE_TOKENS,
                candidate_count=1,
            )

            # Generate response
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE_SPEECH': 'block_none',
                    'SEXUALLY_EXPLICIT': 'block_none',
                    'DANGEROUS_CONTENT': 'block_none'
                }
            )

            # Extract response text
            response_text = response.text

            # Add a disclaimer if no strong matches were found
            min_score = min(ctx.similarity_score for ctx in retrieved_contexts)
            if min_score < 0.7:
                response_text = f"{response_text}\n\n[Note: The confidence in this answer is moderate as the textbook may not cover this topic in detail.]"

            return response_text

        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
    
    async def validate_query(self, query: str, selected_text: str = None) -> Tuple[bool, float, List[str]]:
        """
        Validate if a query can be answered using the available textbook content.
        Returns: (is_valid, confidence, relevant_sources)
        """
        # Retrieve context for the query
        contexts = await self.retrieve_context(query, selected_text)

        if not contexts:
            return False, 0.0, []

        # Calculate average similarity score as confidence
        avg_similarity = sum(ctx.similarity_score for ctx in contexts) / len(contexts)

        # Get unique source documents
        source_documents = list(set(ctx.source_document for ctx in contexts))

        # Consider valid if at least one context has good similarity score
        is_valid = any(ctx.similarity_score >= settings.MIN_SIMILARITY_SCORE for ctx in contexts)

        return is_valid, avg_similarity, source_documents

    async def verify_response_grounding(
        self,
        query: str,
        response: str,
        retrieved_contexts: List[RetrievedContext]
    ) -> Tuple[bool, float]:
        """
        Verify that a response is properly grounded in the retrieved contexts.
        Returns: (is_grounded, confidence_score)
        """
        if not retrieved_contexts:
            return False, 0.0

        # Calculate how well the response is supported by the contexts
        response_words = set(response.lower().split())
        total_context_support = 0.0
        total_weight = 0.0

        for ctx in retrieved_contexts:
            context_words = set(ctx.content.lower().split())

            # Calculate overlap between response and context
            overlap = len(response_words.intersection(context_words))
            max_possible_overlap = min(len(response_words), len(context_words))

            if max_possible_overlap > 0:
                overlap_ratio = overlap / max_possible_overlap
                # Weight the support by the context's similarity score
                context_support = overlap_ratio * ctx.similarity_score
                total_context_support += context_support
                total_weight += ctx.similarity_score

        # Calculate weighted average support
        if total_weight > 0:
            grounding_score = total_context_support / total_weight
        else:
            grounding_score = 0.0

        # Determine if the response is sufficiently grounded
        is_grounded = grounding_score >= settings.MIN_SIMILARITY_SCORE

        return is_grounded, grounding_score


# Dependency for FastAPI
def get_rag_service():
    return RAGService()