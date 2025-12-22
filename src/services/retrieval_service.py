import time
from typing import List, Dict, Any, Optional
from src.services.base_service import BaseService
from src.services.embedding_service import CohereClient
from src.services.storage_service import QdrantService
from src.models.content_chunk import ContentChunk
from src.config.settings import settings
from src.utils.validation import validate_embedding_compatibility


class RetrievalService(BaseService):
    """
    Service for retrieving relevant content chunks from Qdrant based on user queries
    Specifically designed to support the RAG agent with OpenAI integration.
    """

    def __init__(self):
        super().__init__()
        self.embedding_service = CohereClient()
        self.storage_service = QdrantService()

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve context for the RAG agent based on the user's query
        Returns documents, relevance scores, and sources as specified in the data model
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            # Generate embedding for the query
            embedding_start = time.time()
            query_embedding = await self.embedding_service.generate_embeddings_with_retry([query])
            query_vector = query_embedding[0]  # Extract the single embedding
            embedding_time = time.time() - embedding_start

            # Verify embedding compatibility with stored vectors
            if not await self._validate_embedding_compatibility(query_vector):
                self.logger.warning("Query embedding may not be compatible with stored vectors")

            # Search in Qdrant for similar vectors
            search_start = time.time()
            similar_chunks = self.storage_service.search_similar(
                query_vector,
                limit=top_k
            )
            search_time = time.time() - search_start

            # Apply score threshold if specified
            if score_threshold is not None:
                filtered_chunks = [chunk for chunk in similar_chunks if chunk['score'] >= score_threshold]
            else:
                filtered_chunks = similar_chunks

            # Format the results according to the data model
            format_start = time.time()
            documents = []
            relevance_scores = []
            sources = []

            for chunk in filtered_chunks:
                documents.append(chunk.get("content", ""))
                relevance_scores.append(chunk.get("score", 0))

                source_info = {
                    "document_id": chunk.get("id", ""),
                    "page_number": chunk.get("page_number"),  # This may be None if not stored
                    "section_title": chunk.get("section", ""),
                    "excerpt": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", "")
                }
                # Only include optional fields if they have values
                if source_info["page_number"] is not None:
                    sources.append(source_info)
                else:
                    # Remove page_number key if it's None
                    source_info.pop("page_number", None)
                    sources.append(source_info)

            format_time = time.time() - format_start

            total_time = time.time() - start_time

            self.logger.info(
                f"Retrieval completed in {total_time:.3f}s ("
                f"embedding: {embedding_time:.3f}s, "
                f"search: {search_time:.3f}s, "
                f"format: {format_time:.3f}s) - "
                f"returned {len(documents)} results"
            )

            # Return in the format expected by the RAG agent as per data model
            return {
                "documents": documents,
                "relevance_scores": relevance_scores,
                "sources": sources
            }

        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            raise e

    async def retrieve_context_with_selected_text(
        self,
        query: str,
        selected_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve context for the RAG agent considering both the query and selected text
        """
        start_time = time.time()

        try:
            full_query = f"{query} Context: {selected_text}" if selected_text else query
            self.logger.info(f"Starting retrieval for combined query: '{full_query[:50]}{'...' if len(full_query) > 50 else ''}'")

            # Generate embedding for the combined query
            embedding_start = time.time()
            query_embedding = await self.embedding_service.generate_embeddings_with_retry([full_query])
            query_vector = query_embedding[0]  # Extract the single embedding
            embedding_time = time.time() - embedding_start

            # Verify embedding compatibility with stored vectors
            if not await self._validate_embedding_compatibility(query_vector):
                self.logger.warning("Query embedding may not be compatible with stored vectors")

            # Search in Qdrant for similar vectors
            search_start = time.time()
            similar_chunks = self.storage_service.search_similar(
                query_vector,
                limit=top_k
            )
            search_time = time.time() - search_start

            # Apply score threshold if specified
            if score_threshold is not None:
                filtered_chunks = [chunk for chunk in similar_chunks if chunk['score'] >= score_threshold]
            else:
                filtered_chunks = similar_chunks

            # Format the results according to the data model
            format_start = time.time()
            documents = []
            relevance_scores = []
            sources = []

            for chunk in filtered_chunks:
                documents.append(chunk.get("content", ""))
                relevance_scores.append(chunk.get("score", 0))

                source_info = {
                    "document_id": chunk.get("id", ""),
                    "page_number": chunk.get("page_number"),  # This may be None if not stored
                    "section_title": chunk.get("section", ""),
                    "excerpt": chunk.get("content", "")[:200] + "..." if len(chunk.get("content", "")) > 200 else chunk.get("content", "")
                }
                # Only include optional fields if they have values
                if source_info["page_number"] is not None:
                    sources.append(source_info)
                else:
                    # Remove page_number key if it's None
                    source_info.pop("page_number", None)
                    sources.append(source_info)

            format_time = time.time() - format_start

            total_time = time.time() - start_time

            self.logger.info(
                f"Retrieval with selected text completed in {total_time:.3f}s ("
                f"embedding: {embedding_time:.3f}s, "
                f"search: {search_time:.3f}s, "
                f"format: {format_time:.3f}s) - "
                f"returned {len(documents)} results"
            )

            # Return in the format expected by the RAG agent as per data model
            return {
                "documents": documents,
                "relevance_scores": relevance_scores,
                "sources": sources
            }

        except Exception as e:
            self.logger.error(f"Error retrieving context with selected text: {str(e)}")
            raise e

    async def _validate_embedding_compatibility(self, query_embedding: List[float]) -> bool:
        """
        Validate that the query embedding is compatible with stored embeddings
        """
        try:
            # Retrieve a sample of stored embeddings to check dimension compatibility
            # We'll use the storage service to get a few random vectors
            sample_chunks = self.storage_service.search_similar(
                query_embedding,
                limit=1  # Just get one result to check embedding dimensions
            )

            if sample_chunks:
                # Check if the first result has a compatible embedding
                first_chunk = sample_chunks[0]
                stored_embedding = first_chunk.get('embedding', [0] * len(query_embedding))  # Fallback if no embedding in payload
                return len(query_embedding) == len(stored_embedding)

            # If no results found, try to get any random chunk
            # For now we'll just check the Cohere model embedding dimension
            return True  # Assuming Cohere model consistency

        except Exception as e:
            self.logger.error(f"Error validating embedding compatibility: {str(e)}")
            return False