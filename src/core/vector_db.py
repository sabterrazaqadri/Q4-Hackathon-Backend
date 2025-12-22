"""
Qdrant vector database module for the ChatKit RAG integration.
Handles vector storage and similarity search operations.
"""
from typing import List, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

from ..chat.models import RetrievedContext
from .config import settings


class VectorDB:
    """
    Service class to handle Qdrant vector database operations.
    """
    
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=True
        )
        
        # Set the collection name for textbook content
        self.collection_name = settings.TEXTBOOK_COLLECTION_NAME
    
    async def initialize_collection(self):
        """
        Initialize the collection if it doesn't exist.
        """
        # Check if collection exists
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Create the collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)  # Assuming OpenAI ada-002 embeddings
            )
    
    async def store_embedding(
        self, 
        text: str, 
        embedding: List[float], 
        source_document: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None
    ) -> str:
        """
        Store a text embedding in the vector database.
        """
        point_id = str(uuid.uuid4())
        
        # Prepare the payload
        payload = {
            "content": text,
            "source_document": source_document,
        }
        
        if page_number is not None:
            payload["page_number"] = page_number
        if section_title is not None:
            payload["section_title"] = section_title
        
        # Store the embedding
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        
        return point_id
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        min_score: float = 0.5
    ) -> List[RetrievedContext]:
        """
        Search for similar embeddings in the database.
        """
        # Search for similar vectors
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=min_score
        )
        
        # Convert search results to RetrievedContext models
        retrieved_contexts = []
        for result in search_results:
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
    
    async def delete_embedding(self, embedding_id: str):
        """
        Delete an embedding from the database.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[embedding_id]
            )
        )


# Dependency for FastAPI
def get_vector_db():
    return VectorDB()