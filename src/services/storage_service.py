from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
from uuid import UUID
from src.config.settings import settings
from src.models.content_chunk import ContentChunk


class QdrantService:
    """
    Service class for interacting with Qdrant vector database
    """
    
    def __init__(self):
        # Initialize Qdrant client
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
        
        self.collection_name = settings.qdrant_collection_name
        self._initialize_collection()
    
    def _initialize_collection(self):
        """
        Initialize the Qdrant collection if it doesn't exist
        """
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # If collection doesn't exist, create it
            # We'll assume the embedding dimension based on Cohere's default (1024 for multilingual model)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),  # Adjust size as needed
            )
    
    def store_chunk(self, chunk: ContentChunk) -> bool:
        """
        Store a single content chunk in Qdrant
        """
        try:
            # Check if a chunk with this ID already exists to prevent duplicates
            existing_chunks = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(chunk.id)]
            )

            if existing_chunks:
                print(f"Chunk with ID {chunk.id} already exists, skipping...")
                return True  # Return True as it's effectively stored

            # Prepare the payload (metadata)
            payload = {
                "content": chunk.content,
                "source_url": chunk.source_url,
                "section": chunk.section,
                "metadata": chunk.metadata,
                "id": str(chunk.id),
                "created_at": chunk.created_at.isoformat(),
                "updated_at": chunk.updated_at.isoformat()
            }

            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(chunk.id),
                        vector=chunk.embedding,
                        payload=payload
                    )
                ]
            )

            return True
        except Exception as e:
            print(f"Error storing chunk in Qdrant: {str(e)}")
            return False
    
    def store_chunks(self, chunks: List[ContentChunk]) -> bool:
        """
        Store multiple content chunks in Qdrant
        """
        try:
            # First, check which chunks already exist to avoid duplicates
            chunk_ids = [str(chunk.id) for chunk in chunks]
            existing_chunks = self.client.retrieve(
                collection_name=self.collection_name,
                ids=chunk_ids
            )

            # Get IDs of existing chunks
            existing_ids = {str(chunk.id) for chunk in existing_chunks}

            # Only process chunks that don't already exist
            new_chunks = [chunk for chunk in chunks if str(chunk.id) not in existing_ids]

            if not new_chunks:
                print(f"All {len(chunks)} chunks already exist, no new chunks to store")
                return True

            points = []
            for chunk in new_chunks:
                payload = {
                    "content": chunk.content,
                    "source_url": chunk.source_url,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                    "id": str(chunk.id),
                    "created_at": chunk.created_at.isoformat(),
                    "updated_at": chunk.updated_at.isoformat()
                }

                points.append(
                    models.PointStruct(
                        id=str(chunk.id),
                        vector=chunk.embedding,
                        payload=payload
                    )
                )

            # Store all new points in a single operation
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            print(f"Stored {len(points)} new chunks out of {len(chunks)} total")
            return True
        except Exception as e:
            print(f"Error storing chunks in Qdrant: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar content chunks based on embedding similarity
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Extract relevant information from results
            similar_chunks = []
            for result in results:
                chunk_info = {
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "source_url": result.payload.get("source_url", ""),
                    "section": result.payload.get("section", ""),
                    "score": result.score
                }
                similar_chunks.append(chunk_info)
            
            return similar_chunks
        except Exception as e:
            print(f"Error searching in Qdrant: {str(e)}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific content chunk by ID
        """
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id]
            )
            
            if records:
                record = records[0]
                return {
                    "id": record.id,
                    "content": record.payload.get("content", ""),
                    "source_url": record.payload.get("source_url", ""),
                    "section": record.payload.get("section", ""),
                    "metadata": record.payload.get("metadata", {})
                }
            else:
                return None
        except Exception as e:
            print(f"Error retrieving chunk from Qdrant: {str(e)}")
            return None
    
    def delete_chunk_by_id(self, chunk_id: str) -> bool:
        """
        Delete a content chunk by ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[chunk_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting chunk from Qdrant: {str(e)}")
            return False