from typing import List, Dict, Any, Optional
from src.config.settings import settings
from src.utils.text_processing import split_text_by_size
from src.utils.content_processing import clean_text
from src.services.base_service import BaseService
from src.utils.exceptions import ChunkingError
from uuid import uuid4
from datetime import datetime


class ChunkingService(BaseService):
    """
    Service for chunking content into smaller pieces with configurable size
    """
    
    def __init__(self):
        super().__init__()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def chunk_content(
        self, 
        content: str, 
        url: str, 
        section: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk content into smaller pieces with configurable size and overlap
        """
        try:
            # Use provided values or fall back to settings
            actual_chunk_size = chunk_size if chunk_size is not None else self.chunk_size
            actual_overlap = overlap if overlap is not None else self.chunk_overlap
            
            # Clean and validate content
            cleaned_content = clean_text(content)
            if not cleaned_content:
                self.logger.warning(f"No content to chunk for URL: {url}")
                return []
            
            # Split content into chunks
            chunks = split_text_by_size(
                cleaned_content, 
                actual_chunk_size, 
                actual_overlap
            )
            
            # Create chunk representations with metadata
            result_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_id = str(uuid4())
                
                chunk_data = {
                    "id": chunk_id,
                    "content": chunk_text,
                    "source_url": url,
                    "section": section,
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_content_length": len(cleaned_content),
                        "chunk_length": len(chunk_text)
                    },
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                result_chunks.append(chunk_data)
            
            self.logger.info(f"Successfully chunked content from {url} into {len(result_chunks)} chunks")
            return result_chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking content from {url}: {str(e)}")
            raise ChunkingError(f"Failed to chunk content from {url}: {str(e)}")
    
    def validate_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate that a chunk meets quality standards
        """
        try:
            # Check if chunk has required fields
            required_fields = ['id', 'content', 'source_url']
            for field in required_fields:
                if field not in chunk or not chunk[field]:
                    self.logger.error(f"Chunk missing required field: {field}")
                    return False
            
            # Check if content is not too short (meaningful content check)
            content = chunk.get('content', '')
            if len(content.strip()) < 10:  # At least 10 characters
                self.logger.warning(f"Chunk has very short content: {chunk.get('id')}")
                return False
            
            # Check if content is not too long (oversized chunk check)
            if len(content) > settings.chunk_size * 2:  # Allow some flexibility
                self.logger.warning(f"Chunk exceeds expected size: {chunk.get('id')}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating chunk {chunk.get('id', 'unknown')}: {str(e)}")
            return False
    
    def rechunk_content(
        self, 
        existing_chunks: List[Dict[str, Any]], 
        new_chunk_size: int, 
        new_overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Rechunk existing content with new parameters
        """
        try:
            # Combine all existing chunks back into full content
            combined_content = " ".join([chunk['content'] for chunk in existing_chunks])
            
            # Extract common metadata
            if existing_chunks:
                first_chunk = existing_chunks[0]
                url = first_chunk.get('source_url', '')
                section = first_chunk.get('section', None)
            else:
                url = ''
                section = None
            
            # Rechunk the combined content
            return self.chunk_content(
                combined_content,
                url,
                section,
                new_chunk_size,
                new_overlap
            )
        except Exception as e:
            self.logger.error(f"Error rechunking content: {str(e)}")
            raise ChunkingError(f"Failed to rechunk content: {str(e)}")