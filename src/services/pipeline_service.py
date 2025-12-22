from typing import List, Dict, Any
from src.services.crawler_service import CrawlerService
from src.services.chunking_service import ChunkingService
from src.services.embedding_service import CohereClient
from src.services.storage_service import QdrantService
from src.models.content_chunk import ContentChunk
from datetime import datetime
from uuid import uuid4


class PipelineService:
    """
    Service to orchestrate the complete RAG pipeline: crawl → chunk → embed → store
    """
    
    def __init__(self):
        self.crawler_service = CrawlerService()
        self.chunking_service = ChunkingService()
        self.embedding_service = CohereClient()
        self.storage_service = QdrantService()
    
    async def execute_pipeline(self, urls: List[str]) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline for a list of URLs
        """
        result = {
            "crawling": {"processed": 0, "failed": 0, "details": []},
            "chunking": {"processed": 0, "failed": 0, "details": []},
            "embedding": {"processed": 0, "failed": 0, "details": []},
            "storage": {"stored": 0, "skipped": 0, "failed": 0, "details": []},
            "start_time": datetime.now(),
            "end_time": None
        }
        
        try:
            # Step 1: Crawling
            crawled_data = []
            for url in urls:
                try:
                    page_data = await self.crawler_service.crawl_single_page(url)
                    if page_data:
                        crawled_data.append(page_data)
                        result["crawling"]["processed"] += 1
                        result["crawling"]["details"].append({
                            "url": url,
                            "status": "success",
                            "content_length": len(page_data.get("content", ""))
                        })
                    else:
                        result["crawling"]["failed"] += 1
                        result["crawling"]["details"].append({
                            "url": url,
                            "status": "failed"
                        })
                except Exception as e:
                    result["crawling"]["failed"] += 1
                    result["crawling"]["details"].append({
                        "url": url,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Step 2: Chunking
            all_chunks = []
            for page_data in crawled_data:
                try:
                    chunks = self.chunking_service.chunk_content(
                        page_data["content"],
                        url=page_data["url"],
                        section=page_data["metadata"].get("title", "")
                    )
                    
                    # Validate chunks before proceeding
                    valid_chunks = []
                    for chunk in chunks:
                        if self.chunking_service.validate_chunk(chunk):
                            valid_chunks.append(chunk)
                        else:
                            result["chunking"]["failed"] += 1
                    
                    all_chunks.extend(valid_chunks)
                    result["chunking"]["processed"] += len(valid_chunks)
                    
                except Exception as e:
                    result["chunking"]["failed"] += 1
                    result["chunking"]["details"].append({
                        "url": page_data["url"],
                        "status": "error",
                        "error": str(e)
                    })
            
            # Step 3: Embedding
            if all_chunks:
                # Extract content from chunks for embedding
                contents = [chunk["content"] for chunk in all_chunks]
                
                try:
                    embeddings = await self.embedding_service.generate_embeddings_with_retry(contents)
                    
                    # Validate embeddings
                    valid_embeddings_count = 0
                    for i, embedding in enumerate(embeddings):
                        if self.embedding_service.validate_embeddings([embedding]):
                            all_chunks[i]["embedding"] = embedding
                            valid_embeddings_count += 1
                        else:
                            result["embedding"]["failed"] += 1
                            all_chunks[i]["embedding"] = []  # Mark as failed
                    
                    result["embedding"]["processed"] = valid_embeddings_count
                    result["embedding"]["failed"] += len(all_chunks) - valid_embeddings_count
                    
                except Exception as e:
                    result["embedding"]["failed"] = len(all_chunks)
                    result["embedding"]["details"].append({
                        "status": "error",
                        "error": str(e)
                    })
                    # Mark all chunks as having failed embedding
                    for chunk in all_chunks:
                        chunk["embedding"] = []
            else:
                result["embedding"]["processed"] = 0
            
            # Step 4: Storage
            chunks_with_embeddings = []
            for chunk in all_chunks:
                if chunk.get("embedding") and len(chunk["embedding"]) > 0:
                    content_chunk = ContentChunk(
                        id=uuid4(),
                        content=chunk["content"],
                        source_url=chunk["source_url"],
                        section=chunk["section"],
                        embedding=chunk["embedding"],
                        created_at=chunk.get("created_at", datetime.now()),
                        updated_at=chunk.get("updated_at", datetime.now()),
                        metadata=chunk.get("metadata", {})
                    )
                    chunks_with_embeddings.append(content_chunk)
            
            if chunks_with_embeddings:
                storage_success = self.storage_service.store_chunks(chunks_with_embeddings)
                if storage_success:
                    result["storage"]["stored"] = len(chunks_with_embeddings)
                else:
                    result["storage"]["failed"] = len(chunks_with_embeddings)
            else:
                result["storage"]["skipped"] = len([c for c in all_chunks if not c.get("embedding")])
            
            result["end_time"] = datetime.now()
            return result
            
        except Exception as e:
            result["end_time"] = datetime.now()
            result["pipeline_error"] = str(e)
            return result