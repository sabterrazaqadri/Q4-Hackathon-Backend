from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


class ContentChunkBase(BaseModel):
    """Base model for content chunks"""
    content: str
    source_url: str
    section: Optional[str] = None
    module: Optional[str] = None
    chapter: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ContentChunkCreate(ContentChunkBase):
    """Model for creating content chunks"""
    pass


class ContentChunk(ContentChunkBase):
    """Model for content chunks with all fields"""
    id: UUID
    embedding: List[float]
    created_at: datetime
    updated_at: datetime


class QueryResult(BaseModel):
    """Model for query results"""
    query_text: str
    retrieved_chunks: List[ContentChunk]
    confidence_scores: List[float]
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


class QueryRequest(BaseModel):
    """Model for query requests"""
    query: str
    top_k: int = 5
    score_threshold: Optional[float] = None


class CrawlJobBase(BaseModel):
    """Base model for crawl job"""
    source_urls: List[str]


class CrawlJobCreate(CrawlJobBase):
    """Model for creating crawl jobs"""
    pass


class CrawlJob(CrawlJobBase):
    """Model for crawl jobs with all fields"""
    id: UUID
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processed_count: int = 0
    failed_count: int = 0
    error_details: Optional[str] = None