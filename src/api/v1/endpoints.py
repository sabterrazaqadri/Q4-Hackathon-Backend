from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from uuid import UUID
from src.models.content_chunk import CrawlJob, CrawlJobCreate, QueryRequest
from src.services.crawler_service import CrawlerService
from src.services.chunking_service import ChunkingService
from src.services.embedding_service import CohereClient
from src.services.storage_service import QdrantService
from src.config.settings import settings

router = APIRouter()
crawler_service = CrawlerService()
chunking_service = ChunkingService()
embedding_service = CohereClient()
storage_service = QdrantService()


@router.post("/crawl", response_model=CrawlJob)
async def initiate_crawl_job(crawl_job_create: CrawlJobCreate):
    """
    Initiate a new crawl job
    """
    try:
        # Validate input
        if not crawl_job_create.source_urls:
            raise HTTPException(status_code=400, detail="Source URLs are required")
        
        # Check if all URLs are valid
        for url in crawl_job_create.source_urls:
            from src.utils.url_validator import is_valid_url
            if not is_valid_url(url):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
        
        # Initiate the crawl job
        crawl_job = crawler_service.initiate_crawl_job(crawl_job_create)
        
        # Execute the crawl job in the background
        # In a real implementation, we would run this as a background task
        # For now, we'll execute it synchronously for demonstration purposes
        updated_job = await crawler_service.execute_crawl_job(str(crawl_job.id))
        
        return updated_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/crawl/{job_id}", response_model=CrawlJob)
async def get_crawl_job_status(job_id: str):
    """
    Get the status of a crawl job
    """
    try:
        crawl_job = crawler_service.get_crawl_job(job_id)
        if not crawl_job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        return crawl_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunks/process")
async def process_and_chunk_content(urls: List[str]):
    """
    Process and chunk content from provided URLs
    """
    try:
        # First, crawl the content
        crawled_data = []
        for url in urls:
            page_data = await crawler_service.crawl_single_page(url)
            if page_data:
                crawled_data.append(page_data)
        
        # Then chunk the content
        chunk_results = []
        for page_data in crawled_data:
            chunks = chunking_service.chunk_content(
                page_data["content"],
                url=page_data["url"],
                section=page_data["metadata"].get("title", "")
            )
            chunk_results.extend(chunks)
        
        return {"processed_chunks": len(chunk_results), "chunks": chunk_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/generate")
async def generate_embeddings(contents: List[str]):
    """
    Generate embeddings for provided content chunks
    """
    try:
        embeddings = embedding_service.generate_embeddings(contents)
        return {"embeddings_generated": len(embeddings), "embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/store")
async def store_vectors_in_qdrant(chunks_with_embeddings: List[dict]):
    """
    Store content chunks with embeddings in Qdrant
    """
    try:
        from src.models.content_chunk import ContentChunk
        from datetime import datetime

        content_chunks = []
        for chunk_data in chunks_with_embeddings:
            content_chunk = ContentChunk(
                id=chunk_data.get("id"),
                content=chunk_data.get("content", ""),
                source_url=chunk_data.get("source_url", ""),
                section=chunk_data.get("section"),
                embedding=chunk_data.get("embedding", []),
                created_at=chunk_data.get("created_at", datetime.now()),
                updated_at=chunk_data.get("updated_at", datetime.now()),
                metadata=chunk_data.get("metadata")
            )
            content_chunks.append(content_chunk)

        success = storage_service.store_chunks(content_chunks)
        if success:
            return {"stored_chunks": len(content_chunks), "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store chunks in Qdrant")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/qdrant")
async def check_qdrant_health():
    """
    Check the health/connectivity of the Qdrant service
    """
    try:
        # Try to get the collection info as a simple health check
        collection_info = storage_service.client.get_collection(settings.qdrant_collection_name)
        return {
            "status": "healthy",
            "collection": settings.qdrant_collection_name,
            "vectors_count": collection_info.points_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant health check failed: {str(e)}")


class ValidationRequest(BaseModel):
    query: str
    expected_result_ids: List[str] = []
    top_k: int = 5
    score_threshold: Optional[float] = None


@router.post("/query/validate")
async def validate_query_endpoint(validation_request: ValidationRequest):
    """
    Validate query results for correctness against expected results
    """
    try:
        from src.services.retrieval_service import RetrievalService
        from src.utils.validation import validate_query_result_relevance

        service = RetrievalService()
        results = await service.retrieve_relevant_chunks(
            validation_request.query,
            top_k=validation_request.top_k,
            score_threshold=validation_request.score_threshold
        )

        # Validate the results
        validation_result = validate_query_result_relevance(
            validation_request.query,
            results,
            expected_keywords=None  # Use default keyword extraction
        )

        # Add additional validation info
        validation_result["query"] = validation_request.query
        validation_result["retrieved_count"] = len(results)
        validation_result["expected_match_count"] = len(validation_request.expected_result_ids)

        # Check if expected results are in the retrieved results
        retrieved_ids = [r["id"] for r in results if r["id"] is not None]
        expected_found = [id for id in validation_request.expected_result_ids if id in retrieved_ids]
        validation_result["expected_found"] = expected_found
        validation_result["expected_recall"] = len(expected_found) / len(validation_request.expected_result_ids) if validation_request.expected_result_ids else None

        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")


@router.get("/pipeline/test")
async def test_pipeline_endpoint():
    """
    Test the complete retrieval pipeline with known inputs and validation
    """
    try:
        from src.services.retrieval_service import RetrievalService
        from src.utils.validation import deterministic_validation
        from src.utils.validation_dataset import get_all_test_queries, get_test_case_by_query

        service = RetrievalService()
        results = []

        # Get all test queries
        test_queries = get_all_test_queries()

        for query in test_queries:
            # Get expected results for this query
            test_case = get_test_case_by_query(query)
            if not test_case:
                continue

            expected_results = test_case["expected_results"]

            # Retrieve actual results
            actual_results = await service.retrieve_relevant_chunks(query, top_k=len(expected_results))

            # Perform deterministic validation
            validation_result = deterministic_validation(query, expected_results, actual_results)
            results.append(validation_result)

        # Calculate overall pipeline status
        all_valid = all(result["is_valid"] for result in results)
        avg_confidence = sum(result["confidence"] for result in results) / len(results) if results else 0

        return {
            "pipeline_status": "working" if all_valid else "issues_found",
            "test_results": results,
            "validation_passed": all_valid,
            "average_confidence": avg_confidence,
            "total_tests": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")


@router.post("/pipeline/execute")
async def execute_rag_pipeline(urls: List[str]):
    """
    Execute the complete RAG pipeline: crawl → chunk → embed → store
    """
    try:
        from src.services.pipeline_service import PipelineService

        pipeline_service = PipelineService()
        result = await pipeline_service.execute_pipeline(urls)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.post("/query")
async def query_endpoint(query_request: QueryRequest):
    """
    Process a user query and return relevant content chunks with preserved metadata.

    Request body:
    - query: The search query text
    - top_k: Number of top results to return (default: 5)
    - score_threshold: Minimum similarity score threshold (default: None)

    Returns:
    - query_text: The original query text
    - retrieved_chunks: List of matching content chunks with metadata
    - total_results: Total number of results returned
    - query_timestamp: When the query was processed
    """
    try:
        from src.services.retrieval_service import RetrievalService
        from src.utils.validation import format_query_results_for_output

        service = RetrievalService()
        results = await service.retrieve_relevant_chunks(
            query_request.query,
            top_k=query_request.top_k,
            score_threshold=query_request.score_threshold
        )

        formatted_results = format_query_results_for_output(results, query_request.query)

        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")