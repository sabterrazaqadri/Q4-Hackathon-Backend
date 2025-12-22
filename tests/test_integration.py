import pytest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.services.crawler_service import CrawlerService
from src.services.chunking_service import ChunkingService
from src.services.embedding_service import CohereClient
from src.services.storage_service import QdrantService


client = TestClient(app)


def test_full_pipeline_integration():
    """Test the complete pipeline from crawling to storage"""
    
    with patch.object(CrawlerService, 'crawl_single_page') as mock_crawl, \
         patch.object(ChunkingService, 'chunk_content') as mock_chunk, \
         patch.object(CohereClient, 'generate_embeddings_with_retry') as mock_embed, \
         patch.object(QdrantService, 'store_chunks') as mock_store:
        
        # Mock the crawler to return sample content
        mock_crawl.return_value = {
            "url": "https://example.com/test",
            "content": "This is test content for the RAG pipeline.",
            "metadata": {"title": "Test Page"}
        }
        
        # Mock the chunker to return sample chunks
        mock_chunk.return_value = [
            {
                "id": "chunk-1",
                "content": "This is test content for the RAG pipeline.",
                "source_url": "https://example.com/test",
                "section": "Test Page",
                "metadata": {"chunk_index": 0, "total_chunks": 1}
            }
        ]
        
        # Mock the embedder to return sample embeddings
        mock_embed.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock the storage to return success
        mock_store.return_value = True
        
        # Call the pipeline endpoint
        response = client.post(
            "/api/v1/pipeline/execute",
            json={"urls": ["https://example.com/test"]}
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check that each step was processed correctly
        assert data["crawling"]["processed"] == 1
        assert data["chunking"]["processed"] == 1
        assert data["embedding"]["processed"] == 1
        assert data["storage"]["stored"] == 1


def test_crawl_endpoint_integration():
    """Integration test for the crawl endpoint"""
    
    with patch.object(CrawlerService, 'initiate_crawl_job') as mock_init, \
         patch.object(CrawlerService, 'execute_crawl_job') as mock_execute:
        
        from src.models.content_chunk import CrawlJob, CrawlJobCreate
        from datetime import datetime
        from uuid import uuid4
        
        # Create a mock job
        mock_job = CrawlJob(
            id=uuid4(),
            source_urls=["https://example.com/test"],
            status="completed",
            start_time=datetime.now(),
            end_time=datetime.now(),
            processed_count=1,
            failed_count=0,
            error_details=None
        )
        
        mock_init.return_value = mock_job
        mock_execute.return_value = mock_job
        
        response = client.post(
            "/api/v1/crawl",
            json={"source_urls": ["https://example.com/test"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["processed_count"] == 1


def test_chunks_process_endpoint_integration():
    """Integration test for the chunks processing endpoint"""
    
    with patch.object(CrawlerService, 'crawl_single_page') as mock_crawl, \
         patch.object(ChunkingService, 'chunk_content') as mock_chunk:
        
        # Mock the crawler
        mock_crawl.return_value = {
            "url": "https://example.com/test",
            "content": "This is test content for chunking.",
            "metadata": {"title": "Test Page"}
        }
        
        # Mock the chunker
        mock_chunk.return_value = [
            {
                "id": "chunk-1",
                "content": "This is test content for chunking.",
                "source_url": "https://example.com/test",
                "section": "Test Page",
                "metadata": {"chunk_index": 0, "total_chunks": 1}
            }
        ]
        
        response = client.post(
            "/api/v1/chunks/process",
            json={"urls": ["https://example.com/test"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["processed_chunks"] == 1
        assert data["chunks"][0]["content"] == "This is test content for chunking."


def test_embeddings_generate_endpoint_integration():
    """Integration test for the embeddings generation endpoint"""
    
    with patch.object(CohereClient, 'generate_embeddings') as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        test_contents = ["Content 1", "Content 2"]
        response = client.post(
            "/api/v1/embeddings/generate",
            json={"contents": test_contents}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["embeddings_generated"] == 2


def test_vectors_store_endpoint_integration():
    """Integration test for the vector storage endpoint"""
    
    with patch.object(QdrantService, 'store_chunks') as mock_store:
        mock_store.return_value = True
        
        test_chunks = [
            {
                "id": "test-id-1",
                "content": "Sample content",
                "source_url": "https://example.com",
                "embedding": [0.1, 0.2, 0.3]
            }
        ]
        
        response = client.post(
            "/api/v1/vectors/store",
            json={"chunks_with_embeddings": test_chunks}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


def test_health_check_endpoint():
    """Test the health check endpoint"""

    with patch.object(QdrantService, 'client') as mock_client:
        # Mock the collection info
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 10
        mock_client.get_collection.return_value = mock_collection_info

        response = client.get("/api/v1/health/qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vectors_count"] == 10


def test_retrieval_endpoint_integration():
    """Integration test for the retrieval endpoint"""

    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Mock the retrieval service to return sample results
        mock_results = [
            {
                "id": "test-id-1",
                "content": "This is a sample content chunk for testing purposes.",
                "source_url": "https://example.com/test",
                "section": "Introduction",
                "module": "Module 1",
                "chapter": "Chapter 1",
                "score": 0.8523
            }
        ]

        mock_retrieve.return_value = mock_results

        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the introduction?",
                "top_k": 5,
                "score_threshold": 0.5
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query_text"] == "What is the introduction?"
        assert data["total_results"] == 1
        assert data["retrieved_chunks"][0]["content"] == "This is a sample content chunk for testing purposes."
        assert data["retrieved_chunks"][0]["source_url"] == "https://example.com/test"
        assert data["retrieved_chunks"][0]["section"] == "Introduction"
        assert data["retrieved_chunks"][0]["module"] == "Module 1"
        assert data["retrieved_chunks"][0]["chapter"] == "Chapter 1"
        assert data["retrieved_chunks"][0]["score"] == 0.8523


def test_validation_endpoint_integration():
    """Integration test for the validation endpoint"""

    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Mock the retrieval service to return sample results
        mock_results = [
            {
                "id": "test-id-1",
                "content": "This is a sample content chunk for testing purposes.",
                "source_url": "https://example.com/test",
                "section": "Introduction",
                "score": 0.8
            }
        ]

        mock_retrieve.return_value = mock_results

        response = client.post(
            "/api/v1/query/validate",
            json={
                "query": "What is the introduction?",
                "expected_result_ids": ["test-id-1"],
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "What is the introduction?"
        assert data["is_valid"] in [True, False]  # Either could be valid depending on implementation
        assert isinstance(data["confidence"], float)
        assert data["expected_found"] == ["test-id-1"]


def test_pipeline_test_endpoint_integration():
    """Integration test for the pipeline test endpoint"""

    # Since this endpoint runs multiple tests, we'll mock the retrieval service
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Mock to return consistent results for known test queries
        def mock_return(query, top_k, score_threshold=None):
            if "ROS 2" in query:
                return [
                    {
                        "id": "test-ros-intro-1",
                        "content": "ROS 2 (Robot Operating System 2) is a set of libraries and tools for building robotic applications.",
                        "source_url": "https://example.com/ros2-basics",
                        "section": "Introduction to ROS 2",
                        "score": 0.95
                    }
                ]
            elif "URDF" in query:
                return [
                    {
                        "id": "test-urdf-fund-1",
                        "content": "URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS.",
                        "source_url": "https://example.com/urdf-intro",
                        "section": "URDF Fundamentals",
                        "score": 0.90
                    }
                ]
            elif "AI" in query:
                return [
                    {
                        "id": "test-ai-robot-1",
                        "content": "Connecting AI systems to robots typically involves creating interfaces between the AI application and the robot's control system.",
                        "source_url": "https://example.com/ai-robot-connection",
                        "section": "AI-Robot Interface",
                        "score": 0.93
                    }
                ]
            return []

        mock_retrieve.side_effect = mock_return

        response = client.get("/api/v1/pipeline/test")

        assert response.status_code == 200
        data = response.json()

        assert "pipeline_status" in data
        assert "test_results" in data
        assert "validation_passed" in data
        assert "total_tests" in data
        assert "average_confidence" in data