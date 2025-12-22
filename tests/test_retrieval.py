import pytest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.services.retrieval_service import RetrievalService


client = TestClient(app)


def test_query_endpoint():
    """Test the query endpoint functionality"""
    
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
                "score": 0.8753
            },
            {
                "id": "test-id-2", 
                "content": "Another content chunk for validation.",
                "source_url": "https://example.com/test2",
                "section": "Background",
                "module": "Module 1",
                "chapter": "Chapter 2",
                "score": 0.7542
            }
        ]
        
        mock_retrieve.return_value = mock_results
        
        # Send a query request
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What is the introduction to this content?",
                "top_k": 5,
                "score_threshold": 0.5
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check that the query text is preserved
        assert data["query_text"] == "What is the introduction to this content?"
        
        # Check that we got the expected number of results
        assert data["total_results"] == 2
        
        # Check that the first result has the expected properties
        first_chunk = data["retrieved_chunks"][0]
        assert first_chunk["id"] == "test-id-1"
        assert "sample content chunk" in first_chunk["content"]
        assert first_chunk["source_url"] == "https://example.com/test"
        assert first_chunk["section"] == "Introduction"
        assert first_chunk["module"] == "Module 1"
        assert first_chunk["chapter"] == "Chapter 1"
        assert first_chunk["score"] == 0.8753


def test_query_endpoint_with_defaults():
    """Test the query endpoint with default parameters"""
    
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        mock_results = [
            {
                "id": "test-id-1",
                "content": "Sample content",
                "source_url": "https://example.com/test",
                "section": "Introduction", 
                "module": "Module 1",
                "chapter": "Chapter 1",
                "score": 0.9123
            }
        ]
        
        mock_retrieve.return_value = mock_results
        
        # Send a query request with minimal parameters
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query"
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        assert data["query_text"] == "Test query"
        assert data["total_results"] == 1


def test_query_endpoint_error_handling():
    """Test the query endpoint error handling"""
    
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Make the retrieval service raise an exception
        mock_retrieve.side_effect = Exception("Test error")
        
        # Send a query request
        response = client.post(
            "/api/v1/query",
            json={
                "query": "Test query"
            }
        )
        
        # Verify the error response
        assert response.status_code == 500
        data = response.json()
        assert "Query processing failed" in data["detail"]