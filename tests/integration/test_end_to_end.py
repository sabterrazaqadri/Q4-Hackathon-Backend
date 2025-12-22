"""
Integration tests for the RAG API endpoints.
These tests verify the complete flow of the API endpoints with real components.
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json
from src.api.main import app
from src.agents.rag_agent import RAGAgent


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_query_endpoint_success():
    """Test successful query to the RAG endpoint."""
    client = TestClient(app)
    
    # Mock the RAG agent to avoid actual API calls
    mock_response = {
        "answer": "This is a test answer based on the textbook content.",
        "sources": [
            {
                "document_id": "test_doc_001",
                "page_number": 42,
                "section_title": "Test Section",
                "excerpt": "This is a test excerpt from the textbook."
            }
        ],
        "confidence": 0.95,
        "usage_stats": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "What are the key components of a humanoid robot?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a test answer based on the textbook content."
        assert len(data["sources"]) == 1
        assert data["confidence"] == 0.95
        assert "usage_stats" in data


@pytest.mark.asyncio
async def test_query_endpoint_with_selected_text():
    """Test query with selected text context."""
    client = TestClient(app)
    
    # Mock the RAG agent to avoid actual API calls
    mock_response = {
        "answer": "With the provided context, the answer is more specific.",
        "sources": [
            {
                "document_id": "test_doc_002",
                "page_number": 38,
                "section_title": "Contextual Section",
                "excerpt": "When considering the selected text context..."
            }
        ],
        "confidence": 0.89,
        "usage_stats": {
            "prompt_tokens": 150,
            "completion_tokens": 75,
            "total_tokens": 225
        }
    }
    
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "How does the system handle contextual queries?",
                "selected_text": "The previous section mentioned specific contextual processing techniques."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "With the provided context, the answer is more specific."
        assert data["confidence"] == 0.89


@pytest.mark.asyncio
async def test_query_endpoint_validation_error():
    """Test query endpoint with invalid input."""
    client = TestClient(app)
    
    # Test with empty question
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": ""  # Empty question should cause validation error
        }
    )
    
    assert response.status_code == 400  # Validation error should return 400
    
    # Test with question that's too long
    long_question = "A" * 2001  # 1 character over the limit
    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": long_question
        }
    )
    
    assert response.status_code == 400  # Validation error should return 400


@pytest.mark.asyncio
async def test_query_endpoint_timeout():
    """Test query endpoint with simulated timeout."""
    client = TestClient(app)
    
    # Mock the RAG agent to simulate a timeout
    async def slow_process(*args, **kwargs):
        await asyncio.sleep(35)  # Longer than our 30 second timeout
        return {}  # This shouldn't be reached
    
    with patch.object(RAGAgent, 'process_query', side_effect=slow_process):
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "What happens with a slow query?"
            }
        )
        
        assert response.status_code == 408  # Should return timeout error


@pytest.mark.asyncio
async def test_validation_endpoint():
    """Test the query validation endpoint."""
    client = TestClient(app)
    
    # Valid query should pass validation
    response = client.post(
        "/api/v1/rag/query/validate",
        json={
            "question": "Is this a valid question?",
            "selected_text": "Some context"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    
    # Invalid query should fail validation
    response = client.post(
        "/api/v1/rag/query/validate",
        json={
            "question": "",  # Empty question
            "selected_text": "Some context"
        }
    )
    
    assert response.status_code == 200  # Validation endpoint returns 200 but with valid=False
    data = response.json()
    assert data["valid"] is False


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    
    response = client.get("/api/v1/rag/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "RAG Agent API"


@pytest.mark.asyncio
async def test_external_service_failure():
    """Test how the API handles external service failures."""
    client = TestClient(app)
    
    # Mock an external service failure (like OpenAI API)
    async def fail_process(*args, **kwargs):
        raise Exception("OpenAI API Error: Rate limit exceeded")
    
    with patch.object(RAGAgent, 'process_query', side_effect=fail_process):
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "What happens when the external service fails?"
            }
        )
        
        assert response.status_code == 200  # Returns structured error response
        data = response.json()
        assert data["error"] == "EXTERNAL_SERVICE_ERROR"
        assert "unavailable" in data["message"].lower()


@pytest.mark.asyncio
async def test_performance_monitoring():
    """Test that performance metrics are correctly logged."""
    # This is more of a conceptual test as performance monitoring is logged
    # rather than returned to the client. The actual test would involve
    # checking log outputs, which is complex in a unit test environment.
    # For now, we'll just ensure the endpoint works without errors.
    client = TestClient(app)
    
    # Mock the RAG agent to provide a quick response
    mock_response = {
        "answer": "Performance test response.",
        "sources": [],
        "confidence": 0.9,
        "usage_stats": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
    }
    
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "Testing performance monitoring?"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Performance test response."