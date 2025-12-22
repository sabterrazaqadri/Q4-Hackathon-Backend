"""
Contract test for /chat/validate endpoint.
Based on the OpenAPI contract in contracts/openapi.yaml.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_validate_contract():
    """
    Test that the /chat/validate endpoint follows the OpenAPI contract.
    """
    with TestClient(app) as client:
        # Mock the RAG service to avoid external dependencies
        with patch('src.chat.endpoints.RAGService') as mock_rag_service:
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            # Mock validation response
            mock_rag_service_instance.validate_query.return_value = (True, 0.85, ["document1.pdf", "document2.pdf"])
            
            # Prepare the request payload matching the OpenAPI spec
            payload = {
                "query": "How do humanoid robots maintain balance?",
                "selected_text": "Balance control in humanoid robots involves complex algorithms"
            }
            
            # Make the request to the endpoint
            response = client.post("/api/v1/chat/validate", json=payload)
            
            # Assertions based on the OpenAPI contract
            assert response.status_code == 200
            
            # Parse the response
            response_data = response.json()
            
            # Verify response structure matches the ValidationResponse schema
            assert "is_valid" in response_data
            assert "confidence" in response_data
            assert "relevant_sources" in response_data
            
            # Verify the types and values
            assert isinstance(response_data["is_valid"], bool)
            assert isinstance(response_data["confidence"], float)
            assert isinstance(response_data["relevant_sources"], list)
            
            # Verify the specific values from our mock
            assert response_data["is_valid"] is True
            assert response_data["confidence"] == 0.85
            assert "document1.pdf" in response_data["relevant_sources"]
            assert "document2.pdf" in response_data["relevant_sources"]


def test_chat_validate_missing_query():
    """
    Test that the endpoint returns 400 when query is missing.
    """
    with TestClient(app) as client:
        payload = {}  # No query field
        
        response = client.post("/api/v1/chat/validate", json=payload)
        
        # Should return 400 for bad request
        assert response.status_code == 400


def test_chat_validate_empty_query():
    """
    Test that the endpoint returns 400 when query is empty.
    """
    with TestClient(app) as client:
        payload = {"query": ""}  # Empty query
        
        response = client.post("/api/v1/chat/validate", json=payload)
        
        # Should return 400 for bad request
        assert response.status_code == 400