"""
Contract test for /chat/completions endpoint.
Based on the OpenAPI contract in contracts/openapi.yaml.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from typing import List

from src.main import app
from src.chat.models import UserQuery, AIResponse, RetrievedContext
from src.chat.services import ChatService
from src.rag.services import RAGService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_chat_completions_contract():
    """
    Test that the /chat/completions endpoint follows the OpenAPI contract.
    """
    with TestClient(app) as client:
        # Mock the services to avoid external dependencies
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            # Setup mock responses
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create mock retrieved context
            mock_context = RetrievedContext(
                id="test_context_id",
                content="Test context content from textbook",
                source_document="test_document.pdf",
                page_number=42,
                section_title="Test Section",
                similarity_score=0.85,
                embedding_id="test_embedding_id"
            )
            
            # Mock the RAG service to return the context
            mock_rag_service_instance.retrieve_context.return_value = [mock_context]
            
            # Create mock AI response
            mock_response = AIResponse(
                id="test_response_id",
                content="Test AI response based on textbook content",
                query_id="test_query_id",
                retrieved_context_ids=["test_context_id"],
                timestamp="2023-10-01T12:00:00",
                confidence_score=0.85,
                source_documents=["test_document.pdf"]
            )
            
            # Mock the chat service to return the response
            mock_chat_service_instance.process_query.return_value = mock_response
            
            # Prepare the request payload matching the OpenAPI spec
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain the concept of inverse kinematics in humanoid robotics"
                    }
                ],
                "selected_text": "Inverse kinematics is the mathematical process of calculating the joint angles...",
                "session_id": "sess_abc123",
                "temperature": 0.5
            }
            
            # Make the request to the endpoint
            response = client.post("/api/v1/chat/completions", json=payload)
            
            # Assertions based on the OpenAPI contract
            assert response.status_code == 200
            
            # Parse the response
            response_data = response.json()
            
            # Verify response structure matches the ChatResponse schema
            assert "id" in response_data
            assert "object" in response_data
            assert "created" in response_data
            assert "model" in response_data
            assert "choices" in response_data
            assert "usage" in response_data
            assert "retrieved_context" in response_data
            
            # Verify the response has the expected structure
            assert response_data["object"] == "chat.completion"
            assert len(response_data["choices"]) > 0
            
            # Verify the first choice has the expected structure
            first_choice = response_data["choices"][0]
            assert "index" in first_choice
            assert "message" in first_choice
            assert "finish_reason" in first_choice
            
            # Verify the message structure
            message = first_choice["message"]
            assert "role" in message
            assert "content" in message
            assert message["role"] == "assistant"
            
            # Verify usage structure
            usage = response_data["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage


def test_chat_completions_missing_messages():
    """
    Test that the endpoint returns 400 when messages are missing.
    """
    with TestClient(app) as client:
        payload = {}  # No messages field
        
        response = client.post("/api/v1/chat/completions", json=payload)
        
        # Should return 400 for bad request
        assert response.status_code == 400


def test_chat_completions_empty_messages():
    """
    Test that the endpoint returns 400 when messages array is empty.
    """
    with TestClient(app) as client:
        payload = {"messages": []}  # Empty messages array
        
        response = client.post("/api/v1/chat/completions", json=payload)
        
        # Should return 400 for bad request
        assert response.status_code == 400