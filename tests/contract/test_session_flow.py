"""
Contract test for session-based queries.
Based on the OpenAPI contract in contracts/openapi.yaml.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_session_based_query_contract():
    """
    Test that the /chat/completions endpoint supports session-based queries
    as specified in the OpenAPI contract.
    """
    with TestClient(app) as client:
        # Mock the services to avoid external dependencies
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            # Setup mock instances
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Mock the response
            mock_response = {
                "id": "chatcmpl-123456789",
                "object": "chat.completion",
                "created": 1677825435,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Test response based on session context"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    "total_tokens": 35
                },
                "retrieved_context": []
            }
            
            # Mock the chat service to return a proper response structure
            from src.chat.models import AIResponse, UserQuery, RetrievedContext
            import uuid
            from datetime import datetime
            
            # Create a mock AI response
            mock_ai_response = AIResponse(
                id=str(uuid.uuid4()),
                content="Test response based on session context",
                query_id=str(uuid.uuid4()),
                retrieved_context_ids=[],
                timestamp=datetime.now(),
                confidence_score=0.85,
                source_documents=[]
            )
            
            mock_chat_service_instance.process_query.return_value = mock_ai_response
            
            # Prepare the request payload with session_id
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "What did I ask about earlier?"
                    }
                ],
                "session_id": "sess_abc123_session_test",
                "temperature": 0.7
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


def test_session_validation():
    """
    Test that the session parameter is properly handled.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            from src.chat.models import AIResponse
            import uuid
            from datetime import datetime
            
            mock_ai_response = AIResponse(
                id=str(uuid.uuid4()),
                content="Test response",
                query_id=str(uuid.uuid4()),
                retrieved_context_ids=[],
                timestamp=datetime.now(),
                confidence_score=0.85,
                source_documents=[]
            )
            
            mock_chat_service_instance.process_query.return_value = mock_ai_response
            
            # Test with a valid session ID
            payload_with_session = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Testing with session"
                    }
                ],
                "session_id": "sess_valid_session_123",
                "temperature": 0.5
            }
            
            response = client.post("/api/v1/chat/completions", json=payload_with_session)
            assert response.status_code == 200


def test_missing_messages_with_session():
    """
    Test that the endpoint still validates required fields even with session.
    """
    with TestClient(app) as client:
        payload = {
            "session_id": "sess_test_123"
            # Missing messages field
        }
        
        response = client.post("/api/v1/chat/completions", json=payload)
        
        # Should return 400 for bad request due to missing required field
        assert response.status_code == 422  # FastAPI validation returns 422 for validation errors