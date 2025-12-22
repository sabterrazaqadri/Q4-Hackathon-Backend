"""
Integration test for user query to textbook response journey.
Tests the full flow from user input to textbook-grounded response.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app
from src.chat.models import UserQuery, AIResponse, RetrievedContext


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_full_chat_flow_integration():
    """
    Test the complete flow from user query to textbook-grounded response.
    This integration test verifies that all components work together correctly.
    """
    with TestClient(app) as client:
        # Mock the services to avoid external dependencies during testing
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            # Setup mock instances
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create mock retrieved context that would come from textbook
            mock_context = RetrievedContext(
                id="context_123",
                content="Humanoid robots use inverse kinematics to calculate joint angles needed to position end effectors (like hands) at specific locations. This is essential for tasks like reaching, walking, and manipulation.",
                source_document="chapter_4_kinematics.pdf",
                page_number=45,
                section_title="Inverse Kinematics in Humanoid Robotics",
                similarity_score=0.92,
                embedding_id="emb_456"
            )
            
            # Set up the RAG service mock to return the context
            mock_rag_service_instance.retrieve_context.return_value = [mock_context]
            
            # Create mock AI response
            mock_response = AIResponse(
                id="response_789",
                content="Humanoid robots use inverse kinematics to calculate the joint angles required to position their end effectors (such as hands) at specific locations. This mathematical process is essential for performing tasks like reaching for objects, walking, and manipulation. The system determines how each joint in the robot's body needs to move to achieve the desired position of a limb or the entire body.",
                query_id="query_101",
                retrieved_context_ids=["context_123"],
                timestamp="2023-10-01T12:00:00",
                confidence_score=0.88,
                source_documents=["chapter_4_kinematics.pdf"]
            )
            
            # Set up the chat service mock to return the response
            mock_chat_service_instance.process_query.return_value = mock_response
            
            # Simulate a user query about inverse kinematics
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How do humanoid robots use inverse kinematics?"
                    }
                ],
                "temperature": 0.7
            }
            
            # Make the request to the API
            response = client.post("/api/v1/chat/completions", json=payload)
            
            # Verify the response
            assert response.status_code == 200
            
            # Parse the response
            response_data = response.json()
            
            # Verify the response structure
            assert "choices" in response_data
            assert len(response_data["choices"]) > 0
            
            # Verify the response content is from the textbook
            assistant_message = response_data["choices"][0]["message"]["content"]
            assert "inverse kinematics" in assistant_message.lower()
            assert "humanoid robots" in assistant_message.lower()
            assert "joint angles" in assistant_message.lower()
            
            # Verify that the retrieved context was used
            retrieved_context_list = response_data.get("retrieved_context", [])
            assert len(retrieved_context_list) > 0
            assert any("inverse kinematics" in ctx.get("content", "").lower() for ctx in retrieved_context_list)


@pytest.mark.asyncio
async def test_query_with_selected_text_integration():
    """
    Test the flow when a user provides selected text along with their query.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create mock context based on selected text
            mock_context = RetrievedContext(
                id="context_124",
                content="The control system of a humanoid robot integrates sensory feedback from multiple sources including vision, proprioception, and tactile sensors to maintain balance and execute complex movements.",
                source_document="chapter_7_control_systems.pdf",
                page_number=127,
                section_title="Sensory Integration in Control Systems",
                similarity_score=0.89,
                embedding_id="emb_457"
            )
            
            mock_rag_service_instance.retrieve_context.return_value = [mock_context]
            
            # Create mock response
            mock_response = AIResponse(
                id="response_790",
                content="The control system of a humanoid robot integrates sensory feedback from multiple sources including vision, proprioception, and tactile sensors. This integration is crucial for maintaining balance and executing complex movements. The system processes information from these diverse sensory inputs to coordinate the robot's actions effectively.",
                query_id="query_102",
                retrieved_context_ids=["context_124"],
                timestamp="2023-10-01T12:05:00",
                confidence_score=0.85,
                source_documents=["chapter_7_control_systems.pdf"]
            )
            
            mock_chat_service_instance.process_query.return_value = mock_response
            
            # Query with selected text
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain the control system"
                    }
                ],
                "selected_text": "The control system of a humanoid robot integrates sensory feedback from multiple sources",
                "temperature": 0.6
            }
            
            response = client.post("/api/v1/chat/completions", json=payload)
            
            assert response.status_code == 200
            
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            
            # Verify the response addresses the selected text
            assert "control system" in assistant_message.lower()
            assert "sensory feedback" in assistant_message.lower()


@pytest.mark.asyncio
async def test_validation_endpoint_integration():
    """
    Test the validation endpoint to ensure it properly validates queries.
    """
    with TestClient(app) as client:
        # Mock the RAG service
        with patch('src.chat.endpoints.RAGService') as mock_rag_service:
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            # Mock validation response
            mock_rag_service_instance.validate_query.return_value = (True, 0.85, ["chapter_3_motors.pdf"])
            
            payload = {
                "query": "How do humanoid robots maintain balance?",
                "selected_text": "Balance control in humanoid robots involves complex algorithms"
            }
            
            response = client.post("/api/v1/chat/validate", json=payload)
            
            assert response.status_code == 200
            
            response_data = response.json()
            assert response_data["is_valid"] is True
            assert response_data["confidence"] == 0.85
            assert "chapter_3_motors.pdf" in response_data["relevant_sources"]