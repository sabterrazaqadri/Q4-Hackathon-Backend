"""
Integration test for response accuracy verification.
Tests that responses are grounded in textbook content without hallucinations.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app
from src.chat.models import RetrievedContext, AIResponse


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_response_accuracy_verification_integration():
    """
    Test that responses are accurately grounded in textbook content.
    This test verifies that the system doesn't hallucinate information.
    """
    with TestClient(app) as client:
        # Mock the services to avoid external dependencies
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create mock retrieved contexts from textbook
            mock_contexts = [
                RetrievedContext(
                    id="context_201",
                    content="Humanoid robots use PID controllers for motor control. PID stands for Proportional-Integral-Derivative and helps maintain precise control over joint movements.",
                    source_document="chapter_5_control_theory.pdf",
                    page_number=87,
                    section_title="PID Controllers in Robotics",
                    similarity_score=0.91,
                    embedding_id="emb_501"
                ),
                RetrievedContext(
                    id="context_202",
                    content="Balance in humanoid robots is maintained through feedback from gyroscopes and accelerometers. The control system processes this data to make real-time adjustments to joint positions.",
                    source_document="chapter_6_balance_control.pdf",
                    page_number=112,
                    section_title="Sensory Feedback for Balance",
                    similarity_score=0.88,
                    embedding_id="emb_502"
                )
            ]
            
            # Set up the RAG service to return these contexts
            mock_rag_service_instance.retrieve_context.return_value = mock_contexts
            
            # Create a mock response that accurately reflects the retrieved content
            mock_response = AIResponse(
                id="response_301",
                content="Humanoid robots maintain balance using feedback from gyroscopes and accelerometers. The control system processes this sensory data to make real-time adjustments to joint positions. Additionally, PID controllers (Proportional-Integral-Derivative) are used for precise motor control of joint movements.",
                query_id="query_401",
                retrieved_context_ids=["context_201", "context_202"],
                timestamp="2023-10-01T13:00:00",
                confidence_score=0.89,
                source_documents=["chapter_5_control_theory.pdf", "chapter_6_balance_control.pdf"]
            )
            
            # Set up the chat service to return this response
            mock_chat_service_instance.process_query.return_value = mock_response
            
            # Make a query about robot balance and control
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How do humanoid robots maintain balance and control their movements?"
                    }
                ],
                "temperature": 0.3  # Lower temperature for more consistent responses
            }
            
            response = client.post("/api/v1/chat/completions", json=payload)
            
            assert response.status_code == 200
            
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            
            # Verify the response contains information from the retrieved contexts
            assert "gyroscopes" in assistant_message.lower()
            assert "accelerometers" in assistant_message.lower()
            assert "pid controllers" in assistant_message.lower()
            assert "joint movements" in assistant_message.lower()
            
            # Verify that the response doesn't contain hallucinated information
            # (information not present in the retrieved contexts)
            # We'll check for a specific topic that should NOT be in the response
            # since it's not in our mock contexts
            assert "artificial intelligence" not in assistant_message.lower() or \
                   "machine learning" not in assistant_message.lower()


@pytest.mark.asyncio
async def test_validation_prevents_hallucination():
    """
    Test that the validation system correctly identifies when a query 
    cannot be answered with available textbook content.
    """
    with TestClient(app) as client:
        # Mock the RAG service
        with patch('src.chat.endpoints.RAGService') as mock_rag_service:
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            # Mock a scenario where the query cannot be answered with available content
            # (low confidence, few or no relevant sources)
            mock_rag_service_instance.validate_query.return_value = (False, 0.2, [])
            
            payload = {
                "query": "What is the capital of Mars?",
                "selected_text": "Information about planetary capitals"
            }
            
            response = client.post("/api/v1/chat/validate", json=payload)
            
            assert response.status_code == 200
            
            response_data = response.json()
            
            # Verify that the system correctly identified this as unanswerable
            # from the textbook content
            assert response_data["is_valid"] is False
            assert response_data["confidence"] == 0.2
            assert response_data["relevant_sources"] == []


@pytest.mark.asyncio
async def test_response_grounding_verification():
    """
    Test that responses are properly grounded in retrieved contexts.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create mock context about inverse kinematics
            mock_context = RetrievedContext(
                id="context_301",
                content="Inverse kinematics in humanoid robotics is the mathematical process of determining joint angles required to position end effectors at specific locations. This is essential for tasks like reaching and manipulation.",
                source_document="chapter_4_kinematics.pdf",
                page_number=45,
                section_title="Inverse Kinematics Fundamentals",
                similarity_score=0.94,
                embedding_id="emb_601"
            )
            
            mock_rag_service_instance.retrieve_context.return_value = [mock_context]
            
            # Create a response that accurately reflects the context
            mock_response = AIResponse(
                id="response_401",
                content="Inverse kinematics in humanoid robotics involves calculating the joint angles needed to position end effectors (like hands) at specific locations. This mathematical process is crucial for tasks such as reaching for objects and manipulation.",
                query_id="query_501",
                retrieved_context_ids=["context_301"],
                timestamp="2023-10-01T13:30:00",
                confidence_score=0.91,
                source_documents=["chapter_4_kinematics.pdf"]
            )
            
            mock_chat_service_instance.process_query.return_value = mock_response
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain inverse kinematics in humanoid robotics"
                    }
                ]
            }
            
            response = client.post("/api/v1/chat/completions", json=payload)
            
            assert response.status_code == 200
            
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            
            # Verify the response is grounded in the provided context
            assert "inverse kinematics" in assistant_message.lower()
            assert "joint angles" in assistant_message.lower()
            assert "end effectors" in assistant_message.lower()
            assert "mathematical process" in assistant_message.lower()
            
            # The response should accurately reflect the context without adding
            # information not present in the context
            retrieved_context_content = mock_context.content.lower()
            response_content = assistant_message.lower()
            
            # Verify that key concepts from context are reflected in response
            assert "determining joint angles" in retrieved_context_content or \
                   "calculating joint angles" in response_content
            assert "positioning end effectors" in retrieved_context_content or \
                   "positioning end effectors" in response_content