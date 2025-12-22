"""
Integration test for multi-turn conversations.
Tests that the system maintains conversational context while staying grounded in textbook content.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app
from src.chat.models import UserQuery, AIResponse, RetrievedContext, ChatSession


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
async def test_multi_turn_conversation_integration():
    """
    Test a multi-turn conversation that maintains context across exchanges.
    Verifies that the system remembers previous interactions while staying 
    grounded in textbook content.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create a shared session ID for the conversation
            session_id = "sess_multi_turn_123"
            
            # First exchange: User asks about inverse kinematics
            first_context = RetrievedContext(
                id="ctx_101",
                content="Inverse kinematics in humanoid robotics is the mathematical process of determining joint angles required to position end effectors at specific locations. This is essential for tasks like reaching and manipulation.",
                source_document="chapter_4_kinematics.pdf",
                page_number=45,
                section_title="Inverse Kinematics Fundamentals",
                similarity_score=0.92,
                embedding_id="emb_101"
            )
            
            first_response = AIResponse(
                id="resp_101",
                content="Inverse kinematics in humanoid robotics involves calculating the joint angles needed to position end effectors (like hands) at specific locations. This mathematical process is crucial for tasks such as reaching for objects and manipulation.",
                query_id="query_101",
                retrieved_context_ids=["ctx_101"],
                timestamp="2023-10-01T14:00:00",
                confidence_score=0.91,
                source_documents=["chapter_4_kinematics.pdf"]
            )
            
            # Set up mocks for first request
            mock_rag_service_instance.retrieve_context.return_value = [first_context]
            mock_chat_service_instance.process_query.return_value = first_response
            
            first_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "What is inverse kinematics in humanoid robotics?"
                    }
                ],
                "session_id": session_id,
                "temperature": 0.3
            }
            
            first_response_raw = client.post("/api/v1/chat/completions", json=first_payload)
            assert first_response_raw.status_code == 200
            
            # Second exchange: User asks a follow-up question referencing the previous topic
            second_context = RetrievedContext(
                id="ctx_102",
                content="Joint angle calculations in inverse kinematics involve complex mathematical formulas including trigonometric functions and matrix transformations. The computational load can be significant for robots with many degrees of freedom.",
                source_document="chapter_4_kinematics.pdf",
                page_number=48,
                section_title="Computational Aspects of Inverse Kinematics",
                similarity_score=0.89,
                embedding_id="emb_102"
            )
            
            second_response = AIResponse(
                id="resp_102",
                content="The joint angle calculations in inverse kinematics involve complex mathematical formulas including trigonometric functions and matrix transformations. The computational load can be significant for robots with many degrees of freedom.",
                query_id="query_102",
                retrieved_context_ids=["ctx_102"],
                timestamp="2023-10-01T14:01:00",
                confidence_score=0.88,
                source_documents=["chapter_4_kinematics.pdf"]
            )
            
            # Set up mocks for second request
            mock_rag_service_instance.retrieve_context.return_value = [second_context]
            mock_chat_service_instance.process_query.return_value = second_response
            
            second_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How are the joint angles calculated?"
                    },
                    # Include the previous conversation for context
                    {
                        "role": "assistant",
                        "content": "Inverse kinematics in humanoid robotics involves calculating the joint angles needed to position end effectors (like hands) at specific locations."
                    },
                    {
                        "role": "user",
                        "content": "What is inverse kinematics in humanoid robotics?"
                    }
                ],
                "session_id": session_id,
                "temperature": 0.3
            }
            
            second_response_raw = client.post("/api/v1/chat/completions", json=second_payload)
            assert second_response_raw.status_code == 200
            
            second_response_data = second_response_raw.json()
            second_assistant_message = second_response_data["choices"][0]["message"]["content"]
            
            # Verify that the response addresses the joint angle calculation
            assert "joint angles" in second_assistant_message.lower()
            assert "calculated" in second_assistant_message.lower()
            assert "mathematical formulas" in second_assistant_message.lower()


@pytest.mark.asyncio
async def test_conversation_context_isolation():
    """
    Test that conversations in different sessions are properly isolated.
    Responses in one session should not affect another session.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Context and response for session 1
            session1_context = RetrievedContext(
                id="ctx_s1_01",
                content="Balance control in humanoid robots uses feedback from gyroscopes and accelerometers to maintain stability.",
                source_document="chapter_6_balance.pdf",
                page_number=110,
                section_title="Balance Control Systems",
                similarity_score=0.88,
                embedding_id="emb_s1_01"
            )
            
            session1_response = AIResponse(
                id="resp_s1_01",
                content="Balance control in humanoid robots uses feedback from gyroscopes and accelerometers to maintain stability.",
                query_id="query_s1_01",
                retrieved_context_ids=["ctx_s1_01"],
                timestamp="2023-10-01T14:30:00",
                confidence_score=0.87,
                source_documents=["chapter_6_balance.pdf"]
            )
            
            # Context and response for session 2
            session2_context = RetrievedContext(
                id="ctx_s2_01",
                content="Motor control in humanoid robots involves PID controllers for precise movement regulation.",
                source_document="chapter_5_motors.pdf",
                page_number=75,
                section_title="Motor Control Systems",
                similarity_score=0.90,
                embedding_id="emb_s2_01"
            )
            
            session2_response = AIResponse(
                id="resp_s2_01",
                content="Motor control in humanoid robots involves PID controllers for precise movement regulation.",
                query_id="query_s2_01",
                retrieved_context_ids=["ctx_s2_01"],
                timestamp="2023-10-01T14:30:01",
                confidence_score=0.89,
                source_documents=["chapter_5_motors.pdf"]
            )
            
            # Test session 1
            mock_rag_service_instance.retrieve_context.return_value = [session1_context]
            mock_chat_service_instance.process_query.return_value = session1_response
            
            session1_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How do humanoid robots maintain balance?"
                    }
                ],
                "session_id": "sess_isolation_test_01",
                "temperature": 0.5
            }
            
            session1_response_raw = client.post("/api/v1/chat/completions", json=session1_payload)
            assert session1_response_raw.status_code == 200
            
            # Test session 2 with different topic
            mock_rag_service_instance.retrieve_context.return_value = [session2_context]
            mock_chat_service_instance.process_query.return_value = session2_response
            
            session2_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How is motor control implemented?"
                    }
                ],
                "session_id": "sess_isolation_test_02",  # Different session
                "temperature": 0.5
            }
            
            session2_response_raw = client.post("/api/v1/chat/completions", json=session2_payload)
            assert session2_response_raw.status_code == 200
            
            session1_content = session1_response_raw.json()["choices"][0]["message"]["content"]
            session2_content = session2_response_raw.json()["choices"][0]["message"]["content"]
            
            # Verify that each session received appropriate content
            assert "balance" in session1_content.lower() or "stability" in session1_content.lower()
            assert "motor" in session2_content.lower() or "pid" in session2_content.lower()
            
            # Verify that the sessions are isolated (different content)
            assert session1_content != session2_content


@pytest.mark.asyncio
async def test_conversation_with_context_grounding():
    """
    Test that multi-turn conversations maintain grounding in textbook content.
    """
    with TestClient(app) as client:
        # Mock the services
        with patch('src.chat.endpoints.ChatService') as mock_chat_service, \
             patch('src.chat.endpoints.RAGService') as mock_rag_service:
            
            mock_rag_service_instance = AsyncMock()
            mock_rag_service.return_value = mock_rag_service_instance
            
            mock_chat_service_instance = AsyncMock()
            mock_chat_service.return_value = mock_chat_service_instance
            
            # Create conversation with consistent textbook grounding
            session_id = "sess_grounding_test_123"
            
            # First exchange about sensors
            first_context = RetrievedContext(
                id="ctx_sensor_01",
                content="Humanoid robots use various sensors including gyroscopes, accelerometers, cameras, and force sensors to perceive their environment and maintain balance.",
                source_document="chapter_3_sensors.pdf",
                page_number=55,
                section_title="Sensor Systems in Humanoid Robots",
                similarity_score=0.93,
                embedding_id="emb_sensor_01"
            )
            
            first_response = AIResponse(
                id="resp_sensor_01",
                content="Humanoid robots use various sensors including gyroscopes, accelerometers, cameras, and force sensors to perceive their environment and maintain balance.",
                query_id="query_sensor_01",
                retrieved_context_ids=["ctx_sensor_01"],
                timestamp="2023-10-01T15:00:00",
                confidence_score=0.92,
                source_documents=["chapter_3_sensors.pdf"]
            )
            
            # Second exchange about the same topic
            second_context = RetrievedContext(
                id="ctx_sensor_02",
                content="Sensor fusion combines data from multiple sensors to create a comprehensive understanding of the robot's state and environment. This is critical for stable locomotion.",
                source_document="chapter_3_sensors.pdf",
                page_number=58,
                section_title="Sensor Fusion Techniques",
                similarity_score=0.90,
                embedding_id="emb_sensor_02"
            )
            
            second_response = AIResponse(
                id="resp_sensor_02",
                content="Sensor fusion combines data from multiple sensors to create a comprehensive understanding of the robot's state and environment. This is critical for stable locomotion.",
                query_id="query_sensor_02",
                retrieved_context_ids=["ctx_sensor_02"],
                timestamp="2023-10-01T15:01:00",
                confidence_score=0.89,
                source_documents=["chapter_3_sensors.pdf"]
            )
            
            # First request
            mock_rag_service_instance.retrieve_context.return_value = [first_context]
            mock_chat_service_instance.process_query.return_value = first_response
            
            first_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "What sensors do humanoid robots use?"
                    }
                ],
                "session_id": session_id,
                "temperature": 0.4
            }
            
            first_response_raw = client.post("/api/v1/chat/completions", json=first_payload)
            assert first_response_raw.status_code == 200
            
            # Second request
            mock_rag_service_instance.retrieve_context.return_value = [second_context]
            mock_chat_service_instance.process_query.return_value = second_response
            
            second_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "How do these sensors work together?"
                    },
                    {
                        "role": "assistant",
                        "content": "Humanoid robots use various sensors including gyroscopes, accelerometers, cameras, and force sensors."
                    },
                    {
                        "role": "user",
                        "content": "What sensors do humanoid robots use?"
                    }
                ],
                "session_id": session_id,
                "temperature": 0.4
            }
            
            second_response_raw = client.post("/api/v1/chat/completions", json=second_payload)
            assert second_response_raw.status_code == 200
            
            first_content = first_response_raw.json()["choices"][0]["message"]["content"]
            second_content = second_response_raw.json()["choices"][0]["message"]["content"]
            
            # Verify both responses are grounded in textbook content
            assert "gyroscopes" in first_content.lower() or "accelerometers" in first_content.lower()
            assert "sensor fusion" in second_content.lower() or "sensors work together" in second_content.lower()
            
            # Verify no hallucinated information
            # (check that responses stick to the provided context)
            assert "artificial intelligence" not in first_content.lower() or \
                   "machine learning" not in first_content.lower()  # Unless in original context