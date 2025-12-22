"""
Test script to verify the system structure and component integration
without requiring actual API keys.
"""
import asyncio
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to the path to import modules properly
sys.path.insert(0, 'C:/Users/sabte/OneDrive/Desktop/All About IT/GIAIC Q4/GIAIC Q4 Hackathons/Hackathon 1/Step-1-physical-ai-humanoid-textbook')

from src.config.settings import settings


async def test_system_structure():
    print("Testing system structure and components...")

    # Test 1: Check if all services can be instantiated
    print("\n1. Testing service instantiation...")
    try:
        from src.services.embedding_service import CohereClient
        embedding_service = CohereClient()
        print("+ CohereClient (embedding service) instantiated")
    except Exception as e:
        print(f"- CohereClient (embedding service) failed: {e}")

    # Storage service requires a real connection to Qdrant, so we'll only test with mock
    print("+ Storage service (QdrantService) needs real connection - skip direct instantiation")

    # Retrieval service also requires a real connection to Qdrant, so we'll only test with mock
    print("+ Retrieval service needs real connection - skip direct instantiation")

    # Test 2: Check if RAG agent can be instantiated with all dependencies mocked
    print("\n2. Testing RAG Agent instantiation...")
    try:
        # Mock the dependencies to avoid connection errors
        with patch('src.services.agent_tools.RetrievalService'):
            from src.agents.rag_agent import RAGAgent
            rag_agent = RAGAgent()
            print("+ RAGAgent instantiated with mocked dependencies")
    except Exception as e:
        print(f"- RAGAgent failed: {e}")

    # Test 3: Test embedding service with mock (since we don't have API keys)
    print("\n3. Testing embedding service with mock...")
    try:
        # We'll just test the class structure since the actual class needs API keys
        with patch('cohere.Client') as MockCohere:
            mock_client_instance = MagicMock()
            mock_client_instance.embed.return_value = MagicMock()
            mock_client_instance.embed.return_value.embeddings = [[0.1, 0.2, 0.3]]
            MockCohere.return_value = mock_client_instance

            from src.services.embedding_service import CohereClient
            embedding_service = CohereClient()
            print("+ Embedding service structure works")
    except Exception as e:
        print(f"- Embedding service structure failed: {e}")

    # Test 4: Test storage service with mock
    print("\n4. Testing storage service with mock...")
    try:
        with patch('src.services.storage_service.QdrantClient') as MockQdrantClient:
            mock_client = MagicMock()
            MockQdrantClient.return_value = mock_client

            import src.services.storage_service as storage_module
            # Temporarily replace the QdrantClient in the module
            original_qdrant_client = storage_module.QdrantClient
            storage_module.QdrantClient = MockQdrantClient

            from src.services.storage_service import QdrantService
            storage_service = QdrantService()
            print("+ Storage service (QdrantService) structure works")

            # Restore original
            storage_module.QdrantClient = original_qdrant_client
    except Exception as e:
        print(f"- Storage service structure failed: {e}")

    # Test 5: Test retrieval service with mocks
    print("\n5. Testing retrieval service with mocks...")
    try:
        import src.services.retrieval_service as retrieval_module
        import src.services.storage_service as storage_module
        import src.services.embedding_service as embedding_module

        # Mock the dependencies
        with patch.object(storage_module, 'QdrantClient') as MockQdrantClient, \
             patch.object(embedding_module, 'CohereClient') as MockCohereClient:

            mock_qdrant = MagicMock()
            mock_embedding = MagicMock()

            MockQdrantClient.return_value = mock_qdrant
            MockCohereClient.return_value = mock_embedding

            # Temporarily replace in the modules to avoid import-time issues
            original_qdrant = storage_module.QdrantClient
            original_cohere = embedding_module.CohereClient
            storage_module.QdrantClient = MockQdrantClient
            embedding_module.CohereClient = MockCohereClient

            from src.services.retrieval_service import RetrievalService
            retrieval_service = RetrievalService()
            print("+ Retrieval service structure works")

            # Restore originals
            storage_module.QdrantClient = original_qdrant
            embedding_module.CohereClient = original_cohere
    except Exception as e:
        print(f"- Retrieval service structure failed: {e}")

    # Test 6: Test the API endpoint works structurally
    print("\n6. Testing API structure...")
    try:
        from src.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        print(f"+ API health check works: status {response.status_code}")
    except Exception as e:
        print(f"- API structure failed: {e}")

    # Test 7: Test that API models are properly defined
    print("\n7. Testing API models...")
    try:
        from src.api.models import QueryRequest, AgentResponse, ErrorResponse
        query = QueryRequest(question="test question")
        print(f"+ QueryRequest model works: {query.question}")

        agent_resp = AgentResponse(
            answer="test answer",
            sources=[],
            confidence=0.8
        )
        print(f"+ AgentResponse model works: {agent_resp.confidence}")

        error_resp = ErrorResponse(error="test_error", message="test message")
        print(f"+ ErrorResponse model works: {error_resp.error}")
    except Exception as e:
        print(f"- API models failed: {e}")

    # Test 8: Test the routes functionality
    print("\n8. Testing API routes structure...")
    try:
        from src.api.routes.rag import router
        print("+ RAG routes structure works")
    except Exception as e:
        print(f"- RAG routes structure failed: {e}")

    print("\nSystem structure test completed!")


if __name__ == "__main__":
    asyncio.run(test_system_structure())