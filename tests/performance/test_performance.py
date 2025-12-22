"""
Performance tests for the RAG API endpoints.
These tests verify that the API meets performance requirements.
"""
import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import statistics
from src.api.main import app
from src.agents.rag_agent import RAGAgent


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_response_time_under_threshold():
    """Test that API responses are under the 500ms threshold."""
    client = TestClient(app)
    
    # Mock response to avoid dependency on external services
    mock_response = {
        "answer": "This is a test answer for performance evaluation.",
        "sources": [
            {
                "document_id": "perf_doc_001",
                "page_number": 10,
                "section_title": "Performance Section",
                "excerpt": "Performance is important for user experience."
            }
        ],
        "confidence": 0.85,
        "usage_stats": {
            "prompt_tokens": 80,
            "completion_tokens": 40,
            "total_tokens": 120
        }
    }
    
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        # Measure response time over multiple requests
        response_times = []
        for i in range(10):  # Test with 10 requests
            start_time = time.time()
            response = client.post(
                "/api/v1/rag/query",
                json={
                    "question": f"Performance test question {i}?"
                }
            )
            end_time = time.time()
            response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            assert response.status_code == 200
        
        # Calculate p95 (95th percentile)
        sorted_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[p95_index]
        
        # Verify that 95% of requests are under 500ms
        assert p95_time < 500, f"95th percentile response time ({p95_time:.2f}ms) exceeds 500ms threshold"
        
        print(f"Performance test results:")
        print(f"  Average response time: {statistics.mean(response_times):.2f}ms")
        print(f"  Median response time: {statistics.median(response_times):.2f}ms")
        print(f"  95th percentile: {p95_time:.2f}ms")
        print(f"  Min response time: {min(response_times):.2f}ms")
        print(f"  Max response time: {max(response_times):.2f}ms")


@pytest.mark.asyncio
async def test_concurrent_request_handling():
    """Test how the API handles concurrent requests."""
    client = TestClient(app)
    
    # Mock response to avoid dependency on external services
    mock_response = {
        "answer": "Concurrent request test response.",
        "sources": [],
        "confidence": 0.9,
        "usage_stats": {
            "prompt_tokens": 60,
            "completion_tokens": 30,
            "total_tokens": 90
        }
    }
    
    async def make_request():
        with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = mock_response
            response = client.post(
                "/api/v1/rag/query",
                json={
                    "question": "Concurrent request test?"
                }
            )
            return response.status_code, time.time()
    
    # Make 20 concurrent requests
    start_time = time.time()
    tasks = [make_request() for _ in range(20)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Verify all requests were successful
    status_codes = [result[0] for result in results]
    assert all(code == 200 for code in status_codes), f"Not all requests were successful: {status_codes}"
    
    # Verify overall time is reasonable for 20 concurrent requests
    total_time = end_time - start_time
    print(f"Completed 20 concurrent requests in {total_time:.2f} seconds")
    
    # Each request should ideally finish in a reasonable time
    # Even with concurrency, total time should be significantly less than 20 requests * 500ms
    assert total_time < 10, f"Concurrent requests took too long: {total_time:.2f}s"


@pytest.mark.asyncio
async def test_large_payload_handling():
    """Test API performance with large payloads."""
    client = TestClient(app)
    
    # Create a large question and selected text
    large_question = "A " + "very " * 1000 + "long question?"  # ~4KB question
    large_selected_text = "B " + "large " * 2000 + "context."  # ~10KB context
    
    # Mock response
    mock_response = {
        "answer": "Response to large payload test.",
        "sources": [],
        "confidence": 0.75,
        "usage_stats": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200
        }
    }
    
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        start_time = time.time()
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": large_question,
                "selected_text": large_selected_text
            }
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 1000, f"Large payload response time ({response_time:.2f}ms) exceeded 1000ms"


@pytest.mark.asyncio
async def test_response_time_consistency():
    """Test that response times are consistent across different types of queries."""
    client = TestClient(app)
    
    mock_response = {
        "answer": "Consistency test response.",
        "sources": [],
        "confidence": 0.8,
        "usage_stats": {
            "prompt_tokens": 70,
            "completion_tokens": 35,
            "total_tokens": 105
        }
    }
    
    queries = [
        "Short question?",
        "A slightly longer question about the specific implementation details of humanoid robotics?",
        "What are the key concepts in chapters 1 through 5 and how do they relate to each other in the context of physical AI?",
        "An extremely long question that tests the boundaries of the system's ability to process complex queries efficiently and effectively within the required time constraints while maintaining accuracy and relevance to the Physical AI & Humanoid Robotics textbook content?"
    ]
    
    response_times = []
    with patch.object(RAGAgent, 'process_query', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = mock_response
        
        for i, query in enumerate(queries):
            start_time = time.time()
            response = client.post(
                "/api/v1/rag/query",
                json={
                    "question": query
                }
            )
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate coefficient of variation to check consistency
    mean_time = statistics.mean(response_times)
    std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
    cv = (std_dev / mean_time) * 100 if mean_time > 0 else 0
    
    print(f"Response time consistency test:")
    print(f"  Mean response time: {mean_time:.2f}ms")
    print(f"  Std deviation: {std_dev:.2f}ms")
    print(f"  Coefficient of variation: {cv:.2f}%")
    
    # A CV under 50% is generally considered acceptable for consistency
    assert cv < 50, f"Response times are too inconsistent (CV: {cv:.2f}%)"


def test_validation_endpoint_performance():
    """Test performance of validation endpoint."""
    client = TestClient(app)
    
    start_time = time.time()
    response = client.post(
        "/api/v1/rag/query/validate",
        json={
            "question": "Is this a valid question for performance testing?",
            "selected_text": "Performance context"
        }
    )
    end_time = time.time()
    
    response_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    assert response.status_code == 200
    assert response_time < 100, f"Validation endpoint took too long: {response_time:.2f}ms"
    
    data = response.json()
    assert data["valid"] is True