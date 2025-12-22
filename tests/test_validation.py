import pytest
from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch, MagicMock
from src.services.retrieval_service import RetrievalService
from src.utils.validation import validate_query_result_relevance, deterministic_validation


client = TestClient(app)


def test_validation_endpoint():
    """Test the validation endpoint functionality"""
    
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Mock the retrieval service to return sample results
        mock_results = [
            {
                "id": "test-id-1",
                "content": "This is a sample content chunk for testing purposes.",
                "source_url": "https://example.com/test",
                "section": "Introduction",
                "score": 0.85
            },
            {
                "id": "test-id-2", 
                "content": "Another content chunk for validation.",
                "source_url": "https://example.com/test2",
                "section": "Background",
                "score": 0.75
            }
        ]
        
        mock_retrieve.return_value = mock_results
        
        # Send a validation request
        response = client.post(
            "/api/v1/query/validate",
            json={
                "query": "What is the introduction?",
                "expected_result_ids": ["test-id-1", "test-id-2"],
                "top_k": 5
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check validation results
        assert "is_valid" in data
        assert "confidence" in data
        assert data["query"] == "What is the introduction?"
        assert data["expected_found"] == ["test-id-1", "test-id-2"]


def test_validation_endpoint_with_no_expected_results():
    """Test validation when no expected result IDs provided"""
    
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        mock_results = [
            {
                "id": "test-id-1",
                "content": "Sample content",
                "source_url": "https://example.com/test",
                "section": "Introduction",
                "score": 0.9
            }
        ]
        
        mock_retrieve.return_value = mock_results
        
        # Send a validation request with no expected IDs
        response = client.post(
            "/api/v1/query/validate",
            json={
                "query": "Test query",
                "expected_result_ids": [],
                "top_k": 3
            }
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check that expected_recall is None when no expected IDs provided
        assert data["expected_recall"] is None


def test_deterministic_validation_utility():
    """Test the deterministic validation utility function"""
    
    # Define expected and actual results 
    expected_results = [
        {
            "id": "test-id-1",
            "content": "This is the expected content for testing",
            "score": 0.9
        },
        {
            "id": "test-id-2",
            "content": "Another expected content",
            "score": 0.8
        }
    ]
    
    actual_results = [
        {
            "id": "test-id-1",
            "content": "This is the expected content for testing",  # Exact match
            "score": 0.89  # Close enough to expected
        },
        {
            "id": "test-id-2", 
            "content": "Another expected content",  # Exact match
            "score": 0.79  # Close enough to expected
        }
    ]
    
    # Perform deterministic validation
    result = deterministic_validation("Test query", expected_results, actual_results)
    
    # Verify the validation result
    assert result["is_valid"] is True
    assert result["confidence"] == 1.0  # Perfect match
    assert result["details"]["matches"] == 2
    assert result["details"]["mismatches"] == 0


def test_deterministic_validation_with_mismatch():
    """Test deterministic validation with mismatched results"""
    
    expected_results = [
        {
            "id": "test-id-1",
            "content": "Expected content",
            "score": 0.9
        },
        {
            "id": "test-id-2",
            "content": "Another expected content",
            "score": 0.8
        }
    ]
    
    actual_results = [
        {
            "id": "test-id-3",  # Different ID
            "content": "Different content",
            "score": 0.7
        },
        {
            "id": "test-id-4",  # Different ID
            "content": "Another different content",
            "score": 0.6
        }
    ]
    
    # Perform deterministic validation
    result = deterministic_validation("Test query", expected_results, actual_results)
    
    # Verify the validation result
    assert result["is_valid"] is False
    assert result["confidence"] == 0.0  # No matches
    assert result["details"]["matches"] == 0
    assert result["details"]["mismatches"] == 2


def test_query_result_relevance_validation():
    """Test the query result relevance validation utility"""
    
    # Define results to validate
    results = [
        {
            "content": "ROS 2 is a robot operating system",
            "score": 0.8,
            "source_url": "https://example.com/ros2"
        },
        {
            "content": "Another related result about robotics",
            "score": 0.6,
            "source_url": "https://example.com/robotics"
        }
    ]
    
    # Validate relevance
    result = validate_query_result_relevance("What is ROS 2?", results)
    
    # Check that it's considered valid (with some confidence)
    assert "is_valid" in result
    assert "confidence" in result
    assert result["details"]["total_results"] == 2


def test_pipeline_test_endpoint():
    """Test the pipeline validation endpoint"""
    
    with patch.object(RetrievalService, 'retrieve_relevant_chunks') as mock_retrieve:
        # Mock to return consistent results for known test queries
        def mock_return(query, top_k, score_threshold=None):
            if "ROS 2" in query:
                return [
                    {
                        "id": "test-ros-intro-1",
                        "content": "ROS 2 (Robot Operating System 2) is a set of libraries and tools for building robotic applications.",
                        "source_url": "https://example.com/ros2-basics",
                        "section": "Introduction to ROS 2",
                        "score": 0.95
                    }
                ]
            elif "URDF" in query:
                return [
                    {
                        "id": "test-urdf-fund-1",
                        "content": "URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS.",
                        "source_url": "https://example.com/urdf-intro",
                        "section": "URDF Fundamentals",
                        "score": 0.90
                    }
                ]
            elif "AI" in query:
                return [
                    {
                        "id": "test-ai-robot-1",
                        "content": "Connecting AI systems to robots typically involves creating interfaces between the AI application and the robot's control system.",
                        "source_url": "https://example.com/ai-robot-connection",
                        "section": "AI-Robot Interface",
                        "score": 0.93
                    }
                ]
            return []
        
        mock_retrieve.side_effect = mock_return
        
        # Call the pipeline test endpoint
        response = client.get("/api/v1/pipeline/test")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        
        # Check the overall status
        assert "pipeline_status" in data
        assert "test_results" in data
        assert "validation_passed" in data
        assert "total_tests" in data