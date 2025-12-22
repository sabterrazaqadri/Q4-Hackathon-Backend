"""
Test validation dataset for deterministic validation of query results
"""
from typing import List, Dict, Any


# Predefined test cases for validation
TEST_CASES = [
    {
        "query": "What is ROS 2?",
        "expected_results": [
            {
                "id": "test-ros-intro-1",
                "content": "ROS 2 (Robot Operating System 2) is a set of libraries and tools for building robotic applications.",
                "source_url": "https://example.com/ros2-basics",
                "section": "Introduction to ROS 2",
                "score": 0.95
            },
            {
                "id": "test-ros-intro-2", 
                "content": "ROS 2 provides a framework for developing robot applications with support for multiple programming languages.",
                "source_url": "https://example.com/ros2-architecture",
                "section": "ROS 2 Architecture",
                "score": 0.92
            }
        ]
    },
    {
        "query": "Explain URDF fundamentals",
        "expected_results": [
            {
                "id": "test-urdf-fund-1",
                "content": "URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS.",
                "source_url": "https://example.com/urdf-intro",
                "section": "URDF Fundamentals",
                "score": 0.90
            }
        ]
    },
    {
        "query": "How to connect AI to robots?",
        "expected_results": [
            {
                "id": "test-ai-robot-1",
                "content": "Connecting AI systems to robots typically involves creating interfaces between the AI application and the robot's control system.",
                "source_url": "https://example.com/ai-robot-connection",
                "section": "AI-Robot Interface",
                "score": 0.93
            }
        ]
    }
]


def get_test_case_by_query(query: str) -> Dict[str, Any]:
    """
    Retrieve a test case by its query string
    """
    for test_case in TEST_CASES:
        if test_case["query"] == query:
            return test_case
    return None


def get_all_test_queries() -> List[str]:
    """
    Get all test query strings
    """
    return [test_case["query"] for test_case in TEST_CASES]