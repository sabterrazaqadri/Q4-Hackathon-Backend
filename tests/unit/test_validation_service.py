"""
Unit tests for the validation service.
"""
import pytest
from unittest.mock import AsyncMock

from src.services.validation_service import ValidationService
from src.rag.services import RAGService
from src.chat.models import RetrievedContext, AIResponse


@pytest.mark.asyncio
async def test_validate_query_response_accurate():
    """Test validating a response that is accurately grounded in contexts."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Create test contexts that support the response
    contexts = [
        RetrievedContext(
            id="ctx1",
            content="Humanoid robots use PID controllers for motor control.",
            source_document="doc1.pdf",
            page_number=10,
            section_title="Control Systems",
            similarity_score=0.9,
            embedding_id="emb1"
        )
    ]
    
    # Create a response that accurately reflects the context
    response = "Humanoid robots use PID controllers for motor control."
    
    # Validate the response
    is_accurate, confidence, sources = await validation_service.validate_query_response(
        "How do humanoid robots control motors?",
        response,
        contexts
    )
    
    # Verify the response is considered accurate
    assert is_accurate is True
    assert confidence > 0.7  # Should be high since response matches context
    assert "doc1.pdf" in sources


@pytest.mark.asyncio
async def test_validate_query_response_inaccurate():
    """Test validating a response that is not grounded in contexts."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Create test contexts
    contexts = [
        RetrievedContext(
            id="ctx1",
            content="Humanoid robots use PID controllers for motor control.",
            source_document="doc1.pdf",
            page_number=10,
            section_title="Control Systems",
            similarity_score=0.9,
            embedding_id="emb1"
        )
    ]
    
    # Create a response that is not supported by the context
    response = "Humanoid robots use neural networks for balance control."
    
    # Validate the response
    is_accurate, confidence, sources = await validation_service.validate_query_response(
        "How do humanoid robots maintain balance?",
        response,
        contexts
    )
    
    # Verify the response is considered inaccurate
    assert is_accurate is False
    assert confidence < 0.7  # Should be low since response doesn't match context
    assert "doc1.pdf" in sources  # Source should still be listed


@pytest.mark.asyncio
async def test_validate_query_response_empty_contexts():
    """Test validating a response when no contexts are provided."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Validate with empty contexts
    is_accurate, confidence, sources = await validation_service.validate_query_response(
        "How do humanoid robots work?",
        "They use advanced AI.",
        []  # Empty contexts
    )
    
    # Verify the response is considered inaccurate with low confidence
    assert is_accurate is False
    assert confidence == 0.0
    assert sources == []


def test_calculate_response_accuracy():
    """Test the response accuracy calculation."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Create test contexts
    contexts = [
        RetrievedContext(
            id="ctx1",
            content="Humanoid robots use PID controllers for motor control.",
            source_document="doc1.pdf",
            page_number=10,
            section_title="Control Systems",
            similarity_score=0.9,
            embedding_id="emb1"
        )
    ]
    
    # Calculate accuracy for a matching response
    accuracy = validation_service._calculate_response_accuracy(
        "Humanoid robots use PID controllers for motor control.",
        contexts
    )
    
    # Verify accuracy is high for matching content
    assert accuracy > 0.5  # Should be high since content matches
    
    # Calculate accuracy for a non-matching response
    accuracy = validation_service._calculate_response_accuracy(
        "This response has nothing to do with the context.",
        contexts
    )
    
    # Verify accuracy is low for non-matching content
    assert accuracy < 0.5  # Should be low since content doesn't match


def test_get_supporting_sources():
    """Test identifying supporting sources for a response."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Create test contexts
    contexts = [
        RetrievedContext(
            id="ctx1",
            content="Humanoid robots use PID controllers for motor control.",
            source_document="doc1.pdf",
            page_number=10,
            section_title="Control Systems",
            similarity_score=0.9,
            embedding_id="emb1"
        ),
        RetrievedContext(
            id="ctx2",
            content="Balance is maintained using gyroscopes and accelerometers.",
            source_document="doc2.pdf",
            page_number=15,
            section_title="Balance Systems",
            similarity_score=0.8,
            embedding_id="emb2"
        )
    ]
    
    # Get supporting sources for a response that matches the first context
    sources = validation_service._get_supporting_sources(
        "Humanoid robots use PID controllers for motor control.",
        contexts
    )
    
    # Verify the first source is identified as supporting
    assert "doc1.pdf" in sources
    assert "doc2.pdf" not in sources or len(sources) == 1  # Only first doc should match significantly
    
    # Get supporting sources for a response that matches both contexts
    sources = validation_service._get_supporting_sources(
        "Humanoid robots use PID controllers and gyroscopes.",
        contexts
    )
    
    # Verify both sources are identified as supporting
    assert "doc1.pdf" in sources
    assert "doc2.pdf" in sources


@pytest.mark.asyncio
async def test_validate_query_content():
    """Test validating a query against textbook content."""
    # Create a mock RAG service
    mock_rag_service = AsyncMock(spec=RAGService)
    
    # Mock the validate_query method to return specific values
    mock_rag_service.validate_query.return_value = (True, 0.85, ["doc1.pdf", "doc2.pdf"])
    
    # Create validation service
    validation_service = ValidationService(mock_rag_service)
    
    # Validate a query
    result = await validation_service.validate_query_content(
        "How do humanoid robots maintain balance?",
        "Balance control information"
    )
    
    # Verify the result matches the mocked values
    assert result.is_valid is True
    assert result.confidence == 0.85
    assert "doc1.pdf" in result.relevant_sources
    assert "doc2.pdf" in result.relevant_sources
    
    # Verify the RAG service was called with the correct parameters
    mock_rag_service.validate_query.assert_called_once_with(
        "How do humanoid robots maintain balance?",
        "Balance control information"
    )