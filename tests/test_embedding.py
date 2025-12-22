import pytest
from src.services.embedding_service import CohereClient
from unittest.mock import patch, MagicMock


def test_validate_embeddings():
    """Test embeddings validation functionality"""
    cohere_client = CohereClient()
    
    # Valid embeddings
    valid_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
    
    assert cohere_client.validate_embeddings(valid_embeddings) is True
    assert cohere_client.validate_embeddings(valid_embeddings, 3) is True
    
    # Invalid: different dimensions
    invalid_embeddings = [
        [0.1, 0.2],
        [0.4, 0.5, 0.6]
    ]
    
    assert cohere_client.validate_embeddings(invalid_embeddings) is False
    
    # Invalid: with non-numeric values
    invalid_embeddings2 = [
        [0.1, "invalid", 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    assert cohere_client.validate_embeddings(invalid_embeddings2) is False
    
    # Invalid: empty embeddings
    assert cohere_client.validate_embeddings([]) is False
    
    # Valid with specific dimension
    assert cohere_client.validate_embeddings(valid_embeddings, 3) is True
    assert cohere_client.validate_embeddings(valid_embeddings, 2) is False


@patch('cohere.Client')
def test_generate_single_embedding_structure(mock_cohere_client):
    """Test the structure of the embed method without calling the actual API"""
    # Mock the Cohere client response
    mock_response = MagicMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    
    mock_client_instance = MagicMock()
    mock_client_instance.embed.return_value = mock_response
    mock_cohere_client.return_value = mock_client_instance
    
    # Initialize the client with a mock API key
    with patch('src.config.settings.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v2.0"
        
        cohere_client = CohereClient()
        
        # Test single embedding generation
        result = cohere_client.generate_single_embedding("Test text")
        
        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_client_instance.embed.assert_called_once()


@patch('cohere.Client')
def test_generate_multiple_embeddings_structure(mock_cohere_client):
    """Test the structure of generating multiple embeddings"""
    # Mock the Cohere client response
    mock_response = MagicMock()
    mock_response.embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ]
    
    mock_client_instance = MagicMock()
    mock_client_instance.embed.return_value = mock_response
    mock_cohere_client.return_value = mock_client_instance
    
    # Initialize the client with a mock API key
    with patch('src.config.settings.settings') as mock_settings:
        mock_settings.cohere_api_key = "test-key"
        mock_settings.cohere_model = "embed-multilingual-v2.0"
        
        cohere_client = CohereClient()
        
        # Test multiple embedding generation
        result = cohere_client.generate_embeddings(["Test text 1", "Test text 2"])
        
        assert result == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        mock_client_instance.embed.assert_called_once()


def test_generate_embeddings_empty_input():
    """Test handling of empty input"""
    cohere_client = CohereClient()
    
    # We expect this to fail gracefully since we're not mocking the API
    with pytest.raises(Exception):
        cohere_client.generate_embeddings([])


def test_embedding_error_handling():
    """Test that errors are properly handled and wrapped in EmbeddingError"""
    cohere_client = CohereClient()
    
    # Mock the client.embed method to raise an exception
    with patch.object(cohere_client.client, 'embed') as mock_embed:
        mock_embed.side_effect = Exception("API Error")
        
        with pytest.raises(Exception) as exc_info:
            cohere_client.generate_embeddings(["Test text"])
        
        # Check that the error is wrapped in EmbeddingError
        assert "Failed to generate embeddings" in str(exc_info.value)