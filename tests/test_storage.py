import pytest
from src.services.storage_service import QdrantService
from src.models.content_chunk import ContentChunk
from unittest.mock import patch, MagicMock
from datetime import datetime
from uuid import uuid4


def test_store_chunk():
    """Test storing a single chunk"""
    # We'll test the structure/logic without connecting to an actual Qdrant instance
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the client methods
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Create a test chunk
        chunk = ContentChunk(
            id=uuid4(),
            content="Test content for storage",
            source_url="https://example.com/test",
            section="Test Section",
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"test": True}
        )
        
        # Store the chunk
        result = service.store_chunk(chunk)
        
        # Verify the result
        assert result is True
        # Verify that upsert was called
        assert mock_client_instance.upsert.called


def test_store_chunks():
    """Test storing multiple chunks"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the client methods
        mock_client_instance = MagicMock()
        mock_client_instance.retrieve.return_value = []  # No existing chunks
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Create test chunks
        chunks = [
            ContentChunk(
                id=uuid4(),
                content="Test content 1",
                source_url="https://example.com/test1",
                section="Test Section 1",
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"test": True}
            ),
            ContentChunk(
                id=uuid4(),
                content="Test content 2",
                source_url="https://example.com/test2",
                section="Test Section 2",
                embedding=[0.4, 0.5, 0.6],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"test": True}
            )
        ]
        
        # Store the chunks
        result = service.store_chunks(chunks)
        
        # Verify the result
        assert result is True
        # Verify that upsert was called once
        assert mock_client_instance.upsert.called_once()


def test_store_chunks_with_duplicates():
    """Test storing chunks when some already exist"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the client methods
        mock_client_instance = MagicMock()
        # Mock retrieve to return one existing chunk
        existing_chunk = ContentChunk(
            id=uuid4(),
            content="Existing content",
            source_url="https://example.com/existing",
            section="Existing Section",
            embedding=[0.1, 0.2, 0.3],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"test": True}
        )
        mock_client_instance.retrieve.return_value = [existing_chunk]
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Create test chunks (one matches the existing, one is new)
        chunks = [
            ContentChunk(
                id=existing_chunk.id,  # Same ID as existing chunk
                content="Test content 1",
                source_url="https://example.com/test1",
                section="Test Section 1",
                embedding=[0.1, 0.2, 0.3],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"test": True}
            ),
            ContentChunk(
                id=uuid4(),  # New ID
                content="Test content 2",
                source_url="https://example.com/test2",
                section="Test Section 2",
                embedding=[0.4, 0.5, 0.6],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"test": True}
            )
        ]
        
        # Store the chunks
        result = service.store_chunks(chunks)
        
        # Verify the result (should succeed even with duplicates)
        assert result is True
        # Verify that upsert was called with only the new chunk
        mock_client_instance.upsert.assert_called_once()
        # Get the actual call to verify only one point was upserted
        call_args = mock_client_instance.upsert.call_args
        points = call_args[1]['points']
        assert len(points) == 1  # Only the new chunk should be stored


def test_search_similar():
    """Test searching for similar content"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the search response
        mock_search_result = [
            MagicMock(),
            MagicMock()
        ]
        mock_search_result[0].id = str(uuid4())
        mock_search_result[0].payload = {
            "content": "Similar content 1",
            "source_url": "https://example.com/similar1",
            "section": "Similar Section 1",
            "score": 0.9
        }
        mock_search_result[1].id = str(uuid4())
        mock_search_result[1].payload = {
            "content": "Similar content 2",
            "source_url": "https://example.com/similar2",
            "section": "Similar Section 2",
            "score": 0.8
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = mock_search_result
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Perform search
        query_embedding = [0.1, 0.2, 0.3]
        results = service.search_similar(query_embedding, limit=10)
        
        # Verify the results
        assert len(results) == 2
        assert results[0]["content"] == "Similar content 1"
        assert results[1]["content"] == "Similar content 2"
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8


def test_get_chunk_by_id():
    """Test retrieving a chunk by ID"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the retrieve response
        chunk_id = str(uuid4())
        mock_record = MagicMock()
        mock_record.id = chunk_id
        mock_record.payload = {
            "content": "Retrieved content",
            "source_url": "https://example.com/retrieved",
            "section": "Retrieved Section",
            "metadata": {"retrieved": True}
        }
        
        mock_client_instance = MagicMock()
        mock_client_instance.retrieve.return_value = [mock_record]
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Retrieve the chunk
        result = service.get_chunk_by_id(chunk_id)
        
        # Verify the result
        assert result is not None
        assert result["id"] == chunk_id
        assert result["content"] == "Retrieved content"
        assert result["source_url"] == "https://example.com/retrieved"
        assert result["section"] == "Retrieved Section"
        assert result["metadata"]["retrieved"] is True


def test_get_chunk_by_id_not_found():
    """Test retrieving a chunk that doesn't exist"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the retrieve response to return empty list
        mock_client_instance = MagicMock()
        mock_client_instance.retrieve.return_value = []
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Try to retrieve a non-existent chunk
        result = service.get_chunk_by_id(str(uuid4()))
        
        # Verify the result
        assert result is None


def test_delete_chunk_by_id():
    """Test deleting a chunk by ID"""
    with patch('qdrant_client.QdrantClient') as mock_client:
        # Mock the client methods
        mock_client_instance = MagicMock()
        mock_client_instance.delete.return_value = True
        mock_client.return_value = mock_client_instance
        
        # Initialize the service
        service = QdrantService()
        
        # Delete a chunk
        chunk_id = str(uuid4())
        result = service.delete_chunk_by_id(chunk_id)
        
        # Verify the result
        assert result is True
        # Verify that delete was called
        assert mock_client_instance.delete.called