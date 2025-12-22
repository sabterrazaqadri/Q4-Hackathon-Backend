import pytest
from src.services.chunking_service import ChunkingService
from src.utils.text_processing import split_text_by_size


def test_chunk_content_basic():
    """Test basic content chunking functionality"""
    chunking_service = ChunkingService()
    
    content = "This is a sample content for testing. " * 50  # Create a longer text
    url = "https://example.com/test"
    
    chunks = chunking_service.chunk_content(content, url)
    
    assert len(chunks) > 0
    assert all('id' in chunk for chunk in chunks)
    assert all('content' in chunk for chunk in chunks)
    assert all('source_url' in chunk for chunk in chunks)
    assert all(chunk['source_url'] == url for chunk in chunks)
    assert all(len(chunk['content']) > 0 for chunk in chunks)


def test_chunk_content_with_section():
    """Test content chunking with section information"""
    chunking_service = ChunkingService()
    
    content = "This is a sample content for testing. " * 20
    url = "https://example.com/test"
    section = "Introduction"
    
    chunks = chunking_service.chunk_content(content, url, section=section)
    
    assert len(chunks) > 0
    assert all(chunk['section'] == section for chunk in chunks)


def test_chunk_content_custom_size():
    """Test content chunking with custom size"""
    chunking_service = ChunkingService()
    
    content = "This is a sample content for testing. " * 50  # Create a longer text
    url = "https://example.com/test"
    
    # Use a smaller chunk size to ensure we get multiple chunks
    chunks = chunking_service.chunk_content(content, url, chunk_size=50)
    
    assert len(chunks) > 1  # Should have multiple chunks with smaller size
    assert all(len(chunk['content']) <= 100 for chunk in chunks)  # Content should be reasonably sized


def test_chunk_content_custom_overlap():
    """Test content chunking with custom overlap"""
    chunking_service = ChunkingService()
    
    content = "This is a sample content for testing. " * 50  # Create a longer text
    url = "https://example.com/test"
    
    # Use a specific overlap to test functionality
    chunks = chunking_service.chunk_content(content, url, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    # Check that chunks have overlap information in metadata
    assert all('metadata' in chunk for chunk in chunks)


def test_empty_content():
    """Test chunking with empty content"""
    chunking_service = ChunkingService()
    
    chunks = chunking_service.chunk_content("", "https://example.com/test")
    
    assert len(chunks) == 0


def test_short_content():
    """Test chunking with very short content"""
    chunking_service = ChunkingService()
    
    content = "Short"
    url = "https://example.com/test"
    
    chunks = chunking_service.chunk_content(content, url)
    
    assert len(chunks) == 1
    assert chunks[0]['content'] == content


def test_validate_chunk():
    """Test chunk validation functionality"""
    chunking_service = ChunkingService()
    
    # Valid chunk
    valid_chunk = {
        'id': 'test-id',
        'content': 'This is a valid chunk with sufficient content length.',
        'source_url': 'https://example.com/test'
    }
    
    assert chunking_service.validate_chunk(valid_chunk) is True
    
    # Invalid chunk - missing required field
    invalid_chunk = {
        'id': 'test-id',
        'content': 'This is a valid chunk with sufficient content length.'
        # Missing source_url
    }
    
    assert chunking_service.validate_chunk(invalid_chunk) is False
    
    # Invalid chunk - too short content
    short_chunk = {
        'id': 'test-id',
        'content': 'Hi',
        'source_url': 'https://example.com/test'
    }
    
    assert chunking_service.validate_chunk(short_chunk) is False


def test_rechunk_content():
    """Test rechunking functionality"""
    chunking_service = ChunkingService()
    
    # Create some initial chunks
    content = "This is a sample content for testing rechunking functionality. " * 30
    url = "https://example.com/test"
    
    initial_chunks = chunking_service.chunk_content(content, url, chunk_size=100)
    
    # Rechunk with different parameters
    new_chunks = chunking_service.rechunk_content(initial_chunks, 75, 10)
    
    assert len(new_chunks) > 0
    assert all('id' in chunk for chunk in new_chunks)
    assert all('content' in chunk for chunk in new_chunks)


def test_chunk_metadata():
    """Test that chunks contain proper metadata"""
    chunking_service = ChunkingService()
    
    content = "This is a sample content for testing. " * 20
    url = "https://example.com/test"
    
    chunks = chunking_service.chunk_content(content, url)
    
    for i, chunk in enumerate(chunks):
        assert 'metadata' in chunk
        metadata = chunk['metadata']
        assert 'chunk_index' in metadata
        assert 'total_chunks' in metadata
        assert 'original_content_length' in metadata
        assert 'chunk_length' in metadata
        assert metadata['chunk_index'] == i
        assert metadata['total_chunks'] == len(chunks)
        assert metadata['original_content_length'] == len(content)
        assert metadata['chunk_length'] == len(chunk['content'])