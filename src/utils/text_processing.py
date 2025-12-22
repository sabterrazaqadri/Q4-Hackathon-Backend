import re
from typing import List, Optional


def count_tokens(text: str) -> int:
    """
    Count approximate number of tokens in text.
    This is a simple heuristic that counts words, numbers, and punctuation groups.
    For more accurate tokenization, consider using a dedicated tokenizer library.
    """
    if not text:
        return 0
        
    # This is a simplified tokenization approach
    # Split on whitespace and punctuation, keeping the delimiters
    tokens = re.findall(r'\w+|\S', text)
    return len(tokens)


def split_by_sentences(text: str) -> List[str]:
    """
    Split text into sentences using common sentence endings.
    """
    # Split by sentence endings followed by whitespace
    sentences = re.split(r'[.!?]+\s+', text)
    # Remove empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens while preserving sentence boundaries.
    """
    if not text or max_tokens <= 0:
        return ""
    
    # First try to preserve sentences
    sentences = split_by_sentences(text)
    result = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens <= max_tokens:
            result.append(sentence)
            current_tokens += sentence_tokens
        else:
            # If the sentence is too long, truncate it by words
            if current_tokens == 0:  # If this is the first sentence
                words = sentence.split()
                truncated_sentence = ""
                for word in words:
                    if count_tokens(truncated_sentence + " " + word) <= max_tokens:
                        truncated_sentence += " " + word
                    else:
                        break
                result.append(truncated_sentence.strip())
                break
            else:
                break
    
    return ". ".join(result)


def split_text_by_size(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks of specified token size with optional overlap.
    """
    if not text:
        return []
    
    sentences = split_by_sentences(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding the sentence would exceed chunk size
        if current_tokens + sentence_tokens > chunk_size:
            # If the sentence is too long by itself, split it
            if sentence_tokens > chunk_size:
                # Add the current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Split the long sentence into smaller parts
                words = sentence.split()
                temp_chunk = ""
                temp_tokens = 0
                
                for word in words:
                    word_tokens = count_tokens(word)
                    
                    if temp_tokens + word_tokens > chunk_size:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                        temp_tokens = word_tokens
                    else:
                        if temp_chunk:
                            temp_chunk += " " + word
                        else:
                            temp_chunk = word
                        temp_tokens += word_tokens
                
                # Add the final temp chunk
                if temp_chunk.strip():
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                else:
                    current_chunk = ""
                    current_tokens = 0
            else:
                # Add the current chunk to results
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Start a new chunk with the current sentence
                current_chunk = sentence
                current_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Apply overlap if specified
    if overlap > 0:
        chunks_with_overlap = []
        for i, chunk in enumerate(chunks):
            if i == len(chunks) - 1:  # Last chunk, no overlap needed
                chunks_with_overlap.append(chunk)
            else:
                # Find the last N tokens of the current chunk
                sentences = split_by_sentences(chunk)
                overlap_chunk = ""
                overlap_tokens = 0
                
                # Add sentences in reverse order until we reach overlap token count
                for sentence in reversed(sentences):
                    sentence_tokens = count_tokens(sentence)
                    if overlap_tokens + sentence_tokens <= overlap:
                        if overlap_chunk:
                            overlap_chunk = sentence + " " + overlap_chunk
                        else:
                            overlap_chunk = sentence
                        overlap_tokens += sentence_tokens
                    else:
                        break
                
                # Add overlap to the next chunk
                if overlap_chunk.strip():
                    next_chunk_with_overlap = overlap_chunk + " " + chunks[i + 1]
                    chunks_with_overlap.append(chunk)
                    # Update next chunk with overlap content
                    chunks[i+1] = next_chunk_with_overlap
                else:
                    chunks_with_overlap.append(chunk)
        
        return chunks_with_overlap
    else:
        return chunks