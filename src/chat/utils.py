"""
Helper functions for the chat module.
"""
import re
from typing import List


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks or other security issues.
    """
    # Remove potentially harmful characters or patterns
    sanitized = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length while preserving words.
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space != -1:
        truncated = truncated[:last_space]
    
    return truncated + "..."


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from a text for search purposes.
    This is a simple implementation - in production, use a more sophisticated NLP approach.
    """
    # Remove punctuation and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Split into words and remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    words = [word for word in clean_text.split() if word not in stop_words and len(word) > 2]
    
    # Return unique keywords, limited to max_keywords
    unique_keywords = list(dict.fromkeys(words))  # Preserves order
    return unique_keywords[:max_keywords]