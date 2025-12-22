import re
from urllib.parse import urlparse
from typing import List, Optional


def is_valid_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing fragments and ensuring proper format
    """
    # Remove fragment
    url = url.split('#')[0]
    
    # Ensure proper scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove trailing slashes
    url = url.rstrip('/')
    
    return url


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL
    """
    try:
        result = urlparse(url)
        return result.netloc
    except Exception:
        return None


def get_base_url(url: str) -> str:
    """
    Get base URL (scheme + domain) from full URL
    """
    try:
        result = urlparse(url)
        return f"{result.scheme}://{result.netloc}"
    except Exception:
        return ""