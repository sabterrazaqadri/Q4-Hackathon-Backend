"""
Content processing utilities
"""
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import re


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text content from HTML
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean it up
    text = soup.get_text()
    
    # Break into lines and remove leading/trailing space on each line
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text


def extract_title_from_html(html_content: str) -> Optional[str]:
    """
    Extract title from HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text().strip()
    
    # Fallback: look for h1 tags
    h1_tag = soup.find('h1')
    if h1_tag:
        return h1_tag.get_text().strip()
    
    return None


def extract_main_content_from_html(html_content: str) -> str:
    """
    Extract main content from HTML, focusing on article or main content areas
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for main content areas
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'main|content|article'))
    
    if main_content:
        # Remove navigation, headers, footers, and other non-content elements
        for element in main_content.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract text from the main content area
        text = main_content.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    else:
        # Fallback to general text extraction
        return extract_text_from_html(html_content)


def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\n\r]', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_content_metadata(html_content: str, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    metadata = {
        'url': url,
        'title': '',
        'description': '',
        'keywords': [],
        'author': '',
        'published_date': ''
    }
    
    # Extract title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()
    else:
        h1_tag = soup.find('h1')
        if h1_tag:
            metadata['title'] = h1_tag.get_text().strip()
    
    # Extract description
    desc_tag = soup.find('meta', attrs={'name': 'description'})
    if desc_tag:
        metadata['description'] = desc_tag.get('content', '')
    else:
        # Fallback: first paragraph
        p_tag = soup.find('p')
        if p_tag:
            metadata['description'] = p_tag.get_text()[:160]  # First 160 chars
    
    # Extract keywords
    keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
    if keywords_tag:
        keywords_str = keywords_tag.get('content', '')
        metadata['keywords'] = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
    
    # Extract author
    author_tag = soup.find('meta', attrs={'name': 'author'})
    if not author_tag:
        author_tag = soup.find('meta', attrs={'property': 'author'})
    if author_tag:
        metadata['author'] = author_tag.get('content', '')
    
    # Extract published date
    date_tag = soup.find('meta', attrs={'name': 'date'})
    if not date_tag:
        date_tag = soup.find('meta', attrs={'property': 'article:published_time'})
    if date_tag:
        metadata['published_date'] = date_tag.get('content', '')
    
    return metadata