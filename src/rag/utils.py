"""
Helper functions for the RAG module.
"""
import asyncio
import hashlib
from typing import List
from ..chat.models import RetrievedContext


def compute_text_hash(text: str) -> str:
    """
    Compute a hash for text content to use as unique identifiers.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def calculate_context_relevance_score(query: str, context: str) -> float:
    """
    Calculate a basic relevance score between a query and context.
    This is a simple implementation - in production, use semantic similarity.
    """
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    intersection = query_words.intersection(context_words)
    return len(intersection) / len(query_words)


def filter_contexts_by_relevance(
    contexts: List[RetrievedContext], 
    min_score: float = 0.3
) -> List[RetrievedContext]:
    """
    Filter contexts based on their similarity score.
    """
    return [ctx for ctx in contexts if ctx.similarity_score >= min_score]


def deduplicate_contexts(contexts: List[RetrievedContext]) -> List[RetrievedContext]:
    """
    Remove duplicate contexts based on content hash.
    """
    seen_hashes = set()
    unique_contexts = []
    
    for ctx in contexts:
        content_hash = compute_text_hash(ctx.content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_contexts.append(ctx)
    
    return unique_contexts