from typing import List, Dict, Any, Optional
from src.models.content_chunk import ContentChunk
import difflib


def validate_query_result_relevance(query: str, results: List[Dict[str, Any]], expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate the relevance of query results to the original query
    """
    if not results:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "details": "No results returned for the query"
        }
    
    # Calculate relevance based on keyword matching
    query_lower = query.lower()
    total_score = 0.0
    
    # Add keywords from query to expected keywords
    if expected_keywords is None:
        expected_keywords = []
    
    # Split query into words and add to expected keywords if not already present
    query_words = [word for word in query_lower.split() if len(word) > 3]
    for word in query_words:
        if word not in expected_keywords:
            expected_keywords.append(word)
    
    for result in results:
        content_lower = result.get("content", "").lower()
        
        # Calculate keyword match score
        keyword_matches = 0
        for keyword in expected_keywords:
            if keyword.lower() in content_lower:
                keyword_matches += 1
        
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Use the result's similarity score as a base
        similarity_score = result.get("score", 0.0)
        
        # Combine scores (weight more heavily toward similarity if available)
        combined_score = (similarity_score * 0.7) + (keyword_score * 0.3)
        total_score += combined_score
    
    avg_score = total_score / len(results) if results else 0.0
    
    return {
        "is_valid": avg_score > 0.3,  # Threshold for validity
        "confidence": avg_score,
        "details": {
            "average_score": avg_score,
            "total_results": len(results),
            "expected_keywords": expected_keywords
        }
    }


def validate_result_determinism(query: str, result1: List[Dict[str, Any]], result2: List[Dict[str, Any]]) -> bool:
    """
    Validate that the same query produces consistent results
    """
    if len(result1) != len(result2):
        return False
    
    # Compare each result in order
    for r1, r2 in zip(result1, result2):
        if r1.get("id") != r2.get("id"):
            return False
        if abs(r1.get("score", 0) - r2.get("score", 0)) > 0.01:  # Small tolerance for score differences
            return False
    
    return True


def validate_metadata_preservation(result: Dict[str, Any], required_fields: List[str] = None) -> bool:
    """
    Validate that required metadata fields are preserved in results
    """
    if required_fields is None:
        required_fields = ["source_url", "section", "module", "chapter"]
    
    for field in required_fields:
        if field not in result or result[field] is None:
            return False
    
    return True


def validate_embedding_compatibility(query_embedding: List[float], stored_embeddings: List[List[float]], tolerance: float = 0.01) -> bool:
    """
    Validate that the query embedding is compatible with stored embeddings (same dimension)
    """
    if not stored_embeddings:
        return False

    expected_dimension = len(stored_embeddings[0])
    return len(query_embedding) == expected_dimension


def format_query_results_for_output(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Format query results for clean, structured output
    """
    formatted_results = []

    for result in results:
        formatted_result = {
            "id": result.get("id"),
            "content": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),  # Truncate long content
            "source_url": result.get("source_url", ""),
            "section": result.get("section", ""),
            "module": result.get("module", ""),
            "chapter": result.get("chapter", ""),
            "score": round(result.get("score", 0), 4),  # Round score for cleaner output
        }
        formatted_results.append(formatted_result)

    return {
        "query_text": query,
        "retrieved_chunks": formatted_results,
        "total_results": len(formatted_results),
        "query_timestamp": __import__('datetime').datetime.now().isoformat()
    }


def deterministic_validation(query: str, expected_results: List[Dict[str, Any]], actual_results: List[Dict[str, Any]], tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Perform deterministic validation of query results against expected results
    """
    validation_details = {
        "query": query,
        "is_valid": True,
        "confidence": 1.0,
        "details": {
            "expected_count": len(expected_results),
            "actual_count": len(actual_results),
            "matches": 0,
            "mismatches": 0,
            "errors": []
        }
    }

    # If counts don't match, results are not valid
    if len(expected_results) != len(actual_results):
        validation_details["is_valid"] = False
        validation_details["confidence"] = 0.0
        validation_details["details"]["errors"].append(f"Expected {len(expected_results)} results but got {len(actual_results)}")
        return validation_details

    matches = 0
    total_comparisons = len(expected_results)

    for i, expected in enumerate(expected_results):
        if i >= len(actual_results):
            break

        actual = actual_results[i]

        # Compare IDs if available
        if "id" in expected and "id" in actual:
            if expected["id"] != actual["id"]:
                validation_details["is_valid"] = False
                validation_details["details"]["mismatches"] += 1
                continue

        # Compare content similarity if available
        if "content" in expected and "content" in actual:
            expected_content = expected["content"][:100].lower().strip()  # Compare first 100 chars
            actual_content = actual["content"][:100].lower().strip()

            if expected_content != actual_content:
                # Check for similarity
                similarity = difflib.SequenceMatcher(None, expected_content, actual_content).ratio()
                if similarity < (1 - tolerance):
                    validation_details["details"]["mismatches"] += 1
                    continue

        # Compare scores if available
        if "score" in expected and "score" in actual:
            score_diff = abs(expected["score"] - actual["score"])
            if score_diff > tolerance:
                validation_details["details"]["mismatches"] += 1
                continue

        # If we reach here, it's a match
        validation_details["details"]["matches"] += 1

    # Calculate confidence based on match ratio
    match_ratio = validation_details["details"]["matches"] / total_comparisons if total_comparisons > 0 else 1.0
    validation_details["confidence"] = match_ratio

    # Update validity based on match ratio
    if match_ratio < 0.8:  # Less than 80% match is considered invalid
        validation_details["is_valid"] = False

    return validation_details