"""
Validation-related data models using Pydantic.
Based on the data-model.md specification and validation requirements.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ValidationRequest(BaseModel):
    """
    Request model for validating a query against textbook content.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="The query to validate against textbook content")
    selected_text: Optional[str] = Field(None, min_length=1, max_length=5000, description="Optional selected text to focus the validation")


class ValidationResponse(BaseModel):
    """
    Response model for query validation results.
    """
    is_valid: bool = Field(..., description="Whether the query can be answered with available content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level in the validation result")
    relevant_sources: List[str] = Field(..., description="List of relevant source documents")