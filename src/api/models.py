from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class QueryRequest(BaseModel):
    """
    Represents questions submitted to the RAG agent, containing the question text and optional selected text context
    """
    question: str = Field(
        ...,
        description="The main question text from the user",
        min_length=1,
        max_length=2000
    )
    selected_text: Optional[str] = Field(
        None,
        description="Additional context selected by the user",
        max_length=5000
    )
    user_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional contextual information from the user"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Request metadata (timestamp, source, etc.)"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "What are the key components of a humanoid robot?",
                "selected_text": "In chapter 3, the author discusses sensors and actuators..."
            }
        }


class Source(BaseModel):
    """
    Represents source information for a piece of retrieved context
    """
    document_id: str = Field(
        ...,
        description="Unique identifier for the source document"
    )
    page_number: Optional[int] = Field(
        None,
        description="Page where the information appears"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section where the information appears"
    )
    excerpt: Optional[str] = Field(
        None,
        description="Relevant excerpt from the source"
    )


class UsageStats(BaseModel):
    """
    Token usage statistics
    """
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the prompt"
    )
    completion_tokens: int = Field(
        ...,
        description="Number of tokens in the completion"
    )
    total_tokens: int = Field(
        ...,
        description="Total number of tokens used"
    )


class AgentResponse(BaseModel):
    """
    The final output from the agent, containing the answer and references to the sources used
    """
    answer: str = Field(
        ...,
        description="The agent's response to the user's question"
    )
    sources: List[Source] = Field(
        ...,
        description="List of sources referenced in the answer"
    )
    confidence: float = Field(
        ...,
        description="Agent's confidence level in the response (0-1)",
        ge=0,
        le=1
    )
    usage_stats: Optional[UsageStats] = Field(
        None,
        description="Token usage statistics"
    )

    class Config:
        schema_extra = {
            "example": {
                "answer": "Humanoid robots maintain balance using a combination of gyroscopes, accelerometers, and control algorithms...",
                "sources": [
                    {
                        "document_id": "doc001",
                        "page_number": 45,
                        "section_title": "Balance Control Systems",
                        "excerpt": "Balance is maintained through feedback control using gyroscope and accelerometer data..."
                    }
                ],
                "confidence": 0.95,
                "usage_stats": {
                    "prompt_tokens": 120,
                    "completion_tokens": 85,
                    "total_tokens": 205
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response format
    """
    error: str = Field(
        ...,
        description="Error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Question field is required and cannot be empty"
            }
        }