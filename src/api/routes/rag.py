from fastapi import APIRouter, HTTPException, Request
import time
from typing import Dict, Any
from src.agents.rag_agent import RAGAgent
from src.api.models import QueryRequest, AgentResponse, ErrorResponse, BaseModel
from pydantic import Field
from src.config.settings import settings
import logging
import asyncio
from src.services.validation_helper import validate_query_format

router = APIRouter(prefix="/rag", tags=["RAG Agent"])


class QueryValidationResponse(BaseModel):
    """
    Response model for query validation endpoint
    """
    valid: bool = Field(..., description="Whether the query is valid")
    message: str = Field(..., description="Validation result message")


@router.post(
    "/query",
    response_model=AgentResponse,
    responses={
        200: {"description": "Successful response with answer from the RAG agent"},
        400: {"model": ErrorResponse, "description": "Bad request - invalid input parameters"},
        422: {"model": ErrorResponse, "description": "Unprocessable entity - validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Submit a question to the RAG agent",
    description="Allows users to ask questions and get answers based strictly on the Physical AI & Humanoid Robotics textbook content. The agent will answer strictly from retrieved context, supporting both normal questions and selected-text questions."
)
async def query_rag_agent(request: QueryRequest):
    """
    Submit a question to the RAG agent and get an answer based strictly on the textbook content

    - **question**: The main question text from the user (required, 1-2000 characters)
    - **selected_text**: Additional context selected by the user (optional, up to 5000 characters)
    - **user_context**: Additional contextual information from the user (optional)
    - **metadata**: Request metadata (optional)

    The response will include:
    - **answer**: The agent's response to the user's question
    - **sources**: List of sources referenced in the answer
    - **confidence**: Agent's confidence level in the response (0-1)
    - **usage_stats**: Token usage statistics (optional)
    """
    start_time = time.time()

    try:
        # Use the validation helper for comprehensive validation
        validation_result = validate_query_format(request.dict())
        if not validation_result['is_valid']:
            logging.warning(f"Query validation failed: {validation_result['message']}")
            raise HTTPException(
                status_code=400,
                detail=validation_result['message']
            )

        # Initialize the RAG agent
        rag_agent = RAGAgent()

        # Process the query with timeout handling
        try:
            result = await asyncio.wait_for(
                rag_agent.process_query(
                    question=request.question,
                    selected_text=request.selected_text,
                    user_context=request.user_context
                ),
                timeout=30  # Default timeout of 30 seconds
            )
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logging.error(f"Query timed out after {processing_time:.2f}s")
            raise HTTPException(
                status_code=408,  # Request Timeout
                detail="Query processing timed out. Please try again with a simpler question."
            )

        # Create and return the response
        response = AgentResponse(**result)

        # Validate the response
        if not rag_agent.response_formatter.validate_agent_response(result):
            raise HTTPException(
                status_code=500,
                detail="Error validating agent response format"
            )

        # Performance monitoring
        processing_time = time.time() - start_time
        logging.info(f"Query processed successfully in {processing_time:.2f}s - Question: '{request.question[:50]}{'...' if len(request.question) > 50 else ''}'")

        # Log if processing time exceeds threshold (500ms as per requirements)
        if processing_time > 0.5:
            logging.warning(f"Slow query detected: {processing_time:.2f}s - Question: '{request.question}'")

        return response

    except HTTPException as http_ex:
        # Performance monitoring for errors
        processing_time = time.time() - start_time
        logging.error(f"Query failed after {processing_time:.2f}s: {http_ex.detail}")

        # Return structured error response
        if http_ex.status_code == 400:
            return ErrorResponse(
                error="VALIDATION_ERROR",
                message=http_ex.detail
            )
        elif http_ex.status_code == 408:
            return ErrorResponse(
                error="TIMEOUT_ERROR",
                message=http_ex.detail
            )
        else:
            # For other HTTP errors, return the same error
            raise http_ex
    except Exception as e:
        # Performance monitoring for unhandled errors
        processing_time = time.time() - start_time
        error_msg = str(e)

        # Check if this is an external service error (OpenAI, Qdrant, etc.)
        if any(service in error_msg.lower() for service in ["openai", "api", "connection", "timeout", "rate limit"]):
            logging.error(f"External service failure after {processing_time:.2f}s: {error_msg}", exc_info=True)
            return ErrorResponse(
                error="EXTERNAL_SERVICE_ERROR",
                message="External service is temporarily unavailable. Please try again later."
            )
        else:
            logging.error(f"Unhandled error in RAG query endpoint after {processing_time:.2f}s: {error_msg}", exc_info=True)

            # For internal errors, return structured error response
            return ErrorResponse(
                error="INTERNAL_ERROR",
                message="Internal server error occurred while processing the query"
            )


@router.post(
    "/query/validate",
    response_model=QueryValidationResponse,
    responses={
        200: {"description": "Validation result"},
        422: {"model": ErrorResponse, "description": "Unprocessable entity - validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Validate a query without processing it",
    description="Allows external systems to validate a query before fully processing it, useful for client-side validation."
)
async def validate_query(request: QueryRequest):
    """
    Validate a query without processing it, useful for external systems to check query format
    """
    try:
        # Use the validation helper for comprehensive validation
        validation_result = validate_query_format(request.dict())

        # Map the validation result to the appropriate response format
        if validation_result['is_valid']:
            return QueryValidationResponse(
                valid=True,
                message="Query is valid"
            )
        else:
            # Log validation failures
            logging.warning(f"Query validation failed: {validation_result['message']}")
            return QueryValidationResponse(
                valid=False,
                message=validation_result['message']
            )

    except Exception as e:
        logging.error(f"Error in query validation endpoint: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while validating the query"
        )


@router.get(
    "/health",
    summary="Health check endpoint",
    description="Check if the RAG service is running and healthy"
)
async def health_check():
    """
    Health check endpoint to verify the service is running
    """
    return {"status": "healthy", "service": "RAG Agent API"}