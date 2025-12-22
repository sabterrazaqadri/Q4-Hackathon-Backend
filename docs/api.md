# ChatKit RAG Integration API Documentation

## Overview

The ChatKit RAG Integration API provides a conversational interface to the Physical AI & Humanoid Robotics textbook. It uses Retrieval-Augmented Generation (RAG) to ensure responses are grounded in textbook content without hallucinations.

## API Endpoints

### POST /api/v1/chat/completions

Creates a chat completion using the RAG system.

#### Request Body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your question about the textbook content"
    }
  ],
  "selected_text": "Optional selected text from the textbook",
  "session_id": "Optional session ID to maintain conversation context",
  "user_id": "Optional user ID",
  "temperature": 0.7
}
```

#### Response

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677825435,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The AI response based on textbook content"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "retrieved_context": []
}
```

### POST /api/v1/chat/validate

Validates if a query can be answered using the available textbook content.

#### Request Body

```json
{
  "query": "Your question about the textbook content",
  "selected_text": "Optional selected text to focus the validation"
}
```

#### Response

```json
{
  "is_valid": true,
  "confidence": 0.85,
  "relevant_sources": ["document1.pdf", "document2.pdf"]
}
```

## Session Management

The API supports session-based conversations to maintain context across multiple exchanges. Include a `session_id` in your requests to maintain conversation state.

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input parameters)
- `429`: Rate limit exceeded
- `500`: Internal server error

## Rate Limiting

The API implements rate limiting to prevent abuse (100 requests per minute per IP).

## Security

- Input validation is performed on all requests
- Rate limiting prevents abuse
- Responses are grounded in textbook content to prevent hallucinations