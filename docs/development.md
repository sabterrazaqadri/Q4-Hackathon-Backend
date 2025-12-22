# Developer Documentation: ChatKit RAG Integration

## Architecture

The application follows a modular architecture with clear separation of concerns:

- `src/chat/` - Chat-related models, services, and endpoints
- `src/rag/` - RAG (Retrieval-Augmented Generation) models, services, and agents
- `src/core/` - Core functionality like configuration, database, and exceptions
- `src/services/` - Business logic services
- `tests/` - Unit, integration, and contract tests

## Modules

### Chat Module
- `models.py` - Pydantic models for chat entities (UserQuery, AIResponse, etc.)
- `services.py` - Business logic for processing chat queries
- `endpoints.py` - API endpoints compatible with ChatKit
- `utils.py` - Helper functions for chat operations

### RAG Module
- `models.py` - Models specific to RAG operations
- `services.py` - RAG business logic for context retrieval and response generation
- `agents.py` - OpenAI agent integration
- `utils.py` - Helper functions for RAG operations

### Core Module
- `config.py` - Application configuration using Pydantic settings
- `database.py` - PostgreSQL database integration with SQLAlchemy
- `vector_db.py` - Qdrant vector database operations
- `exceptions.py` - Custom exceptions and logging setup
- `openai_client.py` - OpenAI API integration

## Services

### ChatService
Handles the business logic for processing user queries through the RAG system. It manages session context for multi-turn conversations.

### RAGService
Implements the core RAG functionality:
- Context retrieval from the textbook content using vector search
- Response generation using OpenAI models
- Query validation to ensure responses are grounded in content

### SessionService
Manages conversational context across multiple exchanges. Stores conversation history in memory with automatic cleanup of expired sessions.

### ValidationService
Ensures responses are accurately grounded in the retrieved textbook content, preventing hallucinations.

## Configuration

The application uses environment variables for configuration, managed through Pydantic settings:

- `OPENAI_API_KEY` - API key for OpenAI services
- `QDRANT_URL` - URL for the Qdrant vector database
- `QDRANT_API_KEY` - API key for Qdrant (if required)
- `DATABASE_URL` - Connection string for PostgreSQL database
- Various other settings defined in `src/core/config.py`

## Testing

The application includes three types of tests:

- Unit tests: Test individual functions and classes in isolation
- Integration tests: Test the interaction between multiple components
- Contract tests: Verify API endpoints match the OpenAPI specification

Run tests using pytest:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_chat.py
```

## Running the Application

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Run with gunicorn (or similar)
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Deployment

The application can be deployed using the provided Dockerfile. Ensure all environment variables are properly configured in your deployment environment.

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**: Ensure all required environment variables are configured
2. **Database Connection Issues**: Verify the DATABASE_URL is correct and the database is accessible
3. **Vector Database Issues**: Check QDRANT_URL and credentials
4. **OpenAI API Issues**: Verify OPENAI_API_KEY is valid and has appropriate permissions

### Logging

The application logs important events and errors. Check the logs for debugging information.