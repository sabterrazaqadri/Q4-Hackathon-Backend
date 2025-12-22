import cohere
from typing import List, Optional
from src.config.settings import settings
from src.utils.logging import setup_logging
import time
import asyncio
from src.utils.exceptions import EmbeddingError


class CohereClient:
    """
    Service class for interacting with Cohere API for embedding generation
    """

    def __init__(self):
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.cohere_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere API
        """
        try:
            # Cohere has limits on batch size, so we may need to process in chunks
            batch_size = 96  # Conservative batch size under Cohere's limits
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"  # Using search_document for content chunks
                )

                batch_embeddings = [embedding for embedding in response.embeddings]
                all_embeddings.extend(batch_embeddings)

                # Add a small delay to respect API rate limits
                time.sleep(0.1)

            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        return self.generate_embeddings([text])[0]

    async def generate_embeddings_with_retry(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        Generate embeddings with retry logic for API failures
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return self.generate_embeddings(texts)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break  # Last attempt, exit the loop

                # Calculate delay with exponential backoff
                delay = min(1.0 * (2 ** attempt), 60.0)
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        # If we've exhausted all retries, raise the last exception
        raise last_exception

    def validate_embeddings(self, embeddings: List[List[float]], expected_dimension: Optional[int] = None) -> bool:
        """
        Validate that embeddings have the expected properties
        """
        if not embeddings:
            return False

        # Check that all embeddings have the same length
        first_length = len(embeddings[0])
        if not all(len(embedding) == first_length for embedding in embeddings):
            return False

        # Check against expected dimension if provided
        if expected_dimension is not None and first_length != expected_dimension:
            return False

        # Check that embeddings contain only numbers
        for embedding in embeddings:
            if not all(isinstance(value, (int, float)) for value in embedding):
                return False

        return True