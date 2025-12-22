"""
OpenAI client integration module for the ChatKit RAG integration.
Handles communication with OpenAI APIs for embeddings and completions.
"""
from typing import List
import logging
from openai import AsyncOpenAI
from ..core.config import settings


logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Service class to handle OpenAI API operations.
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def create_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Create an embedding for the given text.
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    async def generate_completion(
        self, 
        messages: List[dict], 
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a completion based on the provided messages.
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    async def validate_api_key(self) -> bool:
        """
        Validate that the OpenAI API key is working properly.
        """
        try:
            # Make a simple request to test the API
            await self.client.models.list()
            return True
        except Exception:
            return False


# Dependency for FastAPI
def get_openai_client():
    return OpenAIClient()