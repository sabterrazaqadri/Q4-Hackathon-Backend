from abc import ABC, abstractmethod
from typing import Any, Dict
import asyncio
import time
from src.utils.logging import setup_logging


# Import validation helper
from .validation_helper import validate_query_format


class BaseService(ABC):
    """
    Base service class providing common functionality for all services
    """
    
    def __init__(self):
        self.logger = setup_logging()  # Using setup_logging function even though it's for configuration
    
    async def retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Execute a function with exponential backoff retry logic
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break  # Last attempt, exit the loop
                    
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * 0.1 * (2 * (0.5 - abs(0.5 - (hash(time.time()) % 1000) / 1000)))
                actual_delay = delay + jitter
                
                print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {actual_delay:.2f} seconds...")
                await asyncio.sleep(actual_delay)
        
        # If we've exhausted all retries, raise the last exception
        raise last_exception
    
    def validate_input(self, data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that required fields are present in input data
        """
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
        return True