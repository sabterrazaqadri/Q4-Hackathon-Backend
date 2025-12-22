"""
Validation helper service for API endpoint validation.
"""
import re
from typing import Dict, Any, Optional
from src.utils.logging import setup_logging


class ValidationHelper:
    """
    Helper class for validating API request formats and parameters.
    """
    
    def __init__(self):
        self.logger = setup_logging()
    
    def validate_query_format(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the format of a query request.
        
        Args:
            query_data: Dictionary containing query parameters
            
        Returns:
            Dictionary with validation result and error details if any
        """
        # Required field validation
        if not query_data.get('question'):
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'Question field is required and cannot be empty',
                'details': {
                    'field': 'question',
                    'value': query_data.get('question'),
                    'constraint': 'required'
                }
            }
        
        question = query_data['question']
        
        # Question length validation
        if len(question) < 1 or len(question) > 2000:
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'Question length must be between 1 and 2000 characters',
                'details': {
                    'field': 'question',
                    'value': f"{len(question)} characters",
                    'constraint': 'length between 1 and 2000'
                }
            }
        
        # Selected text validation (if provided)
        selected_text = query_data.get('selected_text')
        if selected_text is not None:
            if len(selected_text) > 5000:
                return {
                    'is_valid': False,
                    'error': 'VALIDATION_ERROR',
                    'message': 'Selected text length must not exceed 5000 characters',
                    'details': {
                        'field': 'selected_text',
                        'value': f"{len(selected_text)} characters",
                        'constraint': 'max length 5000'
                    }
                }
        
        # Validate user_context is a dict if provided
        user_context = query_data.get('user_context')
        if user_context is not None and not isinstance(user_context, dict):
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'user_context must be a valid JSON object',
                'details': {
                    'field': 'user_context',
                    'value': type(user_context).__name__,
                    'constraint': 'must be dict'
                }
            }
        
        # Validate metadata is a dict if provided
        metadata = query_data.get('metadata')
        if metadata is not None and not isinstance(metadata, dict):
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'metadata must be a valid JSON object',
                'details': {
                    'field': 'metadata',
                    'value': type(metadata).__name__,
                    'constraint': 'must be dict'
                }
            }
        
        # Validate question content doesn't contain problematic characters
        if not self._is_valid_content(question):
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'Question contains invalid characters',
                'details': {
                    'field': 'question',
                    'constraint': 'no control characters or invalid Unicode'
                }
            }
        
        # If selected_text exists, validate its content too
        if selected_text is not None and not self._is_valid_content(selected_text):
            return {
                'is_valid': False,
                'error': 'VALIDATION_ERROR',
                'message': 'Selected text contains invalid characters',
                'details': {
                    'field': 'selected_text',
                    'constraint': 'no control characters or invalid Unicode'
                }
            }
        
        return {
            'is_valid': True,
            'error': None,
            'message': 'Query format is valid',
            'details': None
        }
    
    def _is_valid_content(self, content: str) -> bool:
        """
        Check if content contains only valid characters.
        
        Args:
            content: String to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        # Check for control characters (except common whitespace)
        for char in content:
            if ord(char) < 32 and char not in ['\n', '\r', '\t']:
                return False
        return True


# Create a global instance for easy access
validation_helper = ValidationHelper()


def validate_query_format(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate query format.
    
    Args:
        query_data: Dictionary containing query parameters
        
    Returns:
        Dictionary with validation result and error details if any
    """
    return validation_helper.validate_query_format(query_data)