"""
RAG-related data models using Pydantic.
Based on the data-model.md specification.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# We're reusing the models from the chat module since they're shared
# This file would contain any additional RAG-specific models if needed