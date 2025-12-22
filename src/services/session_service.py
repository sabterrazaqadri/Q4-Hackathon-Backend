"""
Session service for the ChatKit RAG integration.
Manages conversational context and session state across multiple exchanges.
"""
from typing import Dict, List, Optional
import uuid
from datetime import datetime, timedelta

from ..chat.models import ChatSession, UserQuery, AIResponse
from ..core.config import settings
from ..core.constants import SESSION_TIMEOUT_MINUTES, MAX_CONVERSATION_HISTORY


class SessionService:
    """
    Service class to handle session management for multi-turn conversations.
    """
    
    def __init__(self):
        # In-memory storage for sessions (in production, use Redis or database)
        self.sessions: Dict[str, ChatSession] = {}
        # Store conversation history for each session
        self.conversations: Dict[str, List[dict]] = {}
        
        # Session timeout configuration
        self.session_timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    async def create_session(self, user_id: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.
        """
        session_id = str(uuid.uuid4())
        
        session = ChatSession(
            id=session_id,
            created_at=datetime.now(),
            last_interaction=datetime.now(),
            user_id=user_id,
            is_active=True,
            query_count=0
        )
        
        self.sessions[session_id] = session
        self.conversations[session_id] = []
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve an existing session by ID.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() - session.last_interaction > self.session_timeout:
            await self.deactivate_session(session_id)
            return None
        
        return session
    
    async def update_session_interaction(self, session_id: str) -> Optional[ChatSession]:
        """
        Update the last interaction time for a session.
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        session.last_interaction = datetime.now()
        session.query_count += 1
        
        # Reactivate if it was inactive
        if not session.is_active:
            session.is_active = True
        
        return session
    
    async def add_message_to_conversation(self, session_id: str, message: dict):
        """
        Add a message to the conversation history for a session.
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(message)
        
        # Limit conversation history to prevent memory issues
        # (Keep only the last N exchanges)
        if len(self.conversations[session_id]) > MAX_CONVERSATION_HISTORY:
            # Keep only the most recent messages
            self.conversations[session_id] = self.conversations[session_id][-MAX_CONVERSATION_HISTORY:]
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """
        Retrieve the conversation history for a session.
        """
        if session_id not in self.conversations:
            return []
        
        # Return the most recent messages up to the limit
        history = self.conversations[session_id]
        return history[-limit:] if len(history) > limit else history
    
    async def deactivate_session(self, session_id: str):
        """
        Mark a session as inactive.
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
    
    async def cleanup_expired_sessions(self):
        """
        Remove expired sessions from memory.
        This should be run periodically as a background task.
        """
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_interaction > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            if session_id in self.conversations:
                del self.conversations[session_id]


# Global instance of SessionService
# In a real application, this would be managed by a dependency injection system
session_service = SessionService()


# Dependency for FastAPI
def get_session_service():
    return session_service