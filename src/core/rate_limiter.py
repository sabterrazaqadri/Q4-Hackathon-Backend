"""
Rate limiting middleware for the ChatKit RAG integration.
Implements per-IP rate limiting to prevent abuse.
"""
import time
from typing import Dict
from fastapi import Request, HTTPException
from collections import defaultdict
from ..core.config import settings


class RateLimiter:
    """
    Simple in-memory rate limiter based on IP addresses.
    In production, use Redis or similar for distributed rate limiting.
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if a request from the given identifier is allowed.
        """
        current_time = time.time()
        
        # Remove requests older than the time window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < settings.RATE_LIMIT_WINDOW
        ]
        
        # Check if the number of requests is within the limit
        if len(self.requests[identifier]) < settings.RATE_LIMIT_REQUESTS:
            # Add the current request
            self.requests[identifier].append(current_time)
            return True
        
        # Rate limit exceeded
        return False


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Extract the client IP address from the request.
    """
    # Check for X-Forwarded-For header (common with proxies/load balancers)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # Take the first IP in the list (client IP)
        return forwarded.split(",")[0].strip()
    
    # Check for X-Real-IP header
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    
    # Use the client host from the request
    return request.client.host


async def rate_limit_middleware(request: Request, call_next):
    """
    Middleware to enforce rate limiting.
    """
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response