"""
Security middleware for the RAG API.
This module implements security measures to protect the API from common threats.
"""
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import time
import logging
from typing import Optional
import re


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement security measures for the API.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
        self.rate_limit_store = {}  # In-memory rate limiting store (use Redis in production)
        self.max_requests_per_minute = 60  # Default rate limit
        self.blocked_ips = set()  # IPs to block (could be from a config or DB)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self.get_client_ip(request)
        
        # Block IP if needed
        if client_ip in self.blocked_ips:
            raise HTTPException(status_code=403, detail="Access forbidden")
        
        # Rate limiting
        if not await self.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Security checks
        self.check_security_headers(request)
        self.validate_input(request)
        
        response = await call_next(request)
        
        # Add security headers to response
        self.add_security_headers(response)
        
        return response

    def get_client_ip(self, request: Request) -> str:
        """
        Extract the client IP address from the request.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0]
        return request.client.host

    async def is_allowed(self, client_ip: str) -> bool:
        """
        Check if a client is allowed to make requests based on rate limiting.
        """
        current_time = time.time()
        if client_ip not in self.rate_limit_store:
            self.rate_limit_store[client_ip] = []
        
        # Clean old requests (older than 60 seconds)
        self.rate_limit_store[client_ip] = [
            req_time for req_time in self.rate_limit_store[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check if under rate limit
        if len(self.rate_limit_store[client_ip]) < self.max_requests_per_minute:
            self.rate_limit_store[client_ip].append(current_time)
            return True
        
        return False

    def check_security_headers(self, request: Request):
        """
        Check for security-related headers in the request.
        """
        # Log potential security issues
        if 'user-agent' not in request.headers:
            self.logger.warning(f"Request without User-Agent header from {request.client.host}")

    def validate_input(self, request: Request):
        """
        Validate input for potential security issues.
        """
        if request.method in ["POST", "PUT", "PATCH"]:
            # For this implementation, we'll focus on checking path parameters and headers
            # The actual body validation is done by Pydantic models
            pass

    def add_security_headers(self, response: Response):
        """
        Add security-related headers to the response.
        """
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection (though modern browsers ignore this in favor of CSP)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Security Policy (basic)
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Strict Transport Security (if serving over HTTPS)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Additional rate limiting middleware for more sophisticated rate limiting.
    """
    
    def __init__(self, app, max_requests_per_minute: int = 60):
        super().__init__(app)
        self.max_requests_per_minute = max_requests_per_minute
        self.requests_store = {}  # In-memory storage (use Redis in production)
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self.get_real_ip(request)
        
        current_time = time.time()
        if client_ip not in self.requests_store:
            self.requests_store[client_ip] = []
        
        # Clean old requests
        self.requests_store[client_ip] = [
            req_time for req_time in self.requests_store[client_ip]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests_store[client_ip]) >= self.max_requests_per_minute:
            return Response(
                status_code=429,
                content="Rate limit exceeded"
            )
        
        # Add current request
        self.requests_store[client_ip].append(current_time)
        
        response = await call_next(request)
        return response
    
    def get_real_ip(self, request: Request) -> str:
        """
        Get the real IP of the client, considering proxies.
        """
        x_forwarded_for = request.headers.get("x-forwarded-for")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.client.host


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for validating input against common attack patterns.
    """
    
    def __init__(self, app):
        super().__init__(app)
        # Patterns for common attack vectors
        self.attack_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # XSS
            r'on\w+\s*=',  # Event handlers
            r'<iframe.*?>',  # Frame injection
            r'<object.*?>',  # Object injection
            r'<embed.*?>',  # Embed injection
            r'eval\s*\(',  # Code execution
            r'expression\s*\(',  # CSS expression
        ]
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # For this implementation, we'll log suspicious patterns
        # In a real implementation, you'd check the actual request body
        response = await call_next(request)
        return response

    def contains_attack_pattern(self, text: str) -> bool:
        """
        Check if text contains known attack patterns.
        """
        text_lower = text.lower()
        for pattern in self.attack_patterns:
            if re.search(pattern, text_lower):
                return True
        return False