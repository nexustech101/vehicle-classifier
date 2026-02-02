"""Consistent error handling and HTTP exception definitions."""

import logging
import uuid
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str
    timestamp: str


class APIException(HTTPException):
    """Base API exception with error tracking."""
    
    def __init__(
        self,
        status_code: int = 400,
        error_code: str = "UNKNOWN_ERROR",
        message: str = "An error occurred",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        """Initialize API exception."""
        from datetime import datetime
        
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        
        # Log the error
        logger.error(
            f"API Error [{self.request_id}] {error_code}: {message}",
            extra={"details": details}
        )
        
        super().__init__(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "message": message,
                "details": details,
                "request_id": self.request_id,
                "timestamp": self.timestamp
            }
        )


# Specific exception classes
class ValidationError(APIException):
    """Input validation error."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            message=message,
            details=details
        )


class AuthenticationError(APIException):
    """Authentication failed."""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            message=message,
            details=details
        )


class AuthorizationError(APIException):
    """User lacks required permissions."""
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            message=message,
            details=details
        )


class RateLimitError(APIException):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_ERROR",
            message=message,
            details=details
        )


class NotFoundError(APIException):
    """Resource not found."""
    def __init__(self, resource: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=f"{resource} not found",
            details=details
        )


class ConflictError(APIException):
    """Resource conflict."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT_ERROR",
            message=message,
            details=details
        )


class ServiceUnavailableError(APIException):
    """Service temporarily unavailable."""
    def __init__(self, service: str, details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            message=f"{service} service temporarily unavailable",
            details=details
        )


class InternalServerError(APIException):
    """Internal server error."""
    def __init__(self, message: str = "Internal server error", details: Optional[Dict] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            details=details
        )


# Error mapping
ERROR_MESSAGES = {
    "INVALID_IMAGE": "Image file is invalid or unsupported format",
    "IMAGE_TOO_LARGE": "Image file exceeds maximum size limit",
    "PATH_TRAVERSAL": "Invalid file path detected",
    "REDIS_ERROR": "Cache service temporarily unavailable",
    "MODEL_ERROR": "Model processing failed",
    "TOKEN_EXPIRED": "Authentication token has expired",
    "INVALID_TOKEN": "Invalid or malformed authentication token",
    "MISSING_FIELD": "Required field is missing",
    "INVALID_FORMAT": "Invalid data format",
}


def get_error_message(error_code: str, default: str = "An error occurred") -> str:
    """Get standardized error message for error code."""
    return ERROR_MESSAGES.get(error_code, default)
