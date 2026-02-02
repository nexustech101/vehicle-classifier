"""Core modules for vehicle classification API."""

from src.core.security import (
    sanitize_filename,
    validate_image_file,
    check_path_traversal,
    generate_rate_limit_key,
    get_security_headers,
    validate_cors_origins,
    sanitize_logs
)

from src.core.errors import (
    ErrorResponse,
    APIException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    NotFoundError,
    ConflictError,
    ServiceUnavailableError,
    InternalServerError
)

from src.core.monitoring import (
    MetricsCollector,
    RequestLogger,
    record_timing,
    Benchmark,
    check_performance_threshold
)

from src.core.database import Database

from src.core.redis_client import (
    ResilientRedisClient,
    get_redis_client
)

__all__ = [
    # Security
    "sanitize_filename",
    "validate_image_file",
    "check_path_traversal",
    "generate_rate_limit_key",
    "get_security_headers",
    "validate_cors_origins",
    "sanitize_logs",
    # Errors
    "ErrorResponse",
    "APIException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "NotFoundError",
    "ConflictError",
    "ServiceUnavailableError",
    "InternalServerError",
    # Monitoring
    "MetricsCollector",
    "RequestLogger",
    "record_timing",
    "Benchmark",
    "check_performance_threshold",
    # Database
    "Database",
    # Redis
    "ResilientRedisClient",
    "get_redis_client",
]
