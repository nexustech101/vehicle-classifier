"""Security utilities and hardening."""

import re
import os
from pathlib import Path
from typing import Dict, Optional


def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous characters from filename.
    Prevents path traversal attacks.
    """
    # Remove path separators
    filename = filename.replace("\\", "").replace("/", "")
    # Remove parent directory references
    filename = filename.replace("..", "")
    # Remove special shell characters
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    return filename or "file"


def validate_image_file(filename: str, max_size_mb: int = 16) -> bool:
    """Validate image file is safe and acceptable format."""
    if not filename:
        return False
    
    # Allowed extensions
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    ext = Path(filename).suffix.lower()
    
    if ext not in allowed:
        return False
    
    # Check for suspicious patterns
    suspicious = ['..', '\\', '/', '<', '>', ':', '"', '|', '?', '*']
    if any(pattern in filename for pattern in suspicious):
        return False
    
    return True


def sanitize_json_input(data: dict) -> dict:
    """Remove potentially dangerous content from JSON input."""
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
    ]
    
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned = value
            for pattern in dangerous_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            sanitized[key] = cleaned
        else:
            sanitized[key] = value
    
    return sanitized


def check_path_traversal(path: str) -> bool:
    """Check if path contains traversal attempts."""
    # Normalize path
    try:
        normalized = Path(path).resolve()
        # Check if path goes outside allowed directories
        if ".." in str(path) or normalized.is_absolute():
            return True
    except (ValueError, RuntimeError):
        return True
    
    return False


def generate_rate_limit_key(client_ip: str, endpoint: str) -> str:
    """Generate rate limit key for IP + endpoint."""
    return f"rate_limit:{client_ip}:{endpoint}"


def get_security_headers() -> Dict[str, str]:
    """Get security headers for all responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }


def sanitize_logs(log_entry: str) -> str:
    """Remove sensitive data from log entries."""
    # Mask passwords
    log_entry = re.sub(
        r'password["\']?\s*[:=]\s*["\']?[^"\s]*',
        'password=***',
        log_entry,
        flags=re.IGNORECASE
    )
    
    # Mask API keys
    log_entry = re.sub(
        r'(api[_-]?key|token)["\']?\s*[:=]\s*["\']?[^"\s]*',
        r'\1=***',
        log_entry,
        flags=re.IGNORECASE
    )
    
    # Mask connection strings with passwords
    log_entry = re.sub(
        r'(://[^:]+:)[^@]+(@)',
        r'\1***\2',
        log_entry
    )
    
    return log_entry


def get_environment_variable(var_name: str, default: Optional[str] = None) -> str:
    """Get environment variable with safe defaults."""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Required environment variable {var_name} not set")
    return value


def validate_cors_origins(allowed_origins: list) -> list:
    """Validate and filter CORS origins."""
    if not allowed_origins:
        return ["http://localhost:3000"]  # Safe default
    
    # Never allow *
    if "*" in allowed_origins:
        raise ValueError("CORS wildcard (*) not allowed in production")
    
    # Validate URLs
    validated = []
    url_pattern = re.compile(
        r'^https?://'  # http or https
        r'(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*'  # subdomains
        r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?$'  # optional port
    )
    
    for origin in allowed_origins:
        if url_pattern.match(origin):
            validated.append(origin)
    
    return validated or ["http://localhost:3000"]
