"""Tests for security features."""

import pytest
from src.core.security import (
    sanitize_filename,
    validate_image_file,
    sanitize_json_input,
    check_path_traversal
)


class TestInputSanitization:
    """Test input sanitization."""
    
    def test_sanitize_filename_removes_path(self):
        """Test filename sanitization removes path traversal."""
        unsafe = "../../../etc/passwd"
        safe = sanitize_filename(unsafe)
        assert ".." not in safe
        assert "/" not in safe
    
    def test_sanitize_filename_removes_special_chars(self):
        """Test sanitization removes dangerous characters."""
        unsafe = "test<script>.jpg"
        safe = sanitize_filename(unsafe)
        assert "<" not in safe
        assert ">" not in safe
    
    def test_sanitize_filename_preserves_extension(self):
        """Test sanitization preserves file extension."""
        safe = sanitize_filename("my_vehicle.jpg")
        assert safe.endswith(".jpg")
    
    def test_sanitize_json_input(self):
        """Test JSON input sanitization."""
        unsafe = {'user_input': '<script>alert(1)</script>'}
        safe = sanitize_json_input(unsafe)
        assert '<script>' not in str(safe)


class TestImageValidation:
    """Test image file validation."""
    
    def test_valid_image_format(self, test_image_path):
        """Test valid image format passes validation."""
        is_valid = validate_image_file(test_image_path)
        assert is_valid is True
    
    def test_invalid_file_extension(self):
        """Test invalid file extension is rejected."""
        is_valid = validate_image_file("test.exe")
        assert is_valid is False
    
    def test_image_size_limits(self, test_image_path):
        """Test image size validation."""
        is_valid = validate_image_file(test_image_path, max_size_mb=10)
        assert is_valid is True
    
    def test_rejected_formats(self):
        """Test dangerous formats are rejected."""
        dangerous = ["test.zip", "test.exe", "test.sh", "test.bat"]
        for filename in dangerous:
            is_valid = validate_image_file(filename)
            assert is_valid is False


class TestPathTraversal:
    """Test path traversal prevention."""
    
    def test_detect_path_traversal(self):
        """Test path traversal detection."""
        unsafe_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32"
        ]
        
        for path in unsafe_paths:
            has_traversal = check_path_traversal(path)
            assert has_traversal is True
    
    def test_safe_paths_allowed(self):
        """Test safe paths are allowed."""
        safe_paths = [
            "uploads/vehicle.jpg",
            "vehicle_2024.jpg",
            "my_car_model.jpg"
        ]
        
        for path in safe_paths:
            has_traversal = check_path_traversal(path)
            assert has_traversal is False


class TestRateLimitingRules:
    """Test rate limiting enforcement."""
    
    def test_rate_limit_key_generation(self):
        """Test rate limit key generation."""
        from src.core.security import generate_rate_limit_key
        
        key1 = generate_rate_limit_key("192.168.1.1", "/api/classify")
        key2 = generate_rate_limit_key("192.168.1.1", "/api/classify")
        
        assert key1 == key2
        assert "192.168.1.1" in key1
    
    def test_different_clients_different_limits(self):
        """Test different clients have separate rate limits."""
        from src.core.security import generate_rate_limit_key
        
        key1 = generate_rate_limit_key("192.168.1.1", "/api/classify")
        key2 = generate_rate_limit_key("192.168.1.2", "/api/classify")
        
        assert key1 != key2


class TestSecurityHeaders:
    """Test security header configuration."""
    
    def test_content_type_options_header(self):
        """Test X-Content-Type-Options header."""
        from src.core.security import get_security_headers
        
        headers = get_security_headers()
        assert "x-content-type-options" in headers
        assert headers["x-content-type-options"] == "nosniff"
    
    def test_frame_options_header(self):
        """Test X-Frame-Options header."""
        from src.core.security import get_security_headers
        
        headers = get_security_headers()
        assert "x-frame-options" in headers
        assert headers["x-frame-options"] == "DENY"
    
    def test_xss_protection_header(self):
        """Test X-XSS-Protection header."""
        from src.core.security import get_security_headers
        
        headers = get_security_headers()
        assert "x-xss-protection" in headers
    
    def test_csp_header(self):
        """Test Content-Security-Policy header."""
        from src.core.security import get_security_headers
        
        headers = get_security_headers()
        assert "content-security-policy" in headers


class TestSecretsManagement:
    """Test secrets management."""
    
    def test_secrets_not_logged(self):
        """Test sensitive data is not logged."""
        from src.core.security import sanitize_logs
        
        log_entry = "redis_password=secret123&api_key=abc123"
        sanitized = sanitize_logs(log_entry)
        
        assert "secret123" not in sanitized
        assert "abc123" not in sanitized
    
    def test_connection_strings_masked(self):
        """Test connection strings are masked in logs."""
        from src.core.security import sanitize_logs
        
        log = "Connected to redis://user:password@localhost:6379"
        sanitized = sanitize_logs(log)
        
        assert "password" not in sanitized or "***" in sanitized
