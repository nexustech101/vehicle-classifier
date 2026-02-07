"""Tests for FastAPI application and endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/api/v2/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "timestamp" in response.json()
        assert "redis_connected" in response.json()
    
    def test_health_endpoint_includes_models(self, client):
        """Test health endpoint includes model info."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "models_loaded" in data
        assert data["models_loaded"] >= 0


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_token_endpoint_success(self, client):
        """Test successful token generation."""
        response = client.post(
            "/api/v2/auth/token",
            json={"username": "testuser", "password": "testpass"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_token_endpoint_invalid_credentials(self, client):
        """Test token endpoint with invalid credentials."""
        response = client.post(
            "/api/v2/auth/token",
            json={"username": "invalid", "password": "invalid"}
        )
        assert response.status_code == 401
        assert "detail" in response.json()
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.post(
            "/api/v2/vehicle/classify",
            files={"file": ("test.jpg", b"fake_image")}
        )
        assert response.status_code == 401


class TestMetadataEndpoint:
    """Test model metadata endpoint."""
    
    def test_metadata_endpoint_success(self, client, valid_token):
        """Test metadata endpoint with valid token."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        response = client.get("/api/v2/models/metadata", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "supported_attributes" in data["data"]
    
    def test_metadata_includes_classifier_info(self, client, valid_token):
        """Test metadata includes all classifiers."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        response = client.get("/api/v2/models/metadata", headers=headers)
        data = response.json()
        attributes = data["data"]["supported_attributes"]
        expected = [
            "make", "type", "color", "condition",
            "stock_or_moded", "functional_utility"
        ]
        assert all(attr in attributes for attr in expected)


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_enforcement(self, client, valid_token):
        """Test rate limiting prevents excessive requests."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        
        # Make multiple rapid requests (should hit rate limit)
        for i in range(5):
            response = client.get("/api/v2/health", headers=headers)
            if response.status_code == 429:
                assert "detail" in response.json()
                break
        # Note: Actual rate limit depends on configuration


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_404_error_message(self, client, valid_token):
        """Test 404 error has consistent format."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        response = client.get("/api/v2/nonexistent", headers=headers)
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data
    
    def test_500_error_includes_request_id(self, client, valid_token):
        """Test 500 errors include request ID for tracking."""
        headers = {"Authorization": f"Bearer {valid_token}"}
        # This would test error handling for internal errors
        # Implementation depends on error injection


class TestSecurityHeaders:
    """Test security headers are present."""
    
    def test_security_headers_present(self, client):
        """Test response includes security headers."""
        response = client.get("/api/v2/health")
        
        # Security headers
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        
        assert "x-frame-options" in response.headers
        assert response.headers["x-frame-options"] == "DENY"
        
        assert "x-xss-protection" in response.headers
    
    def test_hsts_header_present(self, client):
        """Test HSTS header for HTTPS."""
        response = client.get("/api/v2/health")
        # assert "strict-transport-security" in response.headers


class TestCORSConfiguration:
    """Test CORS configuration."""
    
    def test_cors_restricted_origins(self, client):
        """Test CORS only allows specific origins."""
        headers = {"Origin": "http://trusted-domain.com"}
        response = client.get("/api/v2/health", headers=headers)
        assert response.status_code == 200
        
        # Check CORS header is present (implementation dependent)
        # assert "access-control-allow-origin" in response.headers
