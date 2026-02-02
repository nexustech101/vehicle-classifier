"""Tests for authentication and authorization."""

import pytest
from datetime import timedelta
from src.api.auth import (
    create_access_token,
    verify_token,
    get_current_user,
    authenticate_user,
    hash_password,
    verify_password
)


class TestTokenGeneration:
    """Test JWT token generation."""
    
    def test_create_access_token(self):
        """Test token creation."""
        token = create_access_token(data={"sub": "testuser"})
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_token_with_expiration(self):
        """Test token with custom expiration."""
        token = create_access_token(
            data={"sub": "testuser"},
            expires_delta=timedelta(hours=24)
        )
        assert isinstance(token, str)
    
    def test_token_includes_claims(self):
        """Test token includes required claims."""
        token = create_access_token(data={"sub": "testuser", "role": "admin"})
        decoded = verify_token(token)
        assert decoded["sub"] == "testuser"
        assert decoded["role"] == "admin"


class TestTokenVerification:
    """Test JWT token verification."""
    
    def test_verify_valid_token(self):
        """Test verification of valid token."""
        token = create_access_token(data={"sub": "testuser"})
        decoded = verify_token(token)
        assert decoded["sub"] == "testuser"
    
    def test_verify_expired_token(self):
        """Test verification rejects expired tokens."""
        from datetime import timedelta
        token = create_access_token(
            data={"sub": "testuser"},
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        with pytest.raises(Exception):  # Should raise JWT error
            verify_token(token)
    
    def test_verify_invalid_token(self):
        """Test verification rejects invalid tokens."""
        with pytest.raises(Exception):
            verify_token("invalid.token.here")


class TestPasswordHandling:
    """Test password hashing and verification."""
    
    def test_password_hashing(self):
        """Test password is hashed."""
        password = "testpass123"
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > len(password)
    
    def test_password_verification(self):
        """Test password verification."""
        password = "testpass123"
        hashed = hash_password(password)
        assert verify_password(password, hashed)
    
    def test_wrong_password_fails(self):
        """Test wrong password fails verification."""
        password = "testpass123"
        hashed = hash_password(password)
        assert not verify_password("wrongpass", hashed)
    
    def test_password_case_sensitive(self):
        """Test passwords are case-sensitive."""
        password = "TestPass123"
        hashed = hash_password(password)
        assert not verify_password("testpass123", hashed)


class TestUserAuthentication:
    """Test user authentication logic."""
    
    def test_authenticate_valid_user(self, mock_redis):
        """Test authentication with valid credentials."""
        user = authenticate_user("testuser", "testpass")
        assert user is not None
        assert user["username"] == "testuser"
    
    def test_authenticate_invalid_user(self):
        """Test authentication with invalid credentials."""
        user = authenticate_user("invalid", "invalid")
        assert user is None
    
    def test_authenticate_wrong_password(self):
        """Test authentication with wrong password."""
        user = authenticate_user("testuser", "wrongpass")
        assert user is None


class TestRoleBasedAccess:
    """Test role-based authorization."""
    
    def test_admin_token_has_admin_role(self):
        """Test admin token contains admin role."""
        token = create_access_token(
            data={"sub": "admin", "role": "admin"}
        )
        decoded = verify_token(token)
        assert decoded.get("role") == "admin"
    
    def test_user_token_has_user_role(self):
        """Test user token contains user role."""
        token = create_access_token(
            data={"sub": "user", "role": "user"}
        )
        decoded = verify_token(token)
        assert decoded.get("role") == "user"
    
    def test_role_extraction(self):
        """Test role can be extracted from token."""
        token = create_access_token(
            data={"sub": "testuser", "role": "admin"}
        )
        decoded = verify_token(token)
        assert "role" in decoded
