"""Pytest configuration and fixtures."""

import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app, db
from src.api.auth import create_access_token, hash_password
from src.core.database import Database


@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    db_path = Path("test_reports.db")
    test_database = Database(db_path)
    yield test_database
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(autouse=True)
def setup_test_users():
    """Ensure test users exist in the app database."""
    if not db.user_exists("test_user"):
        db.create_user(
            username="test_user",
            email="test@test.com",
            password=hash_password("testpass"),
            role="user"
        )
    if not db.user_exists("admin_user"):
        db.create_user(
            username="admin_user",
            email="admin_test@test.com",
            password=hash_password("adminpass"),
            role="admin"
        )


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def valid_token():
    """Create valid test JWT token."""
    return create_access_token(data={"sub": "test_user", "role": "user"})


@pytest.fixture
def admin_token():
    """Create admin JWT token."""
    return create_access_token(data={"sub": "admin_user", "role": "admin"})


@pytest.fixture
def test_image_path():
    """Create test image path."""
    # Create a minimal test image if it doesn't exist
    from PIL import Image
    import numpy as np
    
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    test_image_path = test_dir / "test_vehicle.jpg"
    if not test_image_path.exists():
        img_array = np.random.randint(0, 256, (100, 90), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(test_image_path)
    
    yield test_image_path
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for testing without actual Redis connection."""
    class MockRedis:
        def __init__(self, *args, **kwargs):
            self.data = {}
        
        def ping(self):
            return True
        
        def get(self, key):
            return self.data.get(key)
        
        def setex(self, key, ttl, value):
            self.data[key] = value
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
                return 1
            return 0
        
        def exists(self, key):
            return 1 if key in self.data else 0
    
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")


@pytest.fixture(autouse=True)
def clear_logs():
    """Clear logs before each test."""
    yield
    # Cleanup log files if needed
