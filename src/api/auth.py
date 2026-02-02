"""JWT authentication and token management."""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Dependency to get current user from token."""
    try:
        payload = verify_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": payload.get("role", "user")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_role(required_role: str):
    """Dependency to require specific role."""
    async def role_checker(current_user: Dict = Depends(get_current_user)):
        if current_user.get("role") != required_role and current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


# Mock user database (replace with real database in production)
# Pre-hashed passwords for testuser:testpass and admin_user:adminpass
# Generated with: pwd_context.hash("testpass") and pwd_context.hash("adminpass")
USERS_DB_INIT = {
    "testuser": {
        "username": "testuser",
        "plaintext_password": "testpass",
        "role": "user",
        "active": True
    },
    "admin_user": {
        "username": "admin_user",
        "plaintext_password": "adminpass",
        "role": "admin",
        "active": True
    }
}

# Initialize USERS_DB with hashed passwords
USERS_DB = {}
for username, user_data in USERS_DB_INIT.items():
    USERS_DB[username] = {
        "username": user_data["username"],
        "password": hash_password(user_data["plaintext_password"]),
        "role": user_data["role"],
        "active": user_data["active"]
    }


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with username and password."""
    user = USERS_DB.get(username)
    if not user:
        return None
    
    if not verify_password(password, user["password"]):
        return None
    
    if not user.get("active"):
        return None
    
    return user
