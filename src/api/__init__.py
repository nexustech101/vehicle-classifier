"""API package for REST service and prediction endpoints."""

from .service import VehicleClassificationAPI
from .auth import (
    create_access_token,
    verify_token,
    hash_password,
    verify_password,
    authenticate_user,
)

__all__ = [
    'VehicleClassificationAPI',
    'create_access_token',
    'verify_token',
    'hash_password',
    'verify_password',
    'authenticate_user',
]
