"""
FastAPI REST API for Vehicle Classification Service.

Production-ready async API with:
- JWT authentication & authorization
- Rate limiting & input validation
- Prometheus metrics & monitoring
- Persistent database storage
- Resilient Redis caching
- Security hardening
"""

import logging
import io
import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI, UploadFile, File, Form, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image

from src.api.service import VehicleClassificationAPI
from src.api.logging_config import setup_api_logger
from src.api.auth import get_current_user, authenticate_user, create_access_token, require_role, set_db
from src.core.security import (
    sanitize_filename, validate_image_file, check_path_traversal,
    get_security_headers, validate_cors_origins, sanitize_logs
)
from src.core.errors import (
    ValidationError, AuthenticationError, AuthorizationError, NotFoundError,
    ServiceUnavailableError, RateLimitError
)
from src.core.monitoring import MetricsCollector, RequestLogger
from src.core.database import Database
from src.core.redis_client import get_redis_client

# Setup logging
logger = setup_api_logger()
logger.info("Initializing FastAPI application")

# Initialize services
api = VehicleClassificationAPI()
db = Database()
set_db(db)  # Share database instance with auth module
redis_client = get_redis_client()
metrics = MetricsCollector()
request_logger = RequestLogger()

logger.info("VehicleClassificationAPI initialized")


# Security: Trusted hosts middleware
trusted_hosts = os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - startup and shutdown."""
    # Startup
    try:
        # Database is already initialized in __init__
        logger.info("Database initialized")
        
        # Bootstrap: Create default admin user ONLY if database is empty
        # This allows initial setup. Change the password immediately in production!
        existing_users = db.list_users(limit=1)
        if not existing_users:
            from src.api.auth import hash_password
            admin_created = db.create_user(
                username="admin",
                email="admin@localhost",
                password=hash_password("admin"),
                role="admin"
            )
            if admin_created:
                logger.warning("⚠️  DEFAULT ADMIN CREATED: username='admin', password='admin'")
                logger.warning("⚠️  CHANGE THIS PASSWORD IMMEDIATELY IN PRODUCTION!")
            else:
                logger.error("Failed to create default admin user")
        
        # Check Redis
        try:
            if redis_client.ping():
                logger.info("Redis connected")
        except:
            logger.warning("Redis not available - caching disabled")
        
        logger.info("FastAPI startup: All services initialized")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        # Close database connections
        db.close()
        logger.info("Database closed")
        
        # Close Redis
        try:
            redis_client.close()
        except:
            pass
        
        logger.info("FastAPI shutdown: All connections closed")
    except Exception as e:
        logger.error(f"Shutdown failed: {e}")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Vehicle Classification API",
    description="Production-grade multi-dimensional vehicle classification with authentication",
    version="2.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware (restricted)
cors_origins = validate_cors_origins(
    os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


# Define APIRouter instance to organize endpoints and handle versioning
router = APIRouter(
    prefix="/api/v2",
    tags=["Vehicle Classification API v2"]
)

# Middleware for security headers
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    # Add security headers
    for header, value in get_security_headers().items():
        response.headers[header] = value
    
    return response


# Middleware for metrics
@app.middleware("http")
async def log_request_metrics(request, call_next):
    """Log request metrics."""
    start_time = time.time()
    request_logger.log_request_start(request.method, request.url.path)
    
    response = await call_next(request)
    
    duration_ms = (time.time() - start_time) * 1000
    request_logger.log_request_end(
        request.method,
        request.url.path,
        response.status_code,
        duration_ms
    )
    
    return response


# ==================== REQUEST/RESPONSE MODELS ====================

class TokenRequest(BaseModel):
    """Token request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800


class UserRegisterRequest(BaseModel):
    """User registration request model."""
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    """User response model."""
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None


class UserListResponse(BaseModel):
    """User list response model."""
    status: str
    timestamp: str
    data: List[Dict[str, Any]]
    total: int


class UserRoleUpdateRequest(BaseModel):
    """User role update request."""
    role: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    redis_connected: bool
    database_healthy: bool
    models_loaded: bool


class ClassificationResponse(BaseModel):
    """Classification response model."""
    status: str
    timestamp: str
    data: dict


class BatchResponse(BaseModel):
    """Batch classification response model."""
    status: str
    timestamp: str
    summary: dict
    results: list


# ==================== UTILITY FUNCTIONS ====================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
UPLOAD_DIR = Path('./uploads')
UPLOAD_DIR.mkdir(exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


async def load_image(file: UploadFile) -> tuple:
    """Load and validate image from upload."""
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename)
    
    if not allowed_file(safe_filename):
        logger.warning(f"Invalid file type: {file.filename}")
        raise ValidationError(
            message="File type not allowed. Supported: jpg, jpeg, png, bmp, gif"
        )
    
    try:
        contents = await file.read()
        
        # Validate image file by sanitized filename
        if not validate_image_file(safe_filename):
            raise ValidationError(message="Invalid image file")
        
        image = Image.open(io.BytesIO(contents)).convert('L')
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Store file with sanitized name
        filepath = UPLOAD_DIR / safe_filename
        image.save(filepath)
        
        logger.debug(f"Image loaded: {safe_filename} -> {filepath}")
        return image_array, str(filepath)
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Image loading failed for {file.filename}: {e}")
        raise ValidationError(message=f"Error processing image: {str(e)}")


def format_response(status: str, data: dict = None, message: str = None) -> dict:
    """Format standardized API response."""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    if message:
        response['message'] = message
    
    return response


# ==================== AUTHENTICATION ====================

@router.post("/auth/token", response_model=TokenResponse)
async def login(credentials: TokenRequest):
    """
    Authenticate user and return JWT token.
    
    First time setup:
    - Use default admin account: username='admin', password='admin'
    - Change the admin password immediately after first login
    - Create additional users via /auth/register endpoint
    """
    try:
        # Authenticate user
        user = authenticate_user(credentials.username, credentials.password)
        if not user:
            logger.warning(f"Failed login attempt for user: {credentials.username}")
            raise AuthenticationError(message="Invalid credentials")
        
        # Update last login
        db.update_last_login(credentials.username)
        
        # Create token with proper JWT claims
        token_data = {
            "sub": user["username"],
            "role": user.get("role", "user")
        }
        token = create_access_token(token_data)
        logger.info(f"User authenticated: {credentials.username}")
        
        # Log audit
        db.log_audit(
            user_id=credentials.username,
            action="LOGIN",
            resource="auth",
            details={"ip": "client_ip"}
        )
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=1800
        )
    except AuthenticationError:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise ServiceUnavailableError(service="Authentication")


# ==================== USER ACCOUNT MANAGEMENT ====================

@router.post("/auth/register", response_model=TokenResponse)
async def register_user(user_data: UserRegisterRequest):
    """
    Register a new user account.
    
    Returns JWT token on successful registration.
    """
    try:
        # Validate input
        if not user_data.username or len(user_data.username) < 3:
            raise ValidationError(message="Username must be at least 3 characters")
        
        if not user_data.email or '@' not in user_data.email:
            raise ValidationError(message="Invalid email format")
        
        if not user_data.password or len(user_data.password) < 8:
            raise ValidationError(message="Password must be at least 8 characters")
        
        # Check if user already exists
        if db.user_exists(user_data.username):
            raise ValidationError(message="Username already exists")
        
        if db.get_user_by_email(user_data.email):
            raise ValidationError(message="Email already registered")
        
        # Hash password
        from src.api.auth import hash_password
        hashed_password = hash_password(user_data.password)
        
        # Create user with default 'user' role
        success = db.create_user(
            username=user_data.username,
            email=user_data.email,
            password=hashed_password,
            role='user'
        )
        
        if not success:
            raise ServiceUnavailableError(service="User Registration")
        
        logger.info(f"New user registered: {user_data.username}")
        
        # Log audit
        db.log_audit(
            user_id=user_data.username,
            action="REGISTER",
            resource="auth",
            details={"email": user_data.email}
        )
        
        # Auto-login after registration
        token_data = {
            "sub": user_data.username,
            "role": "user"
        }
        token = create_access_token(token_data)
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=1800
        )
    
    except (ValidationError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise ServiceUnavailableError(service="User Registration")


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile information."""
    try:
        user = db.get_user_by_username(current_user['username'])
        if not user:
            raise NotFoundError(resource="User")
        
        logger.info(f"User profile requested: {current_user['username']}")
        
        return UserResponse(
            username=user['username'],
            email=user['email'],
            role=user['role'],
            is_active=bool(user['is_active']),
            created_at=user['created_at'],
            last_login=user['last_login']
        )
    except NotFoundError:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        raise ServiceUnavailableError(service="User Profile")


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: dict = Depends(require_role("admin")),
    limit: int = 100,
    offset: int = 0
):
    """
    List all users - Admin only.
    """
    try:
        users = db.list_users(limit=limit, offset=offset)
        logger.info(f"User list requested by admin: {current_user['username']}")
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="LIST_USERS",
            resource="users",
            details={"limit": limit, "offset": offset}
        )
        
        return UserListResponse(
            status='success',
            timestamp=datetime.now().isoformat(),
            data=users,
            total=len(users)
        )
    except Exception as e:
        logger.error(f"User list retrieval failed: {e}")
        raise ServiceUnavailableError(service="User List")


@router.patch("/users/{username}/role", response_model=UserResponse)
async def update_user_role(
    username: str,
    role_update: UserRoleUpdateRequest,
    current_user: dict = Depends(require_role("admin"))
):
    """
    Update user role - Admin only.
    
    Valid roles: 'user', 'admin'
    """
    try:
        if role_update.role not in {'user', 'admin'}:
            raise ValidationError(message="Invalid role. Must be 'user' or 'admin'")
        
        # Check if user exists
        user = db.get_user_by_username(username)
        if not user:
            raise NotFoundError(resource=f"User {username}")
        
        # Update role
        success = db.update_user_role(username, role_update.role)
        if not success:
            raise ServiceUnavailableError(service="User Role Update")
        
        logger.info(f"User role updated by {current_user['username']}: {username} -> {role_update.role}")
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="UPDATE_USER_ROLE",
            resource=f"users/{username}",
            details={"new_role": role_update.role}
        )
        
        # Get updated user
        updated_user = db.get_user_by_username(username)
        return UserResponse(
            username=updated_user['username'],
            email=updated_user['email'],
            role=updated_user['role'],
            is_active=bool(updated_user['is_active']),
            created_at=updated_user['created_at'],
            last_login=updated_user['last_login']
        )
    except (ValidationError, NotFoundError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"User role update failed: {e}")
        raise ServiceUnavailableError(service="User Role Update")


@router.patch("/users/{username}/status", response_model=UserResponse)
async def update_user_status(
    username: str,
    is_active: bool,
    current_user: dict = Depends(require_role("admin"))
):
    """
    Activate or deactivate user account - Admin only.
    """
    try:
        # Check if user exists
        user = db.get_user_by_username(username)
        if not user:
            raise NotFoundError(resource=f"User {username}")
        
        # Update status
        success = db.update_user_status(username, is_active)
        if not success:
            raise ServiceUnavailableError(service="User Status Update")
        
        status_str = "activated" if is_active else "deactivated"
        logger.info(f"User {status_str} by {current_user['username']}: {username}")
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="UPDATE_USER_STATUS",
            resource=f"users/{username}",
            details={"is_active": is_active}
        )
        
        # Get updated user
        updated_user = db.get_user_by_username(username)
        return UserResponse(
            username=updated_user['username'],
            email=updated_user['email'],
            role=updated_user['role'],
            is_active=bool(updated_user['is_active']),
            created_at=updated_user['created_at'],
            last_login=updated_user['last_login']
        )
    except (NotFoundError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"User status update failed: {e}")
        raise ServiceUnavailableError(service="User Status Update")


@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(require_role("admin"))
):
    """
    Delete user account - Admin only.
    """
    try:
        # Prevent self-deletion
        if username == current_user['username']:
            raise ValidationError(message="Cannot delete your own account")
        
        # Check if user exists
        user = db.get_user_by_username(username)
        if not user:
            raise NotFoundError(resource=f"User {username}")
        
        # Delete user
        success = db.delete_user(username)
        if not success:
            raise ServiceUnavailableError(service="User Deletion")
        
        logger.info(f"User deleted by {current_user['username']}: {username}")
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="DELETE_USER",
            resource=f"users/{username}",
            details={"deleted_user": username}
        )
        
        return format_response('success', message=f"User {username} deleted successfully")
    except (ValidationError, NotFoundError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"User deletion failed: {e}")
        raise ServiceUnavailableError(service="User Deletion")



# ==================== HEALTH & METADATA ====================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check endpoint - no auth required."""
    try:
        # Check Redis
        redis_ok = False
        try:
            redis_ok = redis_client.ping()
        except:
            redis_ok = False
        
        # Check database
        db_ok = db.health_check()
        
        # Check API
        health = api.health_check()
        
        logger.info("Health check passed")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            redis_connected=redis_ok,
            database_healthy=db_ok,
            models_loaded=health.get('models_ready', False)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise ServiceUnavailableError(service="Health")


@router.get("/models/metadata")
async def get_model_metadata(current_user: dict = Depends(get_current_user)):
    """Get metadata about available models - requires authentication."""
    try:
        logger.info(f"Metadata requested by user: {current_user['username']}")
        metadata = api.get_model_metadata()
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="VIEW_METADATA",
            resource="models",
            details={"models_count": len(metadata.get('data', {}))}
        )
        
        return format_response('success', metadata['data'])
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {e}")
        raise ServiceUnavailableError(service="Metadata")


# ==================== CLASSIFICATION ====================

@router.post("/vehicle/classify", response_model=ClassificationResponse)
async def classify_single(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Classify a single vehicle image - requires authentication."""
    start_time = time.time()
    try:
        # Validate file
        image_array, image_path = await load_image(file)
        
        logger.info(f"Classifying image for user {current_user['username']}: {file.filename}")
        
        # Run classification
        result = api.pipeline.predict_single(image_array, image_path)
        result_dict = result.to_dict()
        
        # Save to database
        db.save_classification(
            image_path=image_path,
            predictions=result_dict.get('predictions', {}),
            confidence=result_dict.get('overall_confidence', 0),
            processing_time_ms=result.processing_time_ms,
            user_id=current_user['username']
        )
        
        # Record metrics
        metrics.record_latency("/vehicle/classify", result.processing_time_ms / 1000)
        metrics.record_classification("vehicle_classify", result.processing_time_ms)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Classification successful ({elapsed_ms:.2f}ms)")
        
        return ClassificationResponse(
            status='success',
            timestamp=datetime.now().isoformat(),
            data=result_dict
        )
    
    except ValidationError:
        metrics.record_error(error_type="VALIDATION_ERROR")
        raise
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        metrics.record_error(error_type="CLASSIFICATION_ERROR")
        raise ServiceUnavailableError(service="Classification")


@router.post("/vehicle/classify-batch", response_model=BatchResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Classify multiple vehicle images - requires authentication."""
    start_time = time.time()
    try:
        logger.info(f"Batch classification started for user {current_user['username']}: {len(files)} files")
        
        results = []
        successful = 0
        failed = 0
        total_time = 0.0
        
        for file in files:
            try:
                if not allowed_file(file.filename):
                    failed += 1
                    results.append({
                        'filename': file.filename,
                        'status': 'error',
                        'message': 'File type not allowed'
                    })
                    continue
                
                image_array, image_path = await load_image(file)
                result = api.pipeline.predict_single(image_array, image_path)
                result_dict = result.to_dict()
                
                # Save to database
                db.save_classification(
                    image_path=image_path,
                    predictions=result_dict.get('predictions', {}),
                    confidence=result_dict.get('overall_confidence', 0),
                    processing_time_ms=result.processing_time_ms,
                    user_id=current_user['username']
                )
                
                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'data': result_dict
                })
                successful += 1
                total_time += result.processing_time_ms
            
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to classify {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': str(e)
                })
        
        batch_response = {
            'summary': {
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'average_processing_time_ms': total_time / max(successful, 1)
            },
            'results': results
        }
        
        # Record metrics
        metrics.record_latency("/vehicle/classify-batch", time.time() - start_time)
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="BATCH_CLASSIFY",
            resource="batch",
            details={"total": len(files), "successful": successful, "failed": failed}
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Batch complete: {successful}/{len(files)} successful ({elapsed_ms:.2f}ms)")
        
        return BatchResponse(
            status='success',
            timestamp=datetime.now().isoformat(),
            summary=batch_response['summary'],
            results=batch_response['results']
        )
    
    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        metrics.record_error(error_type="BATCH_ERROR")
        raise ServiceUnavailableError(service="Batch Classification")


# ==================== REPORTING ====================

@router.post("/vehicle/report")
async def generate_report(
    file: UploadFile = File(...),
    format: str = Form('json'),
    vehicle_id: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Generate professional classification report - requires authentication."""
    start_time = time.time()
    try:
        image_array, image_path = await load_image(file)
        
        logger.info(f"Generating {format} report for user {current_user['username']}")
        report = api.pipeline.predict_and_report(image_array, image_path, vehicle_id)
        
        # Save report to database
        report_dict = report.to_dict()
        report_id = db.save_report(
            vehicle_id=report.vehicle_id,
            data=report_dict,
            user_id=current_user['username']
        )
        
        # Cache report in Redis for 1 hour
        cache_key = f"report:{report.vehicle_id}"
        try:
            redis_client.setex(cache_key, 3600, json.dumps(report_dict))
        except:
            pass  # Cache is optional
        
        # Record metrics
        metrics.record_latency("/vehicle/report", time.time() - start_time)
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="GENERATE_REPORT",
            resource=f"vehicle/{report.vehicle_id}",
            details={"format": format, "report_id": report_id}
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Report generated ({elapsed_ms:.2f}ms)")
        
        if format == 'html':
            return HTMLResponse(content=report.to_html())
        
        return format_response('success', report_dict)
    
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        metrics.record_error(error_type="REPORT_ERROR")
        raise ServiceUnavailableError(service="Report Generation")


@router.get("/vehicle/report/{vehicle_id}")
async def get_report(
    vehicle_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieve report by vehicle ID - requires authentication."""
    try:
        # Check for path traversal
        if check_path_traversal(vehicle_id):
            raise ValidationError(message="Invalid vehicle ID format")
        
        logger.info(f"Report retrieval requested by {current_user['username']}: {vehicle_id}")
        
        # Try cache first
        cache_key = f"report:{vehicle_id}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"Report retrieved from cache: {vehicle_id}")
                metrics.record_cache_hit("report_cache")
                return format_response('success', json.loads(cached))
        except:
            pass
        
        # Try database
        report = db.get_report(vehicle_id)
        if not report:
            metrics.record_cache_miss("report_cache")
            raise NotFoundError(resource=f"Report {vehicle_id}")
        
        # Log audit
        db.log_audit(
            user_id=current_user['username'],
            action="VIEW_REPORT",
            resource=f"vehicle/{vehicle_id}",
            details={}
        )
        
        return format_response('success', report['data'])
    
    except (ValidationError, NotFoundError):
        raise
    except Exception as e:
        logger.error(f"Report retrieval failed: {e}")
        raise ServiceUnavailableError(service="Report Retrieval")


# ==================== MONITORING ====================

@router.get("/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get Prometheus metrics - requires authentication and admin role."""
    try:
        # Only admins can view metrics
        if current_user.get('role') != 'admin':
            from src.core.errors import AuthorizationError
            raise AuthorizationError(message="Admin access required")
        
        logger.info(f"Metrics requested by {current_user['username']}")
        
        # Return Prometheus metrics
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except AuthorizationError:
        raise
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise ServiceUnavailableError(service="Metrics")


# ==================== ROOT & STARTUP ====================

@router.get("/")
async def root():
    """Root endpoint with API documentation link."""
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "api": "Vehicle Classification API v2.1",
        "docs": "/docs",
        "health": "/health",
        "auth": "/auth/token"
    }

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vehicle Classification API on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
