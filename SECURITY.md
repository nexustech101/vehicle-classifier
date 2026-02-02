# Security & Authentication Guide

## Overview

The Vehicle Classification API now includes comprehensive security features for production deployments:

- **JWT Authentication** - Token-based access control
- **Role-Based Access Control** - User and Admin roles
- **Input Validation** - Path traversal, XSS, and file type protection
- **Security Headers** - HSTS, CSP, X-Frame-Options
- **Rate Limiting** - IP-based and endpoint-based rate limiting
- **Secrets Management** - Environment-based secrets configuration
- **Redis Resilience** - Automatic reconnection with exponential backoff
- **Persistent Audit Logging** - Comprehensive audit trail in SQLite
- **Docker Security** - Non-root user execution, health checks

## Authentication

### Getting an Access Token

The API uses JWT (JSON Web Tokens) for authentication. All protected endpoints require an `Authorization: Bearer <token>` header.

#### Test Credentials

**User Role (limited access):**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Admin Role (full access):**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin_user",
    "password": "adminpass"
  }'
```

### Using the Token

Include the token in all protected requests:

```bash
# Get model metadata (requires authentication)
curl -X GET http://localhost:8000/api/models/metadata \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Classify an image (requires authentication)
curl -X POST -F "file=@vehicle.jpg" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  http://localhost:8000/api/vehicle/classify
```

### Token Configuration

Configure token expiration via environment variable:

```bash
# Default: 30 minutes
export ACCESS_TOKEN_EXPIRE_MINUTES=60

# In docker-compose.yml:
environment:
  - ACCESS_TOKEN_EXPIRE_MINUTES=60
```

## Secrets Management

### Secret Key Configuration

The API uses a SECRET_KEY for signing JWT tokens. In production, **always change the default secret key**:

```bash
# Development (default):
export SECRET_KEY="your-secret-key-change-in-production"

# Production (generate strong key):
python -c "import secrets; print(secrets.token_urlsafe(32))"
export SECRET_KEY="the-generated-key-above"
```

Update in `docker-compose.yml`:
```yaml
environment:
  - SECRET_KEY=your-production-secret-key-here
```

### Environment Variables

Required environment variables for security:

```bash
# JWT Configuration
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1

# Database
DATABASE_URL=sqlite:///db/vehicle_classifier.db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Role-Based Access Control

### Available Roles

- **user** - Can classify vehicles, generate reports, view own audit logs
- **admin** - Can access metrics, view all audit logs, system administration

### Protected Endpoints

**Public (no auth required):**
- `GET /health` - Health check
- `GET /` - Root endpoint
- `POST /auth/token` - Token generation

**User Authenticated:**
- `POST /api/vehicle/classify` - Single image classification
- `POST /api/vehicle/classify-batch` - Batch classification
- `POST /api/vehicle/report` - Generate report
- `GET /api/vehicle/report/{vehicle_id}` - Get report
- `GET /api/models/metadata` - Get model metadata

**Admin Only:**
- `GET /metrics` - Prometheus metrics (admin role required)

## Input Validation & Security

### Filename Sanitization

Filenames are automatically sanitized to prevent path traversal:

```python
from src.core.security import sanitize_filename

# "../../../etc/passwd" -> "etcpasswd"
# "vehicle.jpg" -> "vehicle.jpg"
safe_name = sanitize_filename(filename)
```

### Image File Validation

Images are validated for:
- File magic numbers (PNG, JPG, BMP, GIF)
- File extension matching
- File size limits (enforced at API level)

Supported formats: `jpg`, `jpeg`, `png`, `bmp`, `gif`

### Path Traversal Detection

Prevent directory traversal attacks:

```python
from src.core.security import check_path_traversal

if check_path_traversal(user_input):
    raise ValidationError("Invalid path")
```

## Security Headers

The API automatically adds security headers to all responses:

- **X-Content-Type-Options: nosniff** - Prevent MIME sniffing
- **X-Frame-Options: DENY** - Prevent clickjacking
- **Content-Security-Policy: default-src 'self'** - XSS protection
- **Strict-Transport-Security: max-age=31536000** - HTTPS enforcement (on HTTPS)
- **X-XSS-Protection: 1; mode=block** - Browser XSS protection
- **Referrer-Policy: strict-origin-when-cross-origin** - Referrer protection

## Rate Limiting

### Configuration

Rate limiting is based on IP address and endpoint combination:

```python
from src.core.security import generate_rate_limit_key

# Generate rate limit key for 10 requests per minute
key = generate_rate_limit_key(ip="192.168.1.1", endpoint="/api/vehicle/classify")
# Result: "rate_limit:192.168.1.1:/api/vehicle/classify"
```

Implement rate limiting middleware:

```python
from redis import Redis
redis = Redis()

if redis.incr(key) > 10:  # 10 requests per minute
    raise RateLimitError("Rate limit exceeded")

redis.expire(key, 60)  # 1 minute window
```

## CORS Configuration

### Allowed Origins

Configure allowed origins to prevent unauthorized cross-origin requests:

```bash
# Default:
export CORS_ORIGINS="http://localhost:3000,http://localhost:8000"

# In docker-compose.yml:
environment:
  - CORS_ORIGINS=http://localhost:3000,https://example.com
```

Origins are validated during request processing:

```python
from src.core.security import validate_cors_origins

allowed = validate_cors_origins(["http://localhost:3000", "https://example.com"])
# Rejects wildcard origins and validates URLs
```

## Audit Logging

All user actions are logged to the SQLite audit table:

### Audit Log Fields

- `user_id` - Username who performed action
- `action` - Action type (VIEW_METADATA, BATCH_CLASSIFY, GENERATE_REPORT, etc.)
- `resource` - Resource accessed (models, batch, vehicle/{id}, etc.)
- `details` - JSON details about the action
- `ip_address` - Client IP address
- `created_at` - Timestamp of action

### Querying Audit Logs

```python
from src.core.database import Database

db = Database()
logs = db.get_audit_logs(user_id="testuser", limit=100)
for log in logs:
    print(f"{log['created_at']}: {log['action']} on {log['resource']}")
```

## Redis Resilience

### Automatic Reconnection

Redis client automatically reconnects with exponential backoff:

```python
from src.core.redis_client import get_redis_client

redis = get_redis_client()
# Automatically handles:
# - Connection timeouts
# - Server unavailability
# - Graceful fallback if Redis is down
```

### Configuration

Exponential backoff retry strategy:

- Attempt 1: Wait 1 second
- Attempt 2: Wait 2 seconds (2^1)
- Attempt 3: Wait 4 seconds (2^2)
- Fallback: Return None/False if all retries exhausted

## Database Security

### SQLite Configuration

Default database location:

```bash
# Development:
DATABASE_URL=sqlite:///db/vehicle_classifier.db

# In Docker:
/app/db/vehicle_classifier.db
```

### Database Backup

Backup the database file:

```bash
# Local backup
cp db/vehicle_classifier.db db/vehicle_classifier.db.backup

# Docker backup
docker exec vehicle-classifier-api cp /app/db/vehicle_classifier.db /app/db/backup.db
```

## Docker Security

### Non-Root User Execution

The application runs as non-root user `appuser`:

```dockerfile
# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser  # Switch to non-root user
```

### Health Check Configuration

The Docker container includes health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Production Deployment Checklist

- [ ] Change `SECRET_KEY` to strong random value
- [ ] Update `CORS_ORIGINS` to production domain
- [ ] Set `TRUSTED_HOSTS` to production hostnames
- [ ] Enable HTTPS/TLS (use nginx reverse proxy)
- [ ] Update Redis password (if not using default)
- [ ] Configure database backups
- [ ] Enable log aggregation
- [ ] Monitor metrics endpoint (/metrics)
- [ ] Set up alerting for error rates
- [ ] Test rate limiting configuration
- [ ] Verify audit logging is working
- [ ] Configure log retention policies
- [ ] Test disaster recovery procedures

## Common Issues

### "Invalid authentication credentials" Error

Verify that:
1. Token is included in Authorization header: `Authorization: Bearer <token>`
2. Token is not expired (tokens expire after 30 minutes by default)
3. Token is valid (not tampered with)
4. SECRET_KEY matches the key used to generate the token

### CORS Errors

Check that:
1. Origin is in `CORS_ORIGINS` environment variable
2. `CORS_ORIGINS` doesn't include wildcard (`*`)
3. Protocol is correct (http vs https)
4. Port number matches

### Rate Limit Exceeded

If receiving rate limit errors:
1. Check if Redis is running and healthy
2. Reduce request rate or contact admin to increase limits
3. Use batch endpoints for multiple classifications

## Security Best Practices

1. **Never commit secrets** - Use environment variables or `.env` files (added to .gitignore)
2. **Rotate tokens regularly** - Implement token refresh mechanism in production
3. **Monitor audit logs** - Regularly review audit logs for suspicious activity
4. **Update dependencies** - Keep FastAPI, Keras, TensorFlow updated
5. **Use HTTPS** - Always use HTTPS in production (not just HTTP)
6. **Backup database** - Regularly backup SQLite database
7. **Monitor Redis** - Ensure Redis is highly available (replicated in production)
8. **Limit exposure** - Use firewall rules to limit API access
9. **Enable logging** - Ensure verbose logging is enabled for debugging
10. **Test security** - Regularly test authentication and authorization

## Support

For security issues or questions:
- Check the [README.md](README.md) for API documentation
- Review test files in `tests/` directory for usage examples
- Check logs in `logs/` directory for error details
