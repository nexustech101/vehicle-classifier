# Deployment & Operations Guide

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Monitoring & Observability](#monitoring--observability)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)

## Local Development

### Prerequisites

- Python 3.10+
- Redis (optional, for caching)
- 4 GB RAM minimum

### Setup

```bash
# Clone repository
git clone <repo-url>
cd vehicle-classifier

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (optional, in another terminal)
redis-server

# Create necessary directories
mkdir -p logs uploads checkpoints db
```

### Running the API

```bash
# Development mode with auto-reload
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Visit http://localhost:8000/docs for interactive documentation
```

### Running Tests

```bash
# Install test dependencies (already in requirements.txt)
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Environment Configuration

Create a `.env` file in project root:

```bash
# API Configuration
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Logging
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=sqlite:///db/vehicle_classifier.db
```

Load environment variables:

```python
# In your code
import os
from dotenv import load_dotenv

load_dotenv()
secret_key = os.getenv("SECRET_KEY")
```

## Docker Deployment

### Quick Start

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Check service status
docker-compose ps

# Stop services
docker-compose down

# Remove volumes (warning: deletes data)
docker-compose down -v
```

### Service Health

Check service health:

```bash
# API health check
curl http://localhost:8000/health

# Redis health check
docker-compose exec redis redis-cli ping
```

### Docker Compose Structure

```yaml
version: '3.9'
services:
  redis:
    # Redis cache service
    # Port: 6379
    # Health check: redis-cli ping
    
  app:
    # FastAPI application
    # Port: 8000
    # Depends on: redis (healthy)
    # Non-root user: appuser
```

### Volume Mounts

Persistent data directories:

```
./uploads/          # Uploaded images
./logs/             # Application logs
./checkpoints/      # Model checkpoints
./db/               # SQLite database
```

These are automatically created and mounted in docker-compose.yml.

### Environment Variables in Production

Update `docker-compose.yml` before deployment:

```yaml
environment:
  - SECRET_KEY=your-strong-production-secret
  - CORS_ORIGINS=https://example.com,https://api.example.com
  - TRUSTED_HOSTS=example.com,api.example.com,app
  - REDIS_HOST=redis-production
  - LOG_LEVEL=INFO
  - DATABASE_URL=sqlite:///db/vehicle_classifier.db
```

## Production Deployment

### Prerequisites for Production

- Linux server (Ubuntu 20.04 LTS or later recommended)
- Docker & Docker Compose installed
- SSL/TLS certificate (for HTTPS)
- Reverse proxy (nginx recommended)
- Backup storage (for database)
- Monitoring tools (Prometheus, Grafana optional)

### Architecture Overview

```
Client
  ↓
Nginx Reverse Proxy (HTTPS)
  ↓
FastAPI Application (port 8000)
  ├── SQLite Database (/app/db/)
  └── Redis Cache (port 6379)
```

### Nginx Configuration

Create `nginx.conf`:

```nginx
upstream vehicle_api {
    server app:8000;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy Configuration
    location / {
        proxy_pass http://vehicle_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://vehicle_api;
        access_log off;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

### Docker Compose Production Setup

Update `docker-compose.yml` for production:

```yaml
version: '3.9'

services:
  redis:
    image: redis:7-alpine
    container_name: vehicle-classifier-redis
    # Don't expose port publicly
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --requirepass your_redis_password
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: always

  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: vehicle-classifier-api
    expose:
      - "8000"  # Don't expose publicly, use reverse proxy
    environment:
      - SECRET_KEY=${SECRET_KEY}  # Use .env file
      - CORS_ORIGINS=https://example.com
      - TRUSTED_HOSTS=api.example.com
      - REDIS_HOST=redis
      - REDIS_PASSWORD=your_redis_password
      - LOG_LEVEL=WARNING
      - DATABASE_URL=sqlite:///db/vehicle_classifier.db
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./db:/app/db
    depends_on:
      redis:
        condition: service_healthy
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: vehicle-classifier-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - /etc/ssl/certs:/etc/ssl/certs
      - /etc/ssl/private:/etc/ssl/private
    depends_on:
      - app
    restart: always

volumes:
  redis_data:
```

### Database Backup Strategy

Automated daily backups:

```bash
# Create backup script: backup.sh
#!/bin/bash
BACKUP_DIR="/backups/vehicle-classifier"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
cp /app/db/vehicle_classifier.db $BACKUP_DIR/vehicle_classifier_$TIMESTAMP.db.gz

# Keep only last 7 days
find $BACKUP_DIR -name "*.db.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/vehicle_classifier_$TIMESTAMP.db.gz"
```

Schedule with cron:

```bash
# Add to crontab (crontab -e)
0 2 * * * /path/to/backup.sh  # Daily at 2 AM
```

## Monitoring & Observability

### Health Check Endpoint

Check API health:

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456",
  "redis_connected": true,
  "database_healthy": true,
  "models_loaded": true
}
```

### Prometheus Metrics

Access metrics endpoint (requires admin authentication):

```bash
# First, get auth token
TOKEN=$(curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin_user","password":"adminpass"}' \
  | jq -r '.access_token')

# Get metrics
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/metrics
```

Metrics include:

- `vehicle_api_requests_total` - Total requests by endpoint
- `vehicle_api_request_duration_seconds` - Request latency histogram
- `vehicle_api_classification_latency_ms` - Classification time
- `vehicle_api_errors_total` - Error count by type
- `vehicle_api_cache_hits_total` - Cache hits
- `vehicle_api_cache_misses_total` - Cache misses
- `vehicle_api_redis_connections` - Active Redis connections

### Structured Logging

Logs are written to `logs/` directory:

```
logs/
├── api.log              # API server logs
├── training.log         # Model training logs
├── evaluation.log       # Model evaluation logs
└── error.log           # Error logs only
```

Log format: JSON (for easy parsing)

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "level": "INFO",
  "logger": "vehicle_api",
  "message": "Classification successful",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user": "testuser",
  "endpoint": "/api/vehicle/classify",
  "duration_ms": 245.3
}
```

### Audit Logging

Query audit logs:

```bash
# Get Python shell
python

# Query audit logs
from src.core.database import Database
db = Database()

# Get all actions for a user
logs = db.get_audit_logs(user_id="testuser")
for log in logs:
    print(f"{log['created_at']}: {log['action']} - {log['resource']}")

# Get classification history
classifications = db.get_classifications(user_id="testuser", limit=50)
for c in classifications:
    print(f"File: {c['image_path']}, Confidence: {c['confidence']:.2%}")
```

## Troubleshooting

### API Won't Start

```bash
# Check for port conflicts
lsof -i :8000

# Check logs
docker-compose logs app

# Verify dependencies installed
pip list | grep -E "fastapi|uvicorn"
```

### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping

# Check logs
docker-compose logs redis

# Check Redis configuration
docker-compose exec redis redis-cli CONFIG GET "*"
```

### High Memory Usage

```bash
# Check process memory
docker stats

# Clear old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clear old uploads
find uploads/ -type f -mtime +7 -delete

# Restart services
docker-compose restart
```

### Slow Requests

```bash
# Check request latency from logs
grep "duration_ms" logs/api.log | sort -t: -k3 -rn | head -10

# Check classification latency
grep "Classification successful" logs/api.log | tail -10

# Check Redis response times
docker-compose exec redis redis-cli --latency
```

### Database Size Growing

```bash
# Check database size
ls -lh db/vehicle_classifier.db

# Get statistics
python -c "from src.core.database import Database; db = Database(); print(db.get_statistics())"

# Archive old classifications
# (Implement based on retention policy)
```

## Performance Tuning

### API Performance

Adjust FastAPI workers:

```bash
# In docker-compose.yml:
CMD ["uvicorn", "src.api.app:app", 
     "--host", "0.0.0.0", 
     "--port", "8000",
     "--workers", "4",        # Number of worker processes
     "--loop", "uvloop"]      # Fast event loop
```

### Redis Performance

Optimize Redis for caching:

```bash
# In docker-compose.yml redis service:
command: redis-server 
  --maxmemory 512mb
  --maxmemory-policy allkeys-lru  # Evict keys when full
  --appendonly yes                 # Persistence
  --appendfsync everysec          # Balance performance/durability
```

### Database Performance

Create SQLite indexes:

```python
from src.core.database import Database

db = Database()
db.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_classifications_user_id 
    ON classifications(user_id)
""")
db.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_audit_log_user_id 
    ON audit_log(user_id)
""")
db.conn.commit()
```

### Model Performance

Load models more efficiently:

```python
# Already implemented in VehicleClassificationAPI
# Models are cached after first load
# Use singleton pattern for model instances

from src.api.service import VehicleClassificationAPI

api = VehicleClassificationAPI()  # Models loaded once
# Subsequent calls reuse loaded models
```

## Scaling Strategies

### Horizontal Scaling (Multiple Instances)

With load balancer (nginx):

```
Load Balancer (nginx)
├── API Instance 1 (port 8001)
├── API Instance 2 (port 8002)
└── API Instance 3 (port 8003)
     ↓
Shared Redis (central caching)
Shared Database (central storage)
```

### Vertical Scaling (More Resources)

Increase resources in docker-compose:

```yaml
services:
  app:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### Caching Strategy

Optimize Redis caching:

```python
# Cache expensive operations
from src.core.redis_client import get_redis_client
import json

redis = get_redis_client()

# Cache model metadata (rarely changes)
metadata_key = "model:metadata"
metadata = redis.get(metadata_key)
if not metadata:
    metadata = api.get_model_metadata()
    redis.setex(metadata_key, 86400, json.dumps(metadata))  # 24 hours
```

## Support & Monitoring

### Log Aggregation

For production, use centralized logging:

```python
# Example: Send to ELK Stack or Splunk
import logging
from pythonjsonlogger import jsonlogger

handler = logging.FileHandler('logs/api.log')
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
```

### Alerting

Set up alerts for:

- High error rate (> 1% errors)
- High latency (> 5s average)
- Redis connection failures
- Database disk space > 90%
- Memory usage > 80%
- CPU usage > 85%

### Status Page

Monitor with:

- Prometheus + Grafana (metrics visualization)
- Datadog / New Relic (APM)
- Sentry (error tracking)
- ELK Stack (log aggregation)

See [SECURITY.md](SECURITY.md) for security-specific deployment checklist.
