# Quick Start Guide

## 5-Minute Setup

### Option 1: Development (Fastest)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SECRET_KEY="dev-key-change-in-production"
export DATABASE_URL="sqlite:///db/vehicle_classifier.db"

# Create directories
mkdir -p logs uploads db

# Start API
uvicorn src.api.app:app --reload --port 8000

# In browser, visit: http://localhost:8000/docs
```

### Option 2: Docker (Recommended)

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f app

# API is at: http://localhost:8000/docs
```

---

## First Request

### 1. Get Access Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass"
  }'

# Save the access_token from response
# Example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 2. Classify an Image

```bash
# Download a test image first (or use your own vehicle image)
curl -X POST -F "file=@vehicle.jpg" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://localhost:8000/api/vehicle/classify

# Response includes predictions for:
# - Make (Toyota, Honda, Ford, etc.)
# - Model (Civic, Accord, etc.)
# - Type (Sedan, SUV, Truck, etc.)
# - Color, Decade, Country, Condition, Stock/Modified, Utility
```

### 3. View API Documentation

Open browser to: **http://localhost:8000/docs**

- Try out endpoints interactively
- See request/response schemas
- View all available parameters

---

## Common Tasks

### Get Help

```bash
# View setup verification
python verify_setup.py

# Check all dependencies
pip list | grep -E "fastapi|tensorflow|redis|pytest|prometheus"

# View API docs
curl http://localhost:8000/docs

# Check health
curl http://localhost:8000/health
```

### Test Your Setup

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

### View Logs

```bash
# Development
tail -f logs/api.log

# Docker
docker-compose logs -f app
```

### Access Admin Features

```bash
# Get admin token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin_user",
    "password": "adminpass"
  }'

# View metrics (admin only)
curl -H "Authorization: Bearer ADMIN_TOKEN" \
  http://localhost:8000/metrics | head -20

# Useful metrics:
# - vehicle_api_requests_total - Total requests
# - vehicle_api_request_duration_seconds - Latency
# - vehicle_api_errors_total - Error count
# - vehicle_api_cache_hits_total - Cache efficiency
```

### Batch Process Images

```bash
# Classify multiple images
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/vehicle/classify-batch
```

### Generate a Report

```bash
# Create detailed classification report
curl -X POST \
  -F "file=@vehicle.jpg" \
  -F "format=json" \
  -F "vehicle_id=CAR_001" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/vehicle/report

# Get report (JSON)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/vehicle/report/CAR_001

# Get report (HTML) - add ?format=html
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/vehicle/report/CAR_001?format=html" \
  > report.html
```

---

## Troubleshooting

### Port Already in Use

```bash
# Change port
uvicorn src.api.app:app --port 8001

# Or kill existing process
lsof -i :8000
kill <PID>
```

### Redis Not Available

Redis is optional. API will work without it (no caching).

```bash
# Start Redis (requires redis-server installed)
redis-server

# Or in Docker
docker-compose up redis -d
```

### Database Issues

```bash
# Reset database
rm db/vehicle_classifier.db
# It will recreate on first run

# View database
sqlite3 db/vehicle_classifier.db
sqlite> .tables  # See all tables
sqlite> SELECT COUNT(*) FROM classifications;
sqlite> .quit
```

### Tests Failing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_api.py::TestHealthEndpoint -v
```

---

## Environment Variables

**Must Set for Production:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | dev key | JWT signing key |
| `CORS_ORIGINS` | localhost | Allowed origins |
| `TRUSTED_HOSTS` | localhost | Allowed hostnames |
| `DATABASE_URL` | sqlite local | Database connection |

**Optional:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | Token expiration time |
| `REDIS_HOST` | localhost | Redis server |
| `REDIS_PORT` | 6379 | Redis port |
| `LOG_LEVEL` | INFO | Logging verbosity |

**Set Variables:**

```bash
# Option 1: Environment
export SECRET_KEY="your-secret-key"
export CORS_ORIGINS="https://example.com"

# Option 2: .env file (for development)
echo "SECRET_KEY=your-secret-key" > .env
echo "CORS_ORIGINS=https://example.com" >> .env

# Option 3: Docker compose
# Edit docker-compose.yml environment section
```

---

## API Endpoints

**Public (No Auth):**
- `GET /` - API info
- `GET /health` - Health check
- `POST /auth/token` - Get access token

**User (Auth Required):**
- `POST /api/vehicle/classify` - Classify single image
- `POST /api/vehicle/classify-batch` - Classify multiple images
- `POST /api/vehicle/report` - Generate report
- `GET /api/vehicle/report/{id}` - Get report
- `GET /api/models/metadata` - Get model info

**Admin (Auth + Admin Role):**
- `GET /metrics` - Prometheus metrics

---

## Performance Tips

1. **Batch Process** - Classify multiple images together
2. **Cache Enabled** - First request loads models, subsequent are faster
3. **Use HTTPS** - In production, enable TLS/SSL
4. **Monitor Performance** - Check `/metrics` endpoint regularly
5. **Review Logs** - Check logs for slow requests

---

## Security Reminders

⚠️ **Before Production:**

1. Change `SECRET_KEY` to random string
2. Update `CORS_ORIGINS` to your domain
3. Change default credentials in `src/api/auth.py`
4. Set `LOG_LEVEL=WARNING` (less verbose)
5. Enable HTTPS with nginx reverse proxy
6. Set up database backups

See [SECURITY.md](SECURITY.md) for complete checklist.

---

## Getting Help

- **API Docs**: http://localhost:8000/docs
- **Setup Issues**: Run `python verify_setup.py`
- **Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Security**: See [SECURITY.md](SECURITY.md)
- **Implementation**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Run Tests**: `pytest tests/ -v`

---

## Next Steps

1. ✅ Verify setup: `python verify_setup.py`
2. ✅ Run tests: `pytest tests/ -v`
3. ✅ Start API: `uvicorn src.api.app:app --reload`
4. ✅ Try classification: Open `/docs` in browser
5. ✅ Review security: Read [SECURITY.md](SECURITY.md)
6. ✅ Deploy: Follow [DEPLOYMENT.md](DEPLOYMENT.md)

---

**API is ready!** 🚀

Visit http://localhost:8000/docs to get started.
