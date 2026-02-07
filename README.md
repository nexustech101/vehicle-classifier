# 🚗 Vehicle Classifier

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)
![Redis](https://img.shields.io/badge/redis-7--alpine-red.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-ready vehicle classification system powered by **9 deep learning models** with transfer learning (EfficientNetB0/ResNet50). Classifies vehicles across make, model, type, color, decade, country of origin, condition, stock/modified status, and functional utility. Served via a **FastAPI** REST API with JWT authentication, Redis caching, and Docker deployment.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [Authentication & User Management](#authentication--user-management)
- [REST API](#rest-api)
- [Python API](#python-api)
- [Security](#security)
- [Architecture](#architecture)
- [Classification Dimensions](#classification-dimensions)
- [Logging & Monitoring](#logging--monitoring)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Features

| Category | Details |
|----------|---------|
| **ML Models** | 9 classifiers with EfficientNetB0/ResNet50 transfer learning |
| **API** | FastAPI with 13 REST endpoints and interactive Swagger docs |
| **Auth** | JWT authentication with role-based access control (RBAC) |
| **Caching** | Redis distributed caching with TTL and regional tracking |
| **Deployment** | Docker Compose stack (FastAPI + Redis) with health checks |
| **Logging** | Structured JSON logs with audit trail |
| **Reports** | JSON and HTML vehicle classification reports |
| **Batch** | Multi-image batch classification |

---

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/yourusername/vehicle-classifier.git
cd vehicle-classifier
docker-compose up -d
```

The API is available at **http://localhost:8000/docs**.

### Local Development

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- Redis (optional for local dev — caching degrades gracefully)

### Local Setup

```bash
git clone https://github.com/yourusername/vehicle-classifier.git
cd vehicle-classifier

python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Docker Deployment

### Start Services

```bash
docker-compose up -d
docker-compose ps
```

**Expected output:**

| Container | Image | Port | Purpose |
|-----------|-------|------|---------|
| vehicle-classifier-api | vehicle-classifier:latest | 8000 | FastAPI application |
| vehicle-classifier-redis | redis:7-alpine | 6379 | Distributed caching |

### Access Points

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### Common Commands

```bash
docker-compose logs -f api          # Real-time API logs
docker-compose restart api          # Restart API
docker-compose up -d --build        # Rebuild and restart
docker-compose down                 # Stop all services
docker-compose down -v              # Stop and remove volumes
docker stats                        # Resource usage
```

### Troubleshooting Docker

| Issue | Fix |
|-------|-----|
| Port 8000 in use | Change port mapping in `docker-compose.yml` to `"8001:8000"` |
| Redis not available | Run `docker-compose logs redis` to check health |
| Code changes not reflecting | Ensure volume mount `./src:/app/src` is set, then `docker-compose restart api` |
| Database errors | Run `docker-compose down -v && docker-compose up -d` (⚠️ loses data) |

---

## Environment Variables

Create a `.env` file in the project root (docker-compose provides defaults):

```bash
# API / JWT
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
LOG_LEVEL=INFO

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
```

---

## Authentication & User Management

The API uses **JWT (JSON Web Token)** authentication with role-based access control and **Argon2** password hashing.

### Default Admin Account

On first startup a bootstrap admin is created:

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin` |
| Role | `admin` |

> **⚠️ Change the admin password immediately after first login.**

### User Roles

| Role | Permissions |
|------|-------------|
| **user** | Classification endpoints, own profile |
| **admin** | Full access + user management (create, list, update, delete) |

### Login

```bash
curl -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Register

```bash
curl -X POST http://localhost:8000/api/v2/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "email": "john@example.com", "password": "secure_password_123"}'
```

### Using Tokens

All protected endpoints require the `Authorization` header:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' | jq -r '.access_token')

curl -X GET http://localhost:8000/api/v2/users/me \
  -H "Authorization: Bearer $TOKEN"
```

### Token Details

| Property | Value |
|----------|-------|
| Algorithm | HS256 (HMAC SHA-256) |
| Expiration | 30 minutes (configurable) |
| Claims | `sub` (username), `role`, `exp` (expiration) |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v2/users/me` | Current user profile |
| GET | `/api/v2/users?skip=0&limit=10` | List all users (admin) |
| PATCH | `/api/v2/users/{username}/role` | Update role (admin) |
| PATCH | `/api/v2/users/{username}/status` | Activate/deactivate (admin) |
| DELETE | `/api/v2/users/{username}` | Delete user (admin) |

---

## REST API

### Endpoints (13 total)

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/health` | Service health & status | No |
| GET | `/api/models/metadata` | Model info & attributes | No |
| POST | `/api/v2/auth/token` | Login (get JWT) | No |
| POST | `/api/v2/auth/register` | Register new user | No |
| GET | `/api/v2/users/me` | Current user profile | Yes |
| GET | `/api/v2/users` | List all users | Admin |
| PATCH | `/api/v2/users/{username}/role` | Update user role | Admin |
| PATCH | `/api/v2/users/{username}/status` | Activate/deactivate | Admin |
| DELETE | `/api/v2/users/{username}` | Delete user | Admin |
| POST | `/api/vehicle/classify` | Single image classification | No |
| POST | `/api/vehicle/classify-batch` | Batch processing | No |
| POST | `/api/vehicle/report` | Generate JSON/HTML report | No |
| GET | `/api/vehicle/report/{id}` | Retrieve cached report | No |

### Examples

```bash
# Single classification
curl -X POST -F "file=@vehicle.jpg" http://localhost:8000/api/vehicle/classify

# Batch processing
curl -X POST -F "files=@car1.jpg" -F "files=@car2.jpg" \
  http://localhost:8000/api/vehicle/classify-batch

# Generate HTML report
curl -X POST -F "file=@vehicle.jpg" -F "format=html" \
  http://localhost:8000/api/vehicle/report > report.html
```

### Response Example

```json
{
  "status": "success",
  "timestamp": "2026-01-31T14:30:22.123",
  "data": {
    "predictions": {
      "make": {
        "attribute": "Make",
        "value": "Toyota",
        "confidence": 0.92,
        "confidence_metrics": {
          "uncertainty": 0.15,
          "is_confident": true
        }
      }
    },
    "overall_confidence": 0.87
  }
}
```

---

## Python API

### Single Image

```python
from src.api import VehicleClassificationAPI

api = VehicleClassificationAPI()
result = api.classify_image("vehicle.jpg")

if result['status'] == 'success':
    for name, pred in result['data']['predictions'].items():
        print(f"{pred['attribute']}: {pred['value']} ({pred['confidence']:.1%})")
```

### Batch Processing

```python
images = ["car1.jpg", "car2.jpg", "car3.jpg"]
results = api.classify_batch(images)
print(f"Success rate: {results['summary']['successful']}/{results['summary']['total_images']}")
```

### Report Generation

```python
json_report = api.generate_report("vehicle.jpg", vehicle_id="VEH_001", format='json')
html_report = api.generate_report("vehicle.jpg", format='html')
with open("report.html", "w") as f:
    f.write(html_report['data'])
```

### Direct Pipeline Usage

```python
import numpy as np
from src.models import VehiclePredictionPipeline, MakeClassifier, TypeClassifier

pipeline = VehiclePredictionPipeline()
pipeline.initialize_models({
    'make': MakeClassifier(),
    'type': TypeClassifier(),
})

image = np.random.randn(100, 90, 1)
result = pipeline.predict_single(image)
print(f"Confidence: {result.overall_confidence:.1%}")
```

---

## Security

### Overview

| Feature | Implementation |
|---------|---------------|
| Password Hashing | Argon2 (GPU-resistant) |
| Authentication | JWT tokens (HS256, 30-min expiry) |
| Authorization | Role-based access control (user/admin) |
| Input Validation | Filename sanitization, path traversal detection, magic number checks |
| Security Headers | 6 standard headers on all responses |
| CORS | Origin validation (no wildcards) |
| Rate Limiting | IP + endpoint key generation for Redis-backed limiting |
| Audit Logging | All user actions logged with IP, timestamp, and resource |
| Docker | Non-root user execution (`appuser`) |

### Security Headers

All API responses include:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy: default-src 'self'`
- `Strict-Transport-Security: max-age=31536000` (on HTTPS)
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`

### Image Validation

Uploaded images are validated by:
- File magic numbers (PNG, JPG, BMP, GIF)
- File extension matching
- File size limits (16 MB max)

### Audit Logging

All user actions are logged to SQLite with user ID, action type, resource, IP address, and timestamp.

```python
from src.core.database import Database
db = Database()
logs = db.get_audit_logs(user_id="testuser", limit=100)
```

### Production Checklist

- [ ] Change `SECRET_KEY` to a strong random value
- [ ] Update `CORS_ORIGINS` to production domains
- [ ] Set `TRUSTED_HOSTS` to production hostnames
- [ ] Enable HTTPS/TLS via reverse proxy (nginx)
- [ ] Configure database backups
- [ ] Set up log aggregation and alerting
- [ ] Test rate limiting configuration
- [ ] Verify audit logging

---

## Architecture

### Data Flow

```
Image Upload → FastAPI → Load Image → Predict (9 Models)
                                         ↓
                               Aggregate Results
                                    ↓
                            Cache (Redis) + Response
```

### Authentication Flow

```
Credentials → auth.py → Verify Password (Argon2)
                              ↓
                    JWT Token (HS256, 30-min)
                              ↓
               Authorization Header → verify_token()
                              ↓
                   Role-Based Access Control
```

### ML Pipeline

```
INPUT (100×90 Greyscale Image)
    ↓
IMAGE PREPROCESSING (auto-normalize, resize)
    ↓
VEHICLE PREDICTION PIPELINE
    ├── Make Classifier (100 classes)
    ├── Model Classifier (dual-input, ~150 classes)
    ├── Type Classifier (12 classes)
    ├── Color Classifier (10 classes)
    ├── Decade Classifier (70 classes)
    ├── Country Classifier (70 classes)
    ├── Condition Classifier (regression)
    ├── Stock/Modified Classifier (binary)
    └── Functional Utility Classifier (8 classes)
    ↓
CONFIDENCE METRICS → PREDICTION MAPPER → REPORT GENERATOR
    ↓
OUTPUT (VehicleClassificationResult)
```

### Design Patterns

| Pattern | Usage |
|---------|-------|
| Singleton | ModelRegistry — thread-safe caching |
| Factory | Dynamic model instantiation |
| Strategy | Flexible training approaches |
| Pipeline | Multi-model orchestration |
| Data Mapper | Prediction output conversion |
| Builder | Incremental report generation |
| Dependency Injection | Component initialization |
| Repository | Centralized model lifecycle |

---

## Classification Dimensions

| Attribute | Classes | Type | Examples |
|-----------|---------|------|----------|
| Make | 100 | Multi-class | Toyota, Honda, Ford |
| Model | ~150 | Multi-class | Camry, Accord, F-150 |
| Type | 12 | Multi-class | Sedan, SUV, Truck, Van |
| Color | 10 | Multi-class | Black, White, Silver, Red |
| Decade | 70 | Multi-class | 1980s, 1990s, 2000s |
| Country | 70 | Multi-class | USA, Japan, Germany |
| Condition | 1 | Regression | 0.0–1.0 score |
| Stock/Modified | 2 | Binary | Stock, Modified |
| Functional Utility | 8 | Multi-class | Passenger, Commercial, Emergency |

---

## Logging & Monitoring

### Log Files

Structured JSON logs in `./logs/`:

| File | Purpose |
|------|---------|
| `api.log` | API requests, responses, timings |
| `training.log` | Model training progress & metrics |
| `evaluation.log` | Model evaluation results |

### Log Format

```json
{
  "timestamp": "2026-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "api",
  "module": "app",
  "function": "classify_single",
  "line": 145,
  "message": "Classification successful for vehicle.jpg (45.23ms)"
}
```

### Commands

```bash
tail -f logs/api.log                    # Real-time API logs
cat logs/api.log | jq '.' | less        # Pretty-print JSON logs
docker stats vehicle-classifier-api     # Container resource usage
```

---

## Project Structure

```
vehicle-classifier/
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI application (13 endpoints)
│   │   ├── auth.py                 # JWT auth & user verification
│   │   ├── service.py              # VehicleClassificationAPI
│   │   ├── cache.py                # Redis caching utilities
│   │   └── logging_config.py       # Structured logging setup
│   ├── core/
│   │   ├── database.py             # SQLite + users table + CRUD
│   │   ├── redis_client.py         # Redis connection manager
│   │   ├── security.py             # Input validation & security
│   │   ├── errors.py               # Custom exceptions
│   │   └── monitoring.py           # Metrics & request logging
│   ├── models/
│   │   └── classifiers.py          # 9 transfer learning classifiers
│   ├── training/
│   │   └── train.py                # Training pipeline orchestrator
│   └── preprocessing/
│       ├── processor.py            # Image preprocessing
│       └── utils.py                # Utility functions
├── tests/
│   ├── conftest.py                 # pytest fixtures
│   ├── test_api.py                 # API endpoint tests
│   ├── test_auth.py                # Authentication tests
│   ├── test_security.py            # Security tests
│   └── test_monitoring.py          # Monitoring tests
├── checkpoints/                    # Trained model weights
├── db/                             # SQLite database
├── logs/                           # Application logs
├── uploads/                        # Temporary image uploads
├── docker-compose.yml              # Container orchestration
├── Dockerfile                      # Container image
├── main.py                         # Application entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Models not loading | `ModelRegistry().get_cached_model_names()` to check, `.clear_cache()` to reset |
| Out of memory | `ModelRegistry().clear_cache()` to free cached models |
| API not responding | `curl http://localhost:8000/health` to verify |
| Token expired / invalid | Re-authenticate: `POST /api/v2/auth/token` |
| Permission denied | Ensure you're using an admin token for admin endpoints |
| Low confidence | Inspect `raw_probabilities` in the prediction response |

---

**Version:** 2.1 · **Status:** Production Ready · **Last Updated:** February 2026
