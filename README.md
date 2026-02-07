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

- **9 ML classifiers** — EfficientNetB0/ResNet50 transfer learning across make, model, type, color, decade, country, condition, stock/modified, and functional utility
- **FastAPI** — 13 REST endpoints with auto-generated Swagger/ReDoc docs
- **JWT auth** — Role-based access control with Argon2 password hashing
- **Redis caching** — Distributed cache with TTL and regional tracking
- **Docker Compose** — One-command deployment (FastAPI + Redis) with health checks
- **Structured logging** — JSON logs with full audit trail
- **Reports** — JSON and HTML vehicle classification reports with batch support

---

## Quick Start

```bash
# Docker (recommended)
git clone https://github.com/yourusername/vehicle-classifier.git
cd vehicle-classifier
docker-compose up -d
# → http://localhost:8000/docs

# Local development
python -m venv venv && venv\Scripts\activate   # Windows (use source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

```bash
docker-compose up -d          # Start FastAPI (:8000) + Redis (:6379)
docker-compose logs -f api    # Tail logs
docker-compose up -d --build  # Rebuild after code changes
docker-compose down -v        # Tear down (removes volumes)
```

Access: [Swagger UI](http://localhost:8000/docs) · [ReDoc](http://localhost:8000/redoc) · [Health](http://localhost:8000/health)

---

## Environment Variables

Create a `.env` in the project root (docker-compose provides defaults):

```bash
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
LOG_LEVEL=INFO
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
```

---

## Authentication & User Management

JWT authentication with RBAC and Argon2 hashing. A default `admin`/`admin` account is created on first startup — **change it immediately**.

Roles: **user** (classification, own profile) · **admin** (full access + user management)

```bash
# Login
curl -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Register
curl -X POST http://localhost:8000/api/v2/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "john_doe", "email": "john@example.com", "password": "secure_password_123"}'

# Use token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' | jq -r '.access_token')

curl http://localhost:8000/api/v2/users/me -H "Authorization: Bearer $TOKEN"
```

Tokens are HS256 JWTs with 30-minute expiry. Claims: `sub` (username), `role`, `exp`.

---

## REST API

### Public

```
GET  /health                          # Service health
GET  /api/models/metadata             # Model info & supported attributes
POST /api/v2/auth/token               # Login → JWT
POST /api/v2/auth/register            # Register new user
```

### Classification

```
POST /api/vehicle/classify            # Single image
POST /api/vehicle/classify-batch      # Multiple images
POST /api/vehicle/report              # Generate JSON/HTML report
GET  /api/vehicle/report/{id}         # Retrieve cached report
```

### User Management (authenticated)

```
GET    /api/v2/users/me               # Own profile (user+)
GET    /api/v2/users                  # List all users (admin)
PATCH  /api/v2/users/{username}/role   # Update role (admin)
PATCH  /api/v2/users/{username}/status # Activate/deactivate (admin)
DELETE /api/v2/users/{username}        # Delete user (admin)
```

### Examples

```bash
curl -X POST -F "file=@vehicle.jpg" http://localhost:8000/api/vehicle/classify

curl -X POST -F "files=@car1.jpg" -F "files=@car2.jpg" \
  http://localhost:8000/api/vehicle/classify-batch

curl -X POST -F "file=@vehicle.jpg" -F "format=html" \
  http://localhost:8000/api/vehicle/report > report.html
```

### Response

```json
{
  "status": "success",
  "data": {
    "predictions": {
      "make": { "value": "Toyota", "confidence": 0.92, "confidence_metrics": { "uncertainty": 0.15, "is_confident": true } }
    },
    "overall_confidence": 0.87
  }
}
```

---

## Python API

```python
from src.api import VehicleClassificationAPI

api = VehicleClassificationAPI()

# Classify
result = api.classify_image("vehicle.jpg")
for name, pred in result['data']['predictions'].items():
    print(f"{pred['attribute']}: {pred['value']} ({pred['confidence']:.1%})")

# Batch
results = api.classify_batch(["car1.jpg", "car2.jpg"])

# Reports
api.generate_report("vehicle.jpg", vehicle_id="VEH_001", format='json')
api.generate_report("vehicle.jpg", format='html')
```

### Direct Pipeline

```python
from src.models import VehiclePredictionPipeline, MakeClassifier, TypeClassifier

pipeline = VehiclePredictionPipeline()
pipeline.initialize_models({'make': MakeClassifier(), 'type': TypeClassifier()})
result = pipeline.predict_single(image)  # expects 100×90 greyscale ndarray
```

---

## Security

Argon2 password hashing · JWT (HS256, 30-min expiry) · RBAC · filename sanitization & path traversal detection · magic-number image validation (PNG/JPG/BMP/GIF, 16 MB max) · CORS origin validation (no wildcards) · Redis-backed rate limiting · full audit logging to SQLite · non-root Docker execution (`appuser`)

Response headers: `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`, `Strict-Transport-Security`, `X-XSS-Protection`, `Referrer-Policy`

### Production Checklist

- [ ] Rotate `SECRET_KEY` to a strong random value
- [ ] Set `CORS_ORIGINS` and `TRUSTED_HOSTS` to production domains
- [ ] Enable HTTPS via reverse proxy (nginx)
- [ ] Configure database backups and log aggregation

---

## Architecture

```
Image Upload → FastAPI → Preprocessing → 9 Model Ensemble → Aggregate
                                                                ↓
                                              Cache (Redis) + JSON/HTML Response
```

```
Credentials → Argon2 verify → JWT (HS256, 30-min) → Authorization header → RBAC middleware
```

### ML Pipeline

```
100×90 greyscale input
  → Make (100)  Model (~150)  Type (12)  Color (10)  Decade (70)
    Country (70)  Condition (regression)  Stock/Modified (binary)  Utility (8)
  → Confidence metrics → Prediction mapper → Report generator
```

Design patterns: Singleton (ModelRegistry), Factory, Strategy, Pipeline, Data Mapper, Builder, DI, Repository.

---

## Classification Dimensions

```
Make              100 classes     multi-class    Toyota, Honda, Ford, ...
Model            ~150 classes     multi-class    Camry, Accord, F-150, ...
Type               12 classes     multi-class    Sedan, SUV, Truck, Van, ...
Color              10 classes     multi-class    Black, White, Silver, Red, ...
Decade             70 classes     multi-class    1980s, 1990s, 2000s, ...
Country            70 classes     multi-class    USA, Japan, Germany, ...
Condition           1 output      regression     0.0–1.0 score
Stock/Modified      2 classes     binary         Stock, Modified
Functional Utility  8 classes     multi-class    Passenger, Commercial, Emergency, ...
```

---

## Logging & Monitoring

Structured JSON logs written to `./logs/` — `api.log`, `training.log`, `evaluation.log`.

```bash
tail -f logs/api.log               # Stream API logs
cat logs/api.log | jq '.' | less   # Pretty-print
```

---

## Project Structure

```
vehicle-classifier/
├── src/
│   ├── api/                        # FastAPI app, auth, service, cache, logging
│   ├── core/                       # Database, Redis, security, errors, monitoring
│   ├── models/                     # 9 transfer learning classifiers
│   ├── training/                   # Training pipeline
│   └── preprocessing/              # Image preprocessing & utils
├── tests/                          # pytest suite (API, auth, security, monitoring)
├── checkpoints/                    # Trained model weights
├── db/                             # SQLite database
├── logs/                           # Application logs
├── uploads/                        # Temporary image uploads
├── docker-compose.yml
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

---

## Troubleshooting

- **Models not loading** — `ModelRegistry().get_cached_model_names()` to inspect, `.clear_cache()` to reset
- **Out of memory** — `ModelRegistry().clear_cache()`
- **API not responding** — `curl http://localhost:8000/health`
- **Token expired** — re-authenticate via `POST /api/v2/auth/token`
- **Permission denied** — verify you're using an admin token for admin endpoints

---

**Version 2.1** · Production Ready · February 2026
