# Vehicle Classifier: Enterprise-Grade Multi-Dimensional Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Redis](https://img.shields.io/badge/Redis-7.0%2B-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## Overview

Production-ready deep learning system for **9-dimensional vehicle classification** from 100×90 greyscale images. Built with FastAPI, Redis caching, Docker containerization, and comprehensive logging for enterprise deployment.

**Key Features:**
- 🚀 **FastAPI** - Async REST API with auto-generated OpenAPI docs
- 🎯 **9 Specialized Classifiers** - Make, Model, Type, Color, Decade, Country, Condition, Stock/Modified, Functional Utility  
- 💾 **Redis Caching** - Distributed caching with regional usage tracking
- 🐳 **Docker Compose** - Production-ready containerization with auto-healthchecks
- 📊 **Structured Logging** - JSON-formatted logs for observability (training, evaluation, API)
- ⚡ **Transfer Learning** - EfficientNetB0/ResNet50 backbones with regularization
- 🔧 **8 Design Patterns** - Singleton, Factory, Strategy, Pipeline, Data Mapper, Builder, DI, Repository
- 📈 **Professional Reports** - JSON/HTML classification reports with confidence metrics
- 🎨 **Batch Processing** - Multi-file classification with summary statistics
- 🏥 **Health Checks** - Liveness/readiness probes for Kubernetes-ready deployments

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [Docker Deployment](#docker-deployment)
4. [Authentication & User Management](#authentication--user-management)
5. [REST API](#rest-api)
6. [Python API](#python-api)
7. [Project Structure](#project-structure)
8. [Architecture](#architecture)
9. [Logging & Monitoring](#logging--monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### FastAPI Server (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server (FastAPI)
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Navigate to http://localhost:8000/docs for interactive API documentation
```

### Docker Compose (Production)

```bash
# Build and start containerized stack (FastAPI + Redis)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### First Classification (Python)

```python
from src.api.service import VehicleClassificationAPI

api = VehicleClassificationAPI()
result = api.classify_image("vehicle.jpg")
print(result['data']['predictions'])
```

### First Classification (FastAPI)

```bash
curl -X POST -F "file=@vehicle.jpg" http://localhost:8000/api/vehicle/classify
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10+
- **Package Manager:** pip or conda
- **RAM:** 4 GB minimum (8 GB recommended)
- **Docker & Docker Compose:** For containerized deployment
- **Git:** For cloning the repository

### Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-classifier.git
cd vehicle-classifier

# View available branches
git branch -a
```

### Local Installation (Development)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

### Docker Installation (Production)

See [Docker Deployment](#docker-deployment) section below for complete containerized setup.

---

## Docker Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/vehicle-classifier.git
cd vehicle-classifier
```

### Step 2: Build Docker Image (Optional)

The docker-compose will automatically build the image, but you can pre-build it:

```bash
# Build image with tag
docker build -t vehicle-classifier:latest .

# Verify image was created
docker images | grep vehicle-classifier
```

### Step 3: Start Services with Docker Compose

```bash
# Start FastAPI + Redis stack in background
docker-compose up -d

# Verify all services are running
docker-compose ps
```

**Expected Output:**
```
NAME                           IMAGE                    STATUS
vehicle-classifier-api        vehicle-classifier:latest   Up 5 seconds
vehicle-classifier-redis      redis:7-alpine              Up 5 seconds (healthy)
```

### Step 4: Access the Application

- **FastAPI Swagger UI:** http://localhost:8000/docs
- **FastAPI ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### Docker Compose Configuration

The `docker-compose.yml` includes:

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| API | vehicle-classifier-api | 8000 | FastAPI application |
| Redis | vehicle-classifier-redis | 6379 | Distributed caching |

**Key Configuration Details:**

```yaml
services:
  api:
    build: .                           # Build from Dockerfile
    ports:
      - "8000:8000"                   # API port mapping
    volumes:
      - ./src:/app/src                # Live code updates (development)
      - ./uploads:/app/uploads        # Persistent uploads
      - ./logs:/app/logs              # Persistent logs
    environment:
      - SECRET_KEY=your-secret-key-change-in-production
      - ACCESS_TOKEN_EXPIRE_MINUTES=30
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      redis:
        condition: service_healthy

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

### Environment Variables

Create a `.env` file in the project root (optional, docker-compose has defaults):

```bash
# API Settings
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
LOG_LEVEL=INFO

# Redis Settings
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
```

### Common Docker Commands

```bash
# View real-time logs from API
docker-compose logs -f api

# View logs from a specific service
docker-compose logs -f redis

# Check container health and resource usage
docker stats

# Execute command in running container
docker-compose exec api python -c "from src.core.database import Database; print('✓ Database OK')"

# Stop all services
docker-compose stop

# Stop and remove all containers
docker-compose down

# Remove containers AND volumes (clean slate)
docker-compose down -v

# Restart services
docker-compose restart

# Rebuild image and restart
docker-compose up -d --build
```

### Troubleshooting Docker

**Issue: Port 8000 already in use**
```bash
# Find what's using port 8000
netstat -ano | findstr :8000

# Change port in docker-compose.yml:
# ports:
#   - "8001:8000"
```

**Issue: "Redis not available - caching disabled"**
```bash
# Verify Redis is healthy
docker-compose logs redis

# Check Redis connection
docker-compose exec api redis-cli -h redis ping
```

**Issue: Changes to code not reflecting**
```bash
# Ensure volume mount is in docker-compose.yml:
# volumes:
#   - ./src:/app/src

# Restart containers
docker-compose restart api
```

**Issue: Database errors**
```bash
# Check if db directory exists
docker-compose exec api ls -la db/

# Clear database and restart (WARNING: loses all data)
docker-compose down -v
docker-compose up -d
```

---

## Authentication & User Management

The Vehicle Classifier API uses **JWT (JSON Web Token)** authentication with role-based access control.

### Initial Setup (Bootstrap Admin)

On first startup, the system automatically creates a default admin account:

- **Username:** `admin`
- **Password:** `admin`
- **Role:** `admin`

**⚠️ IMPORTANT:** Change the admin password immediately after first login.

### User Roles

| Role | Permissions |
|------|-------------|
| **user** | Access classification endpoints, view own profile |
| **admin** | Full access + user management (create, list, update roles, delete) |

### Authentication Endpoints

#### 1. Login (Get Token)

**Endpoint:** `POST /api/v2/auth/token`

**Request:**
```json
{
  "username": "admin",
  "password": "admin"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Using curl:**
```bash
curl -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

#### 2. Register New User

**Endpoint:** `POST /api/v2/auth/register`

**Request:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure_password_123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Using curl:**
```bash
curl -X POST http://localhost:8000/api/v2/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password_123"
  }'
```

### User Management Endpoints (Admin Only)

#### 1. Get Current User Profile

**Endpoint:** `GET /api/v2/users/me`

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "username": "admin",
  "email": "admin@localhost",
  "role": "admin",
  "is_active": true,
  "created_at": "2026-02-02T20:11:49Z",
  "last_login": "2026-02-02T20:15:30Z"
}
```

**Using curl:**
```bash
curl -X GET http://localhost:8000/api/v2/users/me \
  -H "Authorization: Bearer {access_token}"
```

#### 2. List All Users (Admin Only)

**Endpoint:** `GET /api/v2/users?skip=0&limit=10`

**Headers:**
```
Authorization: Bearer {admin_token}
```

**Response:**
```json
{
  "users": [
    {
      "username": "admin",
      "email": "admin@localhost",
      "role": "admin",
      "is_active": true,
      "created_at": "2026-02-02T20:11:49Z",
      "last_login": "2026-02-02T20:15:30Z"
    },
    {
      "username": "john_doe",
      "email": "john@example.com",
      "role": "user",
      "is_active": true,
      "created_at": "2026-02-02T20:12:00Z",
      "last_login": "2026-02-02T20:13:45Z"
    }
  ],
  "total": 2
}
```

**Using curl:**
```bash
curl -X GET "http://localhost:8000/api/v2/users?skip=0&limit=10" \
  -H "Authorization: Bearer {admin_token}"
```

#### 3. Update User Role (Admin Only)

**Endpoint:** `PATCH /api/v2/users/{username}/role`

**Headers:**
```
Authorization: Bearer {admin_token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "role": "admin"
}
```

**Valid roles:** `"user"` or `"admin"`

**Response:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "role": "admin",
  "is_active": true,
  "created_at": "2026-02-02T20:12:00Z",
  "last_login": "2026-02-02T20:13:45Z"
}
```

**Using curl:**
```bash
curl -X PATCH http://localhost:8000/api/v2/users/john_doe/role \
  -H "Authorization: Bearer {admin_token}" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'
```

#### 4. Deactivate User (Admin Only)

**Endpoint:** `PATCH /api/v2/users/{username}/status`

**Headers:**
```
Authorization: Bearer {admin_token}
Content-Type: application/json
```

**Request Body:**
```json
{
  "is_active": false
}
```

**Using curl:**
```bash
curl -X PATCH http://localhost:8000/api/v2/users/john_doe/status \
  -H "Authorization: Bearer {admin_token}" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

#### 5. Delete User (Admin Only)

**Endpoint:** `DELETE /api/v2/users/{username}`

**Headers:**
```
Authorization: Bearer {admin_token}
```

**Response:**
```json
{
  "message": "User deleted successfully"
}
```

**Using curl:**
```bash
curl -X DELETE http://localhost:8000/api/v2/users/john_doe \
  -H "Authorization: Bearer {admin_token}"
```

### Using Authentication in Requests

All protected endpoints require the `Authorization` header with a valid JWT token:

```bash
# Get token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}' | jq -r '.access_token')

# Use token in API requests
curl -X GET http://localhost:8000/api/v2/users/me \
  -H "Authorization: Bearer $TOKEN"
```

### Token Details

- **Type:** JWT (JSON Web Token)
- **Algorithm:** HS256 (HMAC SHA-256)
- **Expiration:** 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)
- **Claims:** 
  - `sub` - Username
  - `role` - User role (admin/user)
  - `exp` - Expiration timestamp

---

## REST API

### Endpoints (13 total)

| Method | Endpoint | Purpose | Auth Required |
|--------|----------|---------|---|
| GET | `/health` | Service health & status | No |
| GET | `/api/models/metadata` | Model info & attributes | No |
| **POST** | **`/api/v2/auth/token`** | **Login (get JWT token)** | **No** |
| **POST** | **`/api/v2/auth/register`** | **Register new user** | **No** |
| **GET** | **`/api/v2/users/me`** | **Get current user profile** | **Yes** |
| **GET** | **`/api/v2/users`** | **List all users** | **Yes (Admin)** |
| **PATCH** | **`/api/v2/users/{username}/role`** | **Update user role** | **Yes (Admin)** |
| **PATCH** | **`/api/v2/users/{username}/status`** | **Activate/deactivate user** | **Yes (Admin)** |
| **DELETE** | **`/api/v2/users/{username}`** | **Delete user** | **Yes (Admin)** |
| POST | `/api/vehicle/classify` | Single image classification | No |
| POST | `/api/vehicle/classify-batch` | Batch processing | No |
| POST | `/api/vehicle/report` | Generate JSON/HTML report | No |
| GET | `/api/vehicle/report/{id}` | Retrieve cached report | No |
| GET | `/docs` | Interactive OpenAPI documentation | No |

### Examples

**Single Classification:**
```bash
curl -X POST -F "file=@vehicle.jpg" http://localhost:8000/api/vehicle/classify
```

**Batch Processing:**
```bash
curl -X POST -F "files=@car1.jpg" -F "files=@car2.jpg" \
  http://localhost:8000/api/vehicle/classify-batch
```

**Generate HTML Report:**
```bash
curl -X POST -F "file=@vehicle.jpg" -F "format=html" \
  http://localhost:8000/api/vehicle/report > report.html
```

**Response Example:**
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
# JSON report
json_report = api.generate_report("vehicle.jpg", vehicle_id="VEH_001", format='json')
report_data = json_report['data']

# HTML report
html_report = api.generate_report("vehicle.jpg", format='html')
with open("report.html", "w") as f:
    f.write(html_report['data'])
```

---

## Logging & Monitoring

### Log Files

Structured JSON logs are written to `./logs/`:

| File | Purpose |
|------|---------|
| `api.log` | REST API requests, responses, timings |
| `training.log` | Model training progress & metrics |
| `evaluation.log` | Model evaluation results |

### View Logs

```bash
# API logs (real-time)
tail -f logs/api.log

# Training logs
tail -f logs/training.log

# Parse JSON logs (pretty-print)
cat logs/api.log | jq '.' | less
```

### Log Structure

All logs are in structured JSON format:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "api",
  "module": "app",
  "function": "classify_single",
  "line": 145,
  "message": "Classification successful for vehicle.jpg (45.23ms)"
}
```

### Monitoring (Docker)

```bash
# View container resource usage
docker stats vehicle-classifier-api

# Check Redis connection
docker-compose exec redis redis-cli ping

# Monitor API health
docker-compose exec app curl http://localhost:8000/health
```

---

## Project Structure

```
vehicle-classifier/
├── src/                                 # Source code
│   ├── __init__.py
│   ├── api/                            # REST API & Web Service
│   │   ├── __init__.py
│   │   ├── app.py                      # FastAPI application (13 endpoints)
│   │   ├── auth.py                     # JWT auth & user verification
│   │   ├── service.py                  # High-level VehicleClassificationAPI
│   │   ├── cache.py                    # Redis caching utilities
│   │   └── logging_config.py           # Structured logging setup
│   │
│   ├── core/                           # Core Functionality
│   │   ├── __init__.py
│   │   ├── database.py                 # SQLite with users table & CRUD
│   │   ├── redis_client.py             # Redis connection manager
│   │   ├── security.py                 # Input validation & security
│   │   ├── errors.py                   # Custom exception classes
│   │   └── monitoring.py               # Metrics & request logging
│   │
│   ├── models/                         # ML Classifiers
│   │   ├── __init__.py
│   │   └── classifiers.py              # 9 transfer learning classifiers
│   │
│   ├── training/                       # Model Training
│   │   ├── __init__.py
│   │   └── train.py                    # Training pipeline orchestrator
│   │
│   └── preprocessing/                  # Data Preparation
│       ├── __init__.py
│       ├── processor.py                # Image preprocessing
│       └── utils.py                    # Utility functions
│
├── tests/                              # Unit & Integration Tests
│   ├── __init__.py
│   ├── conftest.py                     # pytest fixtures & configuration
│   ├── test_api.py                     # API endpoint tests
│   ├── test_auth.py                    # Authentication tests
│   ├── test_security.py                # Security tests
│   └── test_monitoring.py              # Monitoring tests
│
├── checkpoints/                        # Trained Model Weights
│   └── (model files)
│
├── db/                                 # SQLite Database
│   └── reports.db                      # Users table + audit logs
│
├── logs/                               # Application Logs
│   ├── api.log                         # API request/response logs
│   ├── training.log                    # Model training logs
│   └── evaluation.log                  # Evaluation results
│
├── uploads/                            # User-uploaded Images
│   └── (temporary storage)
│
├── docker-compose.yml                  # Container orchestration (FastAPI + Redis)
├── Dockerfile                          # FastAPI container image
├── main.py                             # Application entry point
├── requirements.txt                    # Python dependencies
├── verify_setup.py                     # Setup verification script
├── README.md                           # This file
├── INDEX.md                            # Detailed feature index
├── SECURITY.md                         # Security guidelines
└── api_documentation.html              # Generated API docs (offline)
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/api/` | REST API server with 13 endpoints + JWT authentication |
| `src/core/` | Database, Redis, security, and monitoring |
| `src/models/` | 9 ML classifiers (transfer learning) |
| `src/training/` | Model training pipeline |
| `src/preprocessing/` | Image preprocessing utilities |
| `tests/` | Comprehensive test suite |
| `checkpoints/` | Trained model weights (TensorFlow format) |
| `db/` | SQLite database with users table |
| `logs/` | Structured JSON logs |
| `uploads/` | Temporary upload storage |

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

### Key Components

- **FastAPI** - Async REST API with auto-generated documentation
- **JWT Authentication** - Stateless auth with role-based access control
- **SQLite Database** - User accounts with audit logging
- **Redis** - Distributed caching with TTL support
- **VehiclePredictionPipeline** - Multi-model orchestration
- **ModelRegistry** - Singleton pattern for model management
- **DataAugmentation** - In-model data augmentation during training

### Authentication Flow

```
User Registration/Login
        ↓
Credentials → auth.py (verify password + hash check)
        ↓
JWT Token Created (HS256, 30-min expiration)
        ↓
Token Included in Authorization Header
        ↓
Protected Endpoints → verify_token() → get_current_user()
        ↓
Role-Based Access Control (require_role middleware)
        ↓
Access Granted/Denied
```

### Direct Pipeline Usage

```python
import numpy as np
from src.models import VehiclePredictionPipeline, MakeClassifier, TypeClassifier

pipeline = VehiclePredictionPipeline()
models = {
    'make': MakeClassifier(),
    'type': TypeClassifier(),
}
pipeline.initialize_models(models)

# Predict
image = np.random.randn(100, 90, 1)
result = pipeline.predict_single(image)
print(f"Confidence: {result.overall_confidence:.1%}")
```

### High-Level Pipeline

```
INPUT (100×90 Greyscale Image)
    ↓
IMAGE PREPROCESSING (Auto-normalize, resize)
    ↓
VEHICLEPREDICTIONPIPELINE (Main Orchestrator)
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
CONFIDENCE METRICS (Entropy-based uncertainty)
    ↓
PREDICTION MAPPER (Human-readable output)
    ↓
REPORT GENERATOR (JSON/HTML reports)
    ↓
OUTPUT (VehicleClassificationResult)
```

### Design Patterns

| Pattern | Usage | Benefit |
|---------|-------|---------|
| **Singleton** | ModelRegistry | Thread-safe caching |
| **Factory** | Model instantiation | Dynamic creation |
| **Strategy** | Training approaches | Flexible algorithms |
| **Pipeline** | Multi-model orchestration | Coordination |
| **Data Mapper** | Output conversion | Decoupling |
| **Builder** | Report generation | Incremental construction |
| **Dependency Injection** | Component initialization | Testability |
| **Repository** | Model lifecycle | Centralized management |

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
| Condition | 1 | Regression | 0.0-1.0 score |
| Stock/Modified | 2 | Binary | Stock, Modified |
| Functional Utility | 8 | Multi-class | Passenger, Commercial, Emergency |

---

## Advanced Usage

### Confidence Metrics

```python
result = api.classify_image("vehicle.jpg")
predictions = result['data']['predictions']

for name, pred in predictions.items():
    metrics = pred.get('confidence_metrics')
    if metrics:
        print(f"{name}:")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Uncertainty: {metrics['uncertainty']:.2f}")
        print(f"  Is Confident: {metrics['is_confident']}")
```

### Custom Image Requirements

- **Format:** JPEG, PNG, BMP, GIF (auto-detected)
- **Size:** Any (auto-resized to 100×90)
- **Color:** Any (auto-converted to greyscale)
- **Max Size:** 16 MB per image

### Model Registry & Caching

```python
from src.models import ModelRegistry

registry = ModelRegistry()
cached_models = registry.get_cached_model_names()
print(cached_models)

# Clear cache if needed
registry.clear_cache()
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t vehicle-classifier .
docker run -p 5000:5000 vehicle-classifier
```

### Gunicorn + Nginx

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```


## Troubleshooting

### Models Not Loading

```python
from src.models import ModelRegistry
registry = ModelRegistry()
print(registry.get_cached_model_names())
```

### Out of Memory

```python
ModelRegistry().clear_cache()  # Free all cached models
```

### Low Confidence Predictions

```python
result = api.classify_image("vehicle.jpg")
for name, pred in result['data']['predictions'].items():
    probs = pred['raw_probabilities']
    top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"{name}: {top_3}")
```

### API Not Responding

```bash
curl http://localhost:8000/health
```

### Auth Token Invalid

```bash
# Token expired? Get a new one:
curl -X POST http://localhost:8000/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

### Permission Denied

```bash
# Make sure you're using an admin token for admin endpoints:
# List users (admin only):
curl -X GET http://localhost:8000/api/v2/users \
  -H "Authorization: Bearer {ADMIN_TOKEN}"
```

---

## Next Steps

1. **Clone & Run:** Follow the [Docker Deployment](#docker-deployment) section
2. **First Login:** Use default admin credentials (`admin`/`admin`)
3. **Change Password:** Update admin password immediately
4. **Create Users:** Register additional users via `/api/v2/auth/register`
5. **Manage Roles:** Use `/api/v2/users/{username}/role` to promote to admin
6. **Classify Vehicles:** Use `/api/vehicle/classify` endpoint
7. **API Docs:** Visit http://localhost:8000/docs for interactive documentation

---

**Status:** Production Ready ✓  
**Version:** 2.1  
**Last Updated:** February 2, 2026  
**Features:** JWT Auth, User Management, Vehicle Classification, Redis Caching, Docker Deployment
**Last Updated:** January 31, 2026  
**Version:** 2.0
