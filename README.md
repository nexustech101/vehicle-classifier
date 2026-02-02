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
2. [Installation](#installation)
3. [Docker Deployment](#docker-deployment)
4. [REST API](#rest-api)
5. [Python API](#python-api)
6. [Architecture](#architecture)
7. [Logging & Monitoring](#logging--monitoring)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)

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

## Installation

### Prerequisites
- Python 3.10+
- pip or conda
- 4 GB RAM (8 GB recommended)
- Docker & Docker Compose (optional, for containerized deployment)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.api.service import VehicleClassificationAPI; print('✓ Ready')"
```

---

## Docker Deployment

### Build Custom Image

```bash
# Build Docker image
docker build -t vehicle-classifier:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/uploads:/app/uploads vehicle-classifier:latest
```

### Docker Compose Stack

**Complete production stack with FastAPI + Redis:**

```bash
# Start services
docker-compose up -d

# Verify services running
docker-compose ps

# View app logs
docker-compose logs -f app

# View Redis logs
docker-compose logs -f redis

# Stop all services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

**Environment Variables:**

Set in `docker-compose.yml`:
- `REDIS_HOST` - Redis host (default: redis)
- `REDIS_PORT` - Redis port (default: 6379)
- `LOG_LEVEL` - Logging level (default: INFO)

---

## REST API

### Endpoints (7 total)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service health & status |
| GET | `/api/models/metadata` | Model info & attributes |
| POST | `/api/vehicle/classify` | Single image classification |
| POST | `/api/vehicle/classify-batch` | Batch processing |
| POST | `/api/vehicle/report` | Generate JSON/HTML report |
| GET | `/api/vehicle/report/{id}` | Retrieve cached report |
| GET | `/docs` | Interactive OpenAPI documentation |

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

## Architecture

### Directory Structure

```
vehicle-classifier/
├── src/
│   ├── api/
│   │   ├── app.py                    # FastAPI application
│   │   ├── service.py                # High-level API service
│   │   ├── cache.py                  # Redis caching utilities
│   │   └── logging_config.py         # Structured logging setup
│   ├── models/
│   │   ├── classifiers.py            # 9 ML classifiers (transfer learning)
│   │   └── registry.py               # Model registry & singleton
│   ├── training/
│   │   └── train.py                  # Training pipeline orchestrator
│   └── preprocessing/
│       ├── processor.py              # Image preprocessing
│       └── utils.py                  # Utility functions
├── docker-compose.yml                # Container orchestration
├── Dockerfile                        # FastAPI container image
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

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
- **Redis** - Distributed caching with TTL support
- **VehiclePredictionPipeline** - Multi-model orchestration
- **ModelRegistry** - Singleton pattern for model management
- **DataAugmentation** - In-model data augmentation during training

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

---

## Architecture

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

### Project Structure

```
vehicle-classifier/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifiers.py         # Transfer learning models & inference
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py               # Training pipeline orchestrator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # Flask REST server
│   │   └── service.py             # VehicleClassificationAPI
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── processor.py           # Image preprocessing
│   │   └── utils.py               # Utility functions
│   └── __init__.py
├── main.py                        # Application entry point
├── checkpoints/                   # Model weights
├── logs/                          # Training logs
├── api_documentation.html         # Interactive docs
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

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
curl http://localhost:5000/health
```

---

## Next Steps

1. **Quick Start:** Run `python app.py` and visit `/api/docs`
2. **Understand Architecture:** Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
3. **API Examples:** Check `/api/docs` endpoint
4. **Production Deployment:** Follow deployment guide above
5. **Troubleshoot:** See troubleshooting section

---

**Status:** Production Ready ✓  
**Last Updated:** January 31, 2026  
**Version:** 2.0
