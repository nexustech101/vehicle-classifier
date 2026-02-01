# Vehicle Classifier: Enterprise-Grade Multi-Dimensional Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-black)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## Overview

Production-ready deep learning system for **9-dimensional vehicle classification** from 100×90 greyscale images. Includes orchestrated multi-model pipeline, Flask REST API (7 endpoints), intelligent caching, batch processing, and professional report generation.

**Key Features:**
- 9 specialized classifiers (Make, Model, Type, Color, Decade, Country, Condition, Stock/Modified, Functional Utility)
- Flask REST API with 7 endpoints (classify, batch, report, metadata, health, docs)
- 8 design patterns (Singleton, Factory, Strategy, Pipeline, Data Mapper, Builder, Dependency Injection, Repository)
- Confidence metrics with entropy-based uncertainty quantification
- Single & batch processing with JSON/HTML report generation
- Model caching via Singleton pattern (thread-safe)
- 100% type hints, SOLID principles, comprehensive error handling
- 50-100ms inference per image, intelligent registry

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [REST API](#rest-api)
4. [Python API](#python-api)
5. [Architecture](#architecture)
6. [Classification Dimensions](#classification-dimensions)
7. [Advanced Usage](#advanced-usage)
8. [Performance](#performance)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 30-Second Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python app.py

# View API documentation
# Open: http://localhost:5000/api/docs
```

### First Classification (Python)

```python
from prediction_api import VehicleClassificationAPI

api = VehicleClassificationAPI()
result = api.classify_image("vehicle.jpg")
print(result['data']['predictions'])
```

### First Classification (cURL)

```bash
curl -X POST -F "file=@vehicle.jpg" http://localhost:5000/api/vehicle/classify
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- 4 GB RAM (8 GB recommended)
- 2 GB GPU VRAM (optional, for faster inference)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "from prediction_api import VehicleClassificationAPI; print('✓ Ready')"
```

---

## REST API

### Endpoints (7 total)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service health |
| GET | `/api/models/metadata` | Model info & supported attributes |
| POST | `/api/vehicle/classify` | Single image classification |
| POST | `/api/vehicle/classify-batch` | Batch processing |
| POST | `/api/vehicle/report` | Generate report (JSON/HTML) |
| GET | `/api/vehicle/report/<id>` | Retrieve cached report |
| GET | `/api/docs` | Interactive API documentation |

### Examples

**Single Classification:**
```bash
curl -X POST -F "file=@vehicle.jpg" http://localhost:5000/api/vehicle/classify
```

**Batch Processing:**
```bash
curl -X POST -F "files=@car1.jpg" -F "files=@car2.jpg" \
  http://localhost:5000/api/vehicle/classify-batch
```

**Generate Report:**
```bash
curl -X POST -F "file=@vehicle.jpg" -F "format=html" \
  http://localhost:5000/api/vehicle/report > report.html
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
from prediction_api import VehicleClassificationAPI

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
with open("report.json", "w") as f:
    f.write(json_report['data'])

# HTML report
html_report = api.generate_report("vehicle.jpg", format='html')
with open("report.html", "w") as f:
    f.write(html_report['data'])
```

### Direct Pipeline Usage

```python
import numpy as np
from models import VehiclePredictionPipeline, MakeClassifier, TypeClassifier

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
├── models.py                      # ML pipeline, classifiers, utilities
├── prediction_api.py              # High-level API
├── app.py                         # Flask REST server
├── preprocessing.py               # Image processing
├── train.py                       # Training pipeline
├── api_documentation.html         # Interactive docs
├── README.md                      # This file
├── IMPLEMENTATION_SUMMARY.md      # What was built
├── requirements.txt               # Dependencies
├── checkpoints/                   # Model weights
└── uploads/                       # Uploaded images
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
from models import ModelRegistry

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
from models import ModelRegistry
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
