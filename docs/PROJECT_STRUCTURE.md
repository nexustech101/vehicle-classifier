# Reorganized Project Structure

## File Migration Map

| Old Location | New Location | Module | Purpose |
|---|---|---|---|
| `models.py` | `src/models/classifiers.py` | ML Models | Transfer learning architectures & inference pipeline |
| `train.py` | `src/training/train.py` | Training | Training pipeline orchestrator |
| `app.py` | `src/api/app.py` | REST API | Flask server & endpoints |
| `prediction_api.py` | `src/api/service.py` | API Service | High-level prediction interface |
| `preprocessing.py` | `src/preprocessing/processor.py` | Preprocessing | Image loading & preparation |
| `utils.py` | `src/preprocessing/utils.py` | Utilities | Helper functions |
| `main.py` | `main.py` | Entry Point | Application entry point |

## Directory Structure

```
vehicle-classifier/
├── src/                              # Main source code
│   ├── __init__.py
│   │
│   ├── models/                       # ML models (classifiers, pipeline, reporting)
│   │   ├── __init__.py               # Exports all model classes
│   │   └── classifiers.py            # Transfer learning models (EfficientNet/ResNet)
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   └── train.py                  # Training pipeline, orchestrator
│   │
│   ├── api/                          # REST API & service layer
│   │   ├── __init__.py               # Exports VehicleClassificationAPI
│   │   ├── app.py                    # Flask server with 7 endpoints
│   │   └── service.py                # High-level API (VehicleClassificationAPI)
│   │
│   └── preprocessing/                # Data handling
│       ├── __init__.py               # Exports preprocessors and utils
│       ├── processor.py              # ImagePreprocessor, BatchProcessor, DataAugmentation
│       └── utils.py                  # Image utilities (load, resize, flatten, crop)
│
├── checkpoints/                      # Trained model files (auto-created)
├── logs/                             # Training logs (auto-created)
│
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── IMPLEMENTATION_SUMMARY.md         # Feature overview
```

## Import Changes

### Before (Flat Structure)
```python
from models import MakeClassifier
from preprocessing import ImagePreprocessor
from prediction_api import VehicleClassificationAPI
from train import TrainingOrchestrator
```

### After (Organized Structure)
```python
from src.models import MakeClassifier
from src.preprocessing import ImagePreprocessor
from src.api import VehicleClassificationAPI
from src.training import TrainingOrchestrator
```

## Benefits

✅ **Separation of Concerns** - Models, training, API, and preprocessing in distinct modules
✅ **Scalability** - Easy to add new models or API endpoints without clutter
✅ **Maintainability** - Clear purpose for each directory
✅ **Package Structure** - Proper Python package with `__init__.py` files for clean imports
✅ **Professional Layout** - Industry-standard ML project organization

## Architecture

```
┌─────────────────────────────────────────┐
│  REST API (src/api/app.py)              │
│  ├── POST /api/vehicle/classify         │
│  ├── POST /api/vehicle/classify-batch   │
│  ├── POST /api/vehicle/report           │
│  └── GET /health, /docs, /metadata      │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  API Service (src/api/service.py)       │
│  VehicleClassificationAPI               │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│  ML Pipeline (src/models/classifiers.py)│
│  ├── VehiclePredictionPipeline          │
│  ├── Individual Classifiers             │
│  └── Report Generation                  │
└─────────────┬───────────────────────────┘
              │
              └──────┬───────────────────┐
                     │                   │
        ┌────────────▼──────┐   ┌───────▼──────────┐
        │  Training         │   │  Preprocessing   │
        │  (src/training)   │   │  (src/preprocessing)
        │                   │   │                  │
        │  TrainingConfig   │   │  ImageProcessor  │
        │  ModelTrainer     │   │  BatchProcessor  │
        │  Orchestrator     │   │  Utilities       │
        └───────────────────┘   └──────────────────┘
```
