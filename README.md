# Vehicle Classification Pipeline

## Project Overview

**Vehicle Classifier** is a deep learning project designed to perform **multi-dimensional vehicle classification** from images (100×90 greyscale). The system predicts multiple independent and dependent attributes of vehicles through a sophisticated neural network pipeline.

## Objectives

1. **Multi-Attribute Classification**: Classify vehicles across 8 distinct dimensions:
   - **Make** (brand): 100 possible vehicle manufacturers
   - **Model** (conditional on make): ~150 distinct models
   - **Type** (body style): 12 categories (sedan, SUV, truck, etc.)
   - **Color**: 10 distinct vehicle colors
   - **Decade**: Manufacturing era (70 classes)
   - **Country** (origin): Country of manufacture (70 classes)
   - **Condition**: Vehicle condition score (0-1 regression)
   - **Modification Status**: Stock vs. Modified (binary classification)

2. **Robust Architecture**: Build a scalable, maintainable deep learning system that handles:
   - Multi-task learning with shared image features
   - Conditional predictions (e.g., Model depends on Make)
   - Both classification and regression tasks
   - Flexible input handling and preprocessing

3. **Production-Ready Design**: Implement professional software engineering practices:
   - Clean separation of concerns
   - Extensible class hierarchies
   - Model persistence and versioning
   - Orchestrated prediction pipeline

## Design Patterns

### 1. **Base Class / Template Method Pattern**
- `VehicleClassificationCNNBase`: Foundation for all classification models
- `VehicleRegressionBase`: Specialized base for regression/binary tasks
- Subclasses override `_build_model()` for architecture variations
- Shared `fit()`, `predict()`, `evaluate()` logic reduces duplication

### 2. **Strategy Pattern** (Planned)
- Different loss functions and metrics per model type
- Output transformations (e.g., threshold for binary classification)
- Flexible preprocessing strategies per classifier

### 3. **Pipeline/Orchestrator Pattern** (Planned)
- `VehicleClassificationPipeline`: Chains predictions in dependency order
- Routes output from one model as input to dependent models
- Aggregates results into unified vehicle profile
- Enables complex inference workflows

### 4. **Factory Pattern** (Ready for Extension)
- Simple instantiation of new classifiers without modification
- Consistent interface across all model types
- Easy to add new dimensions (e.g., `TransmissionTypeClassifier`)

### 5. **Dependency Injection** (Planned)
- Pipeline accepts configured classifier instances
- Supports model versioning and hot-swapping
- Testable and flexible architecture

### 6. **Registry/Manager Pattern** (Planned)
- `ModelManager`: Central registry for all trained models
- Handles model loading, versioning, and lifecycle
- Enables easy model updates without code changes

## Architecture

```
models.py
├── VehicleClassificationCNNBase (abstract)
│   ├── MakeClassifier
│   ├── TypeClassifier
│   ├── ColorClassifier
│   ├── DecadeClassifier
│   ├── CountryClassifier
│   └── ModelClassifier (dual-input)
│
└── VehicleRegressionBase (abstract)
    ├── ConditionClassifier
    └── StockOrModedClassifier

pipeline.py (Planned)
├── VehicleClassificationPipeline
│   ├── Loads all classifiers
│   ├── Orchestrates prediction flow
│   └── Returns unified vehicle profile

preprocessing.py
├── ImagePreprocessor
├── BatchProcessor
└── DataAugmentation
```

## Current Status

- ✅ Model architecture and inheritance hierarchy
- ✅ Base class patterns for code reuse
- ✅ 8 specialized classifiers (multi-task setup)
- ✅ Conditional input handling (ModelClassifier)
- 🔄 Pipeline orchestrator (in progress)
- 🔄 Configuration management (pending)
- 🔄 Preprocessing module (pending)
- 🔄 Model manager/registry (pending)

## Key Technologies

- **Framework**: Keras/TensorFlow
- **Input**: 100×90×1 greyscale images
- **Output**: Multi-dimensional vehicle classification
- **Design**: Object-oriented with inheritance and composition

## Next Steps

1. Implement `VehicleClassificationPipeline` orchestrator
2. Build `preprocessing.py` module for image handling
3. Create `config.py` for centralized configuration
4. Develop `ModelManager` for unified model lifecycle
5. Add comprehensive unit and integration tests
6. Document API and usage examples
