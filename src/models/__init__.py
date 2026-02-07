"""Vehicle classification models package."""

from .classifiers import (
    # Enums
    BackboneType,
    ConfidenceLevel,
    # Base classes
    VehicleClassificationCNNBase,
    VehicleRegressionBase,
    # Concrete classifiers
    MakeClassifier,
    ModelClassifier,
    TypeClassifier,
    ColorClassifier,
    DecadeClassifier,
    CountryClassifier,
    ConditionClassifier,
    StockOrModedClassifier,
    FunctionalUtilityClassifier,
    # Data structures
    ConfidenceMetrics,
    VehicleAttributePrediction,
    VehicleClassificationResult,
    VehicleClassificationReport,
    # Utility classes
    PredictionConfidenceAnalyzer,
    PredictionMapper,
    ReportGenerator,
    ModelRegistry,
    # Pipeline
    VehiclePredictionPipeline,
)

__all__ = [
    'BackboneType',
    'ConfidenceLevel',
    'VehicleClassificationCNNBase',
    'VehicleRegressionBase',
    'MakeClassifier',
    'ModelClassifier',
    'TypeClassifier',
    'ColorClassifier',
    'DecadeClassifier',
    'CountryClassifier',
    'ConditionClassifier',
    'StockOrModedClassifier',
    'FunctionalUtilityClassifier',
    'ConfidenceMetrics',
    'VehicleAttributePrediction',
    'VehicleClassificationResult',
    'VehicleClassificationReport',
    'PredictionConfidenceAnalyzer',
    'PredictionMapper',
    'ReportGenerator',
    'ModelRegistry',
    'VehiclePredictionPipeline',
]
