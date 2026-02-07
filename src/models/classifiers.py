"""
Vehicle Classification ML Pipeline
===================================

Production-grade deep learning pipeline for multi-dimensional vehicle classification.
Implements transfer learning with EfficientNetB0/ResNet50 backbones for 9 vehicle attributes.

Version: 2.0.0

Architecture:
- Base Classes: VehicleClassificationCNNBase, VehicleRegressionBase
- 9 Specialized Classifiers: Make, Model, Type, Color, Decade, Country, Condition, Stock/Modified, Functional Utility
- Design Patterns: Singleton (ModelRegistry), Factory (VehiclePredictionPipeline), Strategy (Analyzers)
- Data Structures: Dataclasses with JSON/HTML serialization
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum

import numpy as np
import keras
from keras import layers, Model, callbacks, optimizers, losses, metrics
from keras.layers import RandomBrightness
from keras.applications import EfficientNetB0, ResNet50


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class BackboneType(Enum):
    """Supported pretrained backbone architectures."""
    EFFICIENTNET = "efficientnet"
    RESNET = "resnet"


class ConfidenceLevel(Enum):
    """Confidence level thresholds for predictions."""
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ConfidenceMetrics:
    """
    Confidence and uncertainty metrics for predictions.
    
    Attributes:
        confidence: Highest probability (0.0-1.0)
        rank_1_accuracy: Top-1 prediction probability
        rank_2_accuracy: Top-2 prediction probability
        uncertainty: Shannon entropy normalized to [0, 1]
        is_confident: Whether confidence >= 0.8
    """
    confidence: float
    rank_1_accuracy: float
    rank_2_accuracy: float
    uncertainty: float
    is_confident: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @property
    def confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence >= ConfidenceLevel.HIGH.value:
            return "HIGH"
        elif self.confidence >= ConfidenceLevel.MEDIUM.value:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class VehicleAttributePrediction:
    """
    Prediction result for a single vehicle attribute.
    
    Attributes:
        attribute_name: Name of the attribute (e.g., "Make", "Color")
        predicted_value: Human-readable prediction value
        confidence: Prediction confidence (0.0-1.0)
        raw_probabilities: Top-5 class probabilities with labels
        confidence_metrics: Optional detailed confidence analysis
    """
    attribute_name: str
    predicted_value: Union[str, int, float]
    confidence: float
    raw_probabilities: Dict[str, float]
    confidence_metrics: Optional[ConfidenceMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'attribute_name': self.attribute_name,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'raw_probabilities': self.raw_probabilities,
        }
        if self.confidence_metrics:
            result['confidence_metrics'] = self.confidence_metrics.to_dict()
        return result


@dataclass
class VehicleClassificationResult:
    """
    Complete classification result for a vehicle image.
    
    Attributes:
        image_path: Source image path
        timestamp: ISO format timestamp
        predictions: Dictionary of attribute predictions
        overall_confidence: Average confidence across all predictions
        processing_time_ms: End-to-end processing time in milliseconds
        model_version: Model version for deployment tracking
    """
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    predictions: Dict[str, VehicleAttributePrediction] = field(default_factory=dict)
    overall_confidence: float = 0.0
    processing_time_ms: float = 0.0
    model_version: str = "2.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'image_path': self.image_path,
            'timestamp': self.timestamp,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'overall_confidence': self.overall_confidence,
            'processing_time_ms': self.processing_time_ms,
            'model_version': self.model_version,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class VehicleClassificationReport:
    """
    Professional report with classification results and recommendations.
    
    Attributes:
        vehicle_id: Unique vehicle identifier
        report_date: ISO format date
        classification_results: Complete classification results
        summary: Executive summary of findings
        confidence_score: Aggregated confidence score
        recommendations: List of actionable insights
        metadata: Extensible metadata dictionary
    """
    vehicle_id: str
    report_date: str = field(default_factory=lambda: datetime.now().isoformat())
    classification_results: Optional[VehicleClassificationResult] = None
    summary: str = ""
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'vehicle_id': self.vehicle_id,
            'report_date': self.report_date,
            'classification_results': self.classification_results.to_dict() if self.classification_results else None,
            'summary': self.summary,
            'confidence_score': self.confidence_score,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_html(self) -> str:
        """
        Generate professional HTML report.
        
        Returns:
            HTML string with embedded CSS
        """
        predictions = self.classification_results.predictions if self.classification_results else {}
        
        # Build prediction table rows
        prediction_rows = ""
        for attr_name, pred in predictions.items():
            confidence_class = "high" if pred.confidence >= 0.8 else "medium" if pred.confidence >= 0.6 else "low"
            
            # Top probabilities
            top_probs_html = "<br>".join([
                f"{label}: {prob:.2%}" 
                for label, prob in list(pred.raw_probabilities.items())[:3]
            ])
            
            prediction_rows += f"""
            <tr>
                <td class="attribute">{attr_name}</td>
                <td class="value">{pred.predicted_value}</td>
                <td class="confidence {confidence_class}">{pred.confidence:.2%}</td>
                <td class="probabilities">{top_probs_html}</td>
            </tr>
            """
        
        # Build recommendations list
        recommendations_html = "".join([
            f"<li>{rec}</li>" for rec in self.recommendations
        ])
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vehicle Classification Report - {self.vehicle_id}</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: #f5f5f5;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                header {{
                    border-bottom: 3px solid #2563eb;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                
                h1 {{
                    color: #1e40af;
                    font-size: 28px;
                    margin-bottom: 10px;
                }}
                
                .meta-info {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8fafc;
                    border-radius: 6px;
                }}
                
                .meta-item {{
                    display: flex;
                    flex-direction: column;
                }}
                
                .meta-label {{
                    font-size: 12px;
                    text-transform: uppercase;
                    color: #64748b;
                    font-weight: 600;
                    margin-bottom: 4px;
                }}
                
                .meta-value {{
                    font-size: 16px;
                    color: #0f172a;
                    font-weight: 500;
                }}
                
                .summary {{
                    background: #eff6ff;
                    border-left: 4px solid #2563eb;
                    padding: 20px;
                    margin-bottom: 30px;
                    border-radius: 4px;
                }}
                
                .summary h2 {{
                    color: #1e40af;
                    font-size: 18px;
                    margin-bottom: 10px;
                }}
                
                .summary p {{
                    color: #1e3a8a;
                    line-height: 1.7;
                }}
                
                .predictions {{
                    margin-bottom: 30px;
                }}
                
                .predictions h2 {{
                    color: #0f172a;
                    font-size: 20px;
                    margin-bottom: 20px;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }}
                
                thead {{
                    background: #1e40af;
                    color: white;
                }}
                
                th {{
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    font-size: 14px;
                }}
                
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #e2e8f0;
                }}
                
                tr:hover {{
                    background: #f8fafc;
                }}
                
                .attribute {{
                    font-weight: 600;
                    color: #0f172a;
                }}
                
                .value {{
                    color: #1e40af;
                    font-weight: 500;
                }}
                
                .confidence {{
                    font-weight: 600;
                }}
                
                .confidence.high {{
                    color: #059669;
                }}
                
                .confidence.medium {{
                    color: #d97706;
                }}
                
                .confidence.low {{
                    color: #dc2626;
                }}
                
                .probabilities {{
                    font-size: 12px;
                    color: #64748b;
                    line-height: 1.5;
                }}
                
                .recommendations {{
                    background: #fefce8;
                    border-left: 4px solid #eab308;
                    padding: 20px;
                    border-radius: 4px;
                }}
                
                .recommendations h2 {{
                    color: #854d0e;
                    font-size: 18px;
                    margin-bottom: 15px;
                }}
                
                .recommendations ul {{
                    list-style: none;
                    padding-left: 0;
                }}
                
                .recommendations li {{
                    padding: 8px 0;
                    color: #713f12;
                    position: relative;
                    padding-left: 24px;
                }}
                
                .recommendations li:before {{
                    content: "→";
                    position: absolute;
                    left: 0;
                    color: #eab308;
                    font-weight: bold;
                }}
                
                footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #e2e8f0;
                    text-align: center;
                    color: #64748b;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Vehicle Classification Report</h1>
                </header>
                
                <div class="meta-info">
                    <div class="meta-item">
                        <span class="meta-label">Vehicle ID</span>
                        <span class="meta-value">{self.vehicle_id}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Report Date</span>
                        <span class="meta-value">{self.report_date}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Overall Confidence</span>
                        <span class="meta-value">{self.confidence_score:.2%}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">Processing Time</span>
                        <span class="meta-value">{self.classification_results.processing_time_ms:.2f if self.classification_results else 0:.2f} ms</span>
                    </div>
                </div>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>{self.summary}</p>
                </div>
                
                <div class="predictions">
                    <h2>Classification Results</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Attribute</th>
                                <th>Predicted Value</th>
                                <th>Confidence</th>
                                <th>Top Probabilities</th>
                            </tr>
                        </thead>
                        <tbody>
                            {prediction_rows}
                        </tbody>
                    </table>
                </div>
                
                <div class="recommendations">
                    <h2>Recommendations</h2>
                    <ul>
                        {recommendations_html}
                    </ul>
                </div>
                
                <footer>
                    <p>Generated by Vehicle Classification ML Pipeline v{self.classification_results.model_version if self.classification_results else '2.0.0'}</p>
                    <p>Image: {self.classification_results.image_path if self.classification_results else 'N/A'}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_template


# ============================================================================
# BASE CLASSES
# ============================================================================


@keras.saving.register_keras_serializable()
class VehicleClassificationCNNBase(Model):
    """
    Base class for multi-class vehicle classification using transfer learning.
    
    Features:
    - Supports EfficientNetB0 and ResNet50 pretrained backbones
    - Automatic grayscale to RGB conversion
    - Configurable data augmentation pipeline
    - Progressive fine-tuning with backbone freezing control
    - Learning rate scheduling with CosineDecay
    - Production-ready with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        backbone: Backbone architecture ('efficientnet' or 'resnet')
        freeze_backbone: Whether to freeze backbone weights initially
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        **kwargs: Additional arguments passed to parent Model class
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (100, 90, 1),
        num_classes: int = 100,
        backbone: str = 'efficientnet',
        freeze_backbone: bool = True,
        dropout_rate: float = 0.4,
        l2_reg: float = 0.01,
        **kwargs
    ):
        # Extract custom kwargs before passing to parent
        self.batch_size = kwargs.pop('batch_size', 32)
        self.lr = kwargs.pop('lr', 0.001)
        self.epochs = kwargs.pop('epochs', 50)
        self.verbose = kwargs.pop('verbose', 1)
        self.early_stopping_patience = kwargs.pop('early_stopping_patience', 10)
        self.use_augmentation = kwargs.pop('use_augmentation', True)
        
        super().__init__(**kwargs)
        
        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.backbone_type = backbone
        self.freeze_backbone_flag = freeze_backbone
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.is_trained = False
        
        # Build the model
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Build the classification model architecture.
        
        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape_config, name='input_image')
        
        # Normalize to [0, 1] if needed
        x = layers.Rescaling(1./255, name='normalization')(inputs)
        
        # Data augmentation
        x = self._build_augmentation_layers(x)
        
        # Convert grayscale to RGB (pretrained models expect 3 channels)
        x = layers.Concatenate(name='grayscale_to_rgb')([x, x, x])
        
        # Load pretrained backbone
        if self.backbone_type.lower() == 'efficientnet':
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(self.input_shape_config[0], self.input_shape_config[1], 3),
                pooling='avg'
            )
        elif self.backbone_type.lower() in ('resnet', 'resnet50'):
            backbone = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(self.input_shape_config[0], self.input_shape_config[1], 3),
                pooling='avg'
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
        
        # Freeze backbone if specified
        backbone.trainable = not self.freeze_backbone_flag
        
        # Extract features
        x = backbone(x, training=False)
        
        # Classification head
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output layer (num_classes outputs with softmax)
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.name or 'VehicleClassifier')
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    
    def _build_augmentation_layers(self, inputs: layers.Layer) -> layers.Layer:
        """
        Build data augmentation pipeline.
        
        Args:
            inputs: Input layer
            
        Returns:
            Augmented layer
        """
        if not self.use_augmentation:
            return inputs
        
        x = inputs
        
        # Random flip
        x = layers.RandomFlip('horizontal', name='aug_flip')(x)
        
        # Random rotation
        x = layers.RandomRotation(0.1, name='aug_rotation')(x)
        
        # Random zoom
        x = layers.RandomZoom(0.1, name='aug_zoom')(x)
        
        # Random contrast
        x = layers.RandomContrast(0.2, name='aug_contrast')(x)
        
        # Random brightness
        x = RandomBrightness(0.1, name='aug_brightness')(x)
        
        return x
    
    def unfreeze_backbone(self, num_layers_to_freeze: int = 0):
        """
        Unfreeze backbone for fine-tuning.
        
        Args:
            num_layers_to_freeze: Number of layers to keep frozen from the start
        """
        # Find backbone in model
        for layer in self.model.layers:
            if isinstance(layer, (EfficientNetB0, ResNet50)):
                layer.trainable = True
                
                # Freeze first N layers if specified
                if num_layers_to_freeze > 0:
                    for sublayer in layer.layers[:num_layers_to_freeze]:
                        sublayer.trainable = False
                
                logger.info(f"Unfroze backbone with {num_layers_to_freeze} layers frozen")
                break
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
    
    def fit(
        self,
        img_list: np.ndarray,
        class_list: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        val_split: float = 0.2,
        fine_tune_epochs: int = 0,
        checkpoint_path: Optional[str] = None,
        verbose: int = None,
        **kwargs
    ):
        """
        Train the model with optional progressive fine-tuning.
        
        Args:
            img_list: Training images (N, 9000) or (N, 100, 90, 1)
            class_list: Training labels (1-indexed)
            epochs: Number of training epochs (default: self.epochs)
            batch_size: Batch size (default: self.batch_size)
            val_split: Validation split ratio
            fine_tune_epochs: Additional epochs for fine-tuning (with unfrozen backbone)
            checkpoint_path: Path to save model checkpoints
            verbose: Verbosity level (default: self.verbose)
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Training history
        """
        # Use instance defaults if not provided
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        verbose = verbose if verbose is not None else self.verbose
        
        # Reshape if needed
        X = self._prepare_input(img_list)
        
        # Convert 1-indexed to 0-indexed
        y = class_list - 1
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        if checkpoint_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=verbose
                )
            )
        
        # Add learning rate scheduler
        initial_lr = self.lr
        decay_steps = max(1, len(X) // batch_size * epochs)
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps
        )
        
        self.model.optimizer.learning_rate = lr_schedule
        
        # Initial training with frozen backbone
        logger.info(f"Training with frozen backbone for {epochs} epochs")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callback_list,
            verbose=verbose,
            **kwargs
        )
        
        # Optional fine-tuning
        if fine_tune_epochs > 0:
            logger.info(f"Fine-tuning with unfrozen backbone for {fine_tune_epochs} epochs")
            self.unfreeze_backbone(num_layers_to_freeze=20)
            
            history_finetune = self.model.fit(
                X, y,
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callback_list,
                verbose=verbose,
                **kwargs
            )
            
            # Merge histories
            for key in history.history:
                history.history[key].extend(history_finetune.history[key])
        
        self.is_trained = True
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input images.
        
        Args:
            x: Input images (N, 9000) or (N, 100, 90, 1)
            
        Returns:
            Predicted class labels (1-indexed)
        """
        X = self._prepare_input(x)
        predictions = self.model.predict(X, verbose=0)
        
        # Get class with highest probability and convert to 1-indexed
        class_predictions = np.argmax(predictions, axis=1) + 1
        
        return class_predictions
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input images.
        
        Args:
            x: Input images (N, 9000) or (N, 100, 90, 1)
            
        Returns:
            Class probabilities (N, num_classes)
        """
        X = self._prepare_input(x)
        return self.model.predict(X, verbose=0)
    
    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            x_test: Test images (N, 9000) or (N, 100, 90, 1)
            y_test: Test labels (1-indexed)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        X = self._prepare_input(x_test)
        y = y_test - 1  # Convert to 0-indexed
        
        return self.model.evaluate(X, y, verbose=0)
    
    def save(self, file_path: str):
        """
        Save model to disk.
        
        Args:
            file_path: Path to save model
        """
        self.model.save(file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str, input_shape=(100, 90, 1), num_classes=100) -> 'VehicleClassificationCNNBase':
        """
        Load model from disk.
        
        Args:
            file_path: Path to saved model
            input_shape: Input shape for the new instance
            num_classes: Number of output classes
            
        Returns:
            Loaded model instance
        """
        inner_model = keras.models.load_model(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        # Create a new instance and replace its inner model
        instance = cls(input_shape=input_shape, num_classes=num_classes)
        instance.model = inner_model
        instance.is_trained = True
        return instance
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input by reshaping if needed.
        
        Args:
            x: Input array (N, flattened) or (N, H, W, C)
            
        Returns:
            Reshaped array (N, H, W, C)
        """
        expected_flat = np.prod(self.input_shape_config)
        if len(x.shape) == 2 and x.shape[1] == expected_flat:
            return x.reshape(-1, *self.input_shape_config)
        elif len(x.shape) == 4:
            return x
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. "
                f"Expected (N, {expected_flat}) or (N, {self.input_shape_config})"
            )
    
    def call(self, inputs):
        """Forward pass for Model subclass."""
        return self.model(inputs)
    
    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'input_shape': self.input_shape_config,
            'num_classes': self.num_classes,
            'backbone': self.backbone_type,
            'freeze_backbone': self.freeze_backbone_flag,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'early_stopping_patience': self.early_stopping_patience,
            'use_augmentation': self.use_augmentation,
            'is_trained': self.is_trained,
        }


@keras.saving.register_keras_serializable()
class VehicleRegressionBase(Model):
    """
    Base class for regression and binary classification tasks.
    
    Features:
    - Single-value output (continuous or binary)
    - Flexible loss functions (MSE, MAE, BinaryCrossentropy)
    - Transfer learning with pretrained backbones
    - Custom output transformations
    
    Args:
        input_shape: Input image shape (height, width, channels)
        backbone: Backbone architecture ('efficientnet' or 'resnet')
        freeze_backbone: Whether to freeze backbone weights initially
        output_activation: Output activation ('linear', 'sigmoid')
        loss_function: Loss function name ('mse', 'mae', 'binary_crossentropy')
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        **kwargs: Additional arguments passed to parent Model class
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (100, 90, 1),
        backbone: str = 'efficientnet',
        freeze_backbone: bool = True,
        output_activation: str = 'linear',
        loss_function: str = 'mse',
        dropout_rate: float = 0.4,
        l2_reg: float = 0.01,
        **kwargs
    ):
        # Extract custom kwargs before passing to parent
        self.batch_size = kwargs.pop('batch_size', 32)
        self.lr = kwargs.pop('lr', 0.001)
        self.epochs = kwargs.pop('epochs', 50)
        self.verbose = kwargs.pop('verbose', 1)
        self.early_stopping_patience = kwargs.pop('early_stopping_patience', 10)
        self.use_augmentation = kwargs.pop('use_augmentation', True)
        
        super().__init__(**kwargs)
        
        self.input_shape_config = input_shape
        self.backbone_type = backbone
        self.freeze_backbone_flag = freeze_backbone
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.is_trained = False
        
        # Output transformation (override in subclasses if needed)
        self.output_transform: Callable = lambda x: x
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Build the regression/binary model architecture.
        
        Returns:
            Compiled Keras Model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape_config, name='input_image')
        
        # Normalize to [0, 1]
        x = layers.Rescaling(1./255, name='normalization')(inputs)
        
        # Data augmentation
        x = self._build_augmentation_layers(x)
        
        # Convert grayscale to RGB
        x = layers.Concatenate(name='grayscale_to_rgb')([x, x, x])
        
        # Load pretrained backbone
        if self.backbone_type.lower() == 'efficientnet':
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(self.input_shape_config[0], self.input_shape_config[1], 3),
                pooling='avg'
            )
        elif self.backbone_type.lower() in ('resnet', 'resnet50'):
            backbone = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(self.input_shape_config[0], self.input_shape_config[1], 3),
                pooling='avg'
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
        
        # Freeze backbone if specified
        backbone.trainable = not self.freeze_backbone_flag
        
        # Extract features
        x = backbone(x, training=False)
        
        # Regression head
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output layer (single value)
        outputs = layers.Dense(
            1,
            activation=self.output_activation,
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name=self.name or 'VehicleRegressor')
        
        # Select loss function
        if self.loss_function == 'mse':
            loss = losses.MeanSquaredError()
        elif self.loss_function == 'mae':
            loss = losses.MeanAbsoluteError()
        elif self.loss_function == 'binary_crossentropy':
            loss = losses.BinaryCrossentropy()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['mae'] if self.loss_function in ['mse', 'mae'] else ['accuracy']
        )
        
        return model
    
    def _build_augmentation_layers(self, inputs: layers.Layer) -> layers.Layer:
        """Build data augmentation pipeline."""
        if not self.use_augmentation:
            return inputs
        
        x = inputs
        x = layers.RandomFlip('horizontal', name='aug_flip')(x)
        x = layers.RandomRotation(0.1, name='aug_rotation')(x)
        x = layers.RandomZoom(0.1, name='aug_zoom')(x)
        x = layers.RandomContrast(0.2, name='aug_contrast')(x)
        x = RandomBrightness(0.1, name='aug_brightness')(x)
        return x
    
    def unfreeze_backbone(self, num_layers_to_freeze: int = 0):
        """Unfreeze backbone for fine-tuning."""
        for layer in self.model.layers:
            if isinstance(layer, (EfficientNetB0, ResNet50)):
                layer.trainable = True
                if num_layers_to_freeze > 0:
                    for sublayer in layer.layers[:num_layers_to_freeze]:
                        sublayer.trainable = False
                logger.info(f"Unfroze backbone with {num_layers_to_freeze} layers frozen")
                break
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
    
    def fit(
        self,
        img_list: np.ndarray,
        target_list: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        val_split: float = 0.2,
        fine_tune_epochs: int = 0,
        checkpoint_path: Optional[str] = None,
        verbose: int = None,
        **kwargs
    ):
        """
        Train the model.
        
        Args:
            img_list: Training images (N, 9000) or (N, 100, 90, 1)
            target_list: Target values (continuous or binary)
            epochs: Number of training epochs (default: self.epochs)
            batch_size: Batch size (default: self.batch_size)
            val_split: Validation split ratio
            fine_tune_epochs: Additional epochs for fine-tuning
            checkpoint_path: Path to save model checkpoints
            verbose: Verbosity level (default: self.verbose)
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Training history
        """
        # Use instance defaults if not provided
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        verbose = verbose if verbose is not None else self.verbose
        
        X = self._prepare_input(img_list)
        y = target_list
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        if checkpoint_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=verbose
                )
            )
        
        # Training
        logger.info(f"Training with frozen backbone for {epochs} epochs")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callback_list,
            verbose=verbose,
            **kwargs
        )
        
        # Optional fine-tuning
        if fine_tune_epochs > 0:
            logger.info(f"Fine-tuning with unfrozen backbone for {fine_tune_epochs} epochs")
            self.unfreeze_backbone(num_layers_to_freeze=20)
            
            history_finetune = self.model.fit(
                X, y,
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callback_list,
                verbose=verbose,
                **kwargs
            )
            
            for key in history.history:
                history.history[key].extend(history_finetune.history[key])
        
        self.is_trained = True
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        
        Args:
            x: Input images (N, 9000) or (N, 100, 90, 1)
            
        Returns:
            Predicted values (with output_transform applied)
        """
        X = self._prepare_input(x)
        predictions = self.model.predict(X, verbose=0)
        
        # Apply output transformation
        return self.output_transform(predictions.flatten())
    
    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate model on test data."""
        X = self._prepare_input(x_test)
        return self.model.evaluate(X, y_test, verbose=0)
    
    def save(self, file_path: str):
        """Save model to disk."""
        self.model.save(file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str, input_shape=(100, 90, 1)) -> 'VehicleRegressionBase':
        """Load model from disk."""
        inner_model = keras.models.load_model(file_path)
        logger.info(f"Model loaded from {file_path}")
        
        instance = cls(input_shape=input_shape)
        instance.model = inner_model
        instance.is_trained = True
        return instance
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare input by reshaping if needed."""
        expected_flat = np.prod(self.input_shape_config)
        if len(x.shape) == 2 and x.shape[1] == expected_flat:
            return x.reshape(-1, *self.input_shape_config)
        elif len(x.shape) == 4:
            return x
        else:
            raise ValueError(
                f"Invalid input shape: {x.shape}. "
                f"Expected (N, {expected_flat}) or (N, {self.input_shape_config})"
            )
    
    def call(self, inputs):
        """Forward pass for Model subclass."""
        return self.model(inputs)
    
    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'input_shape': self.input_shape_config,
            'backbone': self.backbone_type,
            'freeze_backbone': self.freeze_backbone_flag,
            'output_activation': self.output_activation,
            'loss_function': self.loss_function,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'epochs': self.epochs,
            'verbose': self.verbose,
            'early_stopping_patience': self.early_stopping_patience,
            'use_augmentation': self.use_augmentation,
            'is_trained': self.is_trained,
        }


# ============================================================================
# CONCRETE CLASSIFIER IMPLEMENTATIONS
# ============================================================================


@keras.saving.register_keras_serializable()
class MakeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle make (100 classes).
    
    Predicts the manufacturer/brand of the vehicle.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


@keras.saving.register_keras_serializable()
class ModelClassifier(Model):
    """
    Classifier for vehicle model (150 classes) conditioned on make.
    
    This classifier accepts dual inputs:
    - Image of the vehicle
    - Make index (as integer)
    
    The make is embedded and concatenated with CNN features for make-aware prediction.
    """
    
    def __init__(
        self,
        num_classes: int = 150,
        num_makes: int = 100,
        backbone: str = 'efficientnet',
        freeze_backbone: bool = True,
        dropout_rate: float = 0.4,
        l2_reg: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.num_makes = num_makes
        self.backbone_type = backbone
        self.freeze_backbone_flag = freeze_backbone
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """Build dual-input model with make conditioning."""
        # Image input
        image_input = layers.Input(shape=(100, 90, 1), name='input_image')
        
        # Make input (integer index, 0-indexed internally)
        make_input = layers.Input(shape=(1,), dtype='int32', name='input_make')
        
        # Process image
        x_img = layers.Rescaling(1./255, name='normalization')(image_input)
        
        # Augmentation
        x_img = layers.RandomFlip('horizontal')(x_img)
        x_img = layers.RandomRotation(0.1)(x_img)
        x_img = layers.RandomZoom(0.1)(x_img)
        
        # Grayscale to RGB
        x_img = layers.Concatenate(name='grayscale_to_rgb')([x_img, x_img, x_img])
        
        # Pretrained backbone
        if self.backbone_type.lower() == 'efficientnet':
            backbone = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(100, 90, 3),
                pooling='avg'
            )
        else:
            backbone = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(100, 90, 3),
                pooling='avg'
            )
        
        backbone.trainable = not self.freeze_backbone_flag
        
        x_img = backbone(x_img, training=False)
        
        # Process make input (embedding)
        x_make = layers.Embedding(
            input_dim=self.num_makes,
            output_dim=32,
            name='make_embedding'
        )(make_input)
        x_make = layers.Flatten(name='make_flatten')(x_make)
        
        # Combine image features and make embedding
        x = layers.Concatenate(name='combine_features')([x_img, x_make])
        
        # Classification head
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Output
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(
            inputs=[image_input, make_input],
            outputs=outputs,
            name='ModelClassifier'
        )
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        img_list: np.ndarray,
        make_list: np.ndarray,
        class_list: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        val_split: float = 0.2,
        **kwargs
    ):
        """
        Train the model.
        
        Args:
            img_list: Training images (N, 9000) or (N, 100, 90, 1)
            make_list: Make indices (1-indexed)
            class_list: Model class labels (1-indexed)
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            **kwargs: Additional arguments for model.fit()
        """
        # Prepare inputs
        X_img = self._prepare_input(img_list)
        X_make = make_list - 1  # Convert to 0-indexed
        y = class_list - 1  # Convert to 0-indexed
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info(f"Training ModelClassifier for {epochs} epochs")
        history = self.model.fit(
            [X_img, X_make],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callback_list,
            verbose=1,
            **kwargs
        )
        
        return history
    
    def predict(self, x: np.ndarray, make_indices: np.ndarray) -> np.ndarray:
        """
        Predict model class labels.
        
        Args:
            x: Input images (N, 9000) or (N, 100, 90, 1)
            make_indices: Make indices (1-indexed)
            
        Returns:
            Predicted model class labels (1-indexed)
        """
        X_img = self._prepare_input(x)
        X_make = make_indices - 1  # Convert to 0-indexed
        
        predictions = self.model.predict([X_img, X_make], verbose=0)
        class_predictions = np.argmax(predictions, axis=1) + 1  # Convert to 1-indexed
        
        return class_predictions
    
    def predict_proba(self, x: np.ndarray, make_indices: np.ndarray) -> np.ndarray:
        """
        Predict model class probabilities.
        
        Args:
            x: Input images (N, 9000) or (N, 100, 90, 1)
            make_indices: Make indices (1-indexed)
            
        Returns:
            Class probabilities (N, num_classes)
        """
        X_img = self._prepare_input(x)
        X_make = make_indices - 1  # Convert to 0-indexed
        
        return self.model.predict([X_img, X_make], verbose=0)
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare image input by reshaping if needed."""
        if len(x.shape) == 2:
            return x.reshape(-1, 100, 90, 1)
        elif len(x.shape) == 4:
            return x
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
    
    def save(self, file_path: str):
        """Save model to disk."""
        self.model.save(file_path)
        logger.info(f"ModelClassifier saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'ModelClassifier':
        """Load model from disk."""
        model = keras.models.load_model(file_path)
        logger.info(f"ModelClassifier loaded from {file_path}")
        
        instance = cls.__new__(cls)
        instance.model = model
        return instance
    
    def call(self, inputs):
        """Forward pass for Model subclass."""
        return self.model(inputs)
    
    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'num_classes': self.num_classes,
            'num_makes': self.num_makes,
            'backbone': self.backbone_type,
            'freeze_backbone': self.freeze_backbone_flag,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
        }


@keras.saving.register_keras_serializable()
class TypeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle type (7 classes).
    
    Types: Sedan, SUV, Truck, Van, Coupe, Convertible, Hatchback
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


@keras.saving.register_keras_serializable()
class ColorClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle color (10 classes).
    
    Colors: Black, White, Silver, Red, Blue, Green, Yellow, Orange, Brown, Other
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


@keras.saving.register_keras_serializable()
class DecadeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle decade (70 classes).
    
    Decades from 1900s to 2690s in 10-year buckets.
    """
    
    def __init__(
        self,
        num_classes: int = 70,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


@keras.saving.register_keras_serializable()
class CountryClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle country of origin (70 classes).
    
    Predicts the manufacturing country.
    """
    
    def __init__(
        self,
        num_classes: int = 70,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


@keras.saving.register_keras_serializable()
class ConditionClassifier(VehicleRegressionBase):
    """
    Regression model for vehicle condition (continuous 0-10 scale).
    
    Predicts vehicle condition from pristine (10) to poor (0).
    """
    
    def __init__(
        self,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            backbone=backbone,
            output_activation='linear',
            loss_function='mse',
            **kwargs
        )
        
        # Clip output to [0, 10] range
        self.output_transform = lambda x: np.clip(x, 0, 10)


@keras.saving.register_keras_serializable()
class StockOrModedClassifier(VehicleRegressionBase):
    """
    Binary classifier for stock vs modified vehicle status.
    
    Outputs: 0 = Stock, 1 = Modified
    """
    
    def __init__(
        self,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            backbone=backbone,
            output_activation='sigmoid',
            loss_function='binary_crossentropy',
            **kwargs
        )
        
        # Threshold at 0.5 for binary classification
        self.output_transform = lambda x: (x >= 0.5).astype(int)


@keras.saving.register_keras_serializable()
class FunctionalUtilityClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle functional utility (10 classes).
    
    Categories: Passenger, Commercial, Emergency, Military, Agricultural,
                Construction, Racing, Recreational, Public Transport, Special Purpose
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = 'efficientnet',
        **kwargs
    ):
        super().__init__(
            input_shape=(100, 90, 1),
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )


# ============================================================================
# UTILITY CLASSES - STRATEGY PATTERN IMPLEMENTATIONS
# ============================================================================


class PredictionConfidenceAnalyzer:
    """
    Strategy class for analyzing prediction confidence and uncertainty.
    
    Implements entropy-based uncertainty quantification and confidence metrics.
    """
    
    @staticmethod
    def calculate_confidence_metrics(probabilities: np.ndarray) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics from probability distribution.
        
        Args:
            probabilities: Class probabilities (1D array)
            
        Returns:
            ConfidenceMetrics dataclass
        """
        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities)[::-1]
        
        # Top-1 and Top-2 probabilities
        rank_1 = sorted_probs[0]
        rank_2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        
        # Calculate Shannon entropy (normalized)
        epsilon = 1e-10  # Numerical stability
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon))
        max_entropy = -np.log(1.0 / len(probabilities))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Confidence threshold
        is_confident = rank_1 >= ConfidenceLevel.HIGH.value
        
        return ConfidenceMetrics(
            confidence=float(rank_1),
            rank_1_accuracy=float(rank_1),
            rank_2_accuracy=float(rank_2),
            uncertainty=float(normalized_entropy),
            is_confident=is_confident
        )
    
    @staticmethod
    def get_top_k_predictions(
        probabilities: np.ndarray,
        class_labels: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Get top-k predictions with labels.
        
        Args:
            probabilities: Class probabilities (1D array)
            class_labels: List of class label strings
            k: Number of top predictions to return
            
        Returns:
            Dictionary mapping labels to probabilities
        """
        # Get top-k indices
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        
        # Create label -> probability mapping
        result = {}
        for idx in top_k_indices:
            if idx < len(class_labels):
                result[class_labels[idx]] = float(probabilities[idx])
        
        return result


class PredictionMapper:
    """
    Strategy class for mapping between different data representations.
    
    Converts between class indices, labels, and human-readable values.
    Supports both instance-based (Strategy pattern) and static usage for backward compatibility.
    """
    
    # Default attribute mappings for backward compatibility
    ATTRIBUTE_MAPPINGS = {
        'make': {i: f"Make_{i}" for i in range(100)},
        'model': {i: f"Model_{i}" for i in range(150)},
        'type': {
            0: "Sedan", 1: "SUV", 2: "Truck", 3: "Van", 4: "Coupe",
            5: "Hatchback", 6: "Convertible", 7: "Wagon", 8: "Pickup", 9: "Crossover",
            10: "Minivan", 11: "Luxury"
        },
        'color': {
            0: "Black", 1: "White", 2: "Silver", 3: "Red", 4: "Blue",
            5: "Gray", 6: "Brown", 7: "Green", 8: "Yellow", 9: "Gold"
        },
        'decade': {i: f"{1900 + i*10}s" for i in range(70)},
        'country': {i: f"Country_{i}" for i in range(70)},
        'condition': None,  # Continuous value
        'stock_or_moded': {0: "Stock", 1: "Modified"},
        'functional_utility': {
            0: "Passenger", 1: "Commercial", 2: "Emergency",
            3: "Military", 4: "Agricultural", 5: "Construction",
            6: "Public Transport", 7: "Recreational"
        }
    }
    
    def __init__(self, class_mapping: Optional[Dict[int, str]] = None):
        """
        Initialize mapper with class index to label mapping.
        
        Args:
            class_mapping: Dictionary mapping class indices to labels
        """
        self.class_mapping = class_mapping or {}
    
    def index_to_label(self, index: int) -> str:
        """
        Convert class index to human-readable label.
        
        Args:
            index: Class index
            
        Returns:
            Human-readable label
        """
        return self.class_mapping.get(index, f"Class_{index}")
    
    def get_all_labels(self) -> List[str]:
        """
        Get all class labels in order.
        
        Returns:
            List of class labels
        """
        if not self.class_mapping:
            return []
        
        max_index = max(self.class_mapping.keys())
        return [self.index_to_label(i) for i in range(1, max_index + 1)]
    
    @staticmethod
    def map_prediction(attribute: str, value: Union[int, float]) -> Union[str, int, float]:
        """Convert numeric prediction to human-readable value (backward compatible)."""
        if attribute not in PredictionMapper.ATTRIBUTE_MAPPINGS:
            return value
        
        mapping = PredictionMapper.ATTRIBUTE_MAPPINGS[attribute]
        if mapping is None:
            return round(float(value), 3)
        
        return mapping.get(int(value), value)
    
    @staticmethod
    def get_probability_dict(attribute: str, probabilities: np.ndarray) -> Dict[str, float]:
        """Create readable mapping of top probabilities (backward compatible)."""
        mapping = PredictionMapper.ATTRIBUTE_MAPPINGS.get(attribute, {})
        if not mapping:
            return {}
        
        top_indices = np.argsort(probabilities)[-5:][::-1]
        return {
            mapping.get(int(idx), f"Value_{idx}"): float(probabilities[idx])
            for idx in top_indices if probabilities[idx] > 0.01
        }


class ReportGenerator:
    """
    Strategy class for generating classification reports.
    
    Creates summaries and recommendations based on prediction results.
    """
    
    @staticmethod
    def generate_summary(predictions: Dict[str, VehicleAttributePrediction]) -> str:
        """
        Generate executive summary from predictions.
        
        Args:
            predictions: Dictionary of attribute predictions
            
        Returns:
            Summary string
        """
        # Extract key information
        make = predictions.get('make')
        model = predictions.get('model')
        vehicle_type = predictions.get('type')
        color = predictions.get('color')
        decade = predictions.get('decade')
        
        # Build summary
        parts = []
        
        if make and model:
            parts.append(f"The vehicle is identified as a {make.predicted_value} {model.predicted_value}")
        elif make:
            parts.append(f"The vehicle is identified as a {make.predicted_value}")
        
        if vehicle_type:
            parts.append(f"classified as a {vehicle_type.predicted_value}")
        
        if color:
            parts.append(f"in {color.predicted_value} color")
        
        if decade:
            parts.append(f"from the {decade.predicted_value} era")
        
        summary = ", ".join(parts) + "."
        
        # Add confidence note
        avg_confidence = np.mean([p.confidence for p in predictions.values()])
        if avg_confidence >= 0.8:
            summary += " The classification shows high confidence across all attributes."
        elif avg_confidence >= 0.6:
            summary += " The classification shows moderate confidence."
        else:
            summary += " Some predictions have low confidence and may require verification."
        
        return summary
    
    @staticmethod
    def generate_recommendations(
        predictions: Dict[str, VehicleAttributePrediction]
    ) -> List[str]:
        """
        Generate actionable recommendations based on predictions.
        
        Args:
            predictions: Dictionary of attribute predictions
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for low confidence predictions
        low_confidence_attrs = [
            name for name, pred in predictions.items()
            if pred.confidence < ConfidenceLevel.MEDIUM.value
        ]
        
        if low_confidence_attrs:
            recommendations.append(
                f"Low confidence detected for: {', '.join(low_confidence_attrs)}. "
                "Consider manual verification or additional images."
            )
        
        # Check for high uncertainty
        high_uncertainty_attrs = [
            name for name, pred in predictions.items()
            if pred.confidence_metrics and pred.confidence_metrics.uncertainty > 0.7
        ]
        
        if high_uncertainty_attrs:
            recommendations.append(
                f"High prediction uncertainty for: {', '.join(high_uncertainty_attrs)}. "
                "Multiple classes have similar probabilities."
            )
        
        # Check condition (if available)
        condition = predictions.get('condition')
        if condition and isinstance(condition.predicted_value, (int, float)):
            if condition.predicted_value < 5:
                recommendations.append(
                    f"Vehicle condition is rated {condition.predicted_value}/10. "
                    "Consider detailed inspection for potential issues."
                )
            elif condition.predicted_value >= 8:
                recommendations.append(
                    f"Vehicle shows excellent condition ({condition.predicted_value}/10). "
                    "Well-maintained vehicle."
                )
        
        # Check stock/modified status
        stock_status = predictions.get('stock_or_modified')
        if stock_status and stock_status.predicted_value == 1:
            recommendations.append(
                "Vehicle appears to be modified. Verify modifications for insurance and valuation purposes."
            )
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "All predictions show good confidence. Classification results are reliable."
            )
        
        return recommendations


# ============================================================================
# MODEL REGISTRY - SINGLETON PATTERN
# ============================================================================


class ModelRegistry:
    """
    Singleton registry for lazy-loading and caching trained models.
    
    Provides centralized model management with Redis-compatible caching support.
    """
    
    _instance = None
    _models: Dict[str, Union[Model, VehicleClassificationCNNBase, VehicleRegressionBase]] = {}
    _model_paths: Dict[str, str] = {}
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_model(
        self,
        name: str,
        model: Union[Model, VehicleClassificationCNNBase, VehicleRegressionBase]
    ):
        """
        Register a model in the registry.
        
        Args:
            name: Unique model identifier
            model: Model instance
        """
        self._models[name] = model
        logger.info(f"Registered model: {name}")
    
    def register_model_path(self, model_name: str, model_path: str):
        """Register path to model file for lazy loading."""
        self._model_paths[model_name] = model_path
    
    def cache_model(self, model_name: str, model: Any):
        """Explicitly cache a model instance."""
        self._models[model_name] = model
    
    def get_model(self, name: str) -> Optional[Union[Model, VehicleClassificationCNNBase, VehicleRegressionBase]]:
        """
        Retrieve a model from the registry, loading from disk if a path was registered.
        
        Args:
            name: Model identifier
            
        Returns:
            Model instance or None if not found
        """
        if name not in self._models:
            if name in self._model_paths:
                self._models[name] = keras.models.load_model(self._model_paths[name])
                logger.info(f"Lazy-loaded model '{name}' from {self._model_paths[name]}")
            else:
                return None
        return self._models.get(name)
    
    def load_model(self, name: str, file_path: str):
        """
        Lazy-load a model from disk and cache it.
        
        Args:
            name: Model identifier
            file_path: Path to saved model
        """
        if name not in self._models:
            model = keras.models.load_model(file_path)
            self._models[name] = model
            logger.info(f"Lazy-loaded model '{name}' from {file_path}")
        else:
            logger.info(f"Model '{name}' already in cache")
    
    def clear_cache(self):
        """Clear all cached models."""
        self._models.clear()
        logger.info("Model registry cache cleared")
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_cached_model_names(self) -> List[str]:
        """Get list of currently cached model names (alias for list_models)."""
        return list(self._models.keys())


# ============================================================================
# VEHICLE PREDICTION PIPELINE - FACTORY + STRATEGY PATTERN
# ============================================================================


class VehiclePredictionPipeline:
    """
    Orchestrator for vehicle classification pipeline.
    
    Implements Factory and Strategy patterns for flexible, maintainable prediction logic.
    
    Design Patterns:
    - Factory: Creates prediction objects (Results, Reports)
    - Strategy: Pluggable components (Analyzers, Mappers, Generators)
    - Dependency Injection: Models injected at initialization
    """
    
    def __init__(
        self,
        confidence_analyzer: Optional[PredictionConfidenceAnalyzer] = None,
        report_generator: Optional[ReportGenerator] = None
    ):
        """
        Initialize prediction pipeline.
        
        Args:
            confidence_analyzer: Strategy for confidence analysis
            report_generator: Strategy for report generation
        """
        self.models: Dict[str, Any] = {}
        self._classifiers: Dict[str, Any] = {}  # Backward compatible alias
        self.mappers: Dict[str, PredictionMapper] = {}
        
        # Strategy components (pluggable)
        self.confidence_analyzer = confidence_analyzer or PredictionConfidenceAnalyzer()
        self.prediction_mapper = PredictionMapper()  # Default static mapper for backward compat
        self.report_generator = report_generator or ReportGenerator()
        
        # Model registry
        self.model_registry = ModelRegistry()
        self.registry = self.model_registry  # Alias
        self._initialized = False
        
        logger.info("VehiclePredictionPipeline initialized")
    
    def initialize_models(
        self,
        classifiers: Dict[str, Union[VehicleClassificationCNNBase, VehicleRegressionBase, ModelClassifier]]
    ):
        """
        Inject model dependencies (Dependency Injection pattern).
        
        Args:
            classifiers: Dictionary mapping attribute names to classifier instances
        """
        self.models = classifiers
        self._classifiers = classifiers  # Backward compatible alias
        
        # Register models in singleton registry
        for name, model in classifiers.items():
            self.model_registry.cache_model(name, model)
        
        self._initialized = True
        logger.info(f"Initialized {len(classifiers)} classifiers")
    
    def add_mapper(self, attribute_name: str, mapper: PredictionMapper):
        """
        Add a prediction mapper for an attribute.
        
        Args:
            attribute_name: Name of the attribute (e.g., 'make', 'color')
            mapper: PredictionMapper instance
        """
        self.mappers[attribute_name] = mapper
    
    def _validate_input(self, image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate input image format and shape."""
        if not isinstance(image, np.ndarray):
            return False, "Input must be a numpy array"
        
        if image.ndim not in [2, 3]:
            return False, f"Image must be 2D or 3D, got {image.ndim}D"
        
        if image.size == 0:
            return False, "Image is empty"
        
        return True, None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Ensure correct shape
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if necessary (assuming 100x90x1)
        if image.shape != (100, 90, 1):
            try:
                from PIL import Image as PILImage
                img = PILImage.fromarray((image[:, :, 0]).astype(np.uint8))
                img = img.resize((90, 100))  # PIL uses (width, height)
                image = np.expand_dims(np.array(img), axis=-1)
            except ImportError:
                logger.warning("PIL not available for image resizing")
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def predict_single(
        self,
        image: np.ndarray,
        image_path: str,
        make_index: Optional[int] = None
    ) -> VehicleClassificationResult:
        """
        Predict all attributes for a single vehicle image.
        
        Args:
            image: Input image (100, 90, 1) or (9000,)
            image_path: Source image path
            make_index: Make index for ModelClassifier (1-indexed, required if model classifier present)
            
        Returns:
            VehicleClassificationResult
        """
        start_time = time.time()
        
        # Ensure image has batch dimension
        if len(image.shape) == 2:
            image = image.reshape(1, -1)  # (1, 9000)
        elif len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # (1, 100, 90, 1)
        
        predictions = {}
        
        # Iterate through all classifiers
        for attr_name, classifier in self.models.items():
            try:
                # Handle ModelClassifier (dual-input)
                if isinstance(classifier, ModelClassifier):
                    if make_index is None:
                        logger.warning("Make index required for ModelClassifier, skipping")
                        continue
                    
                    probs = classifier.predict_proba(image, np.array([make_index]))[0]
                    predicted_class = np.argmax(probs) + 1  # 1-indexed
                
                # Handle classification models
                elif isinstance(classifier, VehicleClassificationCNNBase):
                    probs = classifier.predict_proba(image)[0]
                    predicted_class = np.argmax(probs) + 1  # 1-indexed
                
                # Handle regression models
                elif isinstance(classifier, VehicleRegressionBase):
                    predicted_value = classifier.predict(image)[0]
                    
                    # Create pseudo-probabilities for regression
                    probs = np.array([1.0])
                    predicted_class = predicted_value
                
                else:
                    logger.warning(f"Unknown classifier type for {attr_name}, skipping")
                    continue
                
                # Get confidence metrics
                confidence_metrics = self.confidence_analyzer.calculate_confidence_metrics(probs)
                
                # Map to human-readable label
                mapper = self.mappers.get(attr_name)
                if mapper and isinstance(predicted_class, (int, np.integer)):
                    predicted_label = mapper.index_to_label(int(predicted_class))
                    class_labels = mapper.get_all_labels()
                else:
                    predicted_label = str(predicted_class)
                    class_labels = [f"Class_{i}" for i in range(1, len(probs) + 1)]
                
                # Get top-k probabilities
                top_k_probs = self.confidence_analyzer.get_top_k_predictions(
                    probs, class_labels, k=5
                )
                
                # Create prediction object (Factory pattern)
                predictions[attr_name] = VehicleAttributePrediction(
                    attribute_name=attr_name.replace('_', ' ').title(),
                    predicted_value=predicted_label,
                    confidence=float(probs.max()),
                    raw_probabilities=top_k_probs,
                    confidence_metrics=confidence_metrics
                )
                
            except Exception as e:
                logger.error(f"Error predicting {attr_name}: {str(e)}")
                continue
        
        # Calculate overall confidence
        overall_confidence = np.mean([p.confidence for p in predictions.values()])
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create result object (Factory pattern)
        result = VehicleClassificationResult(
            image_path=image_path,
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            overall_confidence=float(overall_confidence),
            processing_time_ms=processing_time_ms
        )
        
        return result
    
    def predict_batch(
        self,
        images: List[Tuple[np.ndarray, str]],
        make_indices: Optional[List[int]] = None
    ) -> List[VehicleClassificationResult]:
        """
        Predict attributes for a batch of images.
        
        Args:
            images: List of (image, path) tuples
            make_indices: Optional list of make indices for ModelClassifier
            
        Returns:
            List of VehicleClassificationResult objects
        """
        results = []
        
        for i, (image, path) in enumerate(images):
            make_idx = make_indices[i] if make_indices else None
            result = self.predict_single(image, path, make_index=make_idx)
            results.append(result)
        
        logger.info(f"Batch prediction completed for {len(images)} images")
        
        return results
    
    def generate_report(
        self,
        result: VehicleClassificationResult,
        vehicle_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VehicleClassificationReport:
        """
        Generate comprehensive report from classification result.
        
        Args:
            result: Classification result
            vehicle_id: Unique vehicle identifier
            metadata: Optional additional metadata
            
        Returns:
            VehicleClassificationReport
        """
        # Generate summary (Strategy pattern)
        summary = self.report_generator.generate_summary(result.predictions)
        
        # Generate recommendations (Strategy pattern)
        recommendations = self.report_generator.generate_recommendations(result.predictions)
        
        # Create report (Factory pattern)
        report = VehicleClassificationReport(
            vehicle_id=vehicle_id,
            report_date=datetime.now().isoformat(),
            classification_results=result,
            summary=summary,
            confidence_score=result.overall_confidence,
            recommendations=recommendations,
            metadata=metadata or {}
        )
        
        return report
    
    def predict_and_report(self, image: np.ndarray, image_path: str = "unknown",
                          vehicle_id: Optional[str] = None) -> VehicleClassificationReport:
        """Convenience method: predict and generate report in one call."""
        classification_result = self.predict_single(image, image_path)
        if not vehicle_id:
            vehicle_id = f"VEH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.generate_report(classification_result, vehicle_id)