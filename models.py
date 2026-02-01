import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, LayerNormalization, MultiHeadAttention, Reshape, Dropout, Embedding, Concatenate
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
from abc import ABC, abstractmethod

@keras.saving.register_keras_serializable()
class VehicleClassificationCNNBase(Model):
    """
    Base class for CNN-based vehicle classification models.
    Provides common functionality for training, prediction, and model management.
    Expects input images of shape (100, 90, 1) or will reshape from flattened format.
    
    Subclasses should override _build_model() if they need a custom architecture.
    
    Args:
        input_shape (tuple): Shape of input images. Default is (100, 90, 1).
        num_classes (int): Number of output classes.
        **kwargs: Additional arguments including batch_size, lr (learning rate), epochs, and verbose.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=100, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_value = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.is_trained = False
        self.batch_size = kwargs.get('batch_size', 64)
        self.lr = kwargs.get('lr', 0.001)
        self.epochs = kwargs.get('epochs', 10)
        self.verbose = kwargs.get('verbose', 1)
    
    def _build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape_value),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.lr), 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model

    def fit(self, img_list, class_list, epochs=None, batch_size=None, verbose=None):
        """
        Train the model on image data.
        Accepts either flattened (N, 9000) or 4D (N, 100, 90, 1) input.
        """
            
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if img_list.ndim == 2 and img_list.shape[1] == expected_flattened_size:
            img_list = img_list.reshape(-1, *self.input_shape_value)
        
        # Ensure input is 4D
        if img_list.ndim != 4 or img_list.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {img_list.shape}")
        
        # Normalize pixel values to 0-1
        img_list = img_list.astype(np.float32) / 255.0
        
        # Ensure class_list is 1D and contains class indices
        if class_list.ndim != 1:
            class_list = class_list.flatten()
        
        # Convert class labels to 0-indexed if needed
        class_list = np.asarray(class_list) - 1
        
        # Recompile the model to reset optimizer state
        self.model.compile(optimizer=Adam(learning_rate=self.lr), 
                          loss='sparse_categorical_crossentropy', 
                          metrics=['accuracy'])
        
        self.model.fit(img_list, class_list, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.is_trained = True
    
    def predict(self, x):
        """
        Predict class indices for input images. Returns 1-indexed class numbers.
        Accepts either flattened (N, 9000) or 4D (N, 100, 90, 1) input.
        """
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if x.ndim == 2 and x.shape[1] == expected_flattened_size:
            x = x.reshape(-1, *self.input_shape_value)
        
        # Ensure input is 4D
        if x.ndim != 4 or x.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {x.shape}")
        
        # Normalize pixel values
        x = x.astype(np.float32) / 255.0
        
        predictions = np.argmax(self.model.predict(x, verbose=0), axis=-1)
        # Convert back to 1-indexed
        return predictions + 1
    
    def evaluate(self, x_test, y_test):
        """Evaluate model performance."""
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if x_test.ndim == 2 and x_test.shape[1] == expected_flattened_size:
            x_test = x_test.reshape(-1, *self.input_shape_value)
        
        # Normalize
        x_test = x_test.astype(np.float32) / 255.0
        
        # Convert class labels to 0-indexed
        y_test = np.asarray(y_test) - 1
        
        return self.model.evaluate(x_test, y_test, verbose=0)
    
    def save(self, file_path):
        """Save the inner Sequential model to disk."""
        # Save the inner model which Keras knows how to serialize
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load(cls, file_path, input_shape=(100, 90, 1), num_classes=100):
        """Load a saved model from disk. Returns an instance of the calling class."""
        # Load the inner Sequential model
        inner_model = keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}")
        
        # Create a new instance
        instance = cls(input_shape=input_shape, num_classes=num_classes)
        instance.model = inner_model
        instance.is_trained = True
        return instance
    

class VehicleRegressionBase(VehicleClassificationCNNBase):
    """
    Base class for single-output regression/binary models (not multi-class).
    Subclasses specify: loss_fn, metrics, and output_transform.
    """
    def __init__(self, input_shape=(100, 90, 1), loss_fn='mse', metrics=None, **kwargs):
        self.loss_fn = loss_fn
        self.metrics_list = metrics or ['mae']
        self.output_transform = lambda x: x  # Override in subclass if needed
        super().__init__(input_shape=input_shape, num_classes=1, **kwargs)
    
    def _build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape_value),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), 
                     loss=self.loss_fn, metrics=self.metrics_list)
        return model
    
    def fit(self, img_list, label_list, epochs=None, batch_size=None, verbose=None):
        """Generic fit for single-output models."""
        expected_flattened_size = np.prod(self.input_shape_value)
        if img_list.ndim == 2 and img_list.shape[1] == expected_flattened_size:
            img_list = img_list.reshape(-1, *self.input_shape_value)
        
        if img_list.ndim != 4 or img_list.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {img_list.shape}")
        
        img_list = img_list.astype(np.float32) / 255.0
        label_list = np.asarray(label_list, dtype=np.float32).flatten()
        
        self.model.compile(optimizer=Adam(learning_rate=self.lr),
                          loss=self.loss_fn, metrics=self.metrics_list)
        self.model.fit(img_list, label_list, epochs=self.epochs, 
                      batch_size=self.batch_size, verbose=self.verbose)
        self.is_trained = True
    
    def predict(self, x):
        """Generic predict with optional output transformation."""
        expected_flattened_size = np.prod(self.input_shape_value)
        if x.ndim == 2 and x.shape[1] == expected_flattened_size:
            x = x.reshape(-1, *self.input_shape_value)
        
        if x.ndim != 4 or x.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {x.shape}")
        
        x = x.astype(np.float32) / 255.0
        preds = self.model.predict(x, verbose=0).flatten()
        return self.output_transform(preds)
    
    def evaluate(self, x_test, y_test):
        """Generic evaluate."""
        expected_flattened_size = np.prod(self.input_shape_value)
        if x_test.ndim == 2 and x_test.shape[1] == expected_flattened_size:
            x_test = x_test.reshape(-1, *self.input_shape_value)
        
        x_test = x_test.astype(np.float32) / 255.0
        y_test = np.asarray(y_test, dtype=np.float32).flatten()
        return self.model.evaluate(x_test, y_test, verbose=0)


@keras.saving.register_keras_serializable()
class MakeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Makes (brands).
    Inherits from VehicleClassificationCNNBase with default settings optimized for make classification.
    Default expects 100 distinct vehicle makes.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=100, **kwargs):
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@keras.saving.register_keras_serializable()
class ModelClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Models, conditioned on make.
    Takes both image and make as inputs to predict the model of the vehicle.
    
    The model uses:
    - A CNN path to extract image features
    - An embedding layer to encode the make as a learned representation
    - Concatenates both to make the final prediction
    
    This allows a single model to handle make-specific model classifications.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=None, make_embedding_dim=16, max_makes=100, **kwargs):
        self.make_embedding_dim = make_embedding_dim
        self.max_makes = max_makes
        # num_classes now represents max models across all makes
        super().__init__(input_shape=input_shape, num_classes=num_classes or 150, **kwargs)
    
    def _build_model(self):
        # Image input
        img_input = Input(shape=self.input_shape_value, name='image')
        # Make input (0-indexed make number)
        make_input = Input(shape=(1,), dtype='int32', name='make')
        
        # CNN path for image features
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        
        # Make embedding path
        make_embed = Embedding(self.max_makes, self.make_embedding_dim, name='make_embedding')(make_input)
        make_embed = Flatten()(make_embed)
        
        # Concatenate image features and make embedding
        combined = Concatenate()([x, make_embed])
        
        # Dense layers
        combined = Dense(128, activation='relu')(combined)
        combined = Dense(64, activation='relu')(combined)
        output = Dense(self.num_classes, activation='softmax')(combined)
        
        # Create functional model
        model = Model(inputs=[img_input, make_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.lr), 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def fit(self, img_list, make_list, class_list, epochs=None, batch_size=None, verbose=None):
        """
        Train the model on image and make data.
        
        Args:
            img_list: Image data, either flattened (N, 9000) or 4D (N, 100, 90, 1)
            make_list: Make indices (0-indexed), shape (N,)
            class_list: Model class labels, shape (N,)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if img_list.ndim == 2 and img_list.shape[1] == expected_flattened_size:
            img_list = img_list.reshape(-1, *self.input_shape_value)
        
        # Ensure input is 4D
        if img_list.ndim != 4 or img_list.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {img_list.shape}")
        
        # Normalize pixel values to 0-1
        img_list = img_list.astype(np.float32) / 255.0
        
        # Ensure make_list is 1D and 0-indexed
        make_list = np.asarray(make_list, dtype=np.int32).flatten()
        if np.any(make_list < 0):
            make_list = make_list - 1  # Convert to 0-indexed if needed
        
        # Ensure class_list is 1D
        if class_list.ndim != 1:
            class_list = class_list.flatten()
        
        # Convert class labels to 0-indexed if needed
        class_list = np.asarray(class_list) - 1
        
        # Recompile the model to reset optimizer state
        self.model.compile(optimizer=Adam(learning_rate=self.lr), 
                          loss='sparse_categorical_crossentropy', 
                          metrics=['accuracy'])
        
        self.model.fit([img_list, make_list], class_list, 
                      epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        self.is_trained = True
    
    def predict(self, x, make_list):
        """
        Predict model class indices for input images.
        
        Args:
            x: Image data, either flattened (N, 9000) or 4D (N, 100, 90, 1)
            make_list: Make indices (0-indexed), shape (N,)
        
        Returns:
            Predictions as 1-indexed class numbers
        """
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if x.ndim == 2 and x.shape[1] == expected_flattened_size:
            x = x.reshape(-1, *self.input_shape_value)
        
        # Ensure input is 4D
        if x.ndim != 4 or x.shape[1:] != self.input_shape_value:
            raise ValueError(f"Expected shape (N, {self.input_shape_value}), got {x.shape}")
        
        # Normalize pixel values
        x = x.astype(np.float32) / 255.0
        
        # Ensure make_list is 1D and 0-indexed
        make_list = np.asarray(make_list, dtype=np.int32).flatten()
        if np.any(make_list < 0):
            make_list = make_list - 1
        
        predictions = np.argmax(self.model.predict([x, make_list], verbose=0), axis=-1)
        # Convert back to 1-indexed
        return predictions + 1
    
    def evaluate(self, x_test, make_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            x_test: Test image data
            make_test: Test make indices (0-indexed)
            y_test: Test labels
        """
        # Calculate expected flattened size from input_shape
        expected_flattened_size = np.prod(self.input_shape_value)
        
        # Handle flattened input
        if x_test.ndim == 2 and x_test.shape[1] == expected_flattened_size:
            x_test = x_test.reshape(-1, *self.input_shape_value)
        
        # Normalize
        x_test = x_test.astype(np.float32) / 255.0
        
        # Ensure make_test is 1D and 0-indexed
        make_test = np.asarray(make_test, dtype=np.int32).flatten()
        if np.any(make_test < 0):
            make_test = make_test - 1
        
        # Convert class labels to 0-indexed
        y_test = np.asarray(y_test) - 1
        
        return self.model.evaluate([x_test, make_test], y_test, verbose=0)


@keras.saving.register_keras_serializable()
class TypeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Types.
    Inherits from VehicleClassificationCNNBase with default settings optimized for type classification.
    Default expects 12 distinct vehicle types.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=7, **kwargs):
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@keras.saving.register_keras_serializable()
class ColorClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Colors.
    Inherits from VehicleClassificationCNNBase with default settings optimized for color classification.
    Default expects 10 distinct vehicle colors.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=10, **kwargs):
        # White, Black, Silver, Gray, Red, Blue, Brown, Green, Gold, Orange, Yellow, Other
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@keras.saving.register_keras_serializable()
class DecadeClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Decade.
    Inherits from VehicleClassificationCNNBase with default settings optimized for decade classification.
    Default expects 70 distinct vehicle decades. We're using decades because it becomes difficult to 
    predict the exact year a vehicle was manufactured.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=70, **kwargs):
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@keras.saving.register_keras_serializable()
class CountryClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Country.
    Inherits from VehicleClassificationCNNBase with default settings optimized for country classification.
    Default expects 70 distinct vehicle countries.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=70, **kwargs):
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@keras.saving.register_keras_serializable()
class ConditionClassifier(VehicleRegressionBase):
    """
    Regressor for vehicle Condition (1-10 scale).
    Inherits from VehicleRegressionBase with settings optimized for condition regression.
    """
    def __init__(self, input_shape=(100, 90, 1), **kwargs):
        super().__init__(input_shape=input_shape, loss_fn='mse', 
                        metrics=['mae'], **kwargs)


@keras.saving.register_keras_serializable()
class StockOrModedClassifier(VehicleRegressionBase):
    """
    Classifier for vehicle Stock vs Modified.
    Inherits from VehicleRegressionBase with settings optimized for binary classification.
    Outputs 1 for Modified, 0 for Stock.
    """
    def __init__(self, input_shape=(100, 90, 1), **kwargs):
        super().__init__(input_shape=input_shape, loss_fn='binary_crossentropy',
                        metrics=['accuracy'], **kwargs)
        self.output_transform = lambda x: (x > 0.5).astype(int)


@keras.saving.register_keras_serializable()
class FunctionalUtilityClassifier(VehicleClassificationCNNBase):
    """
    Classifier for vehicle Functional Utility.
    Inherits from VehicleClassificationCNNBase with default settings optimized for functional utility classification.
    Default expects 10 distinct vehicle functional utilities.
    """
    def __init__(self, input_shape=(100, 90, 1), num_classes=10, **kwargs):
        super().__init__(input_shape=input_shape, num_classes=num_classes, **kwargs)


@dataclass
class ConfidenceMetrics:
    """Confidence and uncertainty metrics for predictions."""
    confidence: float
    rank_1_accuracy: float
    rank_2_accuracy: float
    uncertainty: float
    is_confident: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleAttributePrediction:
    """Single vehicle attribute prediction with confidence metrics."""
    attribute_name: str
    predicted_value: str | int | float
    confidence: float
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    confidence_metrics: Optional[ConfidenceMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            'attribute': self.attribute_name,
            'value': self.predicted_value,
            'confidence': float(self.confidence),
            'raw_probabilities': self.raw_probabilities
        }
        if self.confidence_metrics:
            data['confidence_metrics'] = self.confidence_metrics.to_dict()
        return data


@dataclass
class VehicleClassificationResult:
    """Complete classification results for a single vehicle image."""
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    predictions: Dict[str, VehicleAttributePrediction] = field(default_factory=dict)
    overall_confidence: float = 0.0
    processing_time_ms: float = 0.0
    model_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'timestamp': self.timestamp,
            'predictions': {k: v.to_dict() for k, v in self.predictions.items()},
            'overall_confidence': float(self.overall_confidence),
            'processing_time_ms': float(self.processing_time_ms),
            'model_version': self.model_version
        }
    
    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class VehicleClassificationReport:
    """Professional report containing classification results and insights."""
    vehicle_id: str
    report_date: str = field(default_factory=lambda: datetime.now().isoformat())
    classification_results: VehicleClassificationResult = None
    summary: str = ""
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vehicle_id': self.vehicle_id,
            'report_date': self.report_date,
            'classification_results': self.classification_results.to_dict() if self.classification_results else None,
            'summary': self.summary,
            'confidence_score': float(self.confidence_score),
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_html(self) -> str:
        """Generate HTML representation of the report."""
        predictions = self.classification_results.predictions if self.classification_results else {}
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin-top: 20px; padding: 15px; border-left: 4px solid #3498db; }}
                .prediction {{ background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .confidence {{ color: #27ae60; font-weight: bold; }}
                .low-confidence {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #34495e; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Vehicle Classification Report</h1>
                <p>Report ID: {self.vehicle_id}</p>
                <p>Generated: {self.report_date}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>{self.summary}</p>
                <p class="confidence">Overall Confidence: {self.confidence_score:.2%}</p>
            </div>
            
            <div class="section">
                <h2>Predictions</h2>
                <table>
                    <tr>
                        <th>Attribute</th>
                        <th>Predicted Value</th>
                        <th>Confidence</th>
                    </tr>
        """
        
        for attr_name, prediction in predictions.items():
            confidence_class = "confidence" if prediction.confidence >= 0.7 else "low-confidence"
            html += f"""
                    <tr>
                        <td>{prediction.attribute_name}</td>
                        <td>{prediction.predicted_value}</td>
                        <td class="{confidence_class}">{prediction.confidence:.2%}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        if self.recommendations:
            html += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
            """
            for rec in self.recommendations:
                html += f"<li>{rec}</li>"
            html += """
                </ul>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        return html


class PredictionConfidenceAnalyzer:
    """Analyzes prediction confidence and generates metrics."""
    
    CONFIDENCE_THRESHOLDS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    }
    
    @staticmethod
    def calculate_confidence_metrics(probabilities: np.ndarray) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics from probability distribution."""
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        
        probs = probabilities[0]
        sorted_probs = np.sort(probs)[::-1]
        
        confidence = float(sorted_probs[0])
        rank_1_accuracy = float(sorted_probs[0])
        rank_2_accuracy = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        
        # Shannon entropy for uncertainty (normalized)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        uncertainty = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        
        is_confident = confidence >= PredictionConfidenceAnalyzer.CONFIDENCE_THRESHOLDS['high']
        
        return ConfidenceMetrics(
            confidence=confidence,
            rank_1_accuracy=rank_1_accuracy,
            rank_2_accuracy=rank_2_accuracy,
            uncertainty=uncertainty,
            is_confident=is_confident
        )


class PredictionMapper:
    """Maps raw model outputs to human-readable predictions."""
    
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
    
    @staticmethod
    def map_prediction(attribute: str, value: int | float) -> str | int | float:
        """Convert numeric prediction to human-readable value."""
        if attribute not in PredictionMapper.ATTRIBUTE_MAPPINGS:
            return value
        
        mapping = PredictionMapper.ATTRIBUTE_MAPPINGS[attribute]
        if mapping is None:
            return round(float(value), 3)
        
        return mapping.get(int(value), value)
    
    @staticmethod
    def get_probability_dict(attribute: str, probabilities: np.ndarray) -> Dict[str, float]:
        """Create readable mapping of top probabilities."""
        mapping = PredictionMapper.ATTRIBUTE_MAPPINGS.get(attribute, {})
        if not mapping:
            return {}
        
        top_indices = np.argsort(probabilities)[-5:][::-1]
        return {
            mapping.get(int(idx), f"Value_{idx}"): float(probabilities[idx])
            for idx in top_indices if probabilities[idx] > 0.01
        }


class ReportGenerator:
    """Generates professional vehicle classification reports."""
    
    @staticmethod
    def generate_summary(predictions: Dict[str, VehicleAttributePrediction]) -> str:
        """Generate executive summary from predictions."""
        if not predictions:
            return "No predictions available."
        
        high_confidence_attrs = [
            p for p in predictions.values() if p.confidence >= 0.8
        ]
        low_confidence_attrs = [
            p for p in predictions.values() if p.confidence < 0.6
        ]
        
        summary = f"Analyzed vehicle attributes with {len(predictions)} classifications. "
        summary += f"{len(high_confidence_attrs)} predictions with high confidence (≥80%), "
        summary += f"{len(low_confidence_attrs)} with low confidence (<60%)."
        
        if high_confidence_attrs:
            top_attr = max(high_confidence_attrs, key=lambda p: p.confidence)
            summary += f" Most confident: {top_attr.attribute_name} = {top_attr.predicted_value} ({top_attr.confidence:.1%})."
        
        return summary
    
    @staticmethod
    def generate_recommendations(predictions: Dict[str, VehicleAttributePrediction]) -> List[str]:
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        low_confidence = [p for p in predictions.values() if p.confidence < 0.6]
        if low_confidence:
            attrs = ", ".join([p.attribute_name for p in low_confidence])
            recommendations.append(
                f"Low confidence predictions for {attrs}. Consider re-capturing images with better lighting."
            )
        
        uncertain_predictions = [
            p for p in predictions.values() 
            if hasattr(p, 'confidence_metrics') and p.confidence_metrics and p.confidence_metrics.uncertainty > 0.5
        ]
        if uncertain_predictions:
            recommendations.append(
                "High uncertainty in some predictions. Multiple similar matches detected."
            )
        
        if len(predictions) < 7:
            recommendations.append(
                f"Only {len(predictions)} attributes predicted. Expected 9 attributes for complete analysis."
            )
        
        if not recommendations:
            recommendations.append("All predictions meet confidence thresholds. Analysis appears reliable.")
        
        return recommendations


class ModelRegistry:
    """Singleton registry for managing loaded models (lazy loading with caching)."""
    _instance = None
    _models: Dict[str, Any] = {}
    _model_paths: Dict[str, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_model_path(self, model_name: str, model_path: str):
        """Register path to model file for lazy loading."""
        self._model_paths[model_name] = model_path
    
    def get_model(self, model_name: str) -> Any:
        """Get model, loading from disk if not cached."""
        if model_name not in self._models:
            if model_name in self._model_paths:
                self._models[model_name] = keras.models.load_model(self._model_paths[model_name])
            else:
                raise ValueError(f"Model '{model_name}' not found in registry.")
        return self._models[model_name]
    
    def cache_model(self, model_name: str, model: Any):
        """Explicitly cache a model instance."""
        self._models[model_name] = model
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._models.clear()
    
    def get_cached_model_names(self) -> List[str]:
        """Get list of currently cached model names."""
        return list(self._models.keys())


class VehiclePredictionPipeline:
    """
    Robust ML prediction pipeline for vehicle classification with API-ready design.
    
    Features:
    - Batch processing of multiple images
    - Intelligent caching and lazy model loading
    - Comprehensive confidence metrics
    - Professional report generation
    - Error handling and validation
    """
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.confidence_analyzer = PredictionConfidenceAnalyzer()
        self.prediction_mapper = PredictionMapper()
        self.report_generator = ReportGenerator()
        self._classifiers: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize_models(self, model_instances: Dict[str, Any]):
        """Initialize pipeline with model instances."""
        self._classifiers = model_instances
        for name, model in model_instances.items():
            self.model_registry.cache_model(name, model)
        self._initialized = True
    
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
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure correct shape
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Resize if necessary (assuming 100x90x1)
        if image.shape != (100, 90, 1):
            from PIL import Image as PILImage
            img = PILImage.fromarray((image[:,:,0] * 255).astype(np.uint8))
            img = img.resize((100, 90))
            image = np.expand_dims(np.array(img) / 255.0, axis=-1)
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)
    
    def predict_single(self, image: np.ndarray, image_path: str = "unknown") -> VehicleClassificationResult:
        """
        Run prediction pipeline on a single image.
        
        Returns:
            VehicleClassificationResult with all predictions and confidence metrics.
        """
        start_time = datetime.now()
        
        # Validation
        is_valid, error_msg = self._validate_input(image)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Preprocessing
        processed_image = self._preprocess_image(image)
        
        result = VehicleClassificationResult(image_path=image_path)
        predictions = {}
        confidences = []
        
        # Make Prediction
        if 'make' in self._classifiers:
            make_model = self._classifiers['make']
            make_probs = make_model.model.predict(processed_image, verbose=0)
            make_pred = np.argmax(make_probs[0])
            confidence = float(make_probs[0][make_pred])
            confidences.append(confidence)
            
            metrics = self.confidence_analyzer.calculate_confidence_metrics(make_probs)
            predictions['make'] = VehicleAttributePrediction(
                attribute_name='Make',
                predicted_value=self.prediction_mapper.map_prediction('make', make_pred),
                confidence=confidence,
                raw_probabilities=self.prediction_mapper.get_probability_dict('make', make_probs[0]),
                confidence_metrics=metrics
            )
        
        # Type Prediction
        if 'type' in self._classifiers:
            type_model = self._classifiers['type']
            type_probs = type_model.model.predict(processed_image, verbose=0)
            type_pred = np.argmax(type_probs[0])
            confidence = float(type_probs[0][type_pred])
            confidences.append(confidence)
            
            metrics = self.confidence_analyzer.calculate_confidence_metrics(type_probs)
            predictions['type'] = VehicleAttributePrediction(
                attribute_name='Type',
                predicted_value=self.prediction_mapper.map_prediction('type', type_pred),
                confidence=confidence,
                raw_probabilities=self.prediction_mapper.get_probability_dict('type', type_probs[0]),
                confidence_metrics=metrics
            )
        
        # Color Prediction
        if 'color' in self._classifiers:
            color_model = self._classifiers['color']
            color_probs = color_model.model.predict(processed_image, verbose=0)
            color_pred = np.argmax(color_probs[0])
            confidence = float(color_probs[0][color_pred])
            confidences.append(confidence)
            
            metrics = self.confidence_analyzer.calculate_confidence_metrics(color_probs)
            predictions['color'] = VehicleAttributePrediction(
                attribute_name='Color',
                predicted_value=self.prediction_mapper.map_prediction('color', color_pred),
                confidence=confidence,
                raw_probabilities=self.prediction_mapper.get_probability_dict('color', color_probs[0]),
                confidence_metrics=metrics
            )
        
        # Condition Prediction (Regression)
        if 'condition' in self._classifiers:
            condition_model = self._classifiers['condition']
            condition_value = condition_model.model.predict(processed_image, verbose=0)
            condition_score = float(condition_value[0][0])
            confidence = min(1.0, abs(condition_score - 0.5) + 0.5)  # Pseudo-confidence
            confidences.append(confidence)
            
            predictions['condition'] = VehicleAttributePrediction(
                attribute_name='Condition',
                predicted_value=round(condition_score, 3),
                confidence=confidence,
                raw_probabilities={'score': condition_score}
            )
        
        # Stock/Moded Prediction
        if 'stock_or_moded' in self._classifiers:
            moded_model = self._classifiers['stock_or_moded']
            moded_value = moded_model.model.predict(processed_image, verbose=0)
            moded_pred = 1 if moded_value[0][0] > 0.5 else 0
            confidence = float(abs(moded_value[0][0] - 0.5) * 2)
            confidences.append(confidence)
            
            predictions['stock_or_moded'] = VehicleAttributePrediction(
                attribute_name='Stock or Modified',
                predicted_value=self.prediction_mapper.map_prediction('stock_or_moded', moded_pred),
                confidence=confidence,
                raw_probabilities={'probability': float(moded_value[0][0])}
            )
        
        # Functional Utility Prediction
        if 'functional_utility' in self._classifiers:
            utility_model = self._classifiers['functional_utility']
            utility_probs = utility_model.model.predict(processed_image, verbose=0)
            utility_pred = np.argmax(utility_probs[0])
            confidence = float(utility_probs[0][utility_pred])
            confidences.append(confidence)
            
            metrics = self.confidence_analyzer.calculate_confidence_metrics(utility_probs)
            predictions['functional_utility'] = VehicleAttributePrediction(
                attribute_name='Functional Utility',
                predicted_value=self.prediction_mapper.map_prediction('functional_utility', utility_pred),
                confidence=confidence,
                raw_probabilities=self.prediction_mapper.get_probability_dict('functional_utility', utility_probs[0]),
                confidence_metrics=metrics
            )
        
        # Calculate overall confidence
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        result.predictions = predictions
        result.overall_confidence = overall_confidence
        result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return result
    
    def predict_batch(self, images: List[Tuple[np.ndarray, str]]) -> List[VehicleClassificationResult]:
        """
        Run prediction pipeline on multiple images.
        
        Args:
            images: List of (image_array, image_path) tuples
        
        Returns:
            List of VehicleClassificationResult objects
        """
        results = []
        for image, path in images:
            try:
                result = self.predict_single(image, path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
        return results
    
    def generate_report(self, classification_result: VehicleClassificationResult, 
                       vehicle_id: Optional[str] = None) -> VehicleClassificationReport:
        """
        Generate professional report from classification results.
        
        Args:
            classification_result: VehicleClassificationResult from prediction
            vehicle_id: Optional custom vehicle identifier
        
        Returns:
            VehicleClassificationReport with summary, recommendations, and HTML
        """
        if not vehicle_id:
            vehicle_id = f"VEH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report = VehicleClassificationReport(
            vehicle_id=vehicle_id,
            classification_results=classification_result,
            confidence_score=classification_result.overall_confidence
        )
        
        report.summary = self.report_generator.generate_summary(classification_result.predictions)
        report.recommendations = self.report_generator.generate_recommendations(classification_result.predictions)
        report.metadata = {
            'num_predictions': len(classification_result.predictions),
            'processing_time_ms': classification_result.processing_time_ms,
            'model_version': classification_result.model_version,
            'image_path': classification_result.image_path
        }
        
        return report
    
    def predict_and_report(self, image: np.ndarray, image_path: str = "unknown",
                          vehicle_id: Optional[str] = None) -> VehicleClassificationReport:
        """Convenience method: predict and generate report in one call."""
        classification_result = self.predict_single(image, image_path)
        return self.generate_report(classification_result, vehicle_id)