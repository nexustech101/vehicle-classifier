"""
Robust ML training pipeline for vehicle classification.

This module orchestrates the training of multiple vehicle classifier models
using design patterns (Strategy, Factory, Pipeline) and DRY principles.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import numpy as np
from preprocessing import ImagePreprocessor, BatchProcessor, DataAugmentation

# Import all classifiers
from models import (
    MakeClassifier, ModelClassifier, TypeClassifier, ColorClassifier,
    DecadeClassifier, CountryClassifier, ConditionClassifier,
    StockOrModedClassifier, FunctionalUtilityClassifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    dataset_root: str
    checkpoint_dir: str = './checkpoints'
    batch_size: int = 32
    epochs: int = 20
    test_ratio: float = 0.2
    learning_rate: float = 0.001
    verbose: int = 1
    augment_data: bool = True
    augment_factor: float = 1.5  # Multiplier for dataset size
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TrainingStrategy(ABC):
    """Abstract base for different training strategies."""
    
    @abstractmethod
    def train(self, model, x_train, y_train, x_test, y_test, config: TrainingConfig) -> Dict:
        """Train model and return metrics."""
        pass


class StandardTrainingStrategy(TrainingStrategy):
    """Standard supervised training strategy."""
    
    def train(self, model, x_train, y_train, x_test, y_test, config: TrainingConfig) -> Dict:
        """Train with standard supervised learning."""
        logger.info(f"Training {model.__class__.__name__} with standard strategy...")
        
        # Train the model
        model.fit(x_train, y_train, epochs=config.epochs, 
                 batch_size=config.batch_size, verbose=config.verbose)
        
        # Evaluate
        train_loss, train_acc = model.evaluate(x_train, y_train)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        metrics = {
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'samples_trained': len(x_train),
            'samples_tested': len(x_test)
        }
        
        return metrics


class AugmentedTrainingStrategy(TrainingStrategy):
    """Training with data augmentation."""
    
    def train(self, model, x_train, y_train, x_test, y_test, config: TrainingConfig) -> Dict:
        """Train with augmented data."""
        logger.info(f"Training {model.__class__.__name__} with augmentation strategy...")
        logger.info(f"Augmentation factor: {config.augment_factor}x")
        
        # Generate augmented samples
        num_augmented = int(len(x_train) * (config.augment_factor - 1))
        augmented_images = []
        augmented_labels = []
        
        for _ in range(num_augmented):
            idx = np.random.randint(0, len(x_train))
            aug_img = DataAugmentation.augment_batch(
                np.expand_dims(x_train[idx], axis=0)
            )[0]
            augmented_images.append(aug_img)
            augmented_labels.append(y_train[idx])
        
        # Combine original and augmented
        x_train_aug = np.concatenate([x_train, np.array(augmented_images)], axis=0)
        y_train_aug = np.concatenate([y_train, np.array(augmented_labels)], axis=0)
        
        logger.info(f"Augmented dataset: {len(x_train)} → {len(x_train_aug)} samples")
        
        # Train with augmented data
        model.fit(x_train_aug, y_train_aug, epochs=config.epochs,
                 batch_size=config.batch_size, verbose=config.verbose)
        
        # Evaluate on original test set
        train_loss, train_acc = model.evaluate(x_train, y_train)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        
        metrics = {
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
            'samples_trained': len(x_train_aug),
            'samples_tested': len(x_test),
            'augmentation_factor': config.augment_factor
        }
        
        return metrics


class ModelTrainer:
    """Trains a single model with a specified strategy."""
    
    def __init__(self, model_class, config: TrainingConfig, 
                strategy: TrainingStrategy = None):
        """Initialize model trainer."""
        self.model_class = model_class
        self.config = config
        self.strategy = strategy or StandardTrainingStrategy()
        self.model = None
        self.metrics = None
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
             x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train model and return metrics."""
        # Instantiate model
        self.model = self.model_class(
            batch_size=self.config.batch_size,
            lr=self.config.learning_rate,
            epochs=self.config.epochs,
            verbose=self.config.verbose
        )
        
        logger.info(f"Starting training for {self.model_class.__name__}")
        
        # Train using strategy
        self.metrics = self.strategy.train(
            self.model, x_train, y_train, x_test, y_test, self.config
        )
        
        logger.info(f"Training complete. Metrics: {self.metrics}")
        return self.metrics
    
    def save_model(self, checkpoint_dir: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_name = self.model_class.__name__
        model_path = Path(checkpoint_dir) / f"{model_name}.h5"
        
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)


class TrainingPipeline:
    """Orchestrates training of multiple models."""
    
    # Model registry: maps model names to classes
    MODEL_REGISTRY = {
        'Make': MakeClassifier,
        'Model': ModelClassifier,
        'Type': TypeClassifier,
        'Color': ColorClassifier,
        'Decade': DecadeClassifier,
        'Country': CountryClassifier,
        'Condition': ConditionClassifier,
        'StockOrModed': StockOrModedClassifier,
        'FunctionalUtility': FunctionalUtilityClassifier,
    }
    
    def __init__(self, config: TrainingConfig):
        """Initialize training pipeline."""
        self.config = config
        self.preprocessor = ImagePreprocessor(config.dataset_root)
        self.trained_models = {}
        self.training_results = {}
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load and split dataset."""
        logger.info("=" * 60)
        logger.info("DATA PREPARATION PHASE")
        logger.info("=" * 60)
        
        # Load all images
        images, labels, class_mapping = self.preprocessor.load_all_images()
        
        # Split data
        x_train, x_test, y_train, y_test = BatchProcessor.split_train_test(
            images, labels, test_ratio=self.config.test_ratio
        )
        
        logger.info(f"Data preparation complete")
        logger.info(f"Classes discovered: {len(class_mapping)}")
        
        return x_train, x_test, y_train, y_test, class_mapping
    
    def train_model(self, model_name: str, x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray, y_test: np.ndarray,
                   use_augmentation: bool = None) -> Dict:
        """Train single model and return metrics."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_REGISTRY.keys())}")
        
        # Determine strategy
        use_aug = use_augmentation if use_augmentation is not None else self.config.augment_data
        strategy = AugmentedTrainingStrategy() if use_aug else StandardTrainingStrategy()
        
        # Create and train
        trainer = ModelTrainer(
            self.MODEL_REGISTRY[model_name],
            self.config,
            strategy=strategy
        )
        
        metrics = trainer.train(x_train, y_train, x_test, y_test)
        
        # Store results
        self.trained_models[model_name] = trainer.model
        self.training_results[model_name] = metrics
        
        return metrics
    
    def train_all_models(self, x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train all registered models."""
        logger.info("=" * 60)
        logger.info("MODEL TRAINING PHASE")
        logger.info("=" * 60)
        
        for idx, model_name in enumerate(self.MODEL_REGISTRY.keys(), 1):
            logger.info(f"\n[{idx}/{len(self.MODEL_REGISTRY)}] Training {model_name}...")
            
            try:
                self.train_model(model_name, x_train, y_train, x_test, y_test)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                self.training_results[model_name] = {'error': str(e)}
                continue
        
        return self.training_results
    
    def save_all_models(self) -> Dict[str, str]:
        """Save all trained models to disk."""
        logger.info("=" * 60)
        logger.info("MODEL PERSISTENCE PHASE")
        logger.info("=" * 60)
        
        saved_paths = {}
        
        for model_name, model in self.trained_models.items():
            try:
                trainer = ModelTrainer(
                    self.MODEL_REGISTRY[model_name],
                    self.config
                )
                trainer.model = model
                path = trainer.save_model(self.config.checkpoint_dir)
                saved_paths[model_name] = path
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
                continue
        
        return saved_paths
    
    def save_results(self) -> str:
        """Save training results to JSON."""
        results_file = Path(self.config.checkpoint_dir) / 'training_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_file}")
        return str(results_file)
    
    def generate_report(self) -> str:
        """Generate formatted training report."""
        report = []
        report.append("\n" + "=" * 60)
        report.append("TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Dataset: {self.config.dataset_root}")
        report.append(f"Total Models: {len(self.MODEL_REGISTRY)}")
        report.append(f"Successful Trainings: {len(self.trained_models)}")
        report.append("\nPER-MODEL RESULTS:")
        report.append("-" * 60)
        
        for model_name in sorted(self.training_results.keys()):
            report.append(f"\n{model_name}:")
            metrics = self.training_results[model_name]
            
            if 'error' in metrics:
                report.append(f"  ERROR: {metrics['error']}")
            else:
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.4f}")
                    else:
                        report.append(f"  {key}: {value}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def run(self):
        """Execute complete training pipeline."""
        logger.info("Starting Vehicle Classification Training Pipeline")
        logger.info(f"Config: {asdict(self.config)}")
        
        # Phase 1: Data preparation
        x_train, x_test, y_train, y_test, class_mapping = self.prepare_data()
        
        # Phase 2: Model training
        self.train_all_models(x_train, y_train, x_test, y_test)
        
        # Phase 3: Save models
        self.save_all_models()
        
        # Phase 4: Save results and report
        self.save_results()
        self.config.save(
            Path(self.config.checkpoint_dir) / 'config.json'
        )
        
        # Print report
        report = self.generate_report()
        print(report)
        
        # Save report to file
        report_path = Path(self.config.checkpoint_dir) / 'training_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Training pipeline complete. Checkpoint directory: {self.config.checkpoint_dir}")


def main():
    """Run training pipeline."""
    # Configuration
    config = TrainingConfig(
        dataset_root=r'C:\Users\charl\Documents\datasets\image-data\vehicle-images',
        checkpoint_dir='./checkpoints',
        batch_size=32,
        epochs=20,
        test_ratio=0.2,
        learning_rate=0.001,
        verbose=1,
        augment_data=True,
        augment_factor=1.5
    )
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
