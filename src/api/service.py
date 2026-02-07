"""
Prediction API - Frontend-ready vehicle classification service.

This module provides a clean API interface for using the ML pipeline in web/mobile applications.
Includes comprehensive logging for API operations and cache management.

Example usage for API endpoints:
    - POST /api/vehicle/classify - Single image classification
    - POST /api/vehicle/classify-batch - Batch processing
    - GET /api/vehicle/report/<vehicle_id> - Retrieve generated report
    - POST /api/vehicle/report - Generate professional HTML report
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime

from src.models.classifiers import (
    VehiclePredictionPipeline,
    VehicleClassificationResult,
    VehicleClassificationReport,
    MakeClassifier,
    TypeClassifier,
    ColorClassifier,
    DecadeClassifier,
    CountryClassifier,
    ConditionClassifier,
    StockOrModedClassifier,
    FunctionalUtilityClassifier,
)
from src.api.logging_config import setup_api_logger

# Setup logging
logger = setup_api_logger()


class VehicleClassificationAPI:
    """High-level API for vehicle classification accessible via REST endpoints."""
    
    def __init__(self, model_checkpoint_dir: Path = Path("./checkpoints")):
        """
        Initialize API with trained models.
        
        Args:
            model_checkpoint_dir: Directory containing saved model checkpoints
        """
        logger.info(f"Initializing VehicleClassificationAPI")
        self.pipeline = VehiclePredictionPipeline()
        self.model_checkpoint_dir = Path(model_checkpoint_dir)
        self._reports_cache: Dict[str, VehicleClassificationReport] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Load trained models into pipeline."""
        try:
            logger.info("Loading trained models...")
            model_instances = {
                'make': MakeClassifier(),
                'type': TypeClassifier(),
                'color': ColorClassifier(),
                'decade': DecadeClassifier(),
                'country': CountryClassifier(),
                'condition': ConditionClassifier(),
                'stock_or_moded': StockOrModedClassifier(),
                'functional_utility': FunctionalUtilityClassifier(),
            }
            self.pipeline.initialize_models(model_instances)
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.warning(f"Warning: Could not load all models: {e}")
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        API endpoint: Classify a single vehicle image.
        
        Args:
            image_path: Path to vehicle image file
        
        Returns:
            Dictionary with predictions and confidence metrics
        """
        try:
            logger.debug(f"Classifying image: {image_path}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Run prediction
            result = self.pipeline.predict_single(image_array, image_path)
            
            logger.info(f"Successfully classified {image_path}")
            
            return {
                'status': 'success',
                'data': result.to_dict()
            }
        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def classify_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        API endpoint: Classify multiple vehicle images (batch processing).
        
        Args:
            image_paths: List of paths to vehicle image files
        
        Returns:
            Dictionary with list of results and summary statistics
        """
        logger.info(f"Starting batch classification with {len(image_paths)} images")
        results = []
        successful = 0
        failed = 0
        total_time = 0.0
        
        for image_path in image_paths:
            try:
                result = self.classify_image(image_path)
                results.append(result)
                if result['status'] == 'success':
                    successful += 1
                    total_time += result['data'].get('processing_time_ms', 0)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Batch classification failed for {image_path}: {e}")
                results.append({
                    'status': 'error',
                    'image': image_path,
                    'message': str(e)
                })
        
        logger.info(f"Batch complete: {successful}/{len(image_paths)} successful")
        
        return {
            'status': 'success',
            'summary': {
                'total_images': len(image_paths),
                'successful': successful,
                'failed': failed,
                'average_processing_time_ms': total_time / max(successful, 1)
            },
            'results': results
        }
    
    def generate_report(self, image_path: str, vehicle_id: Optional[str] = None,
                       format: str = 'json') -> Dict[str, Any]:
        """
        API endpoint: Generate professional vehicle classification report.
        
        Args:
            image_path: Path to vehicle image
            vehicle_id: Optional custom vehicle identifier
            format: Output format ('json', 'html', 'dict')
        
        Returns:
            Report in requested format
        """
        try:
            logger.info(f"Generating {format} report for {image_path}")
            
            # Load image
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Generate report
            report = self.pipeline.predict_and_report(image_array, image_path, vehicle_id)
            
            # Cache report
            self._reports_cache[report.vehicle_id] = report
            logger.info(f"Report generated and cached: {report.vehicle_id}")
            
            if format == 'json':
                return {
                    'status': 'success',
                    'vehicle_id': report.vehicle_id,
                    'data': report.to_json()
                }
            elif format == 'html':
                return {
                    'status': 'success',
                    'vehicle_id': report.vehicle_id,
                    'data': report.to_html(),
                    'content_type': 'text/html'
                }
            elif format == 'dict':
                return {
                    'status': 'success',
                    'vehicle_id': report.vehicle_id,
                    'data': report.to_dict()
                }
            else:
                logger.warning(f"Unsupported report format: {format}")
                return {
                    'status': 'error',
                    'message': f"Unsupported format: {format}"
                }
        
        except Exception as e:
            logger.error(f"Report generation failed for {image_path}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_report(self, vehicle_id: str) -> Dict[str, Any]:
        """
        API endpoint: Retrieve previously generated report.
        
        Args:
            vehicle_id: Vehicle report identifier
        
        Returns:
            Report data or error message
        """
        if vehicle_id in self._reports_cache:
            report = self._reports_cache[vehicle_id]
            logger.debug(f"Report retrieved from cache: {vehicle_id}")
            return {
                'status': 'success',
                'data': report.to_dict()
            }
        else:
            logger.warning(f"Report not found: {vehicle_id}")
            return {
                'status': 'error',
                'message': f"Report not found: {vehicle_id}"
            }
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        API endpoint: Get metadata about available models.
        
        Returns:
            Model information and statistics
        """
        cached_models = self.pipeline.model_registry.get_cached_model_names()
        
        logger.info("Model metadata requested")
        
        return {
            'status': 'success',
            'data': {
                'cached_models': cached_models,
                'pipeline_initialized': self.pipeline._initialized,
                'version': '2.0',
                'supported_attributes': [
                    'make', 'type', 'color', 'decade', 'country',
                    'condition', 'stock_or_moded', 'functional_utility'
                ],
                'max_image_size': (100, 90),
                'supported_formats': ['JPEG', 'PNG', 'BMP'],
                'cached_reports': len(self._reports_cache)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        API endpoint: Health check endpoint for monitoring.
        
        Returns:
            Service health status
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_ready': self.pipeline._initialized,
            'cached_reports': len(self._reports_cache)
        }
