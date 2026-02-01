"""
Prediction API - Frontend-ready vehicle classification service.

This module provides a clean API interface for using the ML pipeline in web/mobile applications.
Demonstrates how to integrate the VehiclePredictionPipeline with actual model instances.

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
import json
from datetime import datetime

from models import (
    VehiclePredictionPipeline,
    VehicleClassificationResult,
    VehicleClassificationReport,
    MakeClassifier,
    TypeClassifier,
    ColorClassifier,
    ConditionClassifier,
    StockOrModedClassifier,
    FunctionalUtilityClassifier,
)


class VehicleClassificationAPI:
    """High-level API for vehicle classification accessible via REST endpoints."""
    
    def __init__(self, model_checkpoint_dir: Path = Path("./checkpoints")):
        """
        Initialize API with trained models.
        
        Args:
            model_checkpoint_dir: Directory containing saved model checkpoints
        """
        self.pipeline = VehiclePredictionPipeline()
        self.model_checkpoint_dir = Path(model_checkpoint_dir)
        self._reports_cache: Dict[str, VehicleClassificationReport] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Load trained models into pipeline."""
        try:
            model_instances = {
                'make': MakeClassifier(),
                'type': TypeClassifier(),
                'color': ColorClassifier(),
                'condition': ConditionClassifier(),
                'stock_or_moded': StockOrModedClassifier(),
                'functional_utility': FunctionalUtilityClassifier(),
            }
            self.pipeline.initialize_models(model_instances)
        except Exception as e:
            print(f"Warning: Could not load all models: {e}")
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        API endpoint: Classify a single vehicle image.
        
        Args:
            image_path: Path to vehicle image file
        
        Returns:
            Dictionary with predictions and confidence metrics
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Run prediction
            result = self.pipeline.predict_single(image_array, image_path)
            
            return {
                'status': 'success',
                'data': result.to_dict()
            }
        except Exception as e:
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
                results.append({
                    'status': 'error',
                    'image': image_path,
                    'message': str(e)
                })
        
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
            # Load image
            image = Image.open(image_path).convert('L')
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Generate report
            report = self.pipeline.predict_and_report(image_array, image_path, vehicle_id)
            
            # Cache report
            self._reports_cache[report.vehicle_id] = report
            
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
                return {
                    'status': 'error',
                    'message': f"Unsupported format: {format}"
                }
        
        except Exception as e:
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
            return {
                'status': 'success',
                'data': report.to_dict()
            }
        else:
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
        
        return {
            'status': 'success',
            'data': {
                'cached_models': cached_models,
                'pipeline_initialized': self.pipeline._initialized,
                'version': '1.0',
                'supported_attributes': [
                    'make', 'type', 'color', 'condition',
                    'stock_or_moded', 'functional_utility'
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


# ==================== EXAMPLE USAGE ====================

def example_single_prediction():
    """Example: Classify a single vehicle image."""
    api = VehicleClassificationAPI()
    
    # Classify image
    result = api.classify_image("path/to/vehicle.jpg")
    print("Classification Result:")
    print(json.dumps(result, indent=2))


def example_batch_processing():
    """Example: Process multiple vehicle images."""
    api = VehicleClassificationAPI()
    
    image_paths = [
        "path/to/vehicle1.jpg",
        "path/to/vehicle2.jpg",
        "path/to/vehicle3.jpg",
    ]
    
    results = api.classify_batch(image_paths)
    print("Batch Results:")
    print(json.dumps(results, indent=2))


def example_report_generation():
    """Example: Generate professional report."""
    api = VehicleClassificationAPI()
    
    # Generate report in different formats
    json_report = api.generate_report("path/to/vehicle.jpg", format='json')
    print("JSON Report:")
    print(json.dumps(json_report, indent=2))
    
    # Generate HTML report
    html_report = api.generate_report("path/to/vehicle.jpg", format='html')
    with open("vehicle_report.html", "w") as f:
        f.write(html_report['data'])
    print("HTML report saved to vehicle_report.html")


def example_api_endpoints():
    """Example: API endpoint usage patterns."""
    api = VehicleClassificationAPI()
    
    # GET /api/health
    health = api.health_check()
    print("Health Status:", health)
    
    # GET /api/models/metadata
    metadata = api.get_model_metadata()
    print("Model Metadata:", json.dumps(metadata, indent=2))
    
    # POST /api/vehicle/classify
    classification = api.classify_image("path/to/vehicle.jpg")
    print("Classification:", json.dumps(classification, indent=2))
    
    # POST /api/vehicle/report
    report = api.generate_report("path/to/vehicle.jpg", format='dict')
    vehicle_id = report.get('vehicle_id')
    
    # GET /api/vehicle/report/{vehicle_id}
    retrieved_report = api.get_report(vehicle_id)
    print("Retrieved Report:", json.dumps(retrieved_report, indent=2))


if __name__ == "__main__":
    # Run examples (uncomment to test)
    # example_single_prediction()
    # example_batch_processing()
    # example_report_generation()
    # example_api_endpoints()
    
    print("Prediction API module loaded successfully.")
    print("Use VehicleClassificationAPI class to create an API service instance.")
