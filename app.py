"""
Flask REST API for Vehicle Classification Service.

This module provides a production-ready REST API interface for the vehicle
classification ML pipeline, suitable for web and mobile applications.

Endpoints:
    GET  /health                    - Service health check
    GET  /api/models/metadata       - Model information
    POST /api/vehicle/classify      - Classify single image
    POST /api/vehicle/classify-batch - Batch classification
    POST /api/vehicle/report        - Generate report
    GET  /api/vehicle/report/<id>   - Retrieve report
    GET  /api/docs                  - API documentation
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import os
from typing import Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging

from prediction_api import VehicleClassificationAPI


app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path('./uploads').absolute()
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize API
api = VehicleClassificationAPI()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}


# ==================== UTILITY FUNCTIONS ====================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_from_request(request_obj) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Load and validate image from Flask request.
    
    Returns:
        (image_array, image_path, error_dict or {})
    """
    # Check if file is in request
    if 'file' not in request_obj.files:
        return None, None, {'error': 'No file provided'}
    
    file = request_obj.files['file']
    
    if file.filename == '':
        return None, None, {'error': 'No file selected'}
    
    if not allowed_file(file.filename):
        return None, None, {'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}
    
    try:
        # Read and convert to greyscale
        image = Image.open(io.BytesIO(file.read())).convert('L')
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Store file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        image.save(filepath)
        
        return image_array, str(filepath), {}
    
    except Exception as e:
        return None, None, {'error': f'Error processing image: {str(e)}'}


def format_api_response(status: str, data: Any = None, message: str = None) -> Dict[str, Any]:
    """Format standardized API response."""
    response = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    if message:
        response['message'] = message
    
    return response


# ==================== HEALTH & METADATA ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Service health check endpoint."""
    try:
        health = api.health_check()
        return jsonify(format_api_response('success', health)), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


@app.route('/api/models/metadata', methods=['GET'])
def get_model_metadata():
    """Get metadata about available models."""
    try:
        metadata = api.get_model_metadata()
        return jsonify(format_api_response('success', metadata['data'])), 200
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


# ==================== CLASSIFICATION ENDPOINTS ====================

@app.route('/api/vehicle/classify', methods=['POST'])
def classify_single():
    """Classify a single vehicle image."""
    try:
        image_array, image_path, error = load_image_from_request(request)
        
        if error:
            return jsonify(format_api_response('error', message=error.get('error'))), 400
        
        # Run prediction
        result = api.pipeline.predict_single(image_array, image_path)
        
        response = format_api_response('success', result.to_dict())
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


@app.route('/api/vehicle/classify-batch', methods=['POST'])
def classify_batch():
    """Classify multiple vehicle images (batch processing)."""
    try:
        if 'files' not in request.files:
            return jsonify(format_api_response('error', message='No files provided')), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify(format_api_response('error', message='No files selected')), 400
        
        results = []
        successful = 0
        failed = 0
        total_time = 0.0
        
        for file in files:
            if not allowed_file(file.filename):
                failed += 1
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': 'File type not allowed'
                })
                continue
            
            try:
                image = Image.open(io.BytesIO(file.read())).convert('L')
                image_array = np.array(image).astype(np.float32) / 255.0
                
                # Save file
                filename = secure_filename(file.filename)
                filepath = app.config['UPLOAD_FOLDER'] / filename
                image.save(filepath)
                
                # Predict
                result = api.pipeline.predict_single(image_array, str(filepath))
                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'data': result.to_dict()
                })
                successful += 1
                total_time += result.processing_time_ms
            
            except Exception as e:
                failed += 1
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'message': str(e)
                })
        
        response = {
            'summary': {
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'average_processing_time_ms': total_time / max(successful, 1)
            },
            'results': results
        }
        
        return jsonify(format_api_response('success', response)), 200
    
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


# ==================== REPORT ENDPOINTS ====================

@app.route('/api/vehicle/report', methods=['POST'])
def generate_report():
    """Generate professional vehicle classification report."""
    try:
        image_array, image_path, error = load_image_from_request(request)
        
        if error:
            return jsonify(format_api_response('error', message=error.get('error'))), 400
        
        # Get optional vehicle ID from request
        vehicle_id = request.form.get('vehicle_id', None)
        report_format = request.form.get('format', 'json')
        
        # Generate report
        report = api.pipeline.predict_and_report(image_array, image_path, vehicle_id)
        api._reports_cache[report.vehicle_id] = report
        
        if report_format == 'html':
            return (
                report.to_html(),
                200,
                {'Content-Type': 'text/html', 'X-Vehicle-ID': report.vehicle_id}
            )
        
        elif report_format == 'json':
            response = {
                'vehicle_id': report.vehicle_id,
                'report': report.to_dict()
            }
            return jsonify(format_api_response('success', response)), 200
        
        else:
            return jsonify(format_api_response('error', message=f'Unsupported format: {report_format}')), 400
    
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


@app.route('/api/vehicle/report/<vehicle_id>', methods=['GET'])
def get_report(vehicle_id: str):
    """Retrieve previously generated report."""
    try:
        report = api._reports_cache.get(vehicle_id)
        
        if not report:
            return jsonify(format_api_response('error', message=f'Report not found: {vehicle_id}')), 404
        
        report_format = request.args.get('format', 'json')
        
        if report_format == 'html':
            return report.to_html(), 200, {'Content-Type': 'text/html'}
        
        return jsonify(format_api_response('success', report.to_dict())), 200
    
    except Exception as e:
        logger.error(f"Report retrieval failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


# ==================== DOCUMENTATION ENDPOINTS ====================

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """Serve API documentation from HTML file."""
    try:
        docs_path = Path(__file__).parent / 'api_documentation.html'
        
        if not docs_path.exists():
            return jsonify(format_api_response('error', message='Documentation not found')), 404
        
        return send_file(str(docs_path), mimetype='text/html')
    
    except Exception as e:
        logger.error(f"Documentation retrieval failed: {str(e)}")
        return jsonify(format_api_response('error', message=str(e))), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found."""
    return jsonify(format_api_response('error', message='Endpoint not found')), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify(format_api_response('error', message='Internal server error')), 500


# ==================== APP STARTUP & SHUTDOWN ====================

@app.before_request
def before_request():
    """Pre-request setup."""
    pass


@app.after_request
def after_request(response):
    """Post-request cleanup."""
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response


# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("Starting Vehicle Classification API...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health")
    logger.info("  GET  /api/models/metadata")
    logger.info("  POST /api/vehicle/classify")
    logger.info("  POST /api/vehicle/classify-batch")
    logger.info("  POST /api/vehicle/report")
    logger.info("  GET  /api/vehicle/report/<id>")
    logger.info("  GET  /api/docs")
    logger.info("\nRunning on http://localhost:5000")
    logger.info("API Documentation: http://localhost:5000/api/docs")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
