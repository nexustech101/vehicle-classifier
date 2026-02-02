"""
Comprehensive logging configuration for the Vehicle Classification system.

Provides structured logging with JSON formatting for production-grade
observability across training, evaluation, and API operations.
"""

import logging
import logging.handlers
import json
from pathlib import Path
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Convert log record to JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logging(name: str, level: str = 'INFO', log_dir: Path = None) -> logging.Logger:
    """
    Setup comprehensive logging for a module.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: Directory for log files (defaults to ./logs)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Skip if already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level))
    
    # Create log directory
    if log_dir is None:
        log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (JSON format)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"{name.replace('.', '_')}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_training_logger(log_dir: Path = None) -> logging.Logger:
    """Setup dedicated logger for training operations."""
    if log_dir is None:
        log_dir = Path('./logs')
    
    logger = logging.getLogger('training')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Training file handler
    handler = logging.handlers.RotatingFileHandler(
        log_dir / 'training.log',
        maxBytes=20 * 1024 * 1024,
        backupCount=10
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Also console output for real-time monitoring
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - [TRAIN] - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    return logger


def setup_evaluation_logger(log_dir: Path = None) -> logging.Logger:
    """Setup dedicated logger for evaluation operations."""
    if log_dir is None:
        log_dir = Path('./logs')
    
    logger = logging.getLogger('evaluation')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluation file handler
    handler = logging.handlers.RotatingFileHandler(
        log_dir / 'evaluation.log',
        maxBytes=20 * 1024 * 1024,
        backupCount=10
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


def setup_api_logger(log_dir: Path = None) -> logging.Logger:
    """Setup dedicated logger for API operations."""
    if log_dir is None:
        log_dir = Path('./logs')
    
    logger = logging.getLogger('api')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # API file handler
    handler = logging.handlers.RotatingFileHandler(
        log_dir / 'api.log',
        maxBytes=20 * 1024 * 1024,
        backupCount=10
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Also console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s - [API] - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)
    
    return logger
