"""Prometheus metrics and performance monitoring."""

import time
import functools
from typing import Callable, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Create registry
registry = CollectorRegistry()

# Metrics
REQUEST_COUNT = Counter(
    'vehicle_classifier_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    'vehicle_classifier_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry
)

CLASSIFICATION_LATENCY = Histogram(
    'vehicle_classifier_classification_duration_ms',
    'Classification latency in milliseconds',
    ['model'],
    buckets=(10, 25, 50, 100, 250, 500),
    registry=registry
)

ERROR_COUNT = Counter(
    'vehicle_classifier_errors_total',
    'Total errors',
    ['error_type'],
    registry=registry
)

CACHE_HITS = Counter(
    'vehicle_classifier_cache_hits_total',
    'Cache hits',
    ['cache_name'],
    registry=registry
)

CACHE_MISSES = Counter(
    'vehicle_classifier_cache_misses_total',
    'Cache misses',
    ['cache_name'],
    registry=registry
)

REDIS_CONNECTIONS = Gauge(
    'vehicle_classifier_redis_connections',
    'Active Redis connections',
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'vehicle_classifier_active_requests',
    'Currently active requests',
    registry=registry
)


class MetricsCollector:
    """Collector for application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_count = REQUEST_COUNT
        self.request_latency = REQUEST_LATENCY
        self.classification_latency = CLASSIFICATION_LATENCY
        self.error_count = ERROR_COUNT
        self.cache_hits = CACHE_HITS
        self.cache_misses = CACHE_MISSES
    
    def record_request(self, method: str, endpoint: str, status_code: int):
        """Record HTTP request."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
    
    def record_latency(self, endpoint: str, duration_seconds: float):
        """Record request latency."""
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_seconds)
    
    def record_classification(self, model_name: str, duration_ms: float):
        """Record classification latency."""
        CLASSIFICATION_LATENCY.labels(model=model_name).observe(duration_ms)
    
    def record_error(self, error_type: str):
        """Record error."""
        ERROR_COUNT.labels(error_type=error_type).inc()
    
    def record_cache_hit(self, cache_name: str):
        """Record cache hit."""
        CACHE_HITS.labels(cache_name=cache_name).inc()
    
    def record_cache_miss(self, cache_name: str):
        """Record cache miss."""
        CACHE_MISSES.labels(cache_name=cache_name).inc()
    
    def set_active_requests(self, count: int):
        """Set number of active requests."""
        ACTIVE_REQUESTS.set(count)
    
    def set_redis_connections(self, count: int):
        """Set number of Redis connections."""
        REDIS_CONNECTIONS.set(count)
    
    @staticmethod
    def export_prometheus() -> str:
        """Export metrics in Prometheus format."""
        from prometheus_client import generate_latest
        return generate_latest(registry).decode('utf-8')


def record_timing(operation_name: str):
    """Decorator to record operation timing."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (time.time() - start) * 1000  # Convert to ms
                collector = MetricsCollector()
                collector.record_classification(operation_name, duration)
        return wrapper
    return decorator


class Benchmark:
    """Context manager for benchmarking code blocks."""
    
    def __init__(self, operation_name: str):
        """Initialize benchmark."""
        self.operation_name = operation_name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing."""
        self.elapsed = (time.time() - self.start_time) * 1000  # ms
        collector = MetricsCollector()
        collector.record_classification(self.operation_name, self.elapsed)


def check_performance_threshold(operation: str, duration_ms: float) -> bool:
    """Check if operation duration exceeds threshold."""
    thresholds = {
        "classify": 200,      # 200ms
        "batch": 5000,        # 5 seconds
        "report": 1000,       # 1 second
        "health": 100         # 100ms
    }
    
    threshold = thresholds.get(operation, 1000)
    return duration_ms > threshold


class RequestLogger:
    """Log all API requests with metrics."""
    
    def __init__(self):
        """Initialize request logger."""
        self.collector = MetricsCollector()
        self.active_requests = 0
    
    def log_request_start(self, method: str, path: str):
        """Log request start."""
        self.active_requests += 1
        self.collector.set_active_requests(self.active_requests)
    
    def log_request_end(self, method: str, path: str, status_code: int, duration_ms: float):
        """Log request completion."""
        self.active_requests -= 1
        self.collector.set_active_requests(self.active_requests)
        self.collector.record_request(method, path, status_code)
        self.collector.record_latency(path, duration_ms / 1000)
        
        if check_performance_threshold(path.split('/')[-1], duration_ms):
            # Log slow request
            pass
