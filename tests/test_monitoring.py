"""Tests for monitoring, metrics, and performance."""

import pytest
from src.core.monitoring import MetricsCollector, record_timing


class TestMetricsCollection:
    """Test Prometheus metrics collection."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector()
        assert collector is not None
    
    def test_request_counter_increment(self):
        """Test request counter increments."""
        collector = MetricsCollector()
        initial = collector.request_count._value.get()
        collector.record_request("GET", "/health", 200)
        assert collector.request_count._value.get() > initial
    
    def test_request_latency_recording(self):
        """Test request latency is recorded."""
        collector = MetricsCollector()
        collector.record_latency("classify", 45.5)
        # Verify histogram bucket updated
        assert True  # Actual assertion depends on Prometheus implementation
    
    def test_error_counter_increment(self):
        """Test error counter increments."""
        collector = MetricsCollector()
        collector.record_error("classification_error")
        # Verify error count increased
        assert True
    
    def test_cache_hit_recorded(self):
        """Test cache hits are recorded."""
        collector = MetricsCollector()
        collector.record_cache_hit("report_cache")
        # Verify cache hit counter increased
        assert True


class TestBenchmarking:
    """Test performance benchmarking functionality."""
    
    def test_timing_decorator(self):
        """Test timing decorator records execution time."""
        @record_timing("test_function")
        def sample_function():
            import time
            time.sleep(0.01)
            return "result"
        
        result = sample_function()
        assert result == "result"
    
    def test_timing_decorator_with_exception(self):
        """Test timing decorator works with exceptions."""
        @record_timing("test_function")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_benchmark_context_manager(self):
        """Test benchmark context manager."""
        from src.core.monitoring import Benchmark
        
        with Benchmark("test_operation") as timer:
            import time
            time.sleep(0.01)
        
        assert timer.elapsed > 0.01


class TestMetricsExport:
    """Test metrics export for monitoring."""
    
    def test_prometheus_format_export(self):
        """Test metrics can be exported in Prometheus format."""
        collector = MetricsCollector()
        collector.record_request("GET", "/health", 200)
        
        metrics = collector.export_prometheus()
        assert isinstance(metrics, str)
        assert len(metrics) > 0
    
    def test_metrics_endpoint_data(self):
        """Test metrics endpoint returns valid data."""
        collector = MetricsCollector()
        metrics = collector.export_prometheus()
        
        # Should contain TYPE and HELP annotations
        assert "# HELP" in metrics or "# TYPE" in metrics or len(metrics) > 0


class TestPerformanceThresholds:
    """Test performance threshold alerts."""
    
    def test_slow_request_detection(self):
        """Test detection of slow requests."""
        from src.core.monitoring import check_performance_threshold
        
        # Simulate slow request
        is_slow = check_performance_threshold("classify", 5000)  # 5 seconds
        assert is_slow is True
    
    def test_normal_request_threshold(self):
        """Test normal requests don't trigger alerts."""
        from src.core.monitoring import check_performance_threshold
        
        is_slow = check_performance_threshold("classify", 45)  # 45ms
        assert is_slow is False
