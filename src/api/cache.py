"""
Redis caching utility for distributed caching and regional data management.

Provides intelligent caching with TTL, regional support, and fallback handling.
"""

import json
import logging
from typing import Optional, Dict, Any
import redis
import os

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis client wrapper with caching utilities."""
    
    def __init__(self, host: str = None, port: int = None, db: int = 0, default_ttl: int = 3600):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host (default from env REDIS_HOST or localhost)
            port: Redis port (default from env REDIS_PORT or 6379)
            db: Redis database number
            default_ttl: Default time-to-live for cache entries (seconds)
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.db = db
        self.default_ttl = default_ttl
        self.client = None
        self.connected = False
        
        self._connect()
    
    def _connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.connected = False
            return False
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/error
        """
        if not self.connected or not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
                return json.loads(value)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: int = None) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time-to-live in seconds (uses default if not specified)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            self.client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key existed, False otherwise
        """
        if not self.connected or not self.client:
            return False
        
        try:
            result = self.client.delete(key)
            logger.debug(f"Cache delete: {key} (existed={result > 0})")
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.connected or not self.client:
            return False
        
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists check failed for {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key in seconds (-1 if no TTL, -2 if not exists)."""
        if not self.connected or not self.client:
            return -2
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.warning(f"Cache TTL check failed for {key}: {e}")
            return -2
    
    def flush_all(self) -> bool:
        """Clear all cache (DANGEROUS - only use in development/testing)."""
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.flushdb()
            logger.warning("Cache flushed - all entries deleted")
            return True
        except Exception as e:
            logger.error(f"Cache flush failed: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "report:*")
        
        Returns:
            Number of keys deleted
        """
        if not self.connected or not self.client:
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                logger.info(f"Cleared {count} keys matching pattern: {pattern}")
                return count
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern failed for {pattern}: {e}")
            return 0
    
    def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache."""
        if not self.connected or not self.client:
            return 0
        
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.warning(f"Cache increment failed for {key}: {e}")
            return 0
    
    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter in cache."""
        if not self.connected or not self.client:
            return 0
        
        try:
            return self.client.decrby(key, amount)
        except Exception as e:
            logger.warning(f"Cache decrement failed for {key}: {e}")
            return 0


class RegionalCache(RedisCache):
    """
    Regional caching system for tracking usage by region.
    Useful for understanding geographic patterns and caching regional reports.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize regional cache."""
        super().__init__(*args, **kwargs)
    
    def record_request(self, region: str, endpoint: str) -> bool:
        """Record API request for a region."""
        key = f"region:{region}:{endpoint}:count"
        return bool(self.incr(key))
    
    def get_region_stats(self, region: str) -> Dict[str, int]:
        """Get usage statistics for a region."""
        if not self.connected or not self.client:
            return {}
        
        try:
            pattern = f"region:{region}:*:count"
            keys = self.client.keys(pattern)
            
            stats = {}
            for key in keys:
                endpoint = key.split(':')[2]
                count = self.client.get(key)
                stats[endpoint] = int(count) if count else 0
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get region stats for {region}: {e}")
            return {}
    
    def cache_regional_result(self, region: str, vehicle_id: str, 
                             result: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache classification result by region."""
        key = f"result:{region}:{vehicle_id}"
        return self.set(key, result, ttl)
    
    def get_regional_result(self, region: str, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result by region."""
        key = f"result:{region}:{vehicle_id}"
        return self.get(key)


# Singleton instance
_cache_instance: Optional[RedisCache] = None
_regional_cache_instance: Optional[RegionalCache] = None


def get_cache() -> RedisCache:
    """Get or create global cache instance (singleton)."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def get_regional_cache() -> RegionalCache:
    """Get or create global regional cache instance (singleton)."""
    global _regional_cache_instance
    if _regional_cache_instance is None:
        _regional_cache_instance = RegionalCache()
    return _regional_cache_instance
