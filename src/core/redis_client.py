"""Resilient Redis client with reconnection and health checks."""

import logging
import time
import os
import redis
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ResilientRedisClient:
    """Redis client with automatic reconnection and health checks."""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize resilient Redis client.
        
        Args:
            host: Redis host (default from REDIS_HOST env)
            port: Redis port (default from REDIS_PORT env)
            max_retries: Maximum connection attempts
            retry_delay: Delay between retries in seconds
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6379))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client: Optional[redis.Redis] = None
        self.connected = False
        self.last_error = None
        
        self._connect_with_retry()
    
    def _connect_with_retry(self) -> bool:
        """Connect to Redis with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30,
                )
                
                # Test connection
                self.client.ping()
                self.connected = True
                self.last_error = None
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
                return True
            
            except (redis.ConnectionError, redis.TimeoutError) as e:
                self.last_error = str(e)
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Redis connection failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Redis after {self.max_retries} attempts")
                    self.connected = False
        
        return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to Redis."""
        if self.connected:
            return True
        
        logger.info("Attempting to reconnect to Redis...")
        return self._connect_with_retry()
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except (redis.ConnectionError, redis.TimeoutError):
            self.connected = False
            return False
    
    def ping(self) -> bool:
        """Ping Redis server. Alias for health_check."""
        return self.health_check()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis with fallback."""
        if not self.connected or not self.client:
            return None
        
        try:
            return self.client.get(key)
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on GET {key}, attempting reconnect")
            self.reconnect()
            return None
    
    def set(self, key: str, value: str, ex: int = 3600) -> bool:
        """Set value in Redis with fallback."""
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.set(key, value, ex=ex)
            return True
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on SET {key}, attempting reconnect")
            self.reconnect()
            return False
    
    def setex(self, key: str, time_seconds: int, value: str) -> bool:
        """Set value with expiration."""
        return self.set(key, value, ex=time_seconds)
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self.connected or not self.client:
            return False
        
        try:
            result = self.client.delete(key)
            return result > 0
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on DELETE {key}")
            self.reconnect()
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.connected or not self.client:
            return False
        
        try:
            return self.client.exists(key) > 0
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on EXISTS {key}")
            return False
    
    def incr(self, key: str) -> int:
        """Increment counter."""
        if not self.connected or not self.client:
            return 0
        
        try:
            return self.client.incr(key)
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on INCR {key}")
            return 0
    
    def decr(self, key: str) -> int:
        """Decrement counter."""
        if not self.connected or not self.client:
            return 0
        
        try:
            return self.client.decr(key)
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on DECR {key}")
            return 0
    
    def keys(self, pattern: str) -> list:
        """Get keys matching pattern."""
        if not self.connected or not self.client:
            return []
        
        try:
            return self.client.keys(pattern)
        except redis.ConnectionError:
            logger.warning(f"Redis connection error on KEYS {pattern}")
            return []
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                self.connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")


# Global instance
_redis_instance: Optional[ResilientRedisClient] = None


def get_redis_client() -> ResilientRedisClient:
    """Get or create global Redis client instance."""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = ResilientRedisClient()
    return _redis_instance
