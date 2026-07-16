"""
Caching utilities for audit-copilot.
Provides Redis-backed caching decorators and request deduplication.
"""

import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional
from config import redis_client
from monitoring import record_cache_hit, record_cache_miss
import structlog

log = structlog.get_logger(__name__)


def make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    # Create a hashable representation of arguments
    arg_str = json.dumps(
        {"args": args, "kwargs": kwargs},
        default=str,
        sort_keys=True
    )
    arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
    return f"{prefix}:{arg_hash}"


def cache(ttl_seconds: int = 3600, prefix: str = "cache"):
    """
    Decorator for caching function results in Redis.
    
    Args:
        ttl_seconds: Time to live in seconds
        prefix: Cache key prefix
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = make_cache_key(prefix, *args, **kwargs)
            
            try:
                # Try to get from cache
                cached = redis_client.get(cache_key)
                if cached:
                    record_cache_hit(prefix)
                    log.debug("cache_hit", key=cache_key, func=func.__name__)
                    return json.loads(cached)
            except Exception as e:
                log.warning("cache_get_failed", error=str(e), key=cache_key)
            
            # Cache miss - compute result
            record_cache_miss(prefix)
            result = await func(*args, **kwargs)
            
            # Store in cache
            try:
                redis_client.setex(
                    cache_key,
                    ttl_seconds,
                    json.dumps(result, default=str)
                )
                log.debug("cache_set", key=cache_key, func=func.__name__, ttl=ttl_seconds)
            except Exception as e:
                log.warning("cache_set_failed", error=str(e), key=cache_key)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = make_cache_key(prefix, *args, **kwargs)
            
            try:
                # Try to get from cache
                cached = redis_client.get(cache_key)
                if cached:
                    record_cache_hit(prefix)
                    log.debug("cache_hit", key=cache_key, func=func.__name__)
                    return json.loads(cached)
            except Exception as e:
                log.warning("cache_get_failed", error=str(e), key=cache_key)
            
            # Cache miss - compute result
            record_cache_miss(prefix)
            result = func(*args, **kwargs)
            
            # Store in cache
            try:
                redis_client.setex(
                    cache_key,
                    ttl_seconds,
                    json.dumps(result, default=str)
                )
                log.debug("cache_set", key=cache_key, func=func.__name__, ttl=ttl_seconds)
            except Exception as e:
                log.warning("cache_set_failed", error=str(e), key=cache_key)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RequestDeduplicator:
    """
    Deduplicates concurrent requests for the same resource.
    Prevents duplicate processing of identical async jobs.
    """
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.prefix = "dedup"
    
    def is_duplicate(self, request_id: str) -> bool:
        """Check if request is currently being processed."""
        key = f"{self.prefix}:{request_id}"
        return redis_client.exists(key) > 0
    
    def mark_processing(self, request_id: str) -> None:
        """Mark request as being processed."""
        key = f"{self.prefix}:{request_id}"
        redis_client.setex(key, self.ttl_seconds, "1")
        log.debug("request_marked", request_id=request_id)
    
    def clear_processing(self, request_id: str) -> None:
        """Clear processing marker."""
        key = f"{self.prefix}:{request_id}"
        redis_client.delete(key)
        log.debug("request_cleared", request_id=request_id)


import asyncio

