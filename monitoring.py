"""
Prometheus metrics and observability middleware for audit-copilot.
Tracks HTTP requests, job queue metrics, and system health.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from config import redis_client, default_queue, high_priority_queue
import time
import structlog

log = structlog.get_logger(__name__)

# Create a registry
REGISTRY = CollectorRegistry()

# HTTP Metrics
http_request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY
)

# Job Queue Metrics
job_queue_depth = Gauge(
    "job_queue_depth",
    "Number of jobs in queue",
    ["queue_name"],
    registry=REGISTRY
)

job_processing_time = Histogram(
    "job_processing_time_seconds",
    "Job processing time in seconds",
    ["job_type"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0),
    registry=REGISTRY
)

job_status_counter = Counter(
    "job_status_total",
    "Job completion status",
    ["job_type", "status"],
    registry=REGISTRY
)

# Cache Metrics
cache_hit_rate = Counter(
    "cache_hits_total",
    "Cache hit count",
    ["cache_type"],
    registry=REGISTRY
)

cache_miss_rate = Counter(
    "cache_misses_total",
    "Cache miss count",
    ["cache_type"],
    registry=REGISTRY
)

# Redis Connection Pool Metrics
redis_pool_connections = Gauge(
    "redis_pool_connections",
    "Active Redis pool connections",
    registry=REGISTRY
)

redis_pool_available = Gauge(
    "redis_pool_available",
    "Available Redis pool connections",
    registry=REGISTRY
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP metrics."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        endpoint = request.url.path
        method = request.method
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            log.error("request_error", path=endpoint, method=method, error=str(e))
            raise
        finally:
            duration = time.time() - start_time
            http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            http_request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            
            if duration > 1.0:
                log.warning("slow_request", path=endpoint, method=method, duration=duration)
        
        return response


def update_queue_metrics():
    """Update job queue depth metrics."""
    try:
        default_depth = len(default_queue.job_ids)
        high_depth = len(high_priority_queue.job_ids)
        
        job_queue_depth.labels(queue_name="default").set(default_depth)
        job_queue_depth.labels(queue_name="high_priority").set(high_depth)
    except Exception as e:
        log.error("queue_metrics_update_failed", error=str(e))


def update_redis_metrics():
    """Update Redis connection pool metrics."""
    try:
        if hasattr(redis_client, 'connection_pool'):
            pool = redis_client.connection_pool
            redis_pool_available.set(pool._available_connections)
            redis_pool_connections.set(pool._created_connections)
    except Exception as e:
        log.error("redis_metrics_update_failed", error=str(e))


def record_job_success(job_type: str, duration: float):
    """Record successful job completion."""
    job_status_counter.labels(job_type=job_type, status="success").inc()
    job_processing_time.labels(job_type=job_type).observe(duration)


def record_job_failure(job_type: str, duration: float, error: str):
    """Record failed job completion."""
    job_status_counter.labels(job_type=job_type, status="failure").inc()
    job_processing_time.labels(job_type=job_type).observe(duration)
    log.error("job_failed", job_type=job_type, duration=duration, error=error)


def record_cache_hit(cache_type: str):
    """Record cache hit."""
    cache_hit_rate.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str):
    """Record cache miss."""
    cache_miss_rate.labels(cache_type=cache_type).inc()


def get_metrics_text():
    """Generate Prometheus metrics output."""
    update_queue_metrics()
    update_redis_metrics()
    return generate_latest(REGISTRY)

