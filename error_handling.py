"""
Resilience patterns for audit-copilot.
Includes circuit breaker, retry logic, and graceful degradation.
"""

import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
import backoff
import structlog

log = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failures detected, reject requests
    HALF_OPEN = "half_open"    # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for external API calls.
    Prevents cascading failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                log.info("circuit_breaker_half_open")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker OPEN. Recovery timeout in {self._time_until_reset()}s"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            log.info("circuit_breaker_closed")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            log.error(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _time_until_reset(self) -> int:
        """Time remaining until reset attempt."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, int(self.recovery_timeout - elapsed))


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    max_tries: int = 3,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    Uses the backoff library for robust retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            @backoff.on_exception(
                backoff.expo,
                exceptions,
                max_tries=max_tries,
                base=base_wait,
                max_time=max_wait,
                on_backoff=lambda details: log.warning(
                    "retry_attempt",
                    attempt=details['tries'],
                    wait_time=details['wait'],
                    func=func.__name__
                )
            )
            def _call():
                return func(*args, **kwargs)
            
            return _call()
        
        return wrapper
    
    return decorator


def with_timeout(timeout_seconds: int):
    """
    Decorator for enforcing timeout on function execution.
    Uses signal for UNIX systems or manual timeout tracking.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            def check_timeout():
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise TimeoutError(
                        f"Function {func.__name__} exceeded timeout of {timeout_seconds}s"
                    )
            
            # For async, this would need different handling
            result = func(*args, **kwargs)
            check_timeout()
            return result
        
        return wrapper
    
    return decorator


class ResilientGeminiClient:
    """
    Wrapper around Gemini API with circuit breaker and retry logic.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        self.api_key = api_key
        self.model_name = model_name
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300,  # 5 minutes
            expected_exception=Exception
        )
        genai.configure(api_key=api_key)
    
    @retry_with_backoff(max_tries=3, base_wait=1.0, max_wait=30.0)
    def generate_content(self, *args, **kwargs):
        """Generate content with resilience."""
        import google.generativeai as genai
        
        def _generate():
            model = genai.GenerativeModel(self.model_name)
            return model.generate_content(*args, **kwargs)
        
        return self.circuit_breaker.call(_generate)


class GracefulShutdown:
    """
    Handles graceful shutdown of the application.
    Allows in-flight requests to complete before shutdown.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.active_requests = 0
        self.is_shutting_down = False
    
    def request_started(self):
        """Call when a request starts."""
        if self.is_shutting_down:
            raise RuntimeError("Server is shutting down, rejecting new requests")
        self.active_requests += 1
        log.debug("request_started", active_requests=self.active_requests)
    
    def request_completed(self):
        """Call when a request completes."""
        self.active_requests -= 1
        log.debug("request_completed", active_requests=self.active_requests)
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown."""
        self.is_shutting_down = True
        log.info("graceful_shutdown_initiated", timeout=self.timeout_seconds)
        
        start_time = time.time()
        while self.active_requests > 0:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                log.warning(
                    "graceful_shutdown_timeout",
                    active_requests=self.active_requests,
                    timeout=self.timeout_seconds
                )
                break
            time.sleep(0.1)
        
        log.info("graceful_shutdown_complete")

