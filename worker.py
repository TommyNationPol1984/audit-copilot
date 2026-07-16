from rq import get_current_job
from config import redis_client, default_queue, high_priority_queue, WORKER_TIMEOUT_SECONDS
import structlog
from audit_logic import run_full_audit_with_quantitative_metrics
from error_handling import retry_with_backoff, CircuitBreaker
from monitoring import record_job_success, record_job_failure
import time
import os

log = structlog.get_logger(__name__)

# Circuit breaker for Gemini API
gemini_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=300,  # 5 minutes
    expected_exception=Exception
)


@retry_with_backoff(
    max_tries=3,
    base_wait=2.0,
    max_wait=60.0,
    exceptions=(Exception,)
)
def process_audit_job(
    audit_job_id: str,
    deck_id: str,
    guidelines: str,
    api_key: str = None,
    pdf_path: str = None
):
    """
    Process an audit job with resilience patterns.
    
    Args:
        audit_job_id: Unique audit job identifier
        deck_id: Deck identifier
        guidelines: Design guidelines for compliance
        api_key: Gemini API key (defaults to env var)
        pdf_path: Path to PDF file
    
    Returns:
        Audit result dictionary
    """
    job = get_current_job()
    job_start_time = time.time()
    
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # Update job metadata
    job.meta.update({
        "status": "processing",
        "deck_id": deck_id,
        "started_at": time.time(),
        "job_id": audit_job_id
    })
    job.save_meta()

    log.info(
        "audit_job_started",
        audit_job_id=audit_job_id,
        deck_id=deck_id,
        timeout_seconds=WORKER_TIMEOUT_SECONDS
    )

    try:
        # Run audit with circuit breaker protection
        def _run_audit():
            return run_full_audit_with_quantitative_metrics(
                pdf_path=pdf_path,
                guidelines=guidelines,
                api_key=api_key
            )
        
        result = gemini_circuit_breaker.call(_run_audit)

        duration = time.time() - job_start_time
        
        # Update job with success
        job.meta.update({
            "status": "finished",
            "result_summary": {
                "total_slides": result.get("total_slides"),
                "average_design_score": result.get("deck_summary", {}).get("average_design_score"),
                "processing_time_seconds": duration
            },
            "completed_at": time.time()
        })
        job.save_meta()

        # Record metrics
        record_job_success("pdf_audit", duration)

        log.info(
            "audit_job_succeeded",
            audit_job_id=audit_job_id,
            deck_id=deck_id,
            duration=duration,
            total_slides=result.get("total_slides")
        )

        return {
            "status": "success",
            "audit_job_id": audit_job_id,
            "deck_id": deck_id,
            "result": result,
            "processing_time_seconds": duration
        }

    except Exception as e:
        duration = time.time() - job_start_time
        error_str = str(e)
        
        # Update job with failure
        job.meta.update({
            "status": "failed",
            "error": error_str,
            "failed_at": time.time(),
            "processing_time_seconds": duration
        })
        job.save_meta()

        # Record metrics
        record_job_failure("pdf_audit", duration, error_str)

        log.error(
            "audit_job_failed",
            audit_job_id=audit_job_id,
            deck_id=deck_id,
            error=error_str,
            duration=duration
        )
        
        # Re-raise for RQ retry logic
        raise


def on_job_failure(job, connection, type, value, traceback):
    """
    Callback for failed jobs - moves to dead letter queue.
    """
    log.error(
        "job_moved_to_dlq",
        job_id=job.id,
        error=str(value)
    )
    
    try:
        from config import dead_letter_queue
        dead_letter_queue.enqueue(
            f"Job moved from main queue after {job.get_status()} attempts",
            job_id=job.id
        )
    except Exception as e:
        log.error("dlq_enqueue_failed", error=str(e))


def on_job_success(job, connection, result):
    """
    Callback for successful jobs - log completion.
    """
    log.info("job_completed_successfully", job_id=job.id)


# Worker startup hook
def setup_worker():
    """Setup worker with callbacks."""
    log.info("worker_starting", timeout_seconds=WORKER_TIMEOUT_SECONDS)


if __name__ == "__main__":
    # This allows running the worker via: python worker.py
    from rq import Worker
    
    w = Worker(
        [high_priority_queue, default_queue],
        connection=redis_client,
        default_result_ttl=500,
        job_monitoring_interval=30,
        disable_default_exception_handler=False
    )
    
    log.info("worker_initialized", queue_names=["audit_high", "audit_default"])
    
    w.work(with_scheduler=True)

