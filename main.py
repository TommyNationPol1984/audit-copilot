"""
Audit Copilot MEGA-ZORD v5.0 - Enterprise-grade production deployment.
Includes performance optimization, observability, reliability, and scalability.
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import Response, PlainTextResponse
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from design_metrics import analyze_slide_design, analyze_contrast_with_font_detection
from deck_analyzer import analyze_entire_deck
import uvicorn
import time
import uuid
from datetime import datetime
import structlog
from monitoring import MetricsMiddleware, get_metrics_text, record_job_success, record_job_failure
from config import redis_client, default_queue, high_priority_queue
from cache import RequestDeduplicator
from error_handling import ResilientGeminiClient, GracefulShutdown
import signal
import asyncio

log = structlog.get_logger(__name__)

# Initialize app
app = FastAPI(
    title="Audit Copilot MEGA-ZORD v5.0",
    description="Enterprise-grade design audit platform with AI analysis",
    version="5.0.0"
)

# Graceful shutdown handler
graceful_shutdown = GracefulShutdown(timeout_seconds=30)

# Request deduplicator for async jobs
request_dedup = RequestDeduplicator(ttl_seconds=300)


# Middleware stack (order matters)
app.add_middleware(GZIPMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)


# Custom middleware for request tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    graceful_shutdown.request_started()
    
    try:
        response = await call_next(request)
        duration = time.time() - request.state.start_time
        
        log.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(duration)
        return response
    finally:
        graceful_shutdown.request_completed()


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint with dependency status.
    Returns detailed health information for monitoring systems.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "5.0.0",
        "features": [
            "contrast_analysis",
            "font_size_detection",
            "batch_processing",
            "async_queue",
            "caching",
            "metrics",
            "resilience"
        ]
    }
    
    # Check Redis connectivity
    try:
        redis_client.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check job queue status
    try:
        default_depth = len(default_queue.job_ids)
        high_depth = len(high_priority_queue.job_ids)
        health_status["queue"] = {
            "default_depth": default_depth,
            "high_priority_depth": high_depth
        }
    except Exception as e:
        health_status["queue"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return health_status, status_code


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    Exposes all application metrics for monitoring systems.
    """
    return PlainTextResponse(get_metrics_text())


@app.post("/analyze/slide", tags=["analysis"])
async def analyze_slide(image_path: str, request: Request):
    """
    Analyze a single slide design for accessibility and aesthetics.
    
    Args:
        image_path: Path to the slide image
        
    Returns:
        Design analysis including contrast ratios, font sizes, and recommendations
    """
    request_id = request.state.request_id
    log.info("analyze_slide_started", request_id=request_id, image_path=image_path)
    
    try:
        from PIL import Image
        img = Image.open(image_path)
        result = analyze_slide_design(img)
        
        log.info("analyze_slide_completed", request_id=request_id, success=True)
        return result
    except FileNotFoundError:
        log.warning("image_not_found", request_id=request_id, image_path=image_path)
        raise HTTPException(status_code=404, detail="Image file not found")
    except Exception as e:
        log.error("analyze_slide_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/deck", tags=["analysis"])
async def analyze_deck(
    pdf_path: str,
    guidelines: str = "",
    priority: str = "default",
    request: Request = None
):
    """
    Analyze an entire PDF deck for design compliance.
    Processes asynchronously using the job queue.
    
    Args:
        pdf_path: Path to the PDF file
        guidelines: Design guidelines for compliance checking
        priority: Queue priority - 'high' or 'default'
        
    Returns:
        Job ID for tracking analysis progress
    """
    request_id = request.state.request_id
    job_id = str(uuid.uuid4())
    
    # Check for duplicate requests
    if request_dedup.is_duplicate(job_id):
        log.warning("duplicate_request_detected", request_id=request_id, job_id=job_id)
        raise HTTPException(
            status_code=429,
            detail="Identical analysis already in progress"
        )
    
    request_dedup.mark_processing(job_id)
    
    log.info(
        "deck_analysis_queued",
        request_id=request_id,
        job_id=job_id,
        pdf_path=pdf_path,
        priority=priority
    )
    
    try:
        # Queue the job
        queue = high_priority_queue if priority == "high" else default_queue
        job = queue.enqueue(
            "worker.process_audit_job",
            audit_job_id=job_id,
            deck_id=pdf_path,
            guidelines=guidelines,
            api_key=None,  # Will be injected by worker
            pdf_path=pdf_path,
            job_timeout=1800  # 30 minutes
        )
        
        return {
            "request_id": request_id,
            "job_id": job_id,
            "rq_job_id": job.id,
            "status": "queued",
            "queue_position": len(queue),
            "priority": priority
        }
    except Exception as e:
        request_dedup.clear_processing(job_id)
        log.error("queue_enqueue_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to queue analysis")


@app.get("/analyze/status/{job_id}", tags=["analysis"])
async def get_analysis_status(job_id: str, request: Request):
    """
    Get the status of an ongoing analysis job.
    
    Args:
        job_id: The job ID returned from /analyze/deck
        
    Returns:
        Job status including progress and results if completed
    """
    request_id = request.state.request_id
    
    try:
        # Try to get job from either queue
        job = None
        for queue in [high_priority_queue, default_queue]:
            if job_id in queue.job_ids:
                job_data = redis_client.hgetall(f"rq:job:{job_id}")
                if job_data:
                    job = job_data
                    break
        
        if not job:
            log.warning("job_not_found", request_id=request_id, job_id=job_id)
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "request_id": request_id,
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "progress": job.get("progress", "N/A"),
            "started_at": job.get("started_at"),
            "result_summary": job.get("result_summary")
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("status_check_failed", request_id=request_id, job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve job status")


@app.post("/analyze/batch", tags=["analysis"])
async def batch_analyze(
    pdf_paths: list[str],
    guidelines: str = "",
    request: Request = None
):
    """
    Queue multiple PDFs for analysis in batch mode.
    Each PDF gets a separate job but processed concurrently.
    
    Args:
        pdf_paths: List of PDF file paths
        guidelines: Design guidelines for all PDFs
        
    Returns:
        List of job IDs for tracking
    """
    request_id = request.state.request_id
    
    if not pdf_paths:
        raise HTTPException(status_code=400, detail="pdf_paths list cannot be empty")
    
    if len(pdf_paths) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 PDFs per batch")
    
    log.info(
        "batch_analysis_started",
        request_id=request_id,
        num_pdfs=len(pdf_paths)
    )
    
    job_ids = []
    
    try:
        for pdf_path in pdf_paths:
            job_id = str(uuid.uuid4())
            request_dedup.mark_processing(job_id)
            
            job = default_queue.enqueue(
                "worker.process_audit_job",
                audit_job_id=job_id,
                deck_id=pdf_path,
                guidelines=guidelines,
                api_key=None,
                pdf_path=pdf_path,
                job_timeout=1800
            )
            
            job_ids.append({
                "job_id": job_id,
                "rq_job_id": job.id,
                "pdf_path": pdf_path
            })
        
        return {
            "request_id": request_id,
            "batch_id": str(uuid.uuid4()),
            "total_jobs": len(job_ids),
            "jobs": job_ids
        }
    except Exception as e:
        log.error("batch_enqueue_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to queue batch")


@app.get("/", tags=["root"])
async def root():
    """API root endpoint with documentation."""
    return {
        "name": "Audit Copilot MEGA-ZORD",
        "version": "5.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metrics": "/metrics"
    }


# Graceful shutdown handlers
def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    log.info("sigterm_received")
    graceful_shutdown.initiate_shutdown()


signal.signal(signal.SIGTERM, handle_sigterm)


if __name__ == "__main__":
    log.info("starting_application", version="5.0.0")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "structlog.stdlib.ProcessorFormatter",
                    "processor": structlog.dev.ConsoleRenderer(),
                },
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": True,
                },
            },
        },
        access_log=True,
        server_header=False  # Security: don't expose server version
    )

