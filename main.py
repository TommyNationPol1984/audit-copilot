"""
Audit Copilot MEGA-ZORD v4.4 - Fast Auto Audit
- Lightweight LLM (gemini-1.5-flash)
- Parallel slide processing
- Cached metrics
- Target: 30-60 seconds for full audit
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from design_metrics import analyze_slide_design
from audit_logic import run_full_audit_fast, analyze_slides_parallel
import uvicorn
import os
import structlog
import time
import uuid
from pathlib import Path

log = structlog.get_logger(__name__)

app = FastAPI(
    title="Audit Copilot MEGA-ZORD v4.4",
    description="Fast AI design audits - 30-60 seconds per deck",
    version="4.4.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    request.state.start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        log.error("request_error", request_id=request.state.request_id, error=str(e))
        raise
    finally:
        duration = time.time() - request.state.start_time
        if duration > 5.0:
            log.warning("slow_request", path=request.url.path, duration=duration)
    return response


# ==================== ENDPOINTS ====================

@app.get("/health")
async def health():
    """Health check with version info."""
    return {
        "status": "ok",
        "version": "4.4.0",
        "model": "gemini-1.5-flash (fast)",
        "features": [
            "fast_auto_audit",
            "parallel_processing",
            "cached_metrics",
            "retry_logic"
        ],
        "expected_audit_time": "30-60 seconds for 10 slides"
    }


@app.post("/analyze/slide")
async def analyze_slide(image_path: str, request: Request):
    """
    Analyze a single slide quickly.
    
    Args:
        image_path: Path to slide image
    
    Returns:
        Design metrics (instant)
    """
    request_id = request.state.request_id
    
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        from PIL import Image
        img = Image.open(image_path)
        result = analyze_slide_design(img)
        
        log.info("slide_analyzed", request_id=request_id)
        return result
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("slide_analysis_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/deck")
async def analyze_deck(
    pdf_path: str,
    guidelines: str = "",
    max_slides: int = 10,
    request: Request = None
):
    """
    Fast auto audit of entire PDF deck.
    
    Args:
        pdf_path: Path to PDF file
        guidelines: Design guidelines (optional)
        max_slides: Max slides to audit (default 10 = ~30-60s, max 20 = ~60-120s)
    
    Returns:
        Full audit with rationale and metrics in 30-120 seconds
    
    Example response:
    {
        "status": "success",
        "rationale": "Guideline 1: The deck follows...",
        "bullets_analyzed": 5,
        "slides_analyzed": 10,
        "metrics_summary": {
            "avg_score": 7.8,
            "avg_contrast": 5.2,
            "avg_margin": 12.5
        },
        "processing_time_seconds": 45.2,
        "total_processing_time_seconds": 50.1,
        "model": "gemini-1.5-flash"
    }
    """
    request_id = request.state.request_id
    
    try:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        if max_slides < 1 or max_slides > 20:
            raise ValueError("max_slides must be 1-20")
        
        log.info(
            "deck_audit_started",
            request_id=request_id,
            pdf=pdf_path,
            max_slides=max_slides,
            model="gemini-1.5-flash"
        )
        
        result = run_full_audit_fast(
            pdf_path=pdf_path,
            guidelines=guidelines,
            api_key=api_key,
            max_slides=max_slides
        )
        
        if result["status"] != "success":
            log.error("deck_audit_failed", request_id=request_id, error=result.get("error"))
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        duration = result.get("total_processing_time_seconds", 0)
        log.info(
            "deck_audit_success",
            request_id=request_id,
            slides=result.get("slides_analyzed"),
            duration=duration
        )
        
        return result
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("deck_audit_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


@app.post("/analyze/deck/batch")
async def batch_audit(
    pdf_paths: list,
    guidelines: str = "",
    max_slides_per: int = 5,
    request: Request = None
):
    """
    Audit multiple PDFs in parallel (fast).
    
    Args:
        pdf_paths: List of PDF paths
        guidelines: Design guidelines (applied to all)
        max_slides_per: Max slides per PDF (lower = faster)
    
    Returns:
        List of audit results
    """
    request_id = request.state.request_id
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        if not pdf_paths or len(pdf_paths) > 10:
            raise ValueError("Provide 1-10 PDF paths")
        
        log.info("batch_audit_started", request_id=request_id, count=len(pdf_paths))
        
        results = []
        for pdf_path in pdf_paths:
            result = run_full_audit_fast(
                pdf_path=pdf_path,
                guidelines=guidelines,
                api_key=api_key,
                max_slides=max_slides_per
            )
            results.append({
                "pdf_path": pdf_path,
                **result
            })
        
        log.info("batch_audit_complete", request_id=request_id, count=len(results))
        return {"results": results, "total_count": len(results)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("batch_audit_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    try:
        if os.path.exists("index.html"):
            return FileResponse("index.html", media_type="text/html")
    except:
        pass
    
    return {
        "name": "Audit Copilot MEGA-ZORD v4.4",
        "version": "4.4.0",
        "model": "gemini-1.5-flash",
        "speed": "30-120 seconds per audit",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "single_slide": "POST /analyze/slide",
            "full_deck": "POST /analyze/deck",
            "batch": "POST /analyze/deck/batch"
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Error handler."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id}
    )


@app.on_event("startup")
async def startup():
    log.info("startup", version="4.4.0", model="gemini-1.5-flash")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

