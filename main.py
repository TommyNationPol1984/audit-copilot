"""
Audit Copilot MEGA-ZORD v4.3 - Production Ready
Simplified, focused endpoints with proper error handling and timeouts.
"""

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from design_metrics import analyze_slide_design, analyze_contrast_with_font_detection
from deck_analyzer import analyze_entire_deck
from audit_logic import run_full_audit_with_quantitative_metrics, generate_strict_bullet_by_bullet_rationale
import uvicorn
import os
import structlog
import time
import uuid
from pathlib import Path

log = structlog.get_logger(__name__)

app = FastAPI(
    title="Audit Copilot MEGA-ZORD v4.3",
    description="AI-powered design audit with quantitative metrics",
    version="4.3.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking middleware
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


# ==================== HEALTH & STATUS ====================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "4.3.0",
        "features": [
            "contrast_analysis",
            "font_size_detection",
            "movable_slides",
            "batch_analysis",
            "error_recovery",
            "timeout_handling"
        ]
    }


# ==================== ANALYSIS ENDPOINTS ====================

@app.post("/analyze/slide")
async def analyze_slide(image_path: str, request: Request):
    """
    Analyze a single slide image.
    
    Args:
        image_path: Path to slide image file
    
    Returns:
        Design metrics including contrast, alignment, whitespace
    """
    request_id = request.state.request_id
    
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        from PIL import Image
        img = Image.open(image_path)
        result = analyze_slide_design(img)
        
        log.info("slide_analyzed_success", request_id=request_id, image=image_path)
        return result
    
    except FileNotFoundError as e:
        log.warning("file_not_found", request_id=request_id, path=image_path)
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        log.error("slide_analysis_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/deck")
async def analyze_deck(
    pdf_path: str,
    guidelines: str = "",
    max_slides: int = 20,
    request: Request = None
):
    """
    Analyze entire PDF deck with design guidelines.
    
    Args:
        pdf_path: Path to PDF file
        guidelines: Design guidelines for compliance
        max_slides: Maximum slides to analyze (limits timeout)
    
    Returns:
        Complete audit with quantitative metrics and rationale
    
    Raises:
        400: Invalid input
        404: File not found
        504: Processing timeout (try with fewer slides)
    """
    request_id = request.state.request_id
    
    try:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        if max_slides < 1 or max_slides > 100:
            raise ValueError("max_slides must be 1-100")
        
        log.info(
            "deck_analysis_started",
            request_id=request_id,
            pdf=pdf_path,
            max_slides=max_slides
        )
        
        result = run_full_audit_with_quantitative_metrics(
            pdf_path=pdf_path,
            guidelines=guidelines,
            api_key=api_key,
            max_slides=max_slides
        )
        
        if result.get("status") == "error":
            log.error("deck_analysis_error", request_id=request_id, error=result.get("error"))
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        log.info(
            "deck_analysis_success",
            request_id=request_id,
            slides=result.get("total_slides"),
            duration=result.get("processing_time_seconds")
        )
        
        return result
    
    except FileNotFoundError as e:
        log.warning("pdf_not_found", request_id=request_id, path=pdf_path)
        raise HTTPException(status_code=404, detail=str(e))
    
    except ValueError as e:
        log.warning("invalid_input", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
    except TimeoutError as e:
        log.error("analysis_timeout", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=504,
            detail="Analysis timed out. Try with fewer slides (max_slides parameter)."
        )
    
    except Exception as e:
        log.error("deck_analysis_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - serves index.html if present, otherwise API docs."""
    try:
        if os.path.exists("index.html"):
            return FileResponse("index.html", media_type="text/html")
    except:
        pass
    
    return {
        "name": "Audit Copilot MEGA-ZORD",
        "version": "4.3.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with request ID."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    log.error(
        "unhandled_exception",
        request_id=request_id,
        type=type(exc).__name__,
        error=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
            "status_code": 500
        }
    )


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup():
    """Application startup."""
    log.info("app_startup", version="4.3.0")


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown."""
    log.info("app_shutdown")


if __name__ == "__main__":
    log.info("starting_server", version="4.3.0", host="0.0.0.0", port=8000)
    
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
                }
            },
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            }
        }
    )

