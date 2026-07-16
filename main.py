"""
Audit Copilot MEGA-ZORD v4.5 - Fast Auto Audit + Fine-Tuning Pipeline
- Lightweight LLM (gemini-1.5-flash)
- Parallel slide processing
- Cached metrics
- Training dataset pipeline for fine-tuning smaller models
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
    title="Audit Copilot MEGA-ZORD v4.5",
    description="Fast AI design audits + fine-tuning pipeline (30-60s)",
    version="4.5.0"
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


# ==================== AUDIT ENDPOINTS ====================

@app.get("/health")
async def health():
    """Health check with version info."""
    return {
        "status": "ok",
        "version": "4.5.0",
        "model": "gemini-1.5-flash (fast)",
        "features": [
            "fast_auto_audit",
            "parallel_processing",
            "cached_metrics",
            "training_pipeline",
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
    store_for_training: bool = True,
    request: Request = None
):
    """
    Fast auto audit of entire PDF deck.
    Automatically stores successful audits in training pipeline.
    
    Args:
        pdf_path: Path to PDF file
        guidelines: Design guidelines (optional)
        max_slides: Max slides to audit (default 10 = ~30-60s, max 20)
        store_for_training: Store output for fine-tuning (default true)
    
    Returns:
        Full audit with rationale, metrics, and training sample ID
    
    Example response:
    {
        "status": "success",
        "rationale": "Guideline 1: The deck follows...",
        "bullets_analyzed": 5,
        "slides_analyzed": 10,
        "metrics_summary": {...},
        "processing_time_seconds": 45.2,
        "model": "gemini-1.5-flash",
        "training_sample_id": "abc123..."  // ID for reference
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
            store_for_training=store_for_training
        )
        
        result = run_full_audit_fast(
            pdf_path=pdf_path,
            guidelines=guidelines,
            api_key=api_key,
            max_slides=max_slides,
            store_for_training=store_for_training
        )
        
        if result["status"] != "success":
            log.error("deck_audit_failed", request_id=request_id, error=result.get("error"))
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        duration = result.get("total_processing_time_seconds", 0)
        log.info(
            "deck_audit_success",
            request_id=request_id,
            slides=result.get("slides_analyzed"),
            duration=duration,
            sample_stored=result.get("training_sample_id") is not None
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
    All successful audits automatically added to training pipeline.
    
    Args:
        pdf_paths: List of PDF paths
        guidelines: Design guidelines (applied to all)
        max_slides_per: Max slides per PDF
    
    Returns:
        List of audit results with training sample IDs
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
                max_slides=max_slides_per,
                store_for_training=True
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


# ==================== PIPELINE / FINE-TUNING ENDPOINTS ====================

@app.get("/pipeline/stats")
async def pipeline_stats(request: Request):
    """
    Get fine-tuning pipeline statistics.
    
    Returns:
        Pipeline health, sample count, quality metrics
    """
    try:
        from fine_tune_pipeline import get_pipeline
        pipeline = get_pipeline()
        stats = pipeline.get_statistics()
        
        return {
            "status": "ok",
            "pipeline": stats,
            "directories": {
                "raw_outputs": "/tmp/audit_copilot_pipeline/raw_outputs",
                "training_data": "/tmp/audit_copilot_pipeline/training_data",
                "validation": "/tmp/audit_copilot_pipeline/validation",
                "metadata": "/tmp/audit_copilot_pipeline/metadata"
            }
        }
    except Exception as e:
        log.error("pipeline_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/export")
async def pipeline_export(
    format: str = "jsonl",
    min_quality: float = 7.0,
    only_validated: bool = False,
    request: Request = None
):
    """
    Export training dataset for fine-tuning.
    
    Args:
        format: Output format (jsonl, completion, chat, parquet, custom)
        min_quality: Minimum quality score 0-10
        only_validated: Only export manually validated samples
    
    Returns:
        File path and export metadata
    
    Example:
    POST /pipeline/export
    {
        "format": "jsonl",
        "min_quality": 8.0,
        "only_validated": false
    }
    """
    request_id = request.state.request_id
    
    try:
        from fine_tune_pipeline import get_pipeline, DatasetFormat
        
        # Validate format
        try:
            dataset_format = DatasetFormat(format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Choose from: {', '.join([f.value for f in DatasetFormat])}"
            )
        
        if min_quality < 0 or min_quality > 10:
            raise HTTPException(status_code=400, detail="min_quality must be 0-10")
        
        pipeline = get_pipeline()
        export_path = pipeline.export_for_training(
            format=dataset_format,
            min_quality=min_quality,
            only_validated=only_validated
        )
        
        if not export_path:
            raise HTTPException(
                status_code=400,
                detail=f"No samples meet criteria: quality>{min_quality}, validated_only={only_validated}"
            )
        
        log.info(
            "pipeline_export_success",
            request_id=request_id,
            format=format,
            file=export_path
        )
        
        return {
            "status": "success",
            "format": format,
            "file": export_path,
            "instructions": {
                "jsonl": "Use with OpenAI Fine-tuning API",
                "completion": "Use with OpenAI Completion fine-tuning",
                "chat": "Use with OpenAI Chat Completion fine-tuning",
                "parquet": "Use with HuggingFace datasets library",
                "custom": "Use as custom JSON structure"
            }[format]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("pipeline_export_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/validate")
async def pipeline_validate(
    sample_id: str,
    is_valid: bool,
    notes: str = "",
    request: Request = None
):
    """
    Manually validate a training sample.
    High-quality validated samples improve fine-tuning results.
    
    Args:
        sample_id: ID from audit response (training_sample_id)
        is_valid: Whether sample is valid for training
        notes: Validation notes (quality issues, etc)
    
    Returns:
        Updated sample status
    """
    request_id = request.state.request_id
    
    try:
        from fine_tune_pipeline import get_pipeline
        
        pipeline = get_pipeline()
        success = pipeline.validate_sample(sample_id, is_valid, notes)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")
        
        log.info(
            "sample_validated",
            request_id=request_id,
            sample_id=sample_id,
            valid=is_valid
        )
        
        return {
            "status": "success",
            "sample_id": sample_id,
            "is_valid": is_valid,
            "notes": notes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("validation_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/download/{sample_id}")
async def pipeline_download(sample_id: str, request: Request):
    """
    Download raw output of a specific training sample.
    
    Args:
        sample_id: Sample ID from audit response
    
    Returns:
        Raw JSON output
    """
    try:
        from pathlib import Path
        
        sample_file = Path(f"/tmp/audit_copilot_pipeline/raw_outputs/{sample_id}.json")
        
        if not sample_file.exists():
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")
        
        return FileResponse(
            sample_file,
            media_type="application/json",
            filename=f"{sample_id}.json"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("download_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ROOT & UTILS ====================

@app.get("/")
async def root():
    """Root endpoint."""
    try:
        if os.path.exists("index.html"):
            return FileResponse("index.html", media_type="text/html")
    except:
        pass
    
    return {
        "name": "Audit Copilot MEGA-ZORD v4.5",
        "version": "4.5.0",
        "model": "gemini-1.5-flash",
        "speed": "30-120 seconds per audit",
        "pipeline": "Fine-tuning dataset collection enabled",
        "docs": "/docs",
        "endpoints": {
            "audit": {
                "health": "GET /health",
                "single_slide": "POST /analyze/slide",
                "full_deck": "POST /analyze/deck",
                "batch": "POST /analyze/deck/batch"
            },
            "pipeline": {
                "stats": "GET /pipeline/stats",
                "export": "POST /pipeline/export",
                "validate": "POST /pipeline/validate",
                "download": "GET /pipeline/download/{sample_id}"
            }
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
    log.info("startup", version="4.5.0", model="gemini-1.5-flash", pipeline="enabled")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

