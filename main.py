"""
Audit Copilot MEGA-ZORD v4.6 - Fast Auto Audit + HF Models + Fine-Tuning Pipeline
- Lightweight LLM (gemini-1.5-flash)
- HuggingFace NER for entity extraction
- HuggingFace embeddings for semantic search
- Fine-tuning pipeline for training smaller models
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
    title="Audit Copilot MEGA-ZORD v4.6",
    description="Fast AI design audits + HuggingFace NLP + fine-tuning (30-60s)",
    version="4.6.0"
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
        "version": "4.6.0",
        "model": "gemini-1.5-flash (fast)",
        "nlp": "HuggingFace (NER, embeddings, classification)",
        "features": [
            "fast_auto_audit",
            "entity_extraction",
            "semantic_search",
            "training_pipeline",
            "parallel_processing"
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
    extract_entities: bool = True,
    request: Request = None
):
    """
    Fast auto audit of entire PDF deck with NLP analysis.
    
    Args:
        pdf_path: Path to PDF file
        guidelines: Design guidelines
        max_slides: Max slides (1-20, default 10)
        store_for_training: Store for fine-tuning (default true)
        extract_entities: Extract design entities via NER (default true)
    
    Returns:
        Full audit with rationale, metrics, entities, and training ID
    
    Example response:
    {
        "status": "success",
        "rationale": "...",
        "entities": {
            "DESIGN_PRINCIPLE": ["alignment", "hierarchy"],
            "COLOR": ["blue", "white"],
            "TYPOGRAPHY": ["sans-serif"]
        },
        "classification": {
            "positive": 0.78,
            "negative": 0.22
        },
        "training_sample_id": "abc123..."
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
            extract_entities=extract_entities
        )
        
        result = run_full_audit_fast(
            pdf_path=pdf_path,
            guidelines=guidelines,
            api_key=api_key,
            max_slides=max_slides,
            store_for_training=store_for_training
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        # Extract entities and classify if enabled
        if extract_entities:
            try:
                from hf_models import get_hf_manager
                
                manager = get_hf_manager()
                rationale = result.get("rationale", "")
                
                # Extract entities (NER)
                entities = manager.extract_entities_from_audit(rationale)
                if entities:
                    result["entities"] = entities
                    log.debug("entities_extracted", count=sum(len(v) for v in entities.values()))
                
                # Classify sentiment/category
                classification = manager.categorize_audit(rationale)
                if classification:
                    result["classification"] = classification
                    log.debug("audit_classified", categories=list(classification.keys()))
                
            except Exception as e:
                log.warning("nlp_analysis_failed", error=str(e))
                # Continue without NLP results
        
        log.info(
            "deck_audit_success",
            request_id=request_id,
            duration=result.get("total_processing_time_seconds")
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
    Audit multiple PDFs with NLP analysis.
    
    Args:
        pdf_paths: List of PDF paths
        guidelines: Design guidelines
        max_slides_per: Max slides per PDF
    
    Returns:
        List of audit results with entities and classifications
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
            results.append({"pdf_path": pdf_path, **result})
        
        log.info("batch_audit_complete", request_id=request_id, count=len(results))
        return {"results": results, "total_count": len(results)}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("batch_audit_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== NLP / ENTITY ENDPOINTS ====================

@app.post("/nlp/extract-entities")
async def extract_entities(
    text: str,
    request: Request = None
):
    """
    Extract design entities (NER) from any text.
    
    Args:
        text: Text to analyze
    
    Returns:
        Extracted entities grouped by type
    
    Example:
    POST /nlp/extract-entities
    {"text": "The design uses sans-serif fonts with blue accent colors..."}
    
    Response:
    {
        "entities": {
            "TYPOGRAPHY": ["sans-serif"],
            "COLOR": ["blue"]
        }
    }
    """
    request_id = request.state.request_id
    
    try:
        from hf_models import get_hf_manager
        
        manager = get_hf_manager()
        entities = manager.extract_entities_from_audit(text)
        
        log.info(
            "entities_extracted",
            request_id=request_id,
            entity_count=sum(len(v) for v in entities.values())
        )
        
        return {
            "status": "success",
            "entities": entities,
            "text_length": len(text)
        }
    
    except Exception as e:
        log.error("entity_extraction_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/classify")
async def classify_text(
    text: str,
    request: Request = None
):
    """
    Classify audit text by sentiment/category.
    
    Args:
        text: Text to classify
    
    Returns:
        Classification scores
    """
    request_id = request.state.request_id
    
    try:
        from hf_models import get_hf_manager
        
        manager = get_hf_manager()
        classification = manager.categorize_audit(text)
        
        log.info("text_classified", request_id=request_id)
        
        return {
            "status": "success",
            "classification": classification
        }
    
    except Exception as e:
        log.error("classification_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/embed")
async def embed_text(
    text: str,
    request: Request = None
):
    """
    Generate semantic embedding for text.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector (768-dim)
    
    Use for semantic search, clustering, similarity
    """
    request_id = request.state.request_id
    
    try:
        from hf_models import get_hf_manager
        
        manager = get_hf_manager()
        embedding = manager.embed_audit(text)
        
        if not embedding:
            raise HTTPException(status_code=500, detail="Embedding generation failed")
        
        log.info("text_embedded", request_id=request_id, dim=len(embedding))
        
        return {
            "status": "success",
            "embedding": embedding,
            "dimension": len(embedding),
            "text_length": len(text)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("embedding_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/search-similar")
async def search_similar_audits(
    query: str,
    audit_database: list,
    top_k: int = 5,
    request: Request = None
):
    """
    Find similar audits using semantic search.
    
    Args:
        query: Query text (audit rationale or guidelines)
        audit_database: List of audit documents with 'rationale' field
        top_k: Number of similar audits to return
    
    Returns:
        Similar audits with similarity scores
    
    Example:
    POST /nlp/search-similar
    {
        "query": "The deck uses inconsistent typography and poor color contrast",
        "audit_database": [
            {"id": "1", "rationale": "..."},
            {"id": "2", "rationale": "..."}
        ],
        "top_k": 3
    }
    """
    request_id = request.state.request_id
    
    try:
        from hf_models import get_hf_manager
        
        if not audit_database:
            raise ValueError("audit_database cannot be empty")
        
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be 1-100")
        
        manager = get_hf_manager()
        results = manager.find_similar_audits(query, audit_database, top_k=top_k)
        
        # Format results
        formatted = [
            {
                "audit": result[0],
                "similarity_score": round(result[1], 4)
            }
            for result in results
        ]
        
        log.info(
            "similar_audits_found",
            request_id=request_id,
            count=len(formatted),
            query_length=len(query)
        )
        
        return {
            "status": "success",
            "query": query,
            "results": formatted,
            "result_count": len(formatted)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("similarity_search_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PIPELINE ENDPOINTS ====================

@app.get("/pipeline/stats")
async def pipeline_stats(request: Request):
    """Get fine-tuning pipeline statistics."""
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
    """Export training dataset for fine-tuning."""
    request_id = request.state.request_id
    
    try:
        from fine_tune_pipeline import get_pipeline, DatasetFormat
        
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
                detail=f"No samples meet criteria"
            )
        
        log.info("pipeline_export_success", request_id=request_id, format=format)
        
        return {
            "status": "success",
            "format": format,
            "file": export_path
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("pipeline_export_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/validate")
async def pipeline_validate(
    sample_id: str,
    is_valid: bool,
    notes: str = "",
    request: Request = None
):
    """Manually validate a training sample."""
    request_id = request.state.request_id
    
    try:
        from fine_tune_pipeline import get_pipeline
        
        pipeline = get_pipeline()
        success = pipeline.validate_sample(sample_id, is_valid, notes)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Sample not found")
        
        log.info("sample_validated", request_id=request_id, valid=is_valid)
        
        return {
            "status": "success",
            "sample_id": sample_id,
            "is_valid": is_valid
        }
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("validation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ROOT ====================

@app.get("/")
async def root():
    """Root endpoint."""
    try:
        if os.path.exists("index.html"):
            return FileResponse("index.html", media_type="text/html")
    except:
        pass
    
    return {
        "name": "Audit Copilot MEGA-ZORD v4.6",
        "version": "4.6.0",
        "model": "gemini-1.5-flash",
        "nlp": "HuggingFace (NER, embeddings, classification)",
        "speed": "30-120 seconds per audit",
        "docs": "/docs",
        "endpoints": {
            "audit": {
                "health": "GET /health",
                "analyze_slide": "POST /analyze/slide",
                "analyze_deck": "POST /analyze/deck",
                "batch_audit": "POST /analyze/deck/batch"
            },
            "nlp": {
                "extract_entities": "POST /nlp/extract-entities",
                "classify": "POST /nlp/classify",
                "embed": "POST /nlp/embed",
                "search_similar": "POST /nlp/search-similar"
            },
            "pipeline": {
                "stats": "GET /pipeline/stats",
                "export": "POST /pipeline/export",
                "validate": "POST /pipeline/validate"
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
    log.info("startup", version="4.6.0", model="gemini-1.5-flash", nlp="enabled")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

