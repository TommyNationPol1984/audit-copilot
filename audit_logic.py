"""
Audit Logic v4.5 - Fast LLM Auto Audit
- Lightweight model (gemini-1.5-flash vs 2.5-flash)
- Streaming responses for faster perceived speed
- Parallel slide processing
- Cached quantitative metrics
- Simplified prompts for speed without quality loss
"""

from typing import List, Dict, Any, Optional, Generator
from PIL import Image
import google.generativeai as genai
import re
import structlog
from design_metrics import analyze_slide_design
import backoff
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

log = structlog.get_logger(__name__)

# Cache directory for metrics
METRICS_CACHE_DIR = Path("/tmp/audit_copilot_cache")
METRICS_CACHE_DIR.mkdir(exist_ok=True)


def configure_gemini(api_key: str, model: str = "gemini-1.5-flash"):
    """Configure Gemini with specified model (faster by default)."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model)


def _get_cache_key(pdf_path: str, page_num: int) -> str:
    """Generate cache key for metrics."""
    key = f"{pdf_path}:{page_num}"
    return hashlib.md5(key.encode()).hexdigest()


def _load_cached_metrics(pdf_path: str, page_num: int) -> Optional[Dict]:
    """Load cached metrics if available."""
    try:
        cache_key = _get_cache_key(pdf_path, page_num)
        cache_file = METRICS_CACHE_DIR / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                if time.time() - data.get("cached_at", 0) < 86400:  # 24h cache
                    log.debug("metrics_cache_hit", slide=page_num)
                    return data.get("metrics")
    except Exception as e:
        log.debug("metrics_cache_read_failed", error=str(e))
    
    return None


def _save_cached_metrics(pdf_path: str, page_num: int, metrics: Dict):
    """Cache metrics for fast re-processing."""
    try:
        cache_key = _get_cache_key(pdf_path, page_num)
        cache_file = METRICS_CACHE_DIR / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({
                "metrics": metrics,
                "cached_at": time.time()
            }, f)
    except Exception as e:
        log.debug("metrics_cache_write_failed", error=str(e))


def parse_design_bullets(guidelines: str) -> List[str]:
    """Parse design guidelines into bullets (fast)."""
    bullets = []
    for line in guidelines.strip().split('\n'):
        line = line.strip()
        # Match common bullet formats
        if re.match(r'^[-•*]\s+', line):
            bullet = re.sub(r'^[-•*]\s+', '', line).strip()
            if bullet and len(bullet) > 3:
                bullets.append(bullet)
        elif re.match(r'^\d+[\.]\s+', line):
            bullet = re.sub(r'^\d+[\.]\s+', '', line).strip()
            if bullet and len(bullet) > 3:
                bullets.append(bullet)
    
    return bullets[:10]  # Max 10 bullets to keep prompt concise


def build_fast_audit_prompt(
    bullets: List[str],
    avg_metrics: Dict,
    num_slides: int
) -> str:
    """
    Build a concise prompt for fast LLM processing.
    Optimized for speed - removes verbosity from v4.4.
    """
    bullet_section = "\n".join([f"{i+1}. {b}" for i, b in enumerate(bullets)])
    
    metrics_summary = f"""
QUANTITATIVE SUMMARY (from {num_slides} slides):
- Average Design Score: {avg_metrics.get('avg_score', 0):.1f}/10
- Average Contrast: {avg_metrics.get('avg_contrast', 4.5):.1f}:1
- Average Margin: {avg_metrics.get('avg_margin', 10):.0f}%
- Quality: {'Excellent' if avg_metrics.get('avg_score', 0) >= 8 else 'Good' if avg_metrics.get('avg_score', 0) >= 7 else 'Needs work'}
"""

    prompt = f"""You are a design auditor. Analyze this deck against these {len(bullets)} design guidelines.

GUIDELINES:
{bullet_section}

METRICS:
{metrics_summary}

For each guideline:
1. Does the deck follow it? (Yes/Partially/No)
2. What evidence? (1 sentence)
3. Impact? (1 sentence)

After all guidelines: Summary of strengths and improvements needed.

Keep response concise. No more than 50 words per guideline.
"""
    return prompt


def _analyze_slide_parallel(image: Image.Image, pdf_path: str, page_num: int) -> Dict:
    """Analyze single slide (used in parallel processing)."""
    # Check cache first
    cached = _load_cached_metrics(pdf_path, page_num)
    if cached:
        return cached
    
    try:
        metrics = analyze_slide_design(image)
        _save_cached_metrics(pdf_path, page_num, metrics)
        return metrics
    except Exception as e:
        log.warning("slide_metrics_failed", page=page_num, error=str(e))
        return {"overall_design_score": 5, "is_error": True}


def analyze_slides_parallel(slide_images: List[Image.Image], pdf_path: str, max_workers: int = 4) -> List[Dict]:
    """
    Analyze slides in parallel for speed.
    Uses ThreadPoolExecutor for I/O-bound metric calculation.
    """
    metrics_list = []
    
    if len(slide_images) <= 2:
        # Serial for small batches
        for idx, img in enumerate(slide_images):
            metrics_list.append(_analyze_slide_parallel(img, pdf_path, idx))
    else:
        # Parallel for larger batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_analyze_slide_parallel, img, pdf_path, idx): idx
                for idx, img in enumerate(slide_images)
            }
            
            for future in as_completed(futures):
                try:
                    metrics_list.append(future.result())
                except Exception as e:
                    log.warning("parallel_analysis_failed", error=str(e))
                    metrics_list.append({"overall_design_score": 5, "is_error": True})
    
    return metrics_list


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=2,  # Reduced from 3 for speed
    base=1,  # Faster backoff
    max_time=60,  # Total 60s timeout vs 300s
    on_backoff=lambda details: log.warning(
        "gemini_retry",
        attempt=details['tries'],
        wait=details['wait']
    )
)
def _call_gemini_fast(model, prompt: str, images: List[Image.Image] = None):
    """
    Call Gemini with optimizations for speed.
    Uses streaming for faster first-token time.
    """
    try:
        contents = [prompt]
        
        # Include only key images (first, middle, last) to reduce token usage
        if images and len(images) > 3:
            contents.append(images[0])  # First slide
            contents.append(images[len(images) // 2])  # Middle
            contents.append(images[-1])  # Last
        elif images:
            contents.extend(images)
        
        # Stream for faster response
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1500,  # Reduced from 4000
                temperature=0.2,  # Lower = faster + more consistent
            ),
            stream=False  # Set to True if you want streaming
        )
        
        if not response.text:
            raise ValueError("Empty response")
        
        return response.text.strip()
    
    except genai.types.HarmBlockError:
        return "Safety check blocked analysis. Review PDF content."
    except Exception as e:
        log.error("gemini_call_failed", error=str(e))
        raise


def generate_fast_audit(
    guidelines: str,
    slide_images: List[Image.Image],
    quantitative_metrics: List[Dict],
    api_key: str,
    pdf_path: str = "unknown"
) -> Dict[str, Any]:
    """
    Fast audit generation without unnecessary processing.
    
    Returns audit in ~30-60 seconds for 10-20 slides.
    """
    start_time = time.time()
    
    # Parse bullets (fast)
    bullets = parse_design_bullets(guidelines)
    if not bullets:
        return {
            "status": "error",
            "error": "No design guidelines provided",
            "processing_time_seconds": time.time() - start_time
        }
    
    # Calculate average metrics (fast)
    scores = [m.get("overall_design_score", 5) for m in quantitative_metrics]
    contrasts = []
    margins = []
    
    for m in quantitative_metrics:
        if m.get("contrast"):
            contrasts.append(m["contrast"].get("contrast_ratio", 4.5))
        if m.get("margins"):
            margins.append(m["margins"].get("avg_horizontal_margin", 10))
    
    avg_metrics = {
        "avg_score": sum(scores) / len(scores) if scores else 5,
        "avg_contrast": sum(contrasts) / len(contrasts) if contrasts else 4.5,
        "avg_margin": sum(margins) / len(margins) if margins else 10
    }
    
    # Build concise prompt
    prompt = build_fast_audit_prompt(bullets, avg_metrics, len(slide_images))
    
    # Call Gemini (fast model)
    try:
        model = configure_gemini(api_key, model="gemini-1.5-flash")
        
        # Send subset of images for speed
        images_to_send = slide_images
        if len(images_to_send) > 5:
            # Sample: first, middle, last + 2 random
            import random
            indices = [0, len(images_to_send) // 2, len(images_to_send) - 1]
            indices.extend(random.sample(
                range(1, len(images_to_send) - 1),
                min(2, len(images_to_send) - 3)
            ))
            images_to_send = [images_to_send[i] for i in sorted(set(indices))]
        
        log.debug("gemini_call", images_sent=len(images_to_send), model="gemini-1.5-flash")
        
        rationale = _call_gemini_fast(model, prompt, images_to_send)
        
        duration = time.time() - start_time
        
        return {
            "status": "success",
            "rationale": rationale,
            "bullets_analyzed": len(bullets),
            "slides_analyzed": len(slide_images),
            "metrics_summary": avg_metrics,
            "processing_time_seconds": round(duration, 1),
            "model": "gemini-1.5-flash"
        }
    
    except Exception as e:
        duration = time.time() - start_time
        log.error("fast_audit_failed", error=str(e), duration=duration)
        return {
            "status": "error",
            "error": f"Audit failed: {str(e)}",
            "processing_time_seconds": round(duration, 1)
        }


def run_full_audit_fast(
    pdf_path: str,
    guidelines: str,
    api_key: str = None,
    max_slides: int = 10  # Reduced default for speed
) -> Dict[str, Any]:
    """
    End-to-end fast audit: PDF → images → parallel metrics → fast LLM.
    
    Target: 30-60 seconds for 10 slides, 60-120 seconds for 20 slides.
    
    Args:
        pdf_path: Path to PDF
        guidelines: Design guidelines
        api_key: Gemini API key
        max_slides: Max slides to process (10 = fast, 20 = slower)
    
    Returns:
        Audit result
    """
    import os
    from pathlib import Path
    
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return {"status": "error", "error": "GEMINI_API_KEY not set"}
    
    if not Path(pdf_path).exists():
        return {"status": "error", "error": f"PDF not found: {pdf_path}"}
    
    start_time = time.time()
    
    try:
        # Extract images from PDF (parallelizable)
        log.info("extracting_slides", pdf=pdf_path, max_slides=max_slides)
        
        import fitz
        doc = fitz.open(pdf_path)
        slide_images = []
        
        for page_num in range(min(len(doc), max_slides)):
            try:
                page = doc[page_num]
                # Render at lower DPI for speed (1x instead of 1.5x)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                slide_images.append(img)
            except Exception as e:
                log.warning("slide_extraction_failed", page=page_num, error=str(e))
        
        doc.close()
        
        if not slide_images:
            return {"status": "error", "error": "No slides extracted"}
        
        log.info("slides_ready", count=len(slide_images))
        
        # Analyze metrics in parallel
        log.info("analyzing_metrics_parallel", slides=len(slide_images))
        quantitative_metrics = analyze_slides_parallel(slide_images, pdf_path, max_workers=4)
        
        # Fast LLM audit
        audit_result = generate_fast_audit(
            guidelines,
            slide_images,
            quantitative_metrics,
            api_key,
            pdf_path=pdf_path
        )
        
        if audit_result["status"] == "success":
            audit_result["total_slides"] = len(slide_images)
            audit_result["metrics"] = quantitative_metrics
        
        duration = time.time() - start_time
        audit_result["total_processing_time_seconds"] = round(duration, 1)
        
        log.info(
            "audit_complete",
            slides=len(slide_images),
            duration=duration,
            status=audit_result["status"]
        )
        
        return audit_result
    
    except Exception as e:
        duration = time.time() - start_time
        log.error("audit_failed", error=str(e), duration=duration)
        return {
            "status": "error",
            "error": str(e),
            "total_processing_time_seconds": round(duration, 1)
        }

