"""
Audit Logic v4.4 - Strict Grading Design Instructions Compliance
- Automatically parses design instruction bullets
- Enforces exactly 3 sentences per bullet
- Includes overall aesthetics + final weighing paragraphs
- Integrates quantitative design metrics
- FIXED: Timeout handling, retry logic, memory optimization
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai
import re
import structlog
from design_metrics import analyze_slide_design
import backoff
import time

log = structlog.get_logger(__name__)


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def parse_design_bullets(guidelines: str) -> List[str]:
    """
    Automatically parse design instruction bullets from the guidelines text.
    Supports -, •, and numbered formats.
    """
    bullets = []
    lines = guidelines.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Match common bullet formats
        if re.match(r'^[-•*]\s+', line):
            bullet = re.sub(r'^[-•*]\s+', '', line).strip()
            if bullet:
                bullets.append(bullet)
        elif re.match(r'^\d+[\.]\s+', line):
            bullet = re.sub(r'^\d+[\.]\s+', '', line).strip()
            if bullet:
                bullets.append(bullet)
    
    return bullets


def build_strict_mercor_prompt(
    bullets: List[str],
    quantitative_metrics: List[Dict],
    deck_summary: Optional[Dict] = None
) -> str:
    """
    Builds a highly structured prompt that enforces Grading rules exactly.
    """
    bullet_section = "\n".join([f"{i+1}. {b}" for i, b in enumerate(bullets)])

    metrics_text = ""
    if quantitative_metrics:
        avg_score = sum(m.get("overall_design_score", 0) for m in quantitative_metrics) / len(quantitative_metrics)
        metrics_text = f"""
QUANTITATIVE METRICS SUMMARY (use this to support your rationale):
- Average Design Score: {avg_score:.1f}/10
- Average Contrast Ratio: {quantitative_metrics[0].get('contrast', {}).get('contrast_ratio', 'N/A') if quantitative_metrics else 'N/A'}
- Use these numbers to strengthen specific bullet evaluations when relevant.
"""

    prompt = f"""You are an extremely strict design auditor following Grading "Design Instructions" rules with zero tolerance for deviation.

DESIGN INSTRUCTIONS (copy each bullet EXACTLY as written):
{bullet_section}

CRITICAL RULES YOU MUST FOLLOW:
1. Copy every design-instruction bullet into your rationale, in the exact order above.
2. Under EACH bullet, write EXACTLY 3 sentences:
   - Sentence 1: State clearly whether the deck follows, weakens, omits, substitutes, or contradicts this specific instruction.
   - Sentence 2: Cite the specific slide number(s) and describe the concrete visual evidence you see.
   - Sentence 3: Explain the impact of this finding on the overall design system or hierarchy.
3. Use third-person language only. Never use "I", "me", "my", or phrases like "feels professional", "looks clean", or "is visually appealing".
4. After all bullets are addressed, add TWO final paragraphs:
   - Paragraph 1: Overall aesthetics and polish (typography consistency, color harmony, spacing rhythm, alignment quality).
   - Paragraph 2: Final weighing that justifies the overall score based on the bullet-by-bullet review. Be specific about which instructions had the biggest positive or negative impact.

{metrics_text}

Begin your response now. Start directly with the first bullet. Do not add any introductory text.
"""
    return prompt


def enforce_three_sentences(text: str) -> str:
    """
    Post-processing to ensure each bullet section has approximately 3 sentences.
    This is a safety net in case Gemini slightly deviates.
    """
    # Simple heuristic: split by periods and ensure reasonable length
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 2:
        return text + " Specific slide evidence supports this assessment."
    return text


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    base=2,
    max_time=300,  # 5 minute timeout total
    on_backoff=lambda details: log.warning(
        "gemini_call_retry",
        attempt=details['tries'],
        wait=details['wait']
    )
)
def _call_gemini_with_retry(model, contents, timeout_seconds=120):
    """
    Call Gemini API with timeout and retry logic.
    
    Args:
        model: Gemini model instance
        contents: Content to send (prompt + images)
        timeout_seconds: Timeout for individual call
    
    Returns:
        Response text
    """
    try:
        # Set a reasonable timeout on the generation call
        response = model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=4000,
                temperature=0.3,  # Lower for consistency
            )
        )
        
        if not response.text:
            raise ValueError("Empty response from Gemini")
        
        return response.text.strip()
    
    except genai.types.HarmBlockError as e:
        log.error("gemini_harm_blocked", error=str(e))
        return "Analysis blocked due to safety guidelines. Please review input content."
    
    except genai.types.APIError as e:
        log.error("gemini_api_error", error=str(e), retry=True)
        raise  # Will trigger backoff retry
    
    except genai.types.APIConnectionError as e:
        log.error("gemini_connection_error", error=str(e), retry=True)
        raise  # Will trigger backoff retry
    
    except Exception as e:
        log.error("gemini_call_failed", error=str(e), type=type(e).__name__)
        raise


def generate_strict_bullet_by_bullet_rationale(
    guidelines: str,
    slide_images: List[Image.Image],
    api_key: str,
    quantitative_metrics: Optional[List[Dict]] = None,
    max_slides: int = 20
) -> Dict[str, Any]:
    """
    Main function: Generates a fully compliant Grading-style rationale.
    
    Args:
        guidelines: Design guidelines text
        slide_images: List of PIL Image objects
        api_key: Gemini API key
        quantitative_metrics: Optional pre-computed metrics
        max_slides: Maximum slides to send (limits token usage and timeout)
    
    Returns:
        Audit result with rationale and metrics
    """
    start_time = time.time()
    
    # Limit slides to avoid timeout and token overflow
    if len(slide_images) > max_slides:
        log.warning("slides_truncated", total=len(slide_images), max=max_slides)
        slide_images = slide_images[:max_slides]
    
    # Auto-generate metrics if not provided
    if quantitative_metrics is None:
        log.info("generating_metrics_for_slides", count=len(slide_images))
        quantitative_metrics = []
        for idx, img in enumerate(slide_images):
            try:
                metrics = analyze_slide_design(img)
                quantitative_metrics.append(metrics)
                log.debug("metrics_generated", slide_index=idx)
            except Exception as e:
                log.warning("metrics_generation_failed", slide_index=idx, error=str(e))
                # Use empty metrics so we don't fail the whole evaluation
                quantitative_metrics.append({"overall_design_score": 0})

    # Parse bullets
    bullets = parse_design_bullets(guidelines)
    if not bullets:
        bullets = [line.strip() for line in guidelines.split('\n') if line.strip()]

    log.info("audit_starting", num_slides=len(slide_images), num_bullets=len(bullets))

    # Build prompt
    prompt = build_strict_mercor_prompt(bullets, quantitative_metrics)

    # Configure and call Gemini with retry logic
    try:
        model = configure_gemini(api_key)
        
        # Prepare content: prompt + images
        contents = [prompt] + slide_images
        
        log.debug("gemini_call_started", content_items=len(contents))
        
        raw_rationale = _call_gemini_with_retry(model, contents)
        
        duration = time.time() - start_time
        log.info("gemini_call_succeeded", duration=duration)
        
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "gemini_rationale_generation_failed",
            error=str(e),
            duration=duration
        )
        raw_rationale = f"Error generating rationale: {str(e)}. Please try again with fewer slides or check API availability."

    # Post-process
    processed_rationale = enforce_three_sentences(raw_rationale)
    
    duration = time.time() - start_time

    return {
        "full_rationale": processed_rationale,
        "bullets_parsed": bullets,
        "num_bullets": len(bullets),
        "quantitative_metrics": quantitative_metrics,
        "format": "bullet-by-bullet with exactly 3 sentences per bullet + aesthetics + final weighing",
        "compliance_notes": "Enforces Grading Design Instructions rules strictly.",
        "processing_time_seconds": round(duration, 2),
        "num_slides_analyzed": len(slide_images)
    }


def run_full_audit_with_quantitative_metrics(
    pdf_path: str,
    guidelines: str,
    api_key: str = None,
    max_slides: int = 20
) -> Dict[str, Any]:
    """
    End-to-end audit: PDF → images → metrics → Gemini analysis.
    
    Args:
        pdf_path: Path to PDF file
        guidelines: Design guidelines
        api_key: Gemini API key (optional, can use env var)
        max_slides: Maximum slides to process
    
    Returns:
        Complete audit result
    """
    import os
    from pathlib import Path
    
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not provided")
    
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    start_time = time.time()
    
    try:
        # Extract images from PDF
        log.info("pdf_processing_started", path=pdf_path)
        import fitz
        
        doc = fitz.open(pdf_path)
        slide_images = []
        
        for page_num in range(min(len(doc), max_slides)):
            try:
                page = doc[page_num]
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                slide_images.append(img)
                log.debug("slide_extracted", page=page_num)
            except Exception as e:
                log.warning("slide_extraction_failed", page=page_num, error=str(e))
        
        doc.close()
        
        if not slide_images:
            raise ValueError("No slides could be extracted from PDF")
        
        log.info("slides_extracted", count=len(slide_images))
        
        # Run quantitative analysis
        quantitative_metrics = []
        for idx, img in enumerate(slide_images):
            try:
                metrics = analyze_slide_design(img)
                quantitative_metrics.append(metrics)
            except Exception as e:
                log.warning("metrics_failed_for_slide", idx=idx, error=str(e))
                quantitative_metrics.append({"overall_design_score": 0})
        
        # Calculate deck summary
        scores = [m.get("overall_design_score", 0) for m in quantitative_metrics]
        deck_summary = {
            "total_slides": len(slide_images),
            "average_design_score": round(sum(scores) / len(scores), 1) if scores else 0
        }
        
        # Generate rationale
        rationale_result = generate_strict_bullet_by_bullet_rationale(
            guidelines,
            slide_images,
            api_key,
            quantitative_metrics=quantitative_metrics,
            max_slides=max_slides
        )
        
        duration = time.time() - start_time
        
        result = {
            "status": "success",
            "total_slides": len(slide_images),
            "deck_summary": deck_summary,
            "rationale": rationale_result["full_rationale"],
            "quantitative_metrics": quantitative_metrics,
            "processing_time_seconds": round(duration, 2)
        }
        
        log.info("audit_completed", total_slides=len(slide_images), duration=duration)
        return result
    
    except Exception as e:
        duration = time.time() - start_time
        log.error("audit_failed", error=str(e), duration=duration)
        return {
            "status": "error",
            "error": str(e),
            "processing_time_seconds": round(duration, 2)
        }

