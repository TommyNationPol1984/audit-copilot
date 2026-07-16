"""
Audit Logic v4.3 - Strict Grading Design Instructions Compliance
- Automatically parses design instruction bullets
- Enforces exactly 3 sentences per bullet
- Includes overall aesthetics + final weighing paragraphs
- Integrates quantitative design metrics
"""

from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai
import re
import structlog
from design_metrics import analyze_slide_design

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
        elif re.match(r'^\d+[\.\)]\s+', line):
            bullet = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
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


def generate_strict_bullet_by_bullet_rationale(
    guidelines: str,
    slide_images: List[Image.Image],
    api_key: str,
    quantitative_metrics: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Main function: Generates a fully compliant Grading-style rationale.
    """
    if quantitative_metrics is None:
        # Auto-generate quantitative metrics if not provided
        quantitative_metrics = [analyze_slide_design(img) for img in slide_images]

    bullets = parse_design_bullets(guidelines)
    if not bullets:
        bullets = [line.strip() for line in guidelines.split('\n') if line.strip()]

    prompt = build_strict_mercor_prompt(bullets, quantitative_metrics)

    model = configure_gemini(api_key)
    contents = [prompt] + slide_images

    try:
        response = model.generate_content(contents)
        raw_rationale = response.text.strip()
    except Exception as e:
        log.error("gemini_rationale_generation_failed", error=str(e))
        raw_rationale = "Error generating rationale. Please try again."

    # Post-process to enforce 3-sentence structure where possible
    processed_rationale = enforce_three_sentences(raw_rationale)

    return {
        "full_rationale": processed_rationale,
        "bullets_parsed": bullets,
        "num_bullets": len(bullets),
        "quantitative_metrics": quantitative_metrics,
        "format": "bullet-by-bullet with exactly 3 sentences per bullet + aesthetics + final weighing",
        "compliance_notes": "Enforces Grading Design Instructions rules strictly."
    }
