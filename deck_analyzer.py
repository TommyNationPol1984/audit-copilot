"""
Deck-level Batch Analyzer for Audit Copilot
Runs quantitative metrics across an entire deck and provides deck-wide insights.
Supports movable slide order.
"""

from typing import List, Dict, Any
from PIL import Image
from design_metrics import analyze_slide_design
import structlog

log = structlog.get_logger(__name__)


def analyze_entire_deck(
    slide_images: List[Image.Image], 
    slide_order: List[int] = None,
    text_regions_per_slide: List[List[Tuple]] = None
) -> Dict[str, Any]:
    """
    Analyze all slides in a deck and return aggregated metrics.
    
    Args:
        slide_images: List of PIL Images in current order
        slide_order: Optional custom order (for movable slides)
        text_regions_per_slide: Optional list of text bounding boxes per slide
    """
    if slide_order:
        # Reorder images according to movable slide preference
        slide_images = [slide_images[i] for i in slide_order]
        if text_regions_per_slide:
            text_regions_per_slide = [text_regions_per_slide[i] for i in slide_order]

    per_slide_results = []
    total_score = 0
    contrast_scores = []
    margin_scores = []

    for idx, img in enumerate(slide_images):
        text_regions = text_regions_per_slide[idx] if text_regions_per_slide else None
        metrics = analyze_slide_design(img, text_regions)
        
        per_slide_results.append({
            "slide_index": idx,
            "overall_design_score": metrics["overall_design_score"],
            "contrast": metrics.get("contrast"),
            "margins": metrics["margins"],
            "alignment": metrics["alignment"],
            "whitespace": metrics["whitespace"]
        })
        
        total_score += metrics["overall_design_score"]
        if metrics.get("contrast"):
            contrast_scores.append(metrics["contrast"]["contrast_ratio"])
        margin_scores.append(metrics["margins"]["avg_horizontal_margin"])

    deck_avg = round(total_score / len(slide_images), 1) if slide_images else 0

    return {
        "deck_summary": {
            "total_slides": len(slide_images),
            "average_design_score": deck_avg,
            "average_contrast_ratio": round(sum(contrast_scores) / len(contrast_scores), 2) if contrast_scores else None,
            "average_margin_pct": round(sum(margin_scores) / len(margin_scores), 1) if margin_scores else None,
            "design_quality_distribution": {
                "excellent": sum(1 for r in per_slide_results if r["overall_design_score"] >= 8.5),
                "good": sum(1 for r in per_slide_results if 7 <= r["overall_design_score"] < 8.5),
                "needs_improvement": sum(1 for r in per_slide_results if r["overall_design_score"] < 7)
            }
        },
        "per_slide_metrics": per_slide_results,
        "movable_slides_note": "Slide order was respected from provided slide_order parameter"
    }


def generate_deck_insights(analysis_result: Dict) -> List[str]:
    """Generate human-readable insights from deck analysis"""
    insights = []
    summary = analysis_result["deck_summary"]
    
    if summary["average_design_score"] >= 8.5:
        insights.append("Excellent overall design consistency across the deck.")
    elif summary["average_design_score"] >= 7:
        insights.append("Good design quality with some opportunities for polish.")
    else:
        insights.append("Several slides need significant design improvements.")

    if summary.get("average_contrast_ratio") and summary["average_contrast_ratio"] < 4.5:
        insights.append("Warning: Average contrast ratio is below WCAG AA standard.")

    if summary.get("average_margin_pct") and summary["average_margin_pct"] < 8:
        insights.append("Margins are consistently too tight across multiple slides.")

    return insights