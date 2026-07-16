"""
Integration Example: How to use new Quantitative Metrics + Movable Slides in Audit Copilot
"""

from design_metrics import analyze_slide_design
from deck_analyzer import analyze_entire_deck, generate_deck_insights
from PIL import Image

# Example: Analyzing a deck with movable slide support

def run_enhanced_audit_with_movable_slides(
    deck_id: str,
    slide_images: List[Image.Image],
    custom_slide_order: List[int] = None,   # From PWA drag-to-reorder
    design_guidelines: str = ""
):
    """
    High-quality audit that respects movable slide order and provides quantitative metrics.
    """
    print(f"Running enhanced audit for deck {deck_id}...")

    # 1. Respect movable slide order (from frontend drag-and-drop)
    if custom_slide_order:
        print(f"Applying custom slide order: {custom_slide_order}")
        ordered_images = [slide_images[i] for i in custom_slide_order]
    else:
        ordered_images = slide_images

    # 2. Run per-slide quantitative analysis
    per_slide_quantitative = []
    for idx, img in enumerate(ordered_images):
        metrics = analyze_slide_design(img)
        per_slide_quantitative.append({
            "slide_number": idx + 1,
            "quantitative_metrics": metrics
        })

    # 3. Run full deck batch analysis (Recommendation 3)
    deck_analysis = analyze_entire_deck(ordered_images, custom_slide_order)
    insights = generate_deck_insights(deck_analysis)

    # 4. (Optional) Feed quantitative data + insights into Gemini for richer rationale
    # This can be added to the existing audit_logic.py

    return {
        "deck_id": deck_id,
        "slide_order_used": custom_slide_order or list(range(len(slide_images))),
        "per_slide_quantitative": per_slide_quantitative,
        "deck_level_analysis": deck_analysis,
        "ai_insights": insights,
        "movable_slides_supported": True
    }


# Example usage
if __name__ == "__main__":
    # Simulate loading slides
    # In real use: load from /tmp/audit_copilot_decks/{deck_id}/
    print("Quantitative Metrics + Movable Slides integration ready.")