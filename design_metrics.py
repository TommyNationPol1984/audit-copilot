"""
High-Quality Quantitative Design Metrics for Audit Copilot v4.2
Includes: Contrast Ratio (WCAG AAA), Font Size Detection, Margin Analysis, Alignment, Whitespace
"""

from typing import Tuple, Dict, Any, Optional, List
from PIL import Image
import numpy as np
from collections import Counter
import structlog

log = structlog.get_logger(__name__)

# Try to import pytesseract (optional but recommended)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


# ==================== WCAG Contrast Ratio ====================

def _relative_luminance(rgb: Tuple[int, int, int]) -> float:
    r, g, b = [x / 255.0 for x in rgb]
    def adjust(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)


def calculate_contrast_ratio(fg_rgb: Tuple[int, int, int], bg_rgb: Tuple[int, int, int]) -> float:
    l1 = _relative_luminance(fg_rgb)
    l2 = _relative_luminance(bg_rgb)
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)


def get_contrast_level(ratio: float, is_large_text: bool = False) -> str:
    if is_large_text:
        if ratio >= 4.5: return "AAA Large"
        elif ratio >= 3: return "AA Large"
        else: return "Fail"
    else:
        if ratio >= 7: return "AAA"
        elif ratio >= 4.5: return "AA"
        elif ratio >= 3: return "AA Large only"
        else: return "Fail"


def check_aaa_compliance(ratio: float, is_large_text: bool = False) -> Dict:
    required = 4.5 if is_large_text else 7.0
    passes = ratio >= required
    return {
        "passes_aaa": passes,
        "required_ratio": required,
        "actual_ratio": round(ratio, 2),
        "text_type": "Large text" if is_large_text else "Normal text",
        "recommendation": "Meets AAA" if passes else f"Increase contrast to at least {required}:1"
    }


def calculate_contrast_from_image_region(
    image: Image.Image, 
    bbox: Tuple[int, int, int, int],
    is_large_text: bool = False
) -> Optional[Dict]:
    try:
        cropped = image.crop(bbox)
        pixels = [p[:3] for p in cropped.getdata() if len(p) >= 3]
        if len(pixels) < 10: return None

        most_common = Counter(pixels).most_common(5)
        bg_color = most_common[0][0]
        text_color = most_common[-1][0]

        ratio = calculate_contrast_ratio(text_color, bg_color)
        level = get_contrast_level(ratio, is_large_text=is_large_text)

        return {
            "contrast_ratio": round(ratio, 2),
            "wcag_level": level,
            "text_rgb": text_color,
            "background_rgb": bg_color,
            "passes_aa": ratio >= 4.5,
            "passes_aaa": ratio >= (4.5 if is_large_text else 7.0),
            "is_large_text": is_large_text,
            "recommended_minimum": 4.5 if is_large_text else 7.0
        }
    except Exception as e:
        log.warning("contrast_calculation_failed", error=str(e))
        return None


# ==================== Font Size Detection ====================

def estimate_font_size_from_height(text_height_px: int, dpi: int = 72) -> float:
    """Estimate font size in points from pixel height."""
    points = (text_height_px / 1.333) * (72 / dpi)
    return round(points, 1)


def detect_font_sizes_from_image(image: Image.Image) -> List[Dict]:
    """
    Detect text and estimate font sizes using pytesseract.
    Returns list of detected text with estimated pt size + is_large_text flag.
    """
    if not PYTESSERACT_AVAILABLE:
        return []

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60 and data['text'][i].strip():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                estimated_pt = estimate_font_size_from_height(h)
                is_large = estimated_pt >= 18 or (estimated_pt >= 14 and h > 22)

                results.append({
                    "text": data['text'][i],
                    "bbox": (x, y, x + w, y + h),
                    "estimated_font_size_pt": estimated_pt,
                    "text_height_px": h,
                    "is_large_text": is_large,
                    "confidence": int(data['conf'][i])
                })
        return results
    except Exception as e:
        log.error("font_size_detection_failed", error=str(e))
        return []


def get_text_size_info(image: Image.Image, bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
    """Get font size info for a region or whole image. Auto-detects large text."""
    if bbox:
        cropped = image.crop(bbox)
        detections = detect_font_sizes_from_image(cropped)
    else:
        detections = detect_font_sizes_from_image(image)

    if not detections:
        return {
            "estimated_font_size_pt": None,
            "is_large_text": False,
            "detection_method": "fallback",
            "note": "Font size detection unavailable. Defaulting to normal text."
        }

    largest = max(detections, key=lambda x: x["estimated_font_size_pt"])
    
    return {
        "estimated_font_size_pt": largest["estimated_font_size_pt"],
        "is_large_text": largest["is_large_text"],
        "text_height_px": largest["text_height_px"],
        "detection_method": "pytesseract" if PYTESSERACT_AVAILABLE else "heuristic",
        "sample_text": largest["text"][:60]
    }


# ==================== PDF Font Size Detection (pymupdf - Highly Accurate) ====================

def extract_text_with_font_sizes_from_pdf(page) -> List[Dict]:
    """
    Extract text blocks with accurate font sizes directly from a pymupdf page.
    Much more accurate than image-based OCR.
    """
    try:
        text_dict = page.get_text("dict")
        results = []

        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size = span["size"]
                    text = span["text"].strip()
                    if not text:
                        continue

                    bbox = span["bbox"]
                    is_large = font_size >= 18 or (font_size >= 14 and (span.get("flags", 0) & 16))

                    results.append({
                        "text": text[:80],
                        "bbox": bbox,
                        "font_size_pt": round(font_size, 1),
                        "font_name": span.get("font", "Unknown"),
                        "is_large_text": is_large,
                        "origin": "pymupdf"
                    })
        return results
    except Exception as e:
        log.error("pdf_font_extraction_failed", error=str(e))
        return []


def get_pdf_text_size_info(pdf_path: str, page_num: int = 0) -> Dict:
    """
    Get the most relevant font size info from a specific page of a PDF.
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]

        detections = extract_text_with_font_sizes_from_pdf(page)
        doc.close()

        if not detections:
            return {
                "estimated_font_size_pt": None,
                "is_large_text": False,
                "detection_method": "pymupdf_fallback",
                "note": "No text detected on page."
            }

        largest = max(detections, key=lambda x: x["font_size_pt"])
        return {
            "estimated_font_size_pt": largest["font_size_pt"],
            "is_large_text": largest["is_large_text"],
            "font_name": largest.get("font_name"),
            "detection_method": "pymupdf",
            "sample_text": largest["text"]
        }
    except Exception as e:
        log.error("pdf_text_size_info_failed", error=str(e))
        return {
            "estimated_font_size_pt": None,
            "is_large_text": False,
            "detection_method": "error",
            "note": str(e)
        }


def analyze_contrast_with_font_detection(
    image: Image.Image, 
    bbox: Tuple[int, int, int, int]
) -> Optional[Dict]:
    """
    Best function: Automatically detects font size and applies correct WCAG AAA thresholds.
    """
    size_info = get_text_size_info(image, bbox)
    is_large = size_info.get("is_large_text", False)

    contrast_result = calculate_contrast_from_image_region(image, bbox, is_large_text=is_large)

    if contrast_result:
        contrast_result["font_size_estimation"] = size_info
        return contrast_result
    return None


# ==================== Other Quantitative Metrics ====================

def analyze_margins(image: Image.Image, content_bbox: Optional[Tuple] = None) -> Dict:
    w, h = image.size
    if content_bbox:
        left = content_bbox[0] / w * 100
        right = (w - content_bbox[2]) / w * 100
        top = content_bbox[1] / h * 100
        bottom = (h - content_bbox[3]) / h * 100
    else:
        left = right = top = bottom = 10.0

    avg_h = (left + right) / 2
    return {
        "left_margin_pct": round(left, 1),
        "right_margin_pct": round(right, 1),
        "avg_horizontal_margin": round(avg_h, 1),
        "margin_score": round(min(avg_h, 15) / 15 * 10, 1)
    }


def estimate_alignment_deviation(image: Image.Image) -> Dict:
    try:
        gray = np.array(image.convert('L'))
        edges = np.abs(np.diff(gray, axis=1))
        score = 10 - min(np.std(edges) / 20, 10)
        return {
            "alignment_deviation_score": round(max(0, score), 1),
            "alignment_quality": "Excellent" if score > 8 else "Good" if score > 6 else "Fair"
        }
    except:
        return {"alignment_deviation_score": 7.0, "alignment_quality": "Unknown"}


def calculate_whitespace_ratio(image: Image.Image) -> Dict:
    try:
        gray = np.array(image.convert('L'))
        white = np.sum(gray > 240)
        pct = (white / gray.size) * 100
        return {
            "whitespace_percentage": round(pct, 1),
            "whitespace_score": min(10, max(0, (pct - 30) / 5))
        }
    except:
        return {"whitespace_percentage": 50.0, "whitespace_score": 7.0}


def analyze_slide_design(image: Image.Image, text_regions: Optional[List[Tuple]] = None) -> Dict[str, Any]:
    """Full quantitative analysis with automatic font size + AAA contrast detection."""
    metrics = {
        "contrast": None,
        "font_size": None,
        "margins": analyze_margins(image),
        "alignment": estimate_alignment_deviation(image),
        "whitespace": calculate_whitespace_ratio(image),
        "overall_design_score": 0.0
    }

    if text_regions and len(text_regions) > 0:
        contrast = analyze_contrast_with_font_detection(image, text_regions[0])
        if contrast:
            metrics["contrast"] = contrast
            metrics["font_size"] = contrast.get("font_size_estimation")

    scores = [
        metrics["margins"]["margin_score"],
        metrics["alignment"]["alignment_deviation_score"],
        metrics["whitespace"]["whitespace_score"]
    ]
    if metrics["contrast"]:
        scores.append(9 if metrics["contrast"]["passes_aaa"] else 6 if metrics["contrast"]["passes_aa"] else 3)

    metrics["overall_design_score"] = round(sum(scores) / len(scores), 1)
    return metrics


# ==================== Smart PDF + Image Hybrid Function (for audit_logic.py) ====================

def get_smart_text_size_and_contrast(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    pdf_path: Optional[str] = None,
    page_num: int = 0
) -> Dict:
    """
    Best of both worlds function:
    - If PDF path is provided → use accurate pymupdf font data
    - Otherwise → fall back to image-based detection
    This is the recommended function to call from audit_logic.py
    """
    if pdf_path:
        pdf_info = get_pdf_text_size_info(pdf_path, page_num)
        is_large = pdf_info.get("is_large_text", False)
        contrast = calculate_contrast_from_image_region(image, bbox, is_large_text=is_large)
        if contrast:
            contrast["font_size_estimation"] = pdf_info
            contrast["detection_source"] = "pymupdf"
            return contrast

    # Fallback to image detection
    return analyze_contrast_with_font_detection(image, bbox) or {}
