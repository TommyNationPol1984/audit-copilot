from fastapi import FastAPI
from design_metrics import analyze_slide_design, analyze_contrast_with_font_detection
from deck_analyzer import analyze_entire_deck
import uvicorn

app = FastAPI(title="Audit Copilot MEGA-ZORD v4.2")

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.2", "features": ["contrast", "font_size_detection", "movable_slides", "batch_analysis"]}

@app.post("/analyze/slide")
def analyze_slide(image_path: str):
    from PIL import Image
    img = Image.open(image_path)
    return analyze_slide_design(img)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)