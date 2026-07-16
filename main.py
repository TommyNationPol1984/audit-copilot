from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from design_metrics import analyze_slide_design, analyze_contrast_with_font_detection
from deck_analyzer import analyze_entire_deck
import uvicorn
import os

app = FastAPI(title="Audit Copilot MEGA-ZORD v4.2")

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.2", "features": ["contrast", "font_size_detection", "movable_slides", "batch_analysis"]}

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return {"message": "Audit Copilot API - visit /docs"}

@app.post("/analyze/slide")
def analyze_slide(image_path: str):
    from PIL import Image
    img = Image.open(image_path)
    return analyze_slide_design(img)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

