from fastapi import FastAPI

from ocr.main_ocr import router as ocr_router
from yolo.main_yolo import router as yolo_router
from llm.recipes import router as llm_router


app = FastAPI(title="Unified Ingredient Processing API")

app.include_router(ocr_router, prefix="/ocr")
app.include_router(yolo_router, prefix="/yolo")
app.include_router(llm_router, prefix="/llm")


@app.get("/")
def root():
    return {
        "message": "Unified Ingredient Processing API is running!",
        "endpoints": {
            "ocr": "/ocr/ocr-receipt/",
            "yolo": "/yolo/detect-base64/",
            "llm": "/llm/recommend",
        },
    }

