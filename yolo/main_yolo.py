import os
import re
import io
import base64
import tempfile
from collections import defaultdict
from typing import Any, Dict, List

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

router = APIRouter()

_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "weight", "best.pt")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", _DEFAULT_MODEL_PATH)
model = YOLO(MODEL_PATH)

class ImageData(BaseModel):
    base64_images: List[str]


def _clean_base64(raw_base64: str) -> str:
    if not raw_base64:
        raise HTTPException(status_code=400, detail="빈 base64 문자열이 포함되어 있습니다.")
    if raw_base64.startswith("data:image"):
        try:
            return raw_base64.split(",", 1)[1]
        except IndexError as split_error:
            raise HTTPException(status_code=400, detail="data URI 형식이 올바르지 않습니다.") from split_error
    return raw_base64


def _detect_single_image(image_bytes: bytes) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        results = model(temp_path)
    finally:
        os.remove(temp_path)

    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    detections: List[Dict[str, Any]] = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]

        bbox = [int(x) for x in box.xyxy[0].tolist()]
        cropped_img = original_image.crop(bbox)

        buf = io.BytesIO()
        cropped_img.save(buf, format="JPEG")
        crop_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        detections.append({"name": class_name, "crop_image": crop_base64})
    return detections


@router.post("/detect-base64/")
def detect_base64_image(data: ImageData):
    if not data.base64_images:
        raise HTTPException(status_code=400, detail="이미지 목록이 비어 있습니다.")

    total_detected = defaultdict(int)
    cropped_images_by_class = defaultdict(list)

    for idx, raw_base64 in enumerate(data.base64_images):
        cleaned_base64 = _clean_base64(raw_base64)
        try:
            image_data = base64.b64decode(cleaned_base64)
        except Exception as decode_error:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 base64 문자열입니다.(index={idx})") from decode_error

        try:
            detections = _detect_single_image(image_data)
        except HTTPException:
            raise
        except Exception as detect_error:
            raise HTTPException(status_code=500, detail=f"YOLO 추론 실패(index={idx}): {detect_error}") from detect_error

        for detection in detections:
            name = detection["name"]
            crop_base64 = detection["crop_image"]
            total_detected[name] += 1
            if not cropped_images_by_class[name]:
                cropped_images_by_class[name].append(crop_base64)

    ingredients = [
        {"name": name, "amount": count, "crop_image": cropped_images_by_class[name]}
        for name, count in total_detected.items()
    ]

    return {"ingredients": ingredients}

@router.get("/")
def root():
    return {"message": "YOLO11 Ingredient Detection API is running!"}


app = FastAPI(title="YOLO11 Ingredient Detection API")
app.include_router(router)
