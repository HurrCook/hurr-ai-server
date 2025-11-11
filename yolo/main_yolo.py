import os
import re
import io
import base64
import tempfile
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image

router = APIRouter()

_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "weight", "best.pt")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", _DEFAULT_MODEL_PATH)
model = YOLO(MODEL_PATH)

class ImageData(BaseModel):
    base64_image: str

@router.post("/detect-base64/")
def detect_base64_image(data: ImageData):
    base64_str = data.base64_image

    if base64_str.startswith("data:image"):
        base64_str = re.sub("^data:image/[^;]+;base64,", "", base64_str)

    try:
        image_data = base64.b64decode(base64_str)
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 base64 문자열입니다.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name

    results = model(temp_path)
    os.remove(temp_path)

    original_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    total_detected = defaultdict(int)
    cropped_images_by_class = defaultdict(list)

    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        total_detected[class_name] += 1

        bbox = [int(x) for x in box.xyxy[0].tolist()]
        cropped_img = original_image.crop(bbox)

        buf = io.BytesIO()
        cropped_img.save(buf, format="JPEG")
        crop_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        if not cropped_images_by_class[class_name]:
            cropped_images_by_class[class_name].append(crop_base64)

    ingredients = []
    for name, count in total_detected.items():
        ingredients.append({
            "name": name,
            "amount": count,
            "crop_image": cropped_images_by_class[name]
        })

    return {"ingredients": ingredients}

@router.get("/")
def root():
    return {"message": "YOLO11 Ingredient Detection API is running!"}


app = FastAPI(title="YOLO11 Ingredient Detection API")
app.include_router(router)
