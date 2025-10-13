import base64
import requests
from pathlib import Path
import json

API_URL = "http://localhost:8001/ocr-receipt/"
IMAGE_FOLDER = Path("./ocr/receipts")
OUTPUT_FILE = Path("./ocr_result.json")  

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

image_paths = [p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]

if not image_paths:
    print("receipts 폴더에 이미지가 없습니다.")
    exit()

merged_ingredients = {}

for img_path in image_paths:
    with open(img_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    try:
        res = requests.post(API_URL, json={"base64_image": base64_image})
        res.raise_for_status()
        data = res.json()

        for item in data.get("ingredients", []):
            name = item["name"]
            amount = item.get("amount", 1)
            merged_ingredients[name] = merged_ingredients.get(name, 0) + amount

    except Exception as e:
        print(f"{img_path.name} 처리 중 오류 발생: {e}")

result = {
    "ingredients": [
        {"name": k, "amount": v} for k, v in merged_ingredients.items()
    ]
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"결과가 저장되었습니다: {OUTPUT_FILE.resolve()}")
