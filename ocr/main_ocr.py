import os
import base64
import json
from collections import defaultdict
from typing import List
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경변수가 필요합니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

prompt = '''
당신은 영수증을 분석하는 AI입니다.  
사용자가 제공한 영수증 텍스트에서 **식자재, 음료, 즉석식품, 가공식품**만 추출하세요.  

출력 형식(JSON):
{
  "ingredients": [
    { "name": "상품명", "amount": 수량 }
  ]
}

규칙:
1. 상품명은 불필요한 설명(브랜드, 용량, g/ml)은 제거하고 핵심 식자재/음료명만 남깁니다.  
   예: "코카콜라350ml" → "코카콜라", "큰 사각햇반300g" → "사각햇반".  
2. 동일한 종류가 여러 번 나오면 수량을 합산합니다.  
3. 가능한 한 사람이 인식하기 쉬운 식품명으로 정리합니다.  
4. 비식품 항목(종량제 봉투, 세금, 할인, 포인트 등)은 제외합니다.
'''

class ImageData(BaseModel):
    base64_images: List[str]

@router.post("/ocr-receipt/")
def analyze_receipt(data: ImageData):
    if not data.base64_images:
        raise HTTPException(status_code=400, detail="이미지 목록이 비어 있습니다.")

    def _prepare_image_payload(raw_base64: str) -> str:
        if not raw_base64:
            raise HTTPException(status_code=400, detail="빈 base64 문자열이 포함되어 있습니다.")
        if raw_base64.startswith("data:image"):
            try:
                return raw_base64.split(",", 1)[1]
            except IndexError:
                raise HTTPException(status_code=400, detail="data URI 형식이 올바르지 않습니다.")
        return raw_base64

    merged_ingredients = defaultdict(int)

    for idx, base64_str in enumerate(data.base64_images):
        processed_base64 = _prepare_image_payload(base64_str)
        contents = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{processed_base64}"}}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": contents}],
                temperature=0.0,
                max_tokens=800
            )

            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(result_text)

            for item in parsed.get("ingredients", []):
                name = item.get("name")
                amount = item.get("amount", 1)
                if name:
                    merged_ingredients[name] += amount

        except json.JSONDecodeError as json_error:
            raise HTTPException(status_code=500, detail=f"OCR JSON 파싱 실패(index={idx}): {json_error}") from json_error
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"OCR 분석 실패(index={idx}): {error}") from error

    ingredients_list = [
        {"name": name, "amount": amount}
        for name, amount in merged_ingredients.items()
    ]

    return {"ingredients": ingredients_list}


@router.get("/")
def root():
    return {"message": "OCR Receipt Ingredient API is running!"}


app = FastAPI(title="OCR Receipt Ingredient API")
app.include_router(router)
