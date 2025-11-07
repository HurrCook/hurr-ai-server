import os
import base64
import json
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 환경변수가 필요합니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

prompt = """
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
"""


class ImageData(BaseModel):
    base64_image: str


@router.post("/ocr-receipt/")
def analyze_receipt(data: ImageData):
    base64_str = data.base64_image
    if not base64_str:
        raise HTTPException(status_code=400, detail="이미지가 비어 있습니다.")

    contents = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": contents}],
            temperature=0.0,
            max_tokens=800,
        )

        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        return json.loads(result_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 분석 실패: {e}")
