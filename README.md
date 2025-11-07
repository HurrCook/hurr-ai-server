# Conference Ingredient Services

FastAPI 기반의 통합 서버로, 다음 세 가지 기능을 한 번에 제공합니다.

- `ocr/` : 영수증 이미지를 업로드하면 OpenAI Vision API를 활용해 식자재 목록을 추출
- `yolo/` : YOLO11 모델로 이미지 속 식재료를 탐지하고 잘라낸 썸네일을 반환
- `llm/` : 냉장고 보유 재료를 바탕으로 레시피 후보를 검색하고 LLM으로 최종 레시피를 추천

## 폴더 구조

```
Conference/
├─ Dockerfile                  # 단일 컨테이너로 세 서비스 실행
├─ main.py                     # 통합 FastAPI 엔트리포인트
├─ requirements.txt            # 공통 파이썬 의존성
├─ ocr/
│  └─ main_ocr.py              # OpenAI Vision 기반 영수증 OCR 라우터
├─ yolo/
│  ├─ main_yolo.py             # YOLO11 감지 라우터
│  └─ weight/
│     └─ best.pt               # 학습된 YOLO 가중치
├─ llm/
│  ├─ main_llm.py              # 레시피 추천 전용 서버 (독립 실행용)
│  ├─ ingredient.json          # 냉장고 재료 샘플 데이터
│  ├─ recipe_embeddings.npy    # FAISS 인덱스용 임베딩
│  ├─ recipe_index.faiss       # 레시피 검색용 FAISS 인덱스
│  ├─ recipes.csv              # 원본 레시피 데이터셋
│  └─ recipes.py               # LLM 레시피 추천 라우터 로직
```

## 실행 방법

### 1. 로컬 Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 환경 변수 설정 (OpenAI Vision, LLM 호출용)
export OPENAI_API_KEY="실제_키"
# YOLO 가중치가 다른 경로에 있다면 필요 시 지정
# export YOLO_MODEL_PATH="/path/to/best.pt"

uvicorn main:app --host 0.0.0.0 --port 8000
```

- `http://localhost:8000/ocr/ocr-receipt/`
- `http://localhost:8000/yolo/detect-base64/`
- `http://localhost:8000/llm/recommend`

### 2. Docker 이미지 빌드

```bash
docker build -t unified-api .
docker run -d --name unified-api \
  -p 8000:8000 \
  -e OPENAI_API_KEY=실제키 \
  -e YOLO_MODEL_PATH=/app/yolo/weight/best.pt \
  unified-api
```

### 3. 단일 모듈만 실행하고 싶은 경우

- OCR만 실행: `uvicorn ocr.main_ocr:app --host 0.0.0.0 --port 8001`
- YOLO만 실행: `uvicorn yolo.main_yolo:app --host 0.0.0.0 --port 8002`
- LLM 추천만 실행 (독립 서버): `uvicorn llm.main_llm:app --host 0.0.0.0 --port 8003`

## API 예시

### OCR
```bash
curl -X POST http://localhost:8000/ocr/ocr-receipt/ \
  -H "Content-Type: application/json" \
  -d '{"base64_image":"<base64 문자열>"}'
```

### YOLO 감지
```bash
curl -X POST http://localhost:8000/yolo/detect-base64/ \
  -H "Content-Type: application/json" \
  -d '{"base64_image":"<base64 문자열>"}'
```

### LLM 레시피 추천
```bash
curl -X POST http://localhost:8000/llm/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_query":"국물요리 추천해줘"}'
```

## 주의사항

- OpenAI API 키는 항상 유효한 값으로 설정해야 합니다. 키가 올바르지 않으면 OCR 및 레시피 추천이 실패합니다.
- YOLO 가중치 (`best.pt`)는 저장소 용량 제한으로 인해 별도 관리가 필요할 수 있습니다. 다른 경로에 둘 경우 `YOLO_MODEL_PATH` 환경변수로 지정하세요.
- 대용량 레시피 데이터(`recipes.csv`, `recipe_index.faiss`, `recipe_embeddings.npy`)는 Git LFS 또는 외부 스토리지를 활용하는 것을 권장합니다.

## 저장소 정리 여부

- 테스트용 임시 파일(페이로드, 결과 JSON 등)은 모두 정리했습니다.
- 현재 저장소는 위의 폴더 구조로 관리됩니다.

