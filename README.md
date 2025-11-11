# AI Camera & Recipe LLM

FastAPI 기반의 통합 서버로, 다음 세 가지 기능을 한 번에 제공

- `ocr/` : 영수증 이미지를 업로드하면 GPT API를 활용해 식자재 목록을 추출
- `yolo/` : YOLO11 모델로 이미지 속 식재료를 탐지하고 잘라낸 썸네일을 반환
- `llm/` : 냉장고 보유 재료를 바탕으로 레시피 후보를 검색하고 LLM으로 최종 레시피를 추천

## 폴더 구조

```
Conference/
├─ Dockerfile                  # 통합 컨테이너 정의
├─ main.py                     # FastAPI 엔트리포인트 (3개 라우터 포함)
├─ requirements.txt            # 공통 Python 의존성
├─ ocr/
│  └─ main_ocr.py              # OCR 라우터 (/ocr/*)
├─ yolo/
│  ├─ main_yolo.py             # YOLO 감지 라우터 (/yolo/*)
│  └─ weight/
│     └─ best.pt               # YOLO11 학습 가중치
├─ llm/
│  ├─ llm.py                   # 레시피 추천 라우터 (/llm/*)
│  ├─ ingredient.json          # 냉장고 DB 데이터
│  ├─ recipe_index.faiss       # FAISS 검색 인덱스
│  ├─ recipes.csv              # 레시피 데이터셋
│  └─ tools.json               # 도구 DB 데이터
└─ README.md
```

## 체크포인트 & 데이터 다운로드
- 링크 : https://drive.google.com/drive/folders/1pitMIVsfyYcCplb1g-sRuds2XrjzpbCC?usp=sharing

---

## Docker 실행

### 1. 이미지 빌드

```bash
docker build -t unified-api .
```

### 2. 컨테이너 실행

```bash
docker run --rm \
  -e OPENAI_API_KEY="실제_키" \
  -e YOLO_MODEL_PATH="/app/yolo/weight/best.pt" \
  -p 8000:8000 \
  unified-api
```

- `OPENAI_API_KEY`: OCR과 LLM 호출에 필수 
- `YOLO_MODEL_PATH`: YOLO 가중치 경로. 컨테이너 내 기본값은 `/app/yolo/weight/best.pt`
- 실행 후 `http://localhost:8000/docs`에서 Swagger UI 확인 가능

### 3. 종료

```bash
# foreground 실행 중이면 Ctrl+C
# 또는 백그라운드 실행 시
docker stop unified-api
```

---

## 개별 모듈 실행

### OCR (영수증 식자재 추출)
```bash
uvicorn ocr.main_ocr:app --host 0.0.0.0 --port 8001
```
- 엔드포인트: `http://localhost:8001/ocr-receipt/`
- OpenAI 모델을 사용하므로 `OPENAI_API_KEY` 환경변수가 필요

### YOLO (카메라 식재료 감지)
```bash
uvicorn yolo.main_yolo:app --host 0.0.0.0 --port 8002
```
- 엔드포인트: `http://localhost:8002/detect-base64/`
- `YOLO_MODEL_PATH`를 지정하지 않으면 기본값 `yolo/weight/best.pt`를 사용

### LLM 레시피 추천
```bash
uvicorn llm.llm:app --host 0.0.0.0 --port 8003
```
- 엔드포인트: `http://localhost:8003/recommend`
- `ingredient.json`, `recipe_index.faiss`, `recipes.csv`, `tools.json` 파일이 필요하며 `OPENAI_API_KEY`가 설정돼 있어야 함

---

## 환경 변수 / 의존 파일

| 항목 | 설명 |
| --- | --- |
| `OPENAI_API_KEY` | OCR·LLM 모두 OpenAI API 사용하므로 필수 |
| `YOLO_MODEL_PATH` | YOLO 라우터에서 참조하는 가중치 경로 |
| `llm/ingredient.json` | 냉장고 DB 정보 |
| `llm/recipe_index.faiss` | SentenceTransformer 임베딩 기반 FAISS 인덱스 |
| `llm/recipes.csv` | 레시피 데이터 |
| `llm/tools.json` | 도구 DB 정보 |

---

## API 예시 (요청 & 응답 형식)

### OCR (POST `/ocr/ocr-receipt/`)
요청 명령
```bash
curl -X POST http://localhost:8000/ocr/ocr-receipt/ \
  -H "Content-Type: application/json" \
  -d '{
        "base64_images": [
          "<base64 문자열 1>",
          "<base64 문자열 2>"
        ]
      }'
```

응답 예시 
```json
{
  "ingredients": [
    {"name": "계란", "amount": 2},
    {"name": "양파", "amount": 1},
    {"name": "두부", "amount": 1}
  ]
}
```

### YOLO 감지 (POST `/yolo/detect-base64/`)
요청 명령
```bash
curl -X POST http://localhost:8000/yolo/detect-base64/ \
  -H "Content-Type: application/json" \
  -d '{
        "base64_images": [
          "<base64 문자열 1>",
          "<base64 문자열 2>"
        ]
      }'
```

응답 예시
```json
{
  "ingredients": [
    {
      "name": "tomato",
      "amount": 3,
      "crop_image": ["<base64-encoded-crop>"]
    }
  ]
}
```

### LLM 레시피 추천 (POST `/llm/recommend`)
요청 명령
```bash
curl -X POST http://localhost:8000/llm/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "user_query": "국물요리 추천해줘",
        "personal_preferences": "저염, 채소 위주",
        "fridge": {
          "ingredients": [
            {"ingredient": "양파", "quantity": 1, "unit": "ea", "expiration_date": "2025-12-31"}
          ]
        },
        "tools": ["칼", "도마", "냄비"]
      }'
```

응답 예시
```json
{
  "title": "멸치 칼국수",
  "category": "국물 요리",
  "cuisine_type": "한식",
  "ingredients": [
    {"name": "양파", "amount": 1, "unit": "ea"},
    {"name": "팽이버섯", "amount": 150, "unit": "g"},
    {"name": "간장", "amount": 15, "unit": "ml"}
  ],
  "tools": ["냄비", "칼", "도마"],
  "steps": [
    "1. 양파와 버섯을 준비한다.",
    "2. 멸치 육수를 끓인다.",
    "3. 면과 재료를 넣고 간을 맞춰 마무리한다."
  ],
  "time": "30분",
  "calorie": "450 kcal"
}
```

---

## API 요약

| 엔드포인트 | 주요 입력 | 주요 출력 |
| --- | --- | --- |
| `POST /ocr/ocr-receipt/` | `base64_images` (영수증 이미지 배열) | 이미지별 식자재를 취합한 목록 (`ingredients`: 이름, 수량) |
| `POST /yolo/detect-base64/` | `base64_images` (카메라 이미지 배열) | 감지된 식재료 합산 및 클래스별 첫 크롭 (`crop_image`) |
| `POST /llm/recommend` | `user_query`, `personal_preferences`, `fridge`, `tools` (옵션) | 최종 레시피 정보 (재료, 도구, 순서, 시간, 칼로리) |

---


