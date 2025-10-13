Conference/
├── docker-compose.yml # YOLO + OCR 컨테이너 통합 실행 설정
├── .env # OpenAI API Key 
│
├── yolo/ # YOLO 식자재 탐지 서버
│ ├── Ingredients # YOLO API 테스트 이미지
│ ├── main_yolo.py # YOLO FastAPI 메인 코드
│ ├── weights/ # YOLO 모델 가중치 (best.pt)
│ ├── requirements.txt # YOLO API 의존성
│ ├── Dockerfile # YOLO Docker 설정
| └── client_yolo_infer.py # YOLO API 테스트 스크립트
│
├── ocr/ # OCR 영수증 분석 서버
│ ├── receipts # OCR API 테스트 이미지
│ ├── main_ocr.py # OCR FastAPI 메인 코드
│ ├── requirements.txt # OCR API 의존성
│ ├── Dockerfile # OCR Docker 설정
| └── client_ocr_infer.py # OCR API 테스트 스크립트