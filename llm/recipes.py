# ====================================================
# 냉장고 기반 요리 추천 API (리랭크 버전)
# ====================================================

#!pip install fastapi uvicorn openai sentence-transformers faiss-cpu pandas numpy

import os
import json
from functools import lru_cache

import pandas as pd
import numpy as np
import faiss
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# 환경 설정
# -----------------------------
_BASE_DIR = os.path.join(os.path.dirname(__file__), "llm")

FRIDGE_JSON_PATH = os.path.join(_BASE_DIR, "ingredient.json")
FAISS_INDEX_PATH = os.path.join(_BASE_DIR, "recipe_index.faiss")
RECIPES_CSV_PATH = os.path.join(_BASE_DIR, "recipes.csv")

TOP_K = 5
MODEL_NAME = "nlpai-lab/KURE-v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 모델 및 데이터 로드
# -----------------------------
print("📦 모델 및 데이터 로드 중...")


@lru_cache(maxsize=1)
def get_sentence_model():
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"❌ {FAISS_INDEX_PATH} 파일이 존재하지 않습니다.")
    return faiss.read_index(FAISS_INDEX_PATH)


@lru_cache(maxsize=1)
def get_recipes_df():
    if not os.path.exists(RECIPES_CSV_PATH):
        raise FileNotFoundError(f"❌ {RECIPES_CSV_PATH} 파일이 존재하지 않습니다.")
    return pd.read_csv(RECIPES_CSV_PATH)


print("✅ 데이터 로드 함수 준비 완료.")


router = APIRouter()


# -----------------------------
# 요청 바디 모델
# -----------------------------
class IngredientItem(BaseModel):
    ingredient: str
    quantity: float
    unit: str
    expiration_date: str


class RecipeRequest(BaseModel):
    user_query: str = None  # 예: "국물요리 추천해줘"
    ingredients: List[IngredientItem]


# -----------------------------
# 냉장고 JSON 로드
# -----------------------------
def load_fridge(ingredients_data: List[dict]):
    df = pd.DataFrame(ingredients_data)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    today = pd.Timestamp("today").normalize()
    df["days_left"] = (df["expiration_date"] - today).dt.days
    df["weight"] = 1 / (df["days_left"] + 1)
    df.loc[df["days_left"] < 0, "weight"] = 0
    return df


# -----------------------------
# 냉장고 재료 추출
# -----------------------------
def get_all_ingredients(fridge_df):
    return fridge_df["ingredient"].tolist()


# -----------------------------
# FAISS 검색 (재료 기반)
# -----------------------------
def search_recipes(ingredients_data: List[dict], top_k=10):
    fridge_df = load_fridge(ingredients_data)
    selected_ings = get_all_ingredients(fridge_df)
    if len(selected_ings) == 0:
        print("❌ 냉장고 재료가 없습니다.")
        return [], pd.DataFrame()

    weight_map = fridge_df.set_index("ingredient")["weight"].to_dict()
    base_query = ", ".join(selected_ings) + "이 들어가는 요리"
    model = get_sentence_model()
    index = get_faiss_index()
    recipes_df = get_recipes_df()

    base_emb = model.encode([base_query])

    D, I = index.search(np.array(base_emb).astype("float32"), k=top_k * 3)

    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(recipes_df):
            continue
        row = recipes_df.iloc[idx]
        recipe_ings = [ing.strip() for ing in row["재료"].split(",")]
        weight_score = sum(weight_map.get(ing, 0) for ing in recipe_ings) / (
            len(recipe_ings) or 1
        )
        results.append(
            {
                "title": row["요리 제목"],
                "ingredients": row["재료"],
                "instructions": row.get("요리 순서", ""),
                "url": row.get("상세주소", ""),
                "distance": float(D[0][list(I[0]).index(idx)]),
                "weight_score": weight_score,
            }
        )

    results = sorted(results, key=lambda x: (-x["weight_score"], x["distance"]))[
        : top_k * 2
    ]
    return selected_ings, pd.DataFrame(results)


# -----------------------------
# 사용자 의도 기반 리랭크
# -----------------------------
def rerank_recipes(df_recipes, user_query):
    if not user_query or df_recipes.empty:
        return df_recipes

    model = get_sentence_model()
    query_emb = model.encode([user_query])
    recipe_embs = model.encode(df_recipes["title"].tolist())

    scores = np.dot(recipe_embs, query_emb.T).flatten()

    df_recipes["intent_score"] = scores
    df_recipes["final_score"] = (
        df_recipes["intent_score"] * 0.6 + df_recipes["weight_score"] * 0.4
    )

    df_recipes = df_recipes.sort_values("final_score", ascending=False)
    return df_recipes.head(5)


# -----------------------------
# GPT를 통한 최종 레시피 생성
# -----------------------------
def generate_final_recipe(selected_ingredients, df_recipes, user_query=None):
    if df_recipes.empty:
        print("⚠️ 추천 가능한 레시피가 없습니다.")
        return None

    recipes_text = "\n".join(
        [
            f"- {r['title']} (재료: {r['ingredients']})"
            for r in df_recipes.to_dict(orient="records")
        ]
    )

    user_query_text = user_query if user_query else "특별한 조건 없음"

    prompt = f"""
냉장고 재료: {selected_ingredients}
사용자 요청: {user_query_text}

아래는 후보 레시피 목록이야.
냉장고 재료로 대체하거나 그대로 사용할 수 있는 재료가 많고,
사용자 요청(예: 매운 음식, 다이어트식, 간단 요리 등)에 가장 잘 맞는 레시피 하나를 선택해.


선택한 레시피를 아래 JSON 형식으로만 출력해줘.

JSON 형식 예시:
{{
  "title": "양파 계란 간장볶음",
  "category": "볶음 요리",
  "cuisine_type": "한식",
  "ingredients": [
    {{"name": "계란", "amount": 2, "unit": ea"}},
    {{"name": "양파", "amount": 1, "unit": "ea"}},
    {{"name": "간장", "amount": 15, "unit": "ml"}}
  ],
  "steps": [
    "1. 양파를 채 썬다.",
    "2. 계란을 풀고 소금 약간을 넣는다.",
    "3. 팬에 기름을 두르고 볶는다."
  ],
  "time": "8분",
  "calorie": "220 kcal"
}}

후보 레시피:
{recipes_text}

주의:
- 반드시 JSON만 출력하고, 추가 설명 금지.
- category는 '볶음 요리', '국물 요리', '디저트' 등 조리 방식 중심으로.
- cuisine_type은 '한식', '중식', '양식', '일식', '베트남식' 등 국가별 음식 유형으로 명시.
- ingredients의 단위(unit)은 반드시 'ea', 'g', 'ml' 중 하나만 사용.
- 계량이 애매하면 ea 사용.
- 사용자 요청이 있다면 조리법, 재료 비율 등을 그에 맞게 합리적으로 수정해줘.
- ```json``` 이런 포맷 표시하지 말고 JSON 내용만 출력해줘.
"""

    print("🤖 GPT가 최종 레시피 생성 중...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    recipe_text = response.choices[0].message.content.strip()
    try:
        recipe_json = json.loads(recipe_text)
    except json.JSONDecodeError:
        print("⚠️ GPT 응답이 JSON 형식이 아닙니다. 원문 출력:")
        print(recipe_text)
        return None

    return recipe_json


# -----------------------------
# API 엔드포인트
# -----------------------------
@router.post("/recommend")
def recommend_recipe(request: RecipeRequest):
    try:
        user_query = request.user_query or ""
        selected_ings, df_recipes = search_recipes(request.ingredients, top_k=TOP_K)
        df_recipes = rerank_recipes(df_recipes, user_query)
        final_recipe = generate_final_recipe(
            selected_ings, df_recipes, user_query=user_query
        )

        if not final_recipe:
            raise HTTPException(
                status_code=404, detail="추천 가능한 레시피가 없습니다."
            )

        return {
            "selected_ingredients": selected_ings,
            "candidates": df_recipes[["title", "ingredients", "final_score"]].to_dict(
                orient="records"
            ),
            "final_recipe": final_recipe,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# 로컬 실행용
# -----------------------------
# 아래 코드로 서버 실행:
# uvicorn app:app --host 0.0.0.0 --port 8000
# -----------------------------
