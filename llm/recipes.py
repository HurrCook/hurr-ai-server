#!pip install fastapi uvicorn openai sentence-transformers faiss-cpu pandas numpy

import os
import json
from functools import lru_cache
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import faiss
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

_BASE_DIR = os.path.dirname(__file__)

FRIDGE_JSON_PATH = os.path.join(_BASE_DIR, "ingredient.json")
TOOLS_JSON_PATH = os.path.join(_BASE_DIR, "tools.json")
FAISS_INDEX_PATH = os.path.join(_BASE_DIR, "recipe_index.faiss")
RECIPES_CSV_PATH = os.path.join(_BASE_DIR, "recipes.csv")

TOP_K = 10
MODEL_NAME = "nlpai-lab/KURE-v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

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


class RecipeRequest(BaseModel):
    user_query: Optional[str] = None
    personal_preferences: Optional[str] = None
    ingredients: Optional[Union[List[dict], dict]] = None
    tools: Optional[Union[List[str], dict]] = None


def load_fridge(source=None):
    if source is None:
        source = FRIDGE_JSON_PATH

    data = None
    if isinstance(source, (str, os.PathLike)):
        resolved_path = os.fspath(source)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(_BASE_DIR, resolved_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"❌ {resolved_path} 파일이 없습니다.")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    if data is None:
        data = []

    if isinstance(data, dict):
        ingredients = data.get("ingredients", [])
    elif isinstance(data, list):
        ingredients = data
    else:
        raise ValueError("❌ ingredient 데이터 형식이 올바르지 않습니다. dict 또는 list여야 합니다.")

    if not isinstance(ingredients, list):
        raise ValueError("❌ ingredient 데이터의 ingredients 형식이 리스트가 아닙니다.")

    if not ingredients:
        df = pd.DataFrame(columns=["ingredient", "quantity", "unit", "expiration_date"])
    else:
        df = pd.DataFrame(ingredients)

    if "ingredient" not in df.columns:
        raise ValueError("❌ ingredient 데이터에 'ingredient' 필드가 필요합니다.")

    if "expiration_date" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
    else:
        df["expiration_date"] = pd.NaT

    today = pd.Timestamp("today").normalize()
    df["days_left"] = (df["expiration_date"] - today).dt.days
    df["weight"] = 1 / (df["days_left"] + 1)
    df.loc[df["days_left"] < 0, "weight"] = 0
    return df


def load_tools(source=None) -> List[str]:
    if source is None:
        source = TOOLS_JSON_PATH

    data = None
    if isinstance(source, (str, os.PathLike)):
        resolved_path = os.fspath(source)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(_BASE_DIR, resolved_path)
        if not os.path.exists(resolved_path):
            print(f"⚠️ {resolved_path} 파일을 찾을 수 없어 도구 정보를 불러오지 못했습니다.")
            return []
        with open(resolved_path, "r", encoding="utf-8") as tools_file:
            data = json.load(tools_file)
    else:
        data = source

    if data is None:
        return []

    if isinstance(data, dict):
        data = data.get("tools", [])

    if not isinstance(data, list):
        raise ValueError("❌ tools 데이터 형식이 리스트 또는 딕셔너리가 아닙니다.")

    cleaned_tools: List[str] = []
    for tool in data:
        if isinstance(tool, dict):
            value = tool.get("name") or tool.get("tool")
        else:
            value = tool

        if value is None:
            continue

        text = str(value).strip()
        if text:
            cleaned_tools.append(text)

    return cleaned_tools


def get_all_ingredients(fridge_df):
    seen = set()
    unique = []
    for ing in fridge_df["ingredient"].tolist():
        key = ing.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(ing.strip())
    return unique


def search_recipes(fridge_source=None, top_k=10):
    fridge_df = load_fridge(fridge_source)
    selected_ings = get_all_ingredients(fridge_df)
    if len(selected_ings) == 0:
        print("❌ 냉장고 재료가 없습니다.")
        return [], pd.DataFrame()

    weight_map = fridge_df.set_index("ingredient")["weight"].to_dict()
    base_query = ", ".join(selected_ings)
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

        recipe_ings = [ing.strip() for ing in str(row["재료"]).split(",")]
        weight_score = sum(weight_map.get(ing, 0) for ing in recipe_ings) / (len(recipe_ings) or 1)

        results.append({
            "title": row["요리 제목"],
            "ingredients": row["재료"],
            "recipe": row.get("요리 순서", ""), 
            "instructions": row.get("요리 순서", ""), 
            "url": row.get("상세주소", ""),
            "distance": float(D[0][list(I[0]).index(idx)]),
            "weight_score": weight_score
        })

    results = sorted(results, key=lambda x: (-x["weight_score"], x["distance"]))[:top_k * 3]
    return selected_ings, pd.DataFrame(results)


def rerank_recipes(df_recipes, user_query, personal_preferences=None):
    def _select_columns(df):
        keep_cols = [col for col in ["title", "ingredients", "recipe"] if col in df.columns]
        return df[keep_cols].copy() if keep_cols else df.copy()

    if not (user_query or personal_preferences) or df_recipes.empty:
        return _select_columns(df_recipes)

    candidates = df_recipes.reset_index(drop=True).to_dict(orient="records")

    candidate_descriptions = "\n".join([
        (
            f"{idx + 1}. 제목: {row.get('title', '')}\n"
            f"   재료: {row.get('ingredients', '')}\n"
            f"   요약: {str(row.get('recipe', '')).strip()}"
        )
        for idx, row in enumerate(candidates)
    ])

    ranking_prompt = f"""
사용자 요청: {user_query}
개인맞춤 설정: {personal_preferences or "없음"}

후보 레시피 목록:
{candidate_descriptions}

지시사항:
- 사용자 요청에 가장 잘 맞는 레시피부터 순서대로 나열해.
- relevance 점수는 0과 1 사이의 실수로, 사용자 요청과의 적합도를 의미해.
- 최대 5개까지만 포함해.
- JSON만 출력하고 다른 텍스트는 금지.

JSON 형식 예시:
{{
  "ranking": [
    {{ "id": 3, "score": 0.9 }},
    {{ "id": 1, "score": 0.75 }},
    {{ "id": 5, "score": 0.6 }}
  ]
}}
"""

    def _extract_json(text: str):
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`")
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
            if "\n" in cleaned:
                cleaned = cleaned.split("\n", 1)[-1]
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            cleaned = cleaned[start:end]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ranking_prompt}],
        temperature=0.2,
    )
    ranking_text = response.choices[0].message.content.strip()
    ranking_json = _extract_json(ranking_text) if ranking_text else None

    if not (ranking_json and isinstance(ranking_json.get("ranking"), list)):
        raise ValueError("LLM이 유효한 ranking 결과를 반환하지 않았습니다.")

    if ranking_json and isinstance(ranking_json.get("ranking"), list):
        filtered_entries = []
        for entry in ranking_json["ranking"]:
            idx = entry.get("id")
            score = entry.get("score")
            if not isinstance(idx, int):
                continue
            if idx < 1 or idx > len(candidates):
                continue
            if not isinstance(score, (int, float)):
                continue
            score_value = float(score)
            if score_value < 0.3:
                continue
            filtered_entries.append((idx - 1, score_value))

        if filtered_entries:
            score_map = {cand_idx: score for cand_idx, score in filtered_entries}

            ranked_df = df_recipes.reset_index(drop=True).copy()
            ranked_df = ranked_df.loc[list(score_map.keys())].copy()
            if ranked_df.empty:
                return _select_columns(ranked_df)

            ranked_df["intent_score"] = ranked_df.index.map(lambda idx: score_map.get(idx, 0.0))
            ranked_df["final_score"] = (
                ranked_df["intent_score"] * 0.7 + ranked_df["weight_score"] * 0.3
            )

            ranked_df = ranked_df.sort_values("final_score", ascending=False)
            return _select_columns(ranked_df.head(5))

    return _select_columns(df_recipes.iloc[0:0])


def generate_final_recipe(
    selected_ingredients,
    available_tools=None,
    df_recipes=None,
    user_query=None,
    personal_preferences=None,
    fridge_source=None,
):
    if available_tools is None:
        available_tools = load_tools()
    if df_recipes is None:
        df_recipes = pd.DataFrame()
    if df_recipes.empty:
        print("⚠️ 추천 가능한 레시피가 없습니다.")
        return None

    try:
        fridge_df = load_fridge(fridge_source)
    except Exception:
        fridge_df = pd.DataFrame()

    def _clean_value(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            if value.is_integer():
                return str(int(value))
            return str(value)
        return str(value).strip()

    def _format_fridge_items(df: pd.DataFrame) -> List[str]:
        if df is None or df.empty:
            return list(selected_ingredients)

        items = []
        for _, row in df.iterrows():
            name = _clean_value(row.get("ingredient"))
            if not name:
                continue

            quantity = _clean_value(row.get("quantity"))
            unit = _clean_value(row.get("unit"))
            amount_parts = [part for part in [quantity, unit] if part]
            amount_text = f" {' '.join(amount_parts)}" if amount_parts else ""

            expiration_date = row.get("expiration_date")
            days_left = row.get("days_left")

            expiry_text = ""
            if isinstance(expiration_date, pd.Timestamp) and not pd.isna(expiration_date):
                expiry_text = expiration_date.strftime("%Y-%m-%d")
            elif expiration_date:
                expiry_text = str(expiration_date)

            days_text = ""
            if days_left is not None and not pd.isna(days_left):
                try:
                    days_int = int(days_left)
                    if days_int < 0:
                        days_text = f"만료 {abs(days_int)}일 경과"
                    elif days_int == 0:
                        days_text = "D-DAY"
                    else:
                        days_text = f"D-{days_int}"
                except (TypeError, ValueError):
                    days_text = ""

            meta_parts = [part for part in [days_text, expiry_text] if part]
            meta_text = f" ({' · '.join(meta_parts)})" if meta_parts else ""

            items.append(f"{name}{amount_text}{meta_text}")

        return items if items else list(selected_ingredients)

    fridge_items = _format_fridge_items(fridge_df)
    fridge_items_text = ", ".join(fridge_items)

    recipes_text = "\n".join([
        f"- {_clean_value(r.get('title'))} (재료: {_clean_value(r.get('ingredients'))})\n  조리법: {_clean_value(r.get('recipe'))}"
        for r in df_recipes.to_dict(orient="records")
    ])

    user_query_text = user_query if user_query else "특별한 조건 없음"
    personal_pref_text = personal_preferences if personal_preferences else "특별한 개인 설정 없음"

    example_recipe = {
        "title": "양파 계란 간장볶음",
        "category": "볶음 요리",
        "cuisine_type": "한식",
        "ingredients": [
            {"name": "계란", "amount": 2, "unit": "ea"},
            {"name": "양파", "amount": 1, "unit": "ea"},
            {"name": "간장", "amount": 15, "unit": "ml"},
        ],
        "tools": ["프라이팬"],
        "steps": [
            "1. 양파를 채 썰어 준비한다.",
            "2. 계란을 풀고 간장과 함께 볶는다.",
            "3. 팬에 기름을 두르고 모든 재료를 볶는다.",
        ],
        "time": "8분",
        "calorie": "220 kcal",
    }
    example_recipe_json = json.dumps(example_recipe, ensure_ascii=False, indent=2)

    prompt = f"""
# ===========================
# 냉장고 기반 레시피 추천 생성 요청
# ===========================

## 개인맞춤 설정
{personal_pref_text or "없음"}

## 사용자 요청
{user_query_text or "특별한 요청 없음"}

## 냉장고 재료
{fridge_items_text}

## 사용 가능한 조리도구
{available_tools}

## 후보 레시피 목록
{recipes_text if recipes_text.strip() else "후보 레시피가 없습니다."}

# ===========================
# 작업 지시
# ===========================
너는 전문 요리 레시피 추천 AI 어시스턴트야.
아래 단계를 따라 **최적의 레시피 하나만 JSON 형식으로 출력**해.

1. **후보 레시피 필터링 및 참고**
- 후보 레시피가 주어졌다면, 사용자 요청과 개인맞춤 설정을 바탕으로 부적합한 레시피를 제거해.
  - 예: 사용자가 "채식"을 선호하면 고기 포함 레시피 제거.
  - 예: "매운 음식"을 요청했는데 조리법에 고추나 매운 양념이 없다면 제거.
- 필터링 후 남은 후보가 있다면, 해당 후보들을 참고해 레시피를 생성해.
  - 냉장고 재료를 가장 많이 활용하고, 사용 가능한 조리도구로 조리 가능한 레시피를 우선 선택해.
  - 필요하다면 후보 레시피를 조합·수정해 더 적합한 하나의 레시피를 완성해.
- 모든 후보가 제거되었거나 후보 자체가 비어 있다면, 냉장고 재료를 최대한 활용할 수 있는 새로운 간단한 요리를 직접 제안해.

2. **재료 활용 우선순위**
- 냉장고 재료 중 **유통기한이 임박한 재료를 최우선으로 활용**하되, **사용자 요청(user_query)**과 **개인맞춤 설정(personal_preferences)**의 방향을 반드시 함께 고려해.
- 즉, 유통기한이 임박했더라도 사용자 의도와 상충되는 재료(예: 채식 선호 시 육류)는 사용하지 마.
- 냉장고 재료를 최대한 활용하되, 사용자 요청에 부합하는 요리가 우선이야.
- 냉장고에 없는 재료는 절대 추가하지 마. 반드시 **냉장고 내 재료로 대체하거나 생략**해.
- 유통기한이 임박한 재료는 “버리기 전에 빨리 소비해야 하는 재료”로 간주하고, 가능한 범위 내에서 우선적으로 포함하려고 노력해.

3. **요리 정보 구성**
- 아래 항목을 모두 포함한 JSON만 출력해야 해.
- 재료(`ingredients`)에는 **반드시 냉장고 재료 목록에 존재하는 항목만 포함해야 한다.**
- 냉장고 JSON에 없는 재료를 **절대 새로 추가하지 마.**
- 만약 요리에 꼭 필요한 재료가 냉장고에 없다면, **가장 유사한 냉장고 내 재료로 대체하거나 생략해.**
- 단위(unit)는 반드시 `ea`, `g`, `ml` 중 하나를 사용해.
- category와 cuisine_type은 실제 조리법과 재료를 근거로 선택해야 해.
  - category 후보: ["국물 요리", "볶음 요리", "탕/전골", "구이", "찜/조림", "면 요리", "밥 요리", "디저트", "샐러드", "간식", "기타"]
  - cuisine_type 후보: ["한식", "중식", "일식", "양식", "베트남식", "태국식", "인도식", "멕시코식", "퓨전", "기타"]
  - 근거가 모호하면 "기타" 또는 "퓨전" 사용.
  - 조리법과 재료가 불일치하는 category/cuisine_type은 절대 선택하지 마.

4. **레시피 구체화 지침**
- 레시피는 실제 요리 단계처럼 **현실적이고 구체적**으로 작성해.
- 각 단계는 한눈에 따라 할 수 있도록 **조리 동작, 시간, 불 세기, 도구 사용, 재료 투입 시점** 등을 포함해야 해.
- "볶는다", "끓인다" 같은 모호한 표현 대신, 다음 예시처럼 구체적으로 작성해:
  - 예: "중불에서 양파를 2분간 볶아 투명해질 때까지 익힌다."
  - 예: "끓는 물 500ml에 소금을 한 꼬집 넣고 면을 6분간 삶는다."
- **재료 준비 과정(손질, 세척, 썰기)**도 반드시 포함해.
- 각 단계는 짧고 명확한 문장으로 작성하되, **실제 요리 순서에 따라 정렬**해.

5. **출력 형식**
  - 반드시 아래 JSON 형식만 출력하고, 코드블록(````json`) 등은 절대 사용하지 마.
  - title에는 요리 이름만 사용 (예: "간장계란볶음", "두부조림" 등).
  - steps는 번호 순서로 구성해.
  - **time(조리 시간)**은 분 단위로 구체적으로 표기하고, 가능한 한 실제 조리 시간을 기반으로 추정해.
  - **calorie(칼로리)**는 전체 요리 1인분 기준으로 kcal 단위를 명확히 표기해.

# ===========================
# 출력 예시
# ===========================
{example_recipe_json}
"""

    print("🤖 GPT가 최종 레시피 생성 중...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    recipe_text = response.choices[0].message.content.strip()

    try:
        recipe_json = json.loads(recipe_text)
    except json.JSONDecodeError:
        print("⚠️ GPT 응답이 JSON 형식이 아닙니다. 원문 출력:")
        print(recipe_text)
        return None

    return recipe_json


@router.post("/recommend")
def recommend_recipe(request: RecipeRequest):
    try:
        user_query = request.user_query or ""
        fridge_source = request.fridge if request.fridge is not None else FRIDGE_JSON_PATH
        tools_source = request.tools if request.tools is not None else TOOLS_JSON_PATH

        selected_ings, df_recipes = search_recipes(fridge_source, top_k=TOP_K)
        available_tools = load_tools(tools_source)
        df_recipes = rerank_recipes(
            df_recipes,
            user_query,
            personal_preferences=request.personal_preferences,
        )
        final_recipe = generate_final_recipe(
            selected_ings,
            available_tools,
            df_recipes,
            user_query=user_query,
            personal_preferences=request.personal_preferences,
            fridge_source=fridge_source,
        )

        if not final_recipe:
            raise HTTPException(
                status_code=404, detail="추천 가능한 레시피가 없습니다."
            )

        return final_recipe

    except HTTPException as http_exc:
        raise http_exc
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=str(fnf_error))
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error))
    except Exception as unknown_error:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {unknown_error}") from unknown_error

