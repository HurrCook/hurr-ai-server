from fastapi import FastAPI

from llm.recipes import router as llm_router


app = FastAPI(title="LLM Recipe Recommendation API")

app.include_router(llm_router, prefix="/llm")


@app.get("/")
def root():
    return {
        "message": "LLM Recipe Recommendation API is running!",
        "endpoints": {
            "recommend": "/llm/recommend",
        },
    }

