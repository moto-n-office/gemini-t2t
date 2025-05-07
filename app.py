import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vertexai
from vertexai.generative_models import GenerativeModel

# 環境変数の取得
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

# Vertex AIの初期化
vertexai.init(project=PROJECT_ID, location=REGION)

# FastAPIインスタンスの作成
app = FastAPI(title="Gemini API Proxy")

# リクエスト用のモデル定義
class GeminiRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_output_tokens: int = 2048

# レスポンス用のモデル定義
class GeminiResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GeminiResponse)
async def generate_content(request: GeminiRequest):
    try:
        # Gemini モデルの初期化
        model = GenerativeModel(GEMINI_MODEL)
        
        # コンテンツ生成
        response = model.generate_content(
            request.prompt,
            generation_config={
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens
            }
        )
        
        return GeminiResponse(text=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成エラー: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
