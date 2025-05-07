import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import vertexai
from vertexai.generative_models import GenerativeModel

# 環境変数の取得
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

# Vertex AIの初期化
vertexai.init(project=PROJECT_ID, location=REGION)

# FastAPIインスタンスの作成
app = FastAPI(title="Gemini API Proxy")

# リクエスト用のモデル定義
class GeminiRequest(BaseModel):
    prompt: str
    model: Optional[str] = None  # モデル名をパラメータとして追加
    temperature: float = 0.7
    max_output_tokens: int = 2048
    json_mode: bool = False
    json_schema: Optional[Dict[str, Any]] = None

# レスポンス用のモデル定義
class GeminiResponse(BaseModel):
    text: str
    model: str  # 使用されたモデルを返す
    json_data: Optional[Dict[str, Any]] = None

@app.post("/generate", response_model=GeminiResponse)
async def generate_content(request: GeminiRequest):
    try:
        # モデル名の決定（リクエストで指定がなければデフォルト値を使用）
        model_name = request.model if request.model else DEFAULT_GEMINI_MODEL
        
        # Gemini モデルの初期化
        model = GenerativeModel(model_name)
        
        # JSONモードの設定
        contents = []
        
        if request.json_mode:
            # システムメッセージの作成
            if request.json_schema:
                schema_str = json.dumps(request.json_schema)
                system_message = f"""
                You must respond with a valid JSON object that adheres to the following schema:
                {schema_str}
                
                Do not include any explanations, only provide a RFC8259 compliant JSON response 
                following the JSON schema above without deviation.
                """
            else:
                system_message = """
                You must respond with a valid JSON object. 
                Do not include any explanations, only provide a RFC8259 compliant JSON response.
                """
                
            # システムメッセージをプロンプトの前に追加
            contents = [
                system_message,
                request.prompt
            ]
        else:
            contents = [request.prompt]
        
        # リクエスト設定
        generation_config = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens
        }
        
        # コンテンツ生成
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config
        )
        
        # レスポンス処理
        result = GeminiResponse(text=response.text, model=model_name)
        
        # JSONモードの場合、JSON解析を試みる
        if request.json_mode:
            try:
                # テキスト内のJSONを抽出して解析
                json_text = response.text
                # コードブロックからJSONを抽出（もしあれば）
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0].strip()
                
                result.json_data = json.loads(json_text)
            except json.JSONDecodeError:
                # JSON解析に失敗した場合はテキストのみ返す
                pass
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成エラー: {str(e)}")

# 利用可能なモデルの取得エンドポイント
@app.get("/models")
async def get_available_models():
    # 利用可能なモデルのリスト
    # 注: 実際には動的に取得する方法もありますが、ここでは静的リストとして提供
    available_models = [
        {"id": "gemini-2.0-flash-001", "description": "Gemini 2.0 Flash - 高速レスポンス向き"},
        {"id": "gemini-2.0-pro-001", "description": "Gemini 2.0 Pro - 高性能・複雑なタスク向き"},
        {"id": "gemini-1.5-flash-001", "description": "Gemini 1.5 Flash - 以前のバージョン"},
        {"id": "gemini-1.5-pro-001", "description": "Gemini 1.5 Pro - 以前のバージョン"}
    ]
    return {"models": available_models, "default_model": DEFAULT_GEMINI_MODEL}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
