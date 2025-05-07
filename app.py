import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, Part

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
    json_mode: bool = False
    json_schema: Optional[Dict[str, Any]] = None

# レスポンス用のモデル定義
class GeminiResponse(BaseModel):
    text: str
    json_data: Optional[Dict[str, Any]] = None

@app.post("/generate", response_model=GeminiResponse)
async def generate_content(request: GeminiRequest):
    try:
        # Gemini モデルの初期化
        model = GenerativeModel(GEMINI_MODEL)
        
        # JSONモードの設定
        system_instruction = None
        
        if request.json_mode:
            if request.json_schema:
                # JSONスキーマが指定されている場合
                schema_str = json.dumps(request.json_schema)
                system_instruction = f"""
                You must respond with a valid JSON object that adheres to the following schema:
                {schema_str}
                
                Do not include any explanations, only provide a RFC8259 compliant JSON response 
                following the JSON schema above without deviation.
                """
            else:
                # JSONスキーマが指定されていない場合
                system_instruction = """
                You must respond with a valid JSON object. 
                Do not include any explanations, only provide a RFC8259 compliant JSON response.
                """
        
        # リクエスト設定
        generation_config = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens
        }
        
        # コンテンツ生成
        response = model.generate_content(
            request.prompt,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # レスポンス処理
        result = GeminiResponse(text=response.text)
        
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

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
