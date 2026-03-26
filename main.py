import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Agro Farming Chatbot")

# Configuration
HF_REPO_ID = os.getenv("HF_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct:fastest")
HF_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "HF_FALLBACK_MODELS",
        "mistralai/Mistral-7B-Instruct-v0.3:fastest,deepseek-ai/DeepSeek-R1:fastest"
    ).split(",")
    if m.strip()
]
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip("'\" ")
HF_TIMEOUT_SECONDS = int(os.getenv("HF_TIMEOUT_SECONDS", "60"))
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))


def _clean_response_text(text):
    # Some models may return internal reasoning inside <think> tags.
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = text[:start] + text[end + len("</think>"):]
    return text.strip()


def _extract_generated_text(result):
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get("message", {})
            return _clean_response_text((message.get("content") or ""))
    return ""

def query_huggingface(question, model_id):
    """Calls Hugging Face router chat completions for a specific model."""
    if not HF_TOKEN:
        print("HUGGINGFACEHUB_API_TOKEN is not configured.")
        return None

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a knowledgeable agro farming expert. Give practical, concise advice."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": HF_MAX_NEW_TOKENS
    }

    try:
        router_url = "https://router.huggingface.co/v1/chat/completions"
        response = requests.post(router_url, headers=headers, json=payload, timeout=HF_TIMEOUT_SECONDS)
        if response.status_code == 200:
            text = _extract_generated_text(response.json())
            if text:
                return text
        else:
            print(f"HF router HTTP {response.status_code} ({model_id}): {response.text[:300]}")
    except Exception as e:
        print(f"HF router error ({model_id}): {repr(e)}")

    return None

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    models_to_try = [HF_REPO_ID] + [m for m in HF_FALLBACK_MODELS if m != HF_REPO_ID]
    response = None

    for model_id in models_to_try:
        print(f"Querying Hugging Face model: {model_id}")
        response = query_huggingface(request.message, model_id)
        if response:
            return {"response": response, "model": model_id}
    
    if not response:
        return {
            "response": "I could not get a model response from Hugging Face. Try again later or change HF_REPO_ID/HF_FALLBACK_MODELS."
        }

# Serve chatbot API only (no frontend)
@app.get("/")
async def root():
    return {"message": "Agro Farming Chatbot API is running. Use POST /chat to interact."}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": "huggingface",
        "model": HF_REPO_ID,
        "fallback_models": HF_FALLBACK_MODELS,
        "token_configured": bool(HF_TOKEN)
    }

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for HF Spaces, or local override if needed
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
