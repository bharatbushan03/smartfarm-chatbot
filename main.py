import os
import requests
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Agro Farming Chatbot")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# In-memory storage for chat history: {session_id: [messages]}
CHAT_HISTORY = {}
# Maximum number of messages to keep in history per session
MAX_HISTORY_LENGTH = 10


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

def query_huggingface(messages, model_id):
    """Calls Hugging Face router chat completions with a full message history."""
    if not HF_TOKEN:
        logger.error("HUGGINGFACEHUB_API_TOKEN is not configured.")
        return None

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": HF_MAX_NEW_TOKENS
    }

    try:
        router_url = "https://router.huggingface.co/v1/chat/completions"
        response = requests.post(router_url, headers=headers, json=payload, timeout=HF_TIMEOUT_SECONDS)
        if response.status_code == 200:
            text = _extract_generated_text(response.json())
            if text:
                logger.info(f"Successfully queried model: {model_id}")
                return text
        else:
            logger.warning(f"HF router HTTP {response.status_code} ({model_id}): {response.text[:300]}")
    except Exception as e:
        logger.error(f"HF router error ({model_id}): {repr(e)}")

    return None

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    # Initialize history if new session
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = [
            {
                "role": "system",
                "content": "You are a knowledgeable agro farming expert. Give practical, concise advice."
            }
        ]
    
    # Add user message to history
    CHAT_HISTORY[session_id].append({"role": "user", "content": request.message})
    
    # Keep history within limits (excluding system message at index 0)
    if len(CHAT_HISTORY[session_id]) > MAX_HISTORY_LENGTH + 1:
        # Keep system message and slice the last N messages
        CHAT_HISTORY[session_id] = [CHAT_HISTORY[session_id][0]] + CHAT_HISTORY[session_id][-(MAX_HISTORY_LENGTH):]

    models_to_try = [HF_REPO_ID] + [m for m in HF_FALLBACK_MODELS if m != HF_REPO_ID]
    response_text = None

    for model_id in models_to_try:
        logger.info(f"Trying Hugging Face model: {model_id} for session: {session_id}")
        response_text = query_huggingface(CHAT_HISTORY[session_id], model_id)
        if response_text:
            # Add assistant response to history
            CHAT_HISTORY[session_id].append({"role": "assistant", "content": response_text})
            return {
                "response": response_text, 
                "model": model_id,
                "session_id": session_id
            }
    
    # If all models fail, remove the last user message to avoid an inconsistent state
    CHAT_HISTORY[session_id].pop()
    
    return {
        "response": "I could not get a model response from Hugging Face. Try again later.",
        "session_id": session_id
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
