import os
import re
import requests
import logging
import base64
import io
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("Pillow not installed. Image validation will be limited.")

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

# Vision model configuration
# IMPORTANT: Vision LLMs are ONLY on Inference Providers (the router), NOT on the
# legacy serverless API. We must use router.huggingface.co AND specify a provider
# that supports image/multimodal inputs (nebius, together, fireworks-ai).
# Do NOT use :fastest — it routes to text-only providers that reject image payloads.
HF_VISION_PROVIDERS = [
    # (model_id, provider) — tried in order, first success wins
    # provider=None lets HF router auto-select a working provider.
    ("Qwen/Qwen3-VL-8B-Instruct", "fireworks-ai"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "nebius"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "together"),
    ("Qwen/Qwen3-VL-8B-Instruct", None),
]
# BLIP via legacy serverless API (free, smaller, for caption-based fallback)
HF_BLIP_MODELS = [
    "Salesforce/blip-image-captioning-base",   # try base first (always warm)
    "Salesforce/blip-image-captioning-large",
]
MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Gemini Vision Configuration (Option B — free tier, 15 RPM, image support)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip("'\" ")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

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


def _parse_json_response(text: str) -> Optional[dict]:
    """
    Robustly extracts a JSON object from raw model output.
    Handles:
      - Plain JSON: {"is_plant": true}
      - Markdown fences: ```json\\n{...}\\n```
      - Trailing text after closing fence
      - Fenced block without closing backticks
    Returns a parsed dict, or None if no valid JSON was found.
    """
    if not text:
        return None
    # 1. Try to extract from a ```...``` block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 2. Try the raw text as-is
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 3. Scan for the first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None

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


PLANT_ANALYSIS_SYSTEM_PROMPT = """
You are an expert botanist and agronomist specializing in plant and crop identification.
When given an image:
1. If it contains a plant, crop, fruit, vegetable, leaf, seed, or any agricultural subject:
   - Identify the specific plant or crop (scientific and common name if possible).
   - Describe its current growth stage (e.g., seedling, vegetative, flowering, fruiting, harvest-ready).
   - Note any visible health conditions, diseases, pest damage, nutrient deficiencies, or stress signs.
   - Provide 2-3 brief, practical farming tips relevant to this stage.
   - Return ONLY a valid JSON object in this exact format (no markdown, no extra text):
     {"is_plant": true, "plant_name": "", "scientific_name": "", "growth_stage": "", "health_status": "", "observations": "", "farming_tips": []}
2. If the image does NOT contain any plant, crop, or agricultural subject:
   - Return ONLY this JSON object:
     {"is_plant": false, "message": "Please upload a valid image of a plant, crop, fruit, vegetable, or any agricultural subject."}
Never include markdown code fences or any text outside the JSON.
"""


def _query_vision_model(
    image_base64: str,
    mime_type: str,
    model_id: str,
    provider: Optional[str]
) -> Optional[str]:
    """
    Sends a vision request through the HF Inference Providers ROUTER
    with an explicitly named provider that supports multimodal inputs.
    Same router URL as /chat — proven to work with this token.
    """
    if not HF_TOKEN:
        logger.error("HUGGINGFACEHUB_API_TOKEN is not configured.")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}",
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": PLANT_ANALYSIS_SYSTEM_PROMPT.strip()
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image and respond with the JSON format described in your instructions."
                    }
                ]
            }
        ],
        "max_tokens": 512
    }
    if provider:
        payload["provider"] = provider

    try:
        router_url = "https://router.huggingface.co/v1/chat/completions"
        response = requests.post(router_url, headers=headers, json=payload, timeout=HF_TIMEOUT_SECONDS)
        if response.status_code == 200:
            text = _extract_generated_text(response.json())
            if text:
                logger.info(f"Vision OK — model={model_id} provider={provider or 'auto'}")
                return text
            else:
                    logger.warning(f"Vision empty response — model={model_id} provider={provider or 'auto'}")
        else:
            logger.warning(
                    f"Vision HTTP {response.status_code} — model={model_id} provider={provider or 'auto'}: "
                f"{response.text[:400]}"
            )
    except requests.exceptions.Timeout:
            logger.error(f"Vision timeout ({HF_TIMEOUT_SECONDS}s) — model={model_id} provider={provider or 'auto'}")
    except Exception as e:
            logger.error(f"Vision error — model={model_id} provider={provider or 'auto'}: {repr(e)}")

    return None


def _query_blip_caption(image_bytes: bytes, mime_type: str = "image/jpeg") -> Optional[str]:
    """
    Tries each BLIP model on the HF legacy serverless API (free, no provider needed).
    X-Wait-For-Model prevents instant 503 on cold start.
    """
    if not HF_TOKEN:
        return None

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": mime_type,
        "X-Wait-For-Model": "true",
    }

    for blip_model in HF_BLIP_MODELS:
        try:
            url = f"https://api-inference.huggingface.co/models/{blip_model}"
            logger.info(f"Trying BLIP: {blip_model}")
            response = requests.post(url, headers=headers, data=image_bytes, timeout=HF_TIMEOUT_SECONDS)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    caption = data[0].get("generated_text", "").strip()
                    if caption:
                        logger.info(f"BLIP ({blip_model}) caption: {caption}")
                        return caption
                logger.warning(f"BLIP ({blip_model}) empty response: {data}")
            else:
                logger.warning(f"BLIP ({blip_model}) HTTP {response.status_code}: {response.text[:300]}")
        except requests.exceptions.Timeout:
            logger.error(f"BLIP ({blip_model}) timed out after {HF_TIMEOUT_SECONDS}s")
        except Exception as e:
            logger.error(f"BLIP ({blip_model}) error: {repr(e)}")

    return None



def _query_gemini_vision(image_base64: str, mime_type: str) -> Optional[str]:
    """
    Sends a vision request to Google Gemini 2.0 Flash API.
    Free tier: 15 RPM, supports image inputs natively.
    """
    if not GEMINI_API_KEY:
        logger.info("GEMINI_API_KEY not configured, skipping Gemini vision.")
        return None

    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64,
                        }
                    },
                    {
                        "text": PLANT_ANALYSIS_SYSTEM_PROMPT.strip()
                        + "\n\nAnalyze the uploaded image and respond with the JSON format described above."
                    },
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0.2,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent?key={GEMINI_API_KEY}"
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT_SECONDS)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    logger.info(f"Gemini vision OK — model={GEMINI_VISION_MODEL}")
                    return text
            logger.warning(f"Gemini empty response")
        elif response.status_code == 429:
            logger.warning("Gemini rate limited (429) — free tier is 15 RPM")
        else:
            logger.warning(f"Gemini HTTP {response.status_code}: {response.text[:400]}")
    except requests.exceptions.Timeout:
        logger.error(f"Gemini timeout ({HF_TIMEOUT_SECONDS}s)")
    except Exception as e:
        logger.error(f"Gemini error: {repr(e)}")

    return None


def _analyze_caption_with_llm(caption: str) -> dict:
    """
    Sends a BLIP caption to the text LLM to produce a plant analysis JSON.
    Used as a last-resort fallback when vision LLMs are unavailable.
    """
    prompt = (
        f'An image was described as: "{caption}". '
        "Based on this description, determine if it is a plant, crop, fruit, or vegetable. "
        "If yes, identify it and return JSON: "
        '{"is_plant": true, "plant_name": "", "scientific_name": "", "growth_stage": "", '
        '"health_status": "", "observations": "", "farming_tips": []}. '
        "If no, return: {\"is_plant\": false, \"message\": \"Please upload a valid image of a plant, crop, fruit, vegetable, or any agricultural subject.\"}. "
        "Reply with only the JSON, no markdown."
    )
    messages = [
        {"role": "system", "content": "You are an expert agronomist. Reply with JSON only."},
        {"role": "user", "content": prompt}
    ]
    models_to_try = [HF_REPO_ID] + HF_FALLBACK_MODELS
    for model_id in models_to_try:
        text = query_huggingface(messages, model_id)
        if text:
            parsed = _parse_json_response(text)
            if parsed is not None:
                return parsed
    return {
        "is_plant": False,
        "message": "Could not analyze the image. Please try again or upload a clearer plant image."
    }


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

@app.post("/analyze-plant")
async def analyze_plant(file: UploadFile = File(...)):
    """
    Upload an image of a plant, crop, fruit, or vegetable.
    Returns plant identification, growth stage, health status, and farming tips.
    Returns an error message for non-plant images.
    """
    # --- Validate content type ---
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Please upload a JPEG, PNG, WebP, or GIF image."
        )

    # --- Read and size-check image ---
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
        )

    # --- Optionally validate image integrity with Pillow ---
    if PIL_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # Will raise if file is corrupt
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="The uploaded file could not be read as a valid image. Please upload a proper image file."
            )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = content_type

    # --- Tier 1: Try HF vision LLMs via router with explicit vision-capable providers ---
    raw_response = None
    used_model = None
    for (model_id, provider) in HF_VISION_PROVIDERS:
        logger.info(f"Trying HF vision: model={model_id} provider={provider or 'auto'}")
        raw_response = _query_vision_model(image_base64, mime_type, model_id, provider)
        if raw_response:
            used_model = f"{model_id} via {provider or 'auto'}"
            break

    # --- Tier 2: Gemini 2.0 Flash vision (free tier, no Inference Provider needed) ---
    if not raw_response and GEMINI_API_KEY:
        logger.info("HF vision unavailable. Trying Gemini vision.")
        raw_response = _query_gemini_vision(image_base64, mime_type)
        if raw_response:
            used_model = f"gemini-2.0-flash ({GEMINI_VISION_MODEL})"

    # --- Tier 3: BLIP caption → text LLM ---
    if not raw_response:
        logger.info("Vision LLMs unavailable. Trying BLIP + text-LLM fallback.")
        caption = _query_blip_caption(image_bytes, mime_type)
        if caption:
            result = _analyze_caption_with_llm(caption)
            return {
                **result,
                "model_used": "BLIP + text-LLM",
                "filename": file.filename
            }
        raise HTTPException(
            status_code=503,
            detail=(
                "All vision and captioning models failed. "
                "Ensure either HUGGINGFACEHUB_API_TOKEN has Inference Provider access "
                "or GEMINI_API_KEY is set with vision capability."
            )
        )

    # --- Parse vision LLM JSON response ---
    result = _parse_json_response(raw_response)
    if result is None:
        logger.warning(f"Vision model returned non-JSON: {raw_response[:200]}")
        result = {
            "is_plant": False,
            "message": "Please upload a valid image of a plant, crop, fruit, vegetable, or any agricultural subject."
        }

    return {**result, "model_used": used_model, "filename": file.filename}


# Serve chatbot API only (no frontend)
@app.get("/")
async def root():
    return {
        "message": "Agro Farming Chatbot API is running.",
        "endpoints": {
            "POST /chat": "Chat with the farming assistant (text)",
            "POST /analyze-plant": "Upload a plant/crop image for analysis",
            "GET /health": "Health check and model status"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "text_providers": {
            "huggingface_token": bool(HF_TOKEN),
            "chat": {
                "primary_model": HF_REPO_ID,
                "fallback_models": HF_FALLBACK_MODELS,
            }
        },
        "image_providers": {
            "huggingface": {
                "models": [f"{m} ({p})" for m, p in HF_VISION_PROVIDERS],
                "blip_models": HF_BLIP_MODELS,
            },
            "gemini": {
                "model": GEMINI_VISION_MODEL,
                "key_set": bool(GEMINI_API_KEY),
            },
        },
        "max_image_size_mb": MAX_IMAGE_SIZE_MB,
    }

if __name__ == "__main__":
    import uvicorn
    # Use port 7860 for HF Spaces, or local override if needed
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
