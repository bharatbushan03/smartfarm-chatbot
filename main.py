import io
import base64
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import (
    logger, HF_REPO_ID, HF_FALLBACK_MODELS, HF_TOKEN, HF_VISION_PROVIDERS, 
    MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_TYPES, GEMINI_API_KEY, GEMINI_VISION_MODEL, 
    MAX_HISTORY_LENGTH, PORT
)
from schemas import ChatRequest, CropRecommendationRequest, ConditionAlertRequest
from services import (
    query_huggingface, _query_vision_model, _query_gemini_vision, 
    _query_blip_caption, _analyze_caption_with_llm, _derive_conditions, 
    _rank_crops, get_condition_alerts
)
from utils import _parse_json_response

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("Pillow not installed. Image validation will be limited.")

app = FastAPI(title="Agro Farming Chatbot")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for chat history
CHAT_HISTORY = {}

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = [
            {
                "role": "system",
                "content": "You are a knowledgeable agro farming expert. Give practical, concise advice."
            }
        ]
    
    CHAT_HISTORY[session_id].append({"role": "user", "content": request.message})
    
    if len(CHAT_HISTORY[session_id]) > MAX_HISTORY_LENGTH + 1:
        CHAT_HISTORY[session_id] = [CHAT_HISTORY[session_id][0]] + CHAT_HISTORY[session_id][-(MAX_HISTORY_LENGTH):]

    models_to_try = [HF_REPO_ID] + [m for m in HF_FALLBACK_MODELS if m != HF_REPO_ID]
    response_text = None

    for model_id in models_to_try:
        logger.info(f"Trying Hugging Face model: {model_id} for session: {session_id}")
        response_text = query_huggingface(CHAT_HISTORY[session_id], model_id)
        if response_text:
            CHAT_HISTORY[session_id].append({"role": "assistant", "content": response_text})
            return {
                "response": response_text, 
                "model": model_id,
                "session_id": session_id
            }
    
    CHAT_HISTORY[session_id].pop()
    return {
        "response": "I could not get a model response from Hugging Face. Try again later.",
        "session_id": session_id
    }

@app.post("/analyze-plant")
async def analyze_plant(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Please upload a JPEG, PNG, WebP, or GIF image."
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
        )

    if PIL_AVAILABLE:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="The uploaded file could not be read as a valid image. Please upload a proper image file."
            )

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = content_type

    raw_response = None
    used_model = None
    for (model_id, provider) in HF_VISION_PROVIDERS:
        logger.info(f"Trying HF vision: model={model_id} provider={provider or 'auto'}")
        raw_response = _query_vision_model(image_base64, mime_type, model_id, provider)
        if raw_response:
            used_model = f"{model_id} via {provider or 'auto'}"
            break

    if not raw_response and GEMINI_API_KEY:
        logger.info("HF vision unavailable. Trying Gemini vision.")
        raw_response = _query_gemini_vision(image_base64, mime_type)
        if raw_response:
            used_model = f"gemini-2.0-flash ({GEMINI_VISION_MODEL})"

    if not raw_response:
        logger.info("Vision LLMs unavailable. Trying BLIP + text-LLM fallback.")
        caption = _query_blip_caption(image_bytes, mime_type)
        if caption:
            result = _analyze_caption_with_llm(caption, HF_REPO_ID, HF_FALLBACK_MODELS)
            return {
                **result,
                "model_used": "BLIP + text-LLM",
                "filename": file.filename
            }
        raise HTTPException(
            status_code=503,
            detail="All vision and captioning models failed."
        )

    result = _parse_json_response(raw_response)
    if result is None:
        result = {
            "is_plant": False,
            "message": "Please upload a valid image of a plant, crop, fruit, vegetable, or any agricultural subject."
        }

    return {**result, "model_used": used_model, "filename": file.filename}

@app.post("/recommend-crops")
async def recommend_crops(payload: CropRecommendationRequest):
    conditions, warnings, improvement_suggestions = _derive_conditions(payload)
    ranked = _rank_crops(conditions)

    top_recommended = ranked[:3]
    alternatives_pool = ranked[3:5]
    alternatives = [
        {
            "crop": item["crop"],
            "reason": f"Alternative option with suitability score {item['suitability_score']} under current/seasonal assumptions.",
        }
        for item in alternatives_pool
    ]

    if not top_recommended or top_recommended[0]["suitability_score"] < 55:
        warnings.append("Current conditions are weak for major crops; corrective action is recommended before sowing.")
        if not improvement_suggestions:
            improvement_suggestions.append("Conduct soil test calibration and optimize irrigation-fertilizer schedule before planting.")

    return {
        "recommended_crops": top_recommended,
        "alternative_crops": alternatives,
        "warnings": warnings,
        "improvement_suggestions": improvement_suggestions,
    }

@app.post("/condition-alert")
async def condition_alert(payload: ConditionAlertRequest):
    overall_status, all_good, alerts, suggestions, context = get_condition_alerts(payload)
    
    return {
        "overall_status": overall_status,
        "all_good": all_good,
        "context": context,
        "alerts": alerts,
        "improvement_suggestions": suggestions,
    }

@app.get("/")
async def root():
    return {
        "message": "Agro Farming Chatbot API is running.",
        "endpoints": {
            "POST /chat": "Chat with the farming assistant (text)",
            "POST /analyze-plant": "Upload a plant/crop image for analysis",
            "POST /recommend-crops": "Recommend crops using sensor and seasonal conditions",
            "POST /condition-alert": "Alert if sensor conditions are healthy or risky",
            "GET /health": "Health check and model status"
        }
    }

@app.get("/health")
async def health():
    from constants import HF_BLIP_MODELS
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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
