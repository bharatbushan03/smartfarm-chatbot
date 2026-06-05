import requests
import logging
from typing import Optional, List, Tuple
from config import (
    HF_TOKEN, HF_MAX_NEW_TOKENS, HF_TIMEOUT_SECONDS, GEMINI_API_KEY, GEMINI_VISION_MODEL
)
from constants import (
    JK_SEASON_DEFAULTS, CROP_PROFILES, SEASON_ALERT_RANGES, PH_SAFE_RANGE,
    SOIL_MOISTURE_SAFE_RANGE, CONDITION_SUGGESTIONS, HF_BLIP_MODELS, PLANT_ANALYSIS_SYSTEM_PROMPT
)
from utils import (
    _normalize_text, _normalize_npk, _score_numeric, _score_npk,
    _extract_generated_text, _parse_json_response
)

logger = logging.getLogger(__name__)

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


def _query_vision_model(
    image_base64: str,
    mime_type: str,
    model_id: str,
    provider: Optional[str]
) -> Optional[str]:
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


def _analyze_caption_with_llm(caption: str, hf_repo_id: str, hf_fallback_models: List[str]) -> dict:
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
    models_to_try = [hf_repo_id] + hf_fallback_models
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


def _season_key(season: Optional[str]) -> str:
    normalized = _normalize_text(season)
    if not normalized:
        return "annual"

    if "kharif" in normalized:
        return "kharif"
    if "rabi" in normalized:
        return "rabi"
    if "zaid" in normalized or "summer" in normalized:
        return "zaid"
    return "annual"


def _build_condition_alert(parameter: str, value: float, low: float, high: float, unit: str) -> Optional[dict]:
    if low <= value <= high:
        return None

    span = max(high - low, 1.0)
    if value < low:
        direction = "low"
        diff = low - value
    else:
        direction = "high"
        diff = value - high

    severity = "critical" if (diff / span) >= 0.4 else "warning"
    value_text = f"{value:.1f}{unit}" if unit else f"{value:.1f}"
    low_text = f"{low:.1f}{unit}" if unit else f"{low:.1f}"
    high_text = f"{high:.1f}{unit}" if unit else f"{high:.1f}"

    return {
        "parameter": parameter,
        "severity": severity,
        "direction": direction,
        "value": value_text,
        "recommended_range": f"{low_text} - {high_text}",
        "message": (
            f"{parameter.replace('_', ' ').title()} is {direction} ({value_text}). "
            f"Recommended range is {low_text} to {high_text}."
        ),
    }


def _derive_conditions(payload) -> tuple[dict, list[str], list[str]]:
    warnings = []
    suggestions = []

    season_key = _season_key(payload.season)
    defaults = JK_SEASON_DEFAULTS[season_key]

    location = payload.location or "Jammu and Kashmir, India"
    if not payload.location:
        warnings.append("Location missing: used Jammu and Kashmir, India seasonal baseline.")

    source_missing = []
    conditions = {
        "temperature": payload.temperature,
        "humidity": payload.humidity,
        "soil_ph": payload.soil_ph,
        "soil_moisture": payload.soil_moisture,
        "rainfall": payload.rainfall,
        "nitrogen": _normalize_npk(payload.nitrogen),
        "phosphorus": _normalize_npk(payload.phosphorus),
        "potassium": _normalize_npk(payload.potassium),
        "location": location,
        "season": payload.season or season_key,
    }

    for key in ["temperature", "humidity", "soil_ph", "soil_moisture", "rainfall"]:
        if conditions[key] is None:
            conditions[key] = defaults[key]
            source_missing.append(key)

    for key in ["nitrogen", "phosphorus", "potassium"]:
        if conditions[key] is None:
            conditions[key] = defaults[key]
            source_missing.append(key)

    if source_missing:
        warnings.append(
            "Missing sensor fields used seasonal defaults for Jammu and Kashmir: "
            + ", ".join(source_missing)
            + "."
        )

    if not (5.0 <= conditions["soil_ph"] <= 7.8):
        suggestions.append("Adjust soil pH toward 6.0-7.0 using lime (for acidic) or gypsum/sulfur strategy (for alkaline).")
    if conditions["soil_moisture"] < 30:
        suggestions.append("Increase irrigation frequency and add mulching to improve soil moisture retention.")
    if conditions["soil_moisture"] > 80:
        suggestions.append("Improve field drainage to prevent root stress and waterlogging.")

    return conditions, warnings, suggestions


def _rank_crops(conditions: dict) -> List[dict]:
    ranked = []
    for profile in CROP_PROFILES:
        matched_conditions = []

        t_score = _score_numeric(conditions["temperature"], *profile["temperature"])
        if t_score >= 0.95:
            matched_conditions.append("temperature in optimal range")

        h_score = _score_numeric(conditions["humidity"], *profile["humidity"])
        if h_score >= 0.95:
            matched_conditions.append("humidity in suitable range")

        ph_score = _score_numeric(conditions["soil_ph"], *profile["soil_ph"])
        if ph_score >= 0.95:
            matched_conditions.append("optimal pH range")

        sm_score = _score_numeric(conditions["soil_moisture"], *profile["soil_moisture"])
        if sm_score >= 0.95:
            matched_conditions.append("soil moisture aligned")

        rf_score = _score_numeric(conditions["rainfall"], *profile["rainfall"])
        if rf_score >= 0.95:
            matched_conditions.append("rainfall suitability")

        n_score = _score_npk(conditions["nitrogen"], profile["nitrogen"])
        if n_score >= 0.95:
            matched_conditions.append("nitrogen requirement matched")

        p_score = _score_npk(conditions["phosphorus"], profile["phosphorus"])
        if p_score >= 0.95:
            matched_conditions.append("phosphorus requirement matched")

        k_score = _score_npk(conditions["potassium"], profile["potassium"])
        if k_score >= 0.95:
            matched_conditions.append("potassium requirement matched")

        weighted = (
            t_score * 12.5
            + h_score * 10.0
            + ph_score * 12.5
            + sm_score * 12.5
            + rf_score * 12.5
            + n_score * 13.33
            + p_score * 13.33
            + k_score * 13.34
        )
        suitability_score = int(round(weighted))

        if not matched_conditions:
            matched_conditions = ["partial climate and soil alignment"]

        reason = (
            f"{profile['crop']} fits current climate-soil profile with strongest alignment in "
            f"{', '.join(matched_conditions[:3])}."
        )

        ranked.append(
            {
                "crop": profile["crop"],
                "suitability_score": max(0, min(100, suitability_score)),
                "reason": reason,
                "matched_conditions": matched_conditions[:5],
            }
        )

    ranked.sort(key=lambda x: x["suitability_score"], reverse=True)
    return ranked


def get_condition_alerts(payload) -> Tuple[str, bool, List[dict], List[str], dict]:
    season_key = _season_key(payload.season)
    season_ranges = SEASON_ALERT_RANGES.get(season_key, SEASON_ALERT_RANGES["annual"])

    checks = [
        ("temperature", payload.temperature, season_ranges["temperature"][0], season_ranges["temperature"][1], "C"),
        ("humidity", payload.humidity, season_ranges["humidity"][0], season_ranges["humidity"][1], "%"),
        ("soil_ph", payload.soil_ph, PH_SAFE_RANGE[0], PH_SAFE_RANGE[1], ""),
    ]

    if payload.soil_moisture is not None:
        checks.append(("soil_moisture", payload.soil_moisture, SOIL_MOISTURE_SAFE_RANGE[0], SOIL_MOISTURE_SAFE_RANGE[1], "%"))

    alerts = []
    suggestions = []

    for parameter, value, low, high, unit in checks:
        alert = _build_condition_alert(parameter, value, low, high, unit)
        if alert is None:
            continue
        alerts.append(alert)
        suggestions.append(CONDITION_SUGGESTIONS.get(f"{parameter}:{alert['direction']}", "Review agronomic management and sensor calibration."))

    suggestions = list(dict.fromkeys(suggestions))

    if not alerts:
        overall_status = "good"
        suggestions = ["No corrective action needed. Continue routine monitoring."]
    elif any(item["severity"] == "critical" for item in alerts):
        overall_status = "critical"
    else:
        overall_status = "warning"

    context = {
        "season": payload.season or season_key,
        "location": payload.location or "Jammu and Kashmir, India",
    }

    return overall_status, len(alerts) == 0, alerts, suggestions, context
