import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Configuration
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
HF_VISION_PROVIDERS = [
    ("Qwen/Qwen3-VL-8B-Instruct", "fireworks-ai"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "nebius"),
    ("Qwen/Qwen2.5-VL-72B-Instruct", "together"),
    ("Qwen/Qwen3-VL-8B-Instruct", None),
]

HF_BLIP_MODELS = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
]

MAX_IMAGE_SIZE_MB = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Gemini Vision Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip("'\" ")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

# Server Configuration
PORT = int(os.getenv("PORT", 7860))

# Chat History Configuration
MAX_HISTORY_LENGTH = 10
