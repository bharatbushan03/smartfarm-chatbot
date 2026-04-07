# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

Agro Farming Chatbot - a FastAPI chatbot API for agricultural guidance using Hugging Face inference models. Supports text chat and plant/crop image analysis.

## Key Commands

- **Run locally**: `python main.py` (port 7860) or `uvicorn main:app --host 0.0.0.0 --port 10000`
- **Install deps**: `pip install -r requirements.txt`
- **Build Docker**: `docker build -t smartfarm-api .`
- **Run Docker**: `docker run -p 10000:10000 -e HUGGINGFACEHUB_API_TOKEN=<token> smartfarm-api`
- **Required env**:
  - `HUGGINGFACEHUB_API_TOKEN` — Hugging Face token with Inference Providers permission
  - `GEMINI_API_KEY` (optional) — Google Gemini API key for vision fallback (15 RPM free tier)

## Architecture

Single-file application (`main.py`) with four endpoints:

- `POST /chat` — Text chat with session-based conversation history (in-memory dict, max 10 messages per session). Routes through HF router (`router.huggingface.co/v1/chat/completions`) with model fallback chain.
- `POST /analyze-plant` — Upload plant/crop image for identification, growth stage analysis, and health assessment. **Three-tier fallback chain**: 1) HF vision LLMs via Inference Providers → 2) Gemini 2.0 Flash vision → 3) BLIP caption → text LLM analysis.
- `GET /` and `GET /health` — Root info and model status.

### Model Configuration

- **Text LLMs**: Primary `HF_REPO_ID` (default: `meta-llama/Llama-3.1-8B-Instruct:fastest`), fallback `HF_FALLBACK_MODELS` (comma-separated).
- **Vision LLMs**: Hardcoded `HF_VISION_PROVIDERS` list of (model, provider) tuples — tried in order. Providers (nebius, together, fireworks-ai) are explicitly selected because vision inputs require Inference Providers, not the legacy serverless API. Do NOT use `:fastest` suffix for vision models.
- **Gemini Vision**: `GEMINI_API_KEY` enables Gemini 2.0 Flash as a fallback option (15 RPM free tier, no provider selection needed).
- **BLIP**: `Salesforce/blip-image-captioning-base` and `-large` via legacy serverless API (`api-inference.huggingface.co`) as last-resort captioning fallback.

### Important Details

- Chat history is stored in a simple in-memory dict `CHAT_HISTORY` keyed by session_id. No persistence.
- The `_clean_response_text()` function strips `<think>` tags from model outputs (e.g., DeepSeek reasoning traces).
- Production deployment uses gunicorn with uvicorn workers via Dockerfile. Render free tier uses 2 workers.
- CORS allows all origins (`*`).
- Image uploads validated against `ALLOWED_IMAGE_TYPES` and `MAX_IMAGE_SIZE_MB`.
- Gemini vision (`GEMINI_API_KEY`) provides a reliable fallback without Inference Providers — useful when HF vision models are unavailable.

### Deployment

Deployed on Render via `render.yaml`. All model config is environment-variable-driven. Models can be changed by updating `HF_REPO_ID` / `HF_FALLBACK_MODELS` in Render settings — no code change needed.
