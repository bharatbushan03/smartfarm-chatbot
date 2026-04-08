# 🌾 SmartFarm Chatbot API

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-5A67D8)](https://render.com/)
[![Repo Stars](https://img.shields.io/github/stars/bharatbushan03/smartfarm-chatbot?style=social)](https://github.com/bharatbushan03/smartfarm-chatbot/stargazers)

SmartFarm Chatbot is a production-ready **FastAPI** backend for agriculture support.  
It provides AI-powered text guidance, plant image analysis, crop recommendation, and condition alerting for farming decisions.

---

## ✨ Highlights

- Multi-model AI fallback for better reliability.
- Plant/crop image understanding with tiered vision pipeline.
- Sensor + seasonal logic for crop recommendation and risk alerts.
- Dockerized and Render-ready deployment setup.
- Clean REST API with interactive docs via Swagger UI.

## 🚀 Key Features

- **Chat Assistant (`POST /chat`)**
  - Session-based conversation for practical farming guidance.
  - Primary + fallback Hugging Face text models.
- **Plant Analysis (`POST /analyze-plant`)**
  - Accepts image uploads (JPEG, PNG, WebP, GIF).
  - 3-level fallback:
    1. Hugging Face vision providers
    2. Gemini vision (optional key)
    3. BLIP caption + text LLM analysis
- **Crop Recommendation (`POST /recommend-crops`)**
  - Ranks crops using climate, soil, moisture, rainfall, and NPK signals.
  - Returns top recommendations, alternatives, warnings, and suggestions.
- **Condition Alert (`POST /condition-alert`)**
  - Detects unsafe ranges (temperature, humidity, pH, moisture).
  - Returns severity and actionable improvement suggestions.
- **Health & Status (`GET /health`)**
  - Quick check for provider and model readiness.

## 🧱 Tech Stack

- **Backend:** FastAPI, Uvicorn, Gunicorn
- **AI Providers:** Hugging Face Inference Providers, Google Gemini (optional)
- **Vision/Image:** Pillow, BLIP captioning fallback
- **Deployment:** Docker, Render
- **Language:** Python

## 📦 Installation & Setup

### 1) Clone the repository

```bash
git clone https://github.com/bharatbushan03/smartfarm-chatbot.git
cd smartfarm-chatbot
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file (or set variables in your deployment platform):

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
HF_REPO_ID=meta-llama/Llama-3.1-8B-Instruct:fastest
HF_FALLBACK_MODELS=mistralai/Mistral-7B-Instruct-v0.3:fastest,deepseek-ai/DeepSeek-R1:fastest
HF_TIMEOUT_SECONDS=60
HF_MAX_NEW_TOKENS=256
MAX_IMAGE_SIZE_MB=10
GEMINI_API_KEY=optional_gemini_key
GEMINI_VISION_MODEL=gemini-2.0-flash
```

> `HUGGINGFACEHUB_API_TOKEN` is required for Hugging Face inference.

## ▶️ Run the API

### Local (default script behavior)

```bash
python main.py
```

Runs on port `7860` by default.

### Uvicorn (custom port example)

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

### Docker

```bash
docker build -t smartfarm-api .
docker run -p 10000:10000 -e HUGGINGFACEHUB_API_TOKEN=<token> smartfarm-api
```

## 📘 Usage

Once running, open:

- API docs: `http://localhost:10000/docs` (or your configured port)
- Health check: `GET /health`

### Sample API calls

```bash
curl -X POST http://localhost:10000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How can I reduce pest attacks in chili crops?","session_id":"demo"}'
```

```bash
curl -X POST http://localhost:10000/recommend-crops \
  -H "Content-Type: application/json" \
  -d '{"temperature":26,"humidity":68,"soil_ph":6.5,"soil_moisture":58,"nitrogen":"high","phosphorus":"medium","potassium":"medium","rainfall":180,"season":"kharif"}'
```

```bash
curl -X POST http://localhost:10000/condition-alert \
  -H "Content-Type: application/json" \
  -d '{"temperature":38,"humidity":82,"soil_ph":8.0,"soil_moisture":84,"season":"kharif"}'
```

For plant analysis, use multipart upload with endpoint `POST /analyze-plant`.

## 🗂️ Project Structure

```text
smartfarm-chatbot/
├── main.py            # FastAPI app and AI integration logic
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container build configuration
├── render.yaml        # Render deployment configuration
└── README.md          # Project documentation
```

## 🤝 Contribution Guidelines

Contributions are welcome and appreciated.

1. Fork the repository.
2. Create a feature/fix branch.
3. Make focused changes with clear commit messages.
4. Test your changes locally.
5. Open a pull request with context and screenshots/logs when useful.

## 🛣️ Roadmap / Future Improvements

- Add automated test suite and CI workflow badges.
- Add optional persistent chat history (database or cache layer).
- Add multilingual support for farmer-friendly regional responses.
- Improve disease diagnosis confidence scoring for image analysis.
- Add role-based API auth for production multi-tenant usage.

## 🆘 Support / Contact

If you need help or want to report an issue:

- Open an issue in this repository:
  `https://github.com/bharatbushan03/smartfarm-chatbot/issues`

---

Built to make practical, AI-assisted farming guidance more accessible and reliable. 🌱
