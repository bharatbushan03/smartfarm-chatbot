# Season and Crop Data
JK_SEASON_DEFAULTS = {
    "kharif": {
        "temperature": 27.0,
        "humidity": 72.0,
        "soil_ph": 6.6,
        "soil_moisture": 62.0,
        "rainfall": 210.0,
        "nitrogen": "high",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    "rabi": {
        "temperature": 14.0,
        "humidity": 60.0,
        "soil_ph": 6.8,
        "soil_moisture": 46.0,
        "rainfall": 85.0,
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    "zaid": {
        "temperature": 29.0,
        "humidity": 52.0,
        "soil_ph": 6.7,
        "soil_moisture": 42.0,
        "rainfall": 55.0,
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "high",
    },
    "annual": {
        "temperature": 22.0,
        "humidity": 64.0,
        "soil_ph": 6.7,
        "soil_moisture": 50.0,
        "rainfall": 120.0,
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "medium",
    },
}

CROP_PROFILES = [
    {
        "crop": "Rice",
        "temperature": (20.0, 35.0),
        "humidity": (60.0, 90.0),
        "soil_ph": (5.0, 7.0),
        "soil_moisture": (60.0, 90.0),
        "rainfall": (150.0, 350.0),
        "nitrogen": "high",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    {
        "crop": "Maize",
        "temperature": (18.0, 32.0),
        "humidity": (50.0, 75.0),
        "soil_ph": (5.8, 7.2),
        "soil_moisture": (45.0, 70.0),
        "rainfall": (80.0, 220.0),
        "nitrogen": "high",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    {
        "crop": "Wheat",
        "temperature": (10.0, 25.0),
        "humidity": (40.0, 65.0),
        "soil_ph": (6.0, 7.5),
        "soil_moisture": (35.0, 55.0),
        "rainfall": (40.0, 120.0),
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    {
        "crop": "Barley",
        "temperature": (8.0, 24.0),
        "humidity": (35.0, 60.0),
        "soil_ph": (6.0, 8.0),
        "soil_moisture": (30.0, 50.0),
        "rainfall": (30.0, 100.0),
        "nitrogen": "medium",
        "phosphorus": "low",
        "potassium": "medium",
    },
    {
        "crop": "Potato",
        "temperature": (15.0, 25.0),
        "humidity": (45.0, 70.0),
        "soil_ph": (5.0, 6.8),
        "soil_moisture": (50.0, 75.0),
        "rainfall": (60.0, 180.0),
        "nitrogen": "high",
        "phosphorus": "high",
        "potassium": "high",
    },
    {
        "crop": "Mustard",
        "temperature": (10.0, 28.0),
        "humidity": (35.0, 60.0),
        "soil_ph": (6.0, 7.8),
        "soil_moisture": (30.0, 50.0),
        "rainfall": (30.0, 110.0),
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    {
        "crop": "Chickpea",
        "temperature": (15.0, 30.0),
        "humidity": (30.0, 55.0),
        "soil_ph": (6.0, 8.0),
        "soil_moisture": (25.0, 45.0),
        "rainfall": (35.0, 95.0),
        "nitrogen": "low",
        "phosphorus": "medium",
        "potassium": "medium",
    },
    {
        "crop": "Apple",
        "temperature": (7.0, 24.0),
        "humidity": (45.0, 70.0),
        "soil_ph": (5.8, 7.0),
        "soil_moisture": (45.0, 65.0),
        "rainfall": (80.0, 220.0),
        "nitrogen": "medium",
        "phosphorus": "medium",
        "potassium": "high",
    },
]

NPK_LEVELS = ["low", "medium", "high"]

SEASON_ALERT_RANGES = {
    "kharif": {
        "temperature": (20.0, 35.0),
        "humidity": (55.0, 90.0),
    },
    "rabi": {
        "temperature": (8.0, 26.0),
        "humidity": (35.0, 70.0),
    },
    "zaid": {
        "temperature": (24.0, 39.0),
        "humidity": (30.0, 65.0),
    },
    "annual": {
        "temperature": (15.0, 35.0),
        "humidity": (35.0, 80.0),
    },
}

PH_SAFE_RANGE = (5.8, 7.5)
SOIL_MOISTURE_SAFE_RANGE = (30.0, 75.0)

CONDITION_SUGGESTIONS = {
    "temperature:high": "High temperature: increase irrigation frequency, use mulching, and avoid mid-day field operations.",
    "temperature:low": "Low temperature: delay sowing of heat-loving crops and use protective covering where possible.",
    "humidity:high": "High humidity: improve ventilation and spacing to reduce disease pressure.",
    "humidity:low": "Low humidity: reduce evapotranspiration losses using mulching and timely irrigation.",
    "soil_ph:high": "Soil pH is high: apply acidifying amendments and add organic matter to improve nutrient availability.",
    "soil_ph:low": "Soil pH is low: apply agricultural lime based on soil test recommendations.",
    "soil_moisture:high": "Soil moisture is high: improve drainage to prevent root rot and oxygen stress.",
    "soil_moisture:low": "Soil moisture is low: increase irrigation and mulching to improve moisture retention.",
}

# Prompt Templates
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
