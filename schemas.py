from typing import Optional, Union
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class CropRecommendationRequest(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soil_ph: Optional[float] = None
    soil_moisture: Optional[float] = None
    nitrogen: Optional[Union[str, float, int]] = None
    phosphorus: Optional[Union[str, float, int]] = None
    potassium: Optional[Union[str, float, int]] = None
    rainfall: Optional[float] = None
    location: Optional[str] = None
    season: Optional[str] = None


class ConditionAlertRequest(BaseModel):
    temperature: float
    humidity: float
    soil_ph: float
    soil_moisture: Optional[float] = None
    season: Optional[str] = None
    location: Optional[str] = None
