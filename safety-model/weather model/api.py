# api.py (Version 2.3 - Fixed NaN Issues)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
import joblib
import pandas as pd
import numpy as np
import logging
import httpx
from datetime import datetime
import uvicorn
import os
from pathlib import Path
from contextlib import asynccontextmanager

from feature_engineering import prepare_features

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Application Startup ---")
    app.state.predictor = WeatherSafetyPredictor()
    if not app.state.predictor.model:
        logger.warning("Model failed to load.")
    yield
    logger.info("--- Application Shutdown ---")

app = FastAPI(title="Weather Safety API", version="2.3.0 (Fixed)", lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_float_conversion(value):
    """Convert numpy values to Python floats, handling NaN and inf values."""
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)

def clean_dict_for_json(data_dict):
    """Clean a dictionary to make it JSON serializable."""
    cleaned = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.floating, float)):
            cleaned[key] = safe_float_conversion(value)
        elif isinstance(value, (np.integer, int)):
            cleaned[key] = int(value)
        elif pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned

class WeatherSafetyPredictor:
    def __init__(self, model_path: str = "weather_safety_model_kaggle.pkl"):
        self.model_path = Path(__file__).resolve().parent / model_path
        self.model, self.scaler, self.imputer, self.feature_names = None, None, None, None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return
        try:
            data = joblib.load(self.model_path)
            self.model, self.scaler, self.imputer, self.feature_names = data['model'], data['scaler'], data['imputer'], data['feature_names']
            logger.info("Model and all preprocessing objects loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def predict(self, weather_data_list: List[dict]) -> List[dict]:
        if not all([self.model, self.scaler, self.imputer, self.feature_names]):
            raise RuntimeError("Model or preprocessing objects are not loaded.")
        
        try:
            input_df = pd.DataFrame(weather_data_list)
            featured_df = prepare_features(input_df)
            aligned_df = featured_df.reindex(columns=self.feature_names)
            
            # Fill any remaining NaN values before imputation
            aligned_df = aligned_df.fillna(0)
            
            imputed_array = self.imputer.transform(aligned_df)
            imputed_df = pd.DataFrame(imputed_array, columns=self.feature_names, index=aligned_df.index)
            
            # Replace any NaN or inf values after imputation
            imputed_df = imputed_df.replace([np.inf, -np.inf], 0).fillna(0)
            
            scaled_array = self.scaler.transform(imputed_df)
            
            # Handle NaN or inf values in scaled array
            scaled_array = np.nan_to_num(scaled_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            predictions = self.model.predict(scaled_array)
            probabilities = self.model.predict_proba(scaled_array)

            safety_labels = {0: "Very Safe", 1: "Safe", 2: "Moderate Risk", 3: "High Risk", 4: "Very High Risk"}
            results = []
            
            # Modified code block 
            for i, (pred_class, pred_probs) in enumerate(zip(predictions, probabilities)):
                # Clean features for JSON serialization
                response_features = aligned_df.iloc[i].to_dict()
                response_features = clean_dict_for_json(response_features)

                # --- START OF NEW LOGIC ---
                # The model's prediction (0-4) is a RISK score.
                weather_risk_score = int(pred_class)

                # 1. Invert the 0-4 risk score to a 0-4 SAFETY score.
                weather_safety_score_0_4 = 4 - weather_risk_score

                # 2. Scale the 0-4 safety score to a 0-100 SAFETY score.
                final_safety_score_100 = weather_safety_score_0_4 * 25
                # --- END OF NEW LOGIC ---

                # Clean probabilities
                clean_probs = {}
                for j, prob in enumerate(pred_probs):
                    clean_probs[safety_labels[j]] = safe_float_conversion(prob)

                results.append({
                    "safety_score": final_safety_score_100, # Use the new 0-100 score
                    "safety_category": safety_labels[weather_risk_score], # The category is still based on the original 0-4 risk score
                    "confidence": safe_float_conversion(np.max(pred_probs)),
                    "probabilities": clean_probs,
                    "input_features": response_features
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

# WeatherInput model with proper field definitions
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-50, le=60)
    apparent_temperature: Optional[float] = Field(None)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0, le=200)
    wind_bearing: Optional[float] = Field(None, ge=0, le=360)
    visibility: float = Field(..., ge=0)
    cloud_cover: Optional[float] = Field(None, ge=0, le=1)
    pressure: float = Field(..., ge=900, le=1100)
    precip_rain: Optional[int] = Field(default=0)
    precip_snow: Optional[int] = Field(default=0)
    summary_clear: Optional[int] = Field(default=0)
    summary_partly_cloudy: Optional[int] = Field(default=0)
    summary_mostly_cloudy: Optional[int] = Field(default=0)
    summary_overcast: Optional[int] = Field(default=0)
    summary_foggy: Optional[int] = Field(default=0)
    summary_rain: Optional[int] = Field(default=0)
    summary_wind: Optional[int] = Field(default=0)
    summary_snow: Optional[int] = Field(default=0)
    summary_breezy: Optional[int] = Field(default=0)
    
    @field_validator('humidity', mode='before')
    def v_humidity(cls, v): 
        return v / 100.0 if v > 1 else v
    
    @field_validator('apparent_temperature', mode='before')
    def v_apparent_temp(cls, v, info): 
        return info.data.get('temperature') if v is None else v

class OpenMeteoInput(BaseModel): 
    latitude: float
    longitude: float
    days: Optional[int] = 1

class SafetyPrediction(BaseModel): 
    safety_score: int
    safety_category: str
    confidence: float
    probabilities: Dict[str, float]
    input_features: Dict
    data_source: Optional[str] = None

@app.get("/")
async def root(): 
    return {"message": "Weather Safety API", "version": "2.3.0 (Fixed)"}

@app.get("/health")
async def health(): 
    return {
        "status": "healthy" if app.state.predictor.model else "unhealthy", 
        "model_loaded": bool(app.state.predictor.model)
    }

@app.get("/model/info")
async def model_info():
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type": type(app.state.predictor.model).__name__, 
        "features_count": len(app.state.predictor.feature_names), 
        "safety_categories": ["Very Safe", "Safe", "Moderate Risk", "High Risk", "Very High Risk"]
    }

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(weather: WeatherInput):
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    try:
        result = app.state.predictor.predict([weather.model_dump(exclude_none=True)])[0]
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(weather_list: List[WeatherInput]):
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    try:
        results = app.state.predictor.predict([w.model_dump(exclude_none=True) for w in weather_list])
        return {
            "batch_size": len(results), 
            "predictions": results, 
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")

@app.post("/predict/openmeteo", response_model=SafetyPrediction)
async def predict_openmeteo(location: OpenMeteoInput):
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    try:
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": location.latitude, 
            "longitude": location.longitude, 
            "hourly": "temperature_2m,relativehumidity_2m,apparent_temperature,pressure_msl,cloudcover,windspeed_10m,winddirection_10m,visibility,precipitation", 
            "forecast_days": location.days
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, params=params)
            response.raise_for_status()
        
        data = response.json()['hourly']
        latest = {k: v[-1] if v else 0 for k, v in data.items()}
        
        weather_data = {
            "temperature": latest.get('temperature_2m', 20), 
            "humidity": latest.get('relativehumidity_2m', 50), 
            "wind_speed": latest.get('windspeed_10m', 5),
            "apparent_temperature": latest.get('apparent_temperature', 20), 
            "pressure": latest.get('pressure_msl', 1013), 
            "visibility": max(latest.get('visibility', 0) / 1000, 0.1), 
            "wind_bearing": latest.get('winddirection_10m', 0), 
            "cloud_cover": latest.get('cloudcover', 0) / 100, 
            "precip_rain": 1 if latest.get('precipitation', 0) > 0 else 0
        }
        
        prediction = app.state.predictor.predict([weather_data])[0]
        prediction['data_source'] = 'open-meteo'
        return prediction
        
    except Exception as e:
        logger.error(f"Open-Meteo prediction failed: {e}")
        raise HTTPException(500, f"Open-Meteo prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)