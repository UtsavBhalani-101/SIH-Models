# api.py (Regression Version - Dynamic Safety Scores)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
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

app = FastAPI(title="Weather Safety API (Regression)", version="3.0.0 (Dynamic Scores)", lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_float_conversion(value):
    """Convert numpy values to Python floats, handling NaN and inf values."""
    if pd.isna(value) or np.isinf(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def clean_dict_for_json(data_dict):
    """Clean a dictionary to make it JSON serializable."""
    cleaned = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.floating, float, np.float32, np.float64)):
            cleaned[key] = safe_float_conversion(value)
        elif isinstance(value, (np.integer, int, np.int32, np.int64)):
            cleaned[key] = int(value)
        elif pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, np.ndarray):
            # Handle numpy arrays by converting to list
            cleaned[key] = value.tolist()
        else:
            cleaned[key] = value
    return cleaned

def get_safety_category(safety_score):
    """Convert continuous safety score to category."""
    if safety_score >= 80:
        return "Very Safe"
    elif safety_score >= 60:
        return "Safe"
    elif safety_score >= 40:
        return "Moderate Risk"
    elif safety_score >= 20:
        return "High Risk"
    else:
        return "Very High Risk"

def calculate_confidence(safety_score):
    """Calculate confidence based on how close the score is to category boundaries."""
    # Distance from nearest boundary
    boundaries = [0, 20, 40, 60, 80, 100]
    distances = [abs(safety_score - b) for b in boundaries]
    min_distance = min(distances)
    
    # Convert to confidence (closer to boundary = lower confidence)
    confidence = min(0.95, max(0.55, (20 - min_distance) / 20))
    return confidence

class WeatherSafetyPredictor:
    def __init__(self, model_path: str = "weather_safety_model_kaggle.pkl"):
        self.model_path = Path(__file__).resolve().parent / model_path
        self.model, self.scaler, self.imputer, self.feature_names = None, None, None, None
        self.model_type = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.imputer = data['imputer'] 
            self.feature_names = data['feature_names']
            self.model_type = data.get('model_type', 'classification')  # Default to classification for backward compatibility
            logger.info(f"Model loaded successfully. Type: {self.model_type}")
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
            
            if self.model_type == 'regression':
                # Regression model - predict continuous scores directly
                predictions = self.model.predict(scaled_array)
                
                # Ensure predictions are in valid range (0-100)
                predictions = np.clip(predictions, 0, 100)
                
                results = []
                for i, safety_score in enumerate(predictions):
                    # Clean features for JSON serialization
                    response_features = aligned_df.iloc[i].to_dict()
                    response_features = clean_dict_for_json(response_features)
                    
                    # Get category and confidence
                    safety_score_clean = safe_float_conversion(safety_score)
                    safety_category = get_safety_category(safety_score_clean)
                    confidence = calculate_confidence(safety_score_clean)
                    
                    # For regression, we don't have class probabilities, 
                    # so we'll create a mock probability distribution centered on the predicted category
                    probabilities = self._create_mock_probabilities(safety_score_clean)
                    
                    results.append({
                        "safety_score": round(safety_score_clean, 2),  # Dynamic continuous score
                        "safety_category": safety_category,
                        "confidence": round(safe_float_conversion(confidence), 3),
                        "probabilities": probabilities,
                        "input_features": response_features,
                        "model_type": "regression"
                    })
                
            else:
                # Classification model - original logic (for backward compatibility)
                predictions = self.model.predict(scaled_array)
                probabilities = self.model.predict_proba(scaled_array)

                safety_labels = {0: "Very Safe", 1: "Safe", 2: "Moderate Risk", 3: "High Risk", 4: "Very High Risk"}
                results = []
                
                for i, (pred_class, pred_probs) in enumerate(zip(predictions, probabilities)):
                    # Clean features for JSON serialization
                    response_features = aligned_df.iloc[i].to_dict()
                    response_features = clean_dict_for_json(response_features)
                    
                    # Original classification logic with your 0-100 conversion
                    weather_risk_score = int(pred_class)
                    weather_safety_score_0_4 = 4 - weather_risk_score
                    final_safety_score_100 = weather_safety_score_0_4 * 25
                    
                    # Clean probabilities
                    clean_probs = {}
                    for j, prob in enumerate(pred_probs):
                        clean_probs[safety_labels[j]] = safe_float_conversion(prob)
                    
                    results.append({
                        "safety_score": final_safety_score_100,
                        "safety_category": safety_labels[weather_risk_score],
                        "confidence": safe_float_conversion(np.max(pred_probs)),
                        "probabilities": clean_probs,
                        "input_features": response_features,
                        "model_type": "classification"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _create_mock_probabilities(self, safety_score):
        """Create mock probabilities for regression model based on safety score."""
        categories = ["Very High Risk", "High Risk", "Moderate Risk", "Safe", "Very Safe"]
        probs = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Determine primary category
        if safety_score >= 80:
            primary_idx = 4  # Very Safe
        elif safety_score >= 60:
            primary_idx = 3  # Safe
        elif safety_score >= 40:
            primary_idx = 2  # Moderate Risk
        elif safety_score >= 20:
            primary_idx = 1  # High Risk
        else:
            primary_idx = 0  # Very High Risk
        
        # Set primary probability
        probs[primary_idx] = 0.7
        
        # Distribute remaining probability to adjacent categories
        remaining = 0.3
        if primary_idx > 0:
            probs[primary_idx - 1] = remaining / 2
        if primary_idx < 4:
            probs[primary_idx + 1] = remaining / 2
        
        # If at boundary, adjust probabilities
        if primary_idx == 0:
            probs[primary_idx + 1] = remaining
        elif primary_idx == 4:
            probs[primary_idx - 1] = remaining
        
        return {categories[i]: float(probs[i]) for i in range(5)}

# WeatherInput model (same as before)
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
    
    @validator('humidity', pre=True)
    def v_humidity(cls, v): 
        return v / 100.0 if v > 1 else v
    
    @validator('apparent_temperature', pre=True)
    def v_apparent_temp(cls, v, values): 
        return values.get('temperature') if v is None else v

class OpenMeteoInput(BaseModel): 
    latitude: float
    longitude: float
    days: Optional[int] = 1

class SafetyPrediction(BaseModel): 
    safety_score: float  # Changed from int to float for continuous scores
    safety_category: str
    confidence: float
    probabilities: Dict[str, float]
    input_features: Dict
    data_source: Optional[str] = None
    model_type: Optional[str] = None

@app.get("/")
async def root(): 
    return {
        "message": "Weather Safety API (Regression)", 
        "version": "3.0.0 (Dynamic Scores)",
        "features": ["Continuous safety scores 0-100", "Dynamic predictions", "Regression model"]
    }

@app.get("/health")
async def health(): 
    return {
        "status": "healthy" if app.state.predictor.model else "unhealthy", 
        "model_loaded": bool(app.state.predictor.model),
        "model_type": getattr(app.state.predictor, 'model_type', 'unknown')
    }

@app.get("/model/info")
async def model_info():
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type": f"{type(app.state.predictor.model).__name__} ({app.state.predictor.model_type})", 
        "features_count": len(app.state.predictor.feature_names), 
        "safety_categories": ["Very High Risk", "High Risk", "Moderate Risk", "Safe", "Very Safe"],
        "score_range": "0-100 (continuous)" if app.state.predictor.model_type == 'regression' else "0,25,50,75,100 (discrete)",
        "prediction_type": "Dynamic continuous scores" if app.state.predictor.model_type == 'regression' else "Fixed category scores"
    }

@app.post("/predict", response_model=SafetyPrediction)
async def predict_safety(weather: WeatherInput):
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    try:
        result = app.state.predictor.predict([weather.dict(exclude_none=True)])[0]
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(weather_list: List[WeatherInput]):
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    try:
        results = app.state.predictor.predict([w.dict(exclude_none=True) for w in weather_list])
        return {
            "batch_size": len(results), 
            "predictions": results, 
            "timestamp": datetime.now().isoformat(),
            "model_type": app.state.predictor.model_type
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

@app.get("/model/sample-predictions")
async def sample_predictions():
    """Get sample predictions to demonstrate the difference between classification and regression."""
    if not app.state.predictor.model: 
        raise HTTPException(503, "Model not loaded")
    
    sample_weather_conditions = [
        {
            "name": "Perfect Weather",
            "data": {
                "temperature": 22, "humidity": 50, "wind_speed": 8, 
                "visibility": 15, "pressure": 1013, "cloud_cover": 0.2
            }
        },
        {
            "name": "Slightly Cloudy",
            "data": {
                "temperature": 18, "humidity": 65, "wind_speed": 15, 
                "visibility": 12, "pressure": 1010, "cloud_cover": 0.6
            }
        },
        {
            "name": "Stormy Weather",
            "data": {
                "temperature": 5, "humidity": 90, "wind_speed": 65, 
                "visibility": 2, "pressure": 985, "cloud_cover": 1.0, "precip_rain": 1
            }
        },
        {
            "name": "Extreme Cold",
            "data": {
                "temperature": -25, "humidity": 80, "wind_speed": 45, 
                "visibility": 8, "pressure": 1020, "cloud_cover": 0.8, "precip_snow": 1
            }
        }
    ]
    
    try:
        results = {}
        for condition in sample_weather_conditions:
            prediction = app.state.predictor.predict([condition["data"]])[0]
            results[condition["name"]] = {
                "safety_score": prediction["safety_score"],
                "safety_category": prediction["safety_category"],
                "model_type": prediction.get("model_type", "unknown")
            }
        
        return {
            "sample_predictions": results,
            "explanation": {
                "regression": "Continuous scores (e.g., 67.3, 23.8, 89.1) - more precise and realistic",
                "classification": "Fixed scores (0, 25, 50, 75, 100) - discrete categories only"
            }
        }
    except Exception as e:
        logger.error(f"Sample predictions failed: {e}")
        raise HTTPException(500, f"Sample predictions failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)