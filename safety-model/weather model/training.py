# training.py (Version 2.3 - Improved and Compatible)

import pandas as pd
import numpy as np
import requests
import logging
import joblib
import shutil
import os
import re
import warnings
from datetime import datetime, timedelta
from scipy.stats import randint, uniform

from feature_engineering import prepare_features

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

TRAINING_CONFIG = {
    "dataset_path": "weatherHistory.csv",
    "model_output_path": "weather_safety_model_kaggle.pkl",
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
    "param_distributions": {
        'n_estimators': randint(200, 500),
        'max_depth': randint(6, 15),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 2)
    },
    "random_search_iterations": 50,
    "cv_folds": 5
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_model_training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_numeric_conversion(series, default_value=0):
    """Safely convert series to numeric, handling NaN and inf values."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    return numeric_series.fillna(default_value)

class WeatherSafetyModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        
    def load_kaggle_dataset(self, dataset_path):
        """Load and perform initial validation of the Kaggle dataset."""
        logger.info(f"Loading dataset from: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Handle potential column name variations
            column_mapping = {
                'Formatted Date': 'formatted_date', 
                'Summary': 'summary', 
                'Precip Type': 'precip_type', 
                'Temperature (C)': 'temperature',
                'Apparent Temperature (C)': 'apparent_temperature', 
                'Humidity': 'humidity',
                'Wind Speed (km/h)': 'wind_speed', 
                'Wind Bearing (degrees)': 'wind_bearing',
                'Visibility (km)': 'visibility', 
                'Loud Cover': 'cloud_cover',  # Note: This is likely a typo in the original dataset
                'Cloud Cover': 'cloud_cover',  # Handle both possible names
                'Pressure (millibars)': 'pressure', 
                'Daily Summary': 'daily_summary'
            }
            
            # Only rename columns that exist
            existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_mappings)
            
            # Log available columns
            logger.info(f"Available columns: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def preprocess_dataset(self, df):
        """Preprocess the dataset with improved error handling and data validation."""
        logger.info("Preprocessing the dataset...")
        processed_df = df.copy()
        
        # Handle date parsing
        if 'formatted_date' in processed_df.columns:
            try:
                processed_df['formatted_date'] = pd.to_datetime(processed_df['formatted_date'], utc=True)
                processed_df['year'] = processed_df['formatted_date'].dt.year
                processed_df['month'] = processed_df['formatted_date'].dt.month
                processed_df['day'] = processed_df['formatted_date'].dt.day
                processed_df['hour'] = processed_df['formatted_date'].dt.hour
                processed_df['day_of_year'] = processed_df['formatted_date'].dt.dayofyear
                
                # Cyclical encoding for temporal features
                processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24.0)
                processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24.0)
                processed_df['month_sin'] = np.sin(2 * np.pi * processed_df['month'] / 12.0)
                processed_df['month_cos'] = np.cos(2 * np.pi * processed_df['month'] / 12.0)
                
                logger.info("Date features successfully created")
            except Exception as e:
                logger.warning(f"Could not parse date column: {e}")
        
        # Handle precipitation type
        if 'precip_type' in processed_df.columns:
            processed_df['precip_type'] = processed_df['precip_type'].fillna('none')
            processed_df['precip_rain'] = (processed_df['precip_type'] == 'rain').astype(int)
            processed_df['precip_snow'] = (processed_df['precip_type'].isin(['snow', 'sleet'])).astype(int)
            logger.info("Precipitation type features created")
        
        # Handle weather summary with more robust processing
        if 'summary' in processed_df.columns:
            # Get top weather summaries
            top_summaries = processed_df['summary'].value_counts().nlargest(15).index
            logger.info(f"Top weather summaries: {list(top_summaries)}")
            
            for summary_type in top_summaries:
                # Clean summary names for column names
                clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', str(summary_type).lower().strip())
                clean_name = f'summary_{clean_name}'
                processed_df[clean_name] = (processed_df['summary'] == summary_type).astype(int)
            
            logger.info(f"Created {len(top_summaries)} summary indicator features")
        
        # Ensure numeric columns are properly converted and handle outliers
        numeric_columns = ['temperature', 'apparent_temperature', 'humidity', 'wind_speed', 
                          'wind_bearing', 'visibility', 'cloud_cover', 'pressure']
        
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = safe_numeric_conversion(processed_df[col])
                
                # Handle specific column constraints
                if col == 'humidity':
                    processed_df[col] = processed_df[col].clip(0, 1)
                elif col == 'wind_bearing':
                    processed_df[col] = processed_df[col].clip(0, 360)
                elif col == 'cloud_cover':
                    processed_df[col] = processed_df[col].clip(0, 1)
                elif col == 'visibility':
                    processed_df[col] = processed_df[col].clip(0, None)  # Can't be negative
        
        # Handle missing cloud_cover values
        if 'cloud_cover' in processed_df.columns:
            processed_df['cloud_cover'] = processed_df['cloud_cover'].fillna(0.5)  # Assume moderate coverage
        
        logger.info("Data preprocessing completed successfully.")
        logger.info(f"Final preprocessed shape: {processed_df.shape}")
        
        return processed_df

    def create_safety_labels(self, df):
        """Create safety labels with improved scoring logic."""
        logger.info("Creating safety labels based on weather conditions...")
        
        def calculate_safety_score(row):
            score = 0
            
            # Temperature risks
            temp = row.get('temperature', 20)
            if temp < -20 or temp > 45:
                score += 3
            elif temp < -10 or temp > 35:
                score += 2
            elif temp < 0 or temp > 30:
                score += 1
            
            # Wind speed risks
            wind = row.get('wind_speed', 0)
            if wind > 80:
                score += 3
            elif wind > 60:
                score += 2
            elif wind > 40:
                score += 1
            
            # Visibility risks
            vis = row.get('visibility', 10)
            if vis < 0.5:
                score += 3
            elif vis < 1:
                score += 2
            elif vis < 3:
                score += 1
            
            # Apparent temperature (feels like)
            app_temp = row.get('apparent_temperature', temp)
            if app_temp < -25 or app_temp > 50:
                score += 2
            elif app_temp < -15 or app_temp > 40:
                score += 1
            
            # Humidity extremes
            humidity = row.get('humidity', 0.5)
            if humidity > 0.95 or humidity < 0.1:
                score += 1
            
            # Pressure extremes
            pressure = row.get('pressure', 1013)
            if pressure < 980 or pressure > 1040:
                score += 1
            
            # Precipitation risks
            if row.get('precip_snow', 0) == 1:
                score += 2
            elif row.get('precip_rain', 0) == 1:
                score += 1
            
            return min(score, 4)  # Cap at 4 (Very High Risk)
        
        df['safety_score'] = df.apply(calculate_safety_score, axis=1)
        
        # Add safety category labels
        safety_categories = {
            0: "Very Safe",
            1: "Safe", 
            2: "Moderate Risk",
            3: "High Risk",
            4: "Very High Risk"
        }
        df['safety_category'] = df['safety_score'].map(safety_categories)
        
        # Log distribution of safety scores
        score_distribution = df['safety_score'].value_counts().sort_index()
        logger.info("Safety score distribution:")
        for score, count in score_distribution.items():
            logger.info(f"  {score} ({safety_categories[score]}): {count} samples")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training with enhanced validation."""
        logger.info("Preparing features for model training...")
        
        # Apply feature engineering
        df_featured = prepare_features(df)
        
        # Define columns to exclude from features
        exclude_cols = [
            'formatted_date', 'summary', 'daily_summary', 'precip_type', 
            'safety_score', 'safety_category'
        ]
        
        # Select feature columns (numeric only)
        feature_cols = []
        for col in df_featured.columns:
            if col not in exclude_cols:
                if df_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
        
        logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")
        
        X = df_featured[feature_cols].copy()
        
        # Handle any remaining infinite or extreme values
        for col in X.columns:
            X[col] = safe_numeric_conversion(X[col])
        
        # Apply imputation
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X), 
            columns=X.columns,
            index=X.index
        )
        
        # Store feature names for API compatibility
        self.feature_names = list(X_imputed.columns)
        
        y = df_featured['safety_score']
        
        logger.info(f"Final features prepared. Shape: {X_imputed.shape}")
        logger.info(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
        
        return X_imputed, y

    def train_model(self, X, y):
        """Train the model with improved hyperparameter search and validation."""
        logger.info("Starting model training with hyperparameter optimization...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TRAINING_CONFIG["test_size"], 
            random_state=TRAINING_CONFIG["random_state"], 
            stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize XGBoost model
        xgb_model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=TRAINING_CONFIG["random_state"],
            n_jobs=-1
        )
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=TRAINING_CONFIG["cv_folds"], 
            shuffle=True, 
            random_state=TRAINING_CONFIG["random_state"]
        )
        
        # Hyperparameter search
        logger.info("Performing hyperparameter optimization...")
        search = RandomizedSearchCV(
            xgb_model,
            TRAINING_CONFIG["param_distributions"],
            n_iter=TRAINING_CONFIG["random_search_iterations"],
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1,
            random_state=TRAINING_CONFIG["random_state"]
        )
        
        search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = search.best_estimator_
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Final evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Final Test Accuracy: {accuracy:.4f}")
        logger.info(f"Final Test F1-Score: {f1:.4f}")
        logger.info("\nDetailed Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Most Important Features:")
            logger.info(feature_importance.head(10).to_string(index=False))
        
        return self.model, accuracy

    def save_model(self, filename: str):
        """Save model and all preprocessing objects."""
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat(),
            'config': TRAINING_CONFIG
        }
        
        joblib.dump(model_data, filename)
        logger.info(f"Model and preprocessing objects saved to {filename}")
        logger.info(f"Model file size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

def validate_dataset(df):
    """Validate dataset before training."""
    logger.info("Validating dataset...")
    
    if df is None or df.empty:
        logger.error("Dataset is empty or None")
        return False
    
    required_columns = ['temperature', 'humidity', 'wind_speed', 'pressure']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("Dataset validation passed")
    return True

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Weather Safety Model Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize model
        weather_model = WeatherSafetyModel()
        
        # Load dataset
        df = weather_model.load_kaggle_dataset(TRAINING_CONFIG["dataset_path"])
        if not validate_dataset(df):
            logger.error("Dataset validation failed. Exiting.")
            return
        
        # Preprocess data
        df_processed = weather_model.preprocess_dataset(df)
        
        # Create safety labels
        df_labeled = weather_model.create_safety_labels(df_processed)
        
        # Prepare features
        X, y = weather_model.prepare_features(df_labeled)
        
        # Train model
        model, accuracy = weather_model.train_model(X, y)
        
        # Save model
        weather_model.save_model(TRAINING_CONFIG["model_output_path"])
        
        logger.info("=" * 60)
        logger.info(f"Training Pipeline Completed Successfully!")
        logger.info(f"Final Model Accuracy: {accuracy:.4f}")
        logger.info(f"Model saved to: {TRAINING_CONFIG['model_output_path']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()