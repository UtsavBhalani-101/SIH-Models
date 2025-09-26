# training.py (Regression Version - Dynamic Safety Scores)

import pandas as pd
import numpy as np
import logging
import joblib
import warnings
import os
from datetime import datetime
from scipy.stats import randint, uniform

from feature_engineering import prepare_features

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor  # Changed from XGBClassifier to XGBRegressor

warnings.filterwarnings('ignore')

TRAINING_CONFIG = {
    "dataset_path": "weatherHistory.csv",
    "model_output_path": "weather_safety_model_regression.pkl",
    "test_size": 0.2,
    "random_state": 42,
    "param_distributions": {
        'n_estimators': randint(100, 300),
        'max_depth': randint(6, 12),
        'learning_rate': uniform(0.01, 0.15),
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 2)
    },
    "random_search_iterations": 20,
    "cv_folds": 3
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_model_training_regression.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_numeric_conversion(series, default_value=0):
    """Safely convert series to numeric, handling NaN and inf values."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    return numeric_series.fillna(default_value)

class WeatherSafetyRegressionModel:
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
                'Loud Cover': 'cloud_cover',
                'Cloud Cover': 'cloud_cover',
                'Pressure (millibars)': 'pressure', 
                'Daily Summary': 'daily_summary'
            }
            
            # Only rename columns that exist
            existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_mappings)
            
            logger.info(f"Available columns: {list(df.columns)}")
            return df
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def preprocess_dataset(self, df):
        """Preprocess the dataset with improved error handling."""
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
        
        # Handle weather summary
        if 'summary' in processed_df.columns:
            import re
            top_summaries = processed_df['summary'].value_counts().nlargest(15).index
            logger.info(f"Top weather summaries: {list(top_summaries)}")
            
            for summary_type in top_summaries:
                clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', str(summary_type).lower().strip())
                clean_name = f'summary_{clean_name}'
                processed_df[clean_name] = (processed_df['summary'] == summary_type).astype(int)
            
            logger.info(f"Created {len(top_summaries)} summary indicator features")
        
        # Ensure numeric columns are properly converted
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
                    processed_df[col] = processed_df[col].clip(0, None)
        
        # Handle missing cloud_cover values
        if 'cloud_cover' in processed_df.columns:
            processed_df['cloud_cover'] = processed_df['cloud_cover'].fillna(0.5)
        
        logger.info("Data preprocessing completed successfully.")
        return processed_df

    def create_continuous_safety_scores(self, df):
        """Create continuous safety scores (0-100) instead of discrete categories."""
        logger.info("Creating continuous safety scores based on weather conditions...")
        
        def calculate_continuous_safety_score(row):
            # Start with perfect safety (100)
            safety_score = 100.0
            
            # Temperature penalties (more granular)
            temp = row.get('temperature', 20)
            if temp < -25:
                safety_score -= 40
            elif temp < -20:
                safety_score -= 35
            elif temp < -15:
                safety_score -= 30
            elif temp < -10:
                safety_score -= 25
            elif temp < -5:
                safety_score -= 20
            elif temp < 0:
                safety_score -= 15
            elif temp > 45:
                safety_score -= 40
            elif temp > 40:
                safety_score -= 35
            elif temp > 35:
                safety_score -= 25
            elif temp > 32:
                safety_score -= 15
            elif temp > 30:
                safety_score -= 10
            
            # Wind speed penalties (continuous scale)
            wind = row.get('wind_speed', 0)
            if wind > 100:
                safety_score -= 35
            elif wind > 80:
                safety_score -= 30
            elif wind > 70:
                safety_score -= 25
            elif wind > 60:
                safety_score -= 20
            elif wind > 50:
                safety_score -= 15
            elif wind > 40:
                safety_score -= 10
            elif wind > 30:
                safety_score -= 5
            
            # Visibility penalties (gradual)
            vis = row.get('visibility', 10)
            if vis < 0.1:
                safety_score -= 30
            elif vis < 0.5:
                safety_score -= 25
            elif vis < 1:
                safety_score -= 20
            elif vis < 2:
                safety_score -= 15
            elif vis < 3:
                safety_score -= 10
            elif vis < 5:
                safety_score -= 5
            
            # Apparent temperature (feels like) penalties
            app_temp = row.get('apparent_temperature', temp)
            temp_diff = abs(app_temp - temp)
            if temp_diff > 15:
                safety_score -= 20
            elif temp_diff > 10:
                safety_score -= 15
            elif temp_diff > 5:
                safety_score -= 10
            
            if app_temp < -30:
                safety_score -= 25
            elif app_temp > 50:
                safety_score -= 25
            elif app_temp < -20:
                safety_score -= 15
            elif app_temp > 45:
                safety_score -= 15
            
            # Humidity penalties
            humidity = row.get('humidity', 0.5)
            if humidity > 0.95:
                safety_score -= 10
            elif humidity < 0.1:
                safety_score -= 15
            elif humidity > 0.9:
                safety_score -= 5
            elif humidity < 0.2:
                safety_score -= 8
            
            # Pressure penalties
            pressure = row.get('pressure', 1013)
            pressure_deviation = abs(pressure - 1013.25)
            if pressure_deviation > 50:
                safety_score -= 15
            elif pressure_deviation > 30:
                safety_score -= 10
            elif pressure_deviation > 20:
                safety_score -= 5
            
            # Precipitation penalties
            if row.get('precip_snow', 0) == 1:
                safety_score -= 20
            elif row.get('precip_rain', 0) == 1:
                safety_score -= 10
            
            # Cloud cover penalties (minor)
            cloud_cover = row.get('cloud_cover', 0.5)
            if cloud_cover > 0.9:
                safety_score -= 5
            
            # Add some randomness for more realistic continuous scores
            # This helps the model learn more nuanced patterns
            noise = np.random.normal(0, 2)  # Small random variation
            safety_score += noise
            
            # Ensure score stays within 0-100 range
            return max(0.0, min(100.0, safety_score))
        
        df['continuous_safety_score'] = df.apply(calculate_continuous_safety_score, axis=1)
        
        # Also create discrete categories for reference
        df['safety_category'] = pd.cut(
            df['continuous_safety_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very High Risk', 'High Risk', 'Moderate Risk', 'Safe', 'Very Safe'],
            include_lowest=True
        )
        
        # Log distribution of safety scores
        logger.info("Continuous Safety Score Statistics:")
        logger.info(f"Mean: {df['continuous_safety_score'].mean():.2f}")
        logger.info(f"Std: {df['continuous_safety_score'].std():.2f}")
        logger.info(f"Min: {df['continuous_safety_score'].min():.2f}")
        logger.info(f"Max: {df['continuous_safety_score'].max():.2f}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for regression model training."""
        logger.info("Preparing features for regression model training...")
        
        # Apply feature engineering
        df_featured = prepare_features(df)
        
        # Define columns to exclude from features
        exclude_cols = [
            'formatted_date', 'summary', 'daily_summary', 'precip_type', 
            'continuous_safety_score', 'safety_category'
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
        
        y = df_featured['continuous_safety_score']  # Use continuous target
        
        logger.info(f"Final features prepared. Shape: {X_imputed.shape}")
        logger.info(f"Target statistics: Mean={y.mean():.2f}, Std={y.std():.2f}")
        
        return X_imputed, y

    def train_model(self, X, y):
        """Train the regression model."""
        logger.info("Starting regression model training with hyperparameter optimization...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TRAINING_CONFIG["test_size"], 
            random_state=TRAINING_CONFIG["random_state"]
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize XGBoost regression model
        # Use n_jobs=1 for better cleanup in containerized environments
        n_jobs_setting = 1 if os.environ.get('MODAL_RUNTIME') else -1
        xgb_model = XGBRegressor(
            random_state=TRAINING_CONFIG["random_state"],
            n_jobs=n_jobs_setting,
            objective='reg:squarederror'  # For regression
        )
        
        # Setup cross-validation
        cv = KFold(
            n_splits=TRAINING_CONFIG["cv_folds"], 
            shuffle=True, 
            random_state=TRAINING_CONFIG["random_state"]
        )
        
        # Hyperparameter search
        logger.info("Performing hyperparameter optimization...")
        # Use fewer parallel jobs in containerized environments for better cleanup
        search_n_jobs = 1 if os.environ.get('MODAL_RUNTIME') else -1
        search = RandomizedSearchCV(
            xgb_model,
            TRAINING_CONFIG["param_distributions"],
            n_iter=TRAINING_CONFIG["random_search_iterations"],
            cv=cv,
            scoring='neg_mean_squared_error',  # For regression
            n_jobs=search_n_jobs,
            verbose=1,
            random_state=TRAINING_CONFIG["random_state"]
        )
        
        search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = search.best_estimator_
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {-search.best_score_:.4f} (MSE)")
        
        # Perform additional cross-validation testing with the best model
        from sklearn.model_selection import cross_val_score
        logger.info("Performing cross-validation testing with best model...")
        
        # Test multiple scoring metrics
        cv_scores_mse = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=cv, scoring='neg_mean_squared_error', n_jobs=search_n_jobs
        )
        cv_scores_mae = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=cv, scoring='neg_mean_absolute_error', n_jobs=search_n_jobs
        )
        cv_scores_r2 = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=cv, scoring='r2', n_jobs=search_n_jobs
        )
        
        # Log cross-validation results
        logger.info("Cross-Validation Test Results:")
        logger.info(f"  MSE: {-cv_scores_mse.mean():.4f} ± {cv_scores_mse.std():.4f}")
        logger.info(f"  MAE: {-cv_scores_mae.mean():.4f} ± {cv_scores_mae.std():.4f}")
        logger.info(f"  R²:  {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
        logger.info(f"  Individual CV R² scores: {[f'{score:.4f}' for score in cv_scores_r2]}")
        
        # Final evaluation
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Final Test Metrics:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Most Important Features:")
            logger.info(feature_importance.head(10).to_string(index=False))
        
        # Prepare cross-validation results for return
        cv_results = {
            'mse_mean': float(-cv_scores_mse.mean()),
            'mse_std': float(cv_scores_mse.std()),
            'mae_mean': float(-cv_scores_mae.mean()),
            'mae_std': float(cv_scores_mae.std()),
            'r2_mean': float(cv_scores_r2.mean()),
            'r2_std': float(cv_scores_r2.std()),
            'r2_scores': cv_scores_r2.tolist()
        }
        
        return self.model, r2, cv_results

    def save_model(self, filename: str):
        """Save model and all preprocessing objects."""
        if self.model is None:
            raise ValueError("No model to save! Train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'model_type': 'regression',  # Important flag
            'timestamp': datetime.now().isoformat(),
            'config': TRAINING_CONFIG
        }
        
        joblib.dump(model_data, filename)
        logger.info(f"Regression model saved to {filename}")

def main():
    """Main training pipeline for regression model."""
    logger.info("=" * 60)
    logger.info("Starting Weather Safety Regression Model Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize model
        weather_model = WeatherSafetyRegressionModel()
        
        # Load dataset
        df = weather_model.load_kaggle_dataset(TRAINING_CONFIG["dataset_path"])
        if df is None:
            logger.error("Failed to load dataset. Exiting.")
            return
        
        # Preprocess data
        df_processed = weather_model.preprocess_dataset(df)
        
        # Create continuous safety scores
        df_labeled = weather_model.create_continuous_safety_scores(df_processed)
        
        # Prepare features
        X, y = weather_model.prepare_features(df_labeled)
        
        # Train model
        model, r2_score = weather_model.train_model(X, y)
        
        # Save model
        weather_model.save_model(TRAINING_CONFIG["model_output_path"])
        
        logger.info("=" * 60)
        logger.info(f"Regression Training Pipeline Completed Successfully!")
        logger.info(f"Final Model R²: {r2_score:.4f}")
        logger.info(f"Model saved to: {TRAINING_CONFIG['model_output_path']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()