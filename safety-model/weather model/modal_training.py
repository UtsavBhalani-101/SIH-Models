"""Modal Training Script for Weather Safety Model (Modal v1.1)

This script runs the weather safety model training on Modal infrastructure.
It has been migrated to Modal v1.1 APIs and includes all necessary dependencies.
"""

import modal
from pathlib import Path
import sys

# Name of the Modal app
app = modal.App("weather-safety-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.2.0",
            "xgboost>=1.7.0",  # GPU-enabled XGBoost will auto-detect CUDA when available
            "joblib>=1.1.0",
            "requests>=2.28.0",
            "python-dateutil>=2.8.0",
            "pytz>=2022.1",
        ]
    )
)

# Add local project files into the image
project_root = Path(__file__).resolve().parent
image = image.add_local_dir(str(project_root), remote_path="/root/app")

# Add the weather dataset if it exists
dataset_file = project_root / "weatherHistory.csv"
if dataset_file.exists():
    image = image.add_local_file(str(dataset_file), remote_path="/root/app/weatherHistory.csv")

# Create a volume for persisting the trained model
model_volume = modal.Volume.from_name("weather-safety-models", create_if_missing=True)

@app.function(
    image=image,
    min_containers=1,
    timeout=3600,  # 4GB memory
    gpu="T4",  # Add GPU support for faster training
    volumes={"/root/models": model_volume}
)
def train_weather_model():
    import atexit
    import signal
    import os
    import gc
    
    # Adjust sys.path to import from the added project directory
    app_dir = Path("/root/app")
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    # Import the training module
    from training import WeatherSafetyRegressionModel
    import joblib

    # Configure joblib to use fewer processes in Modal environment
    os.environ['MODAL_RUNTIME'] = '1'  # Flag to indicate we're running in Modal
    os.environ['JOBLIB_START_METHOD'] = 'spawn'
    os.environ['OMP_NUM_THREADS'] = '1'

    # Setup cleanup handler for joblib processes
    def cleanup_processes():
        gc.collect()
        
    atexit.register(cleanup_processes)

    print("Starting weather safety model training on Modal with GPU support...")
    print(f"GPU configured: T4")
    print(f"CPU cores: 2.0, Memory: 4GB")

    # Change to the app directory
    os.chdir("/root/app")

    # Initialize the model
    weather_model = WeatherSafetyRegressionModel()

    # Load the dataset
    dataset_path = "weatherHistory.csv"
    print(f"Loading dataset from: {dataset_path}")
    df = weather_model.load_kaggle_dataset(dataset_path)

    if df is None:
        error_msg = "Failed to load dataset. Please check the file path."
        print(error_msg)
        return {"status": "error", "message": error_msg}

    # Preprocess the dataset
    df_processed = weather_model.preprocess_dataset(df)

    # Create safety labels (using continuous scores for regression)
    df_labeled = weather_model.create_continuous_safety_scores(df_processed)

    # Prepare features
    X, y = weather_model.prepare_features(df_labeled)

    # Override training config for Modal environment to reduce parallel processing issues
    import training
    original_config = training.TRAINING_CONFIG.copy()
    training.TRAINING_CONFIG['random_search_iterations'] = 10  # Reduce iterations
    training.TRAINING_CONFIG['cv_folds'] = 2  # Reduce CV folds
    
    try:
        # Train model
        model, r2_score, cv_results = weather_model.train_model(X, y)
    finally:
        # Restore original config
        training.TRAINING_CONFIG = original_config

    # Get the model data for return
    model_data = {
        'model': weather_model.model,
        'scaler': weather_model.scaler,
        'imputer': weather_model.imputer,
        'feature_names': weather_model.feature_names,
        'model_type': 'regression'
    }

    # Get feature importance if available
    feature_importance_dict = []
    if hasattr(weather_model.model, 'feature_importances_'):
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'feature': weather_model.feature_names,
            'importance': weather_model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importance for analysis
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        print("Feature importance saved to 'feature_importance.csv'")
        feature_importance_dict = feature_importance_df.to_dict('records')

    # Save model to Modal volume for persistence
    model_volume_path = "/root/models/weather_safety_model_kaggle.pkl"
    joblib.dump(model_data, model_volume_path)
    print(f"Model saved to Modal volume: {model_volume_path}")

    # Also save locally in the container
    local_model_path = "/root/app/weather_safety_model_kaggle.pkl"
    joblib.dump(model_data, local_model_path)
    print(f"Model saved locally in container: {local_model_path}")

    print("Training completed successfully!")
    print(f"Final model R² score: {r2_score:.4f}")
    
    # Print cross-validation results
    print("\nCross-Validation Test Results:")
    print(f"  MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
    print(f"  MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
    print(f"  R²:  {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

    # Cleanup: Force garbage collection and process cleanup
    import time
    import threading
    
    # Force garbage collection
    gc.collect()
    
    # Wait briefly for background threads to finish
    time.sleep(0.5)
    
    print("Cleanup completed, returning results...")

    # Return the model data and metadata
    return {
        "status": "success",
        "message": "Weather safety regression model trained successfully",
        "r2_score": float(r2_score),
        "cv_results": cv_results,
        "model_data": model_data,
        "feature_importance": feature_importance_dict,
        "model_volume_path": model_volume_path
    }

@app.local_entrypoint()
def run_training():
    """
    Local entry point to trigger training on Modal and save the model locally
    """
    import joblib

    print("Triggering weather safety model training on Modal with GPU acceleration...")
    result = train_weather_model.remote()

    if result["status"] == "success":
        print(f"Training completed with R² score: {result['r2_score']:.4f}")
        
        # Display cross-validation results
        cv_results = result.get('cv_results', {})
        if cv_results:
            print("\nCross-Validation Test Results:")
            print(f"  MSE: {cv_results['mse_mean']:.4f} ± {cv_results['mse_std']:.4f}")
            print(f"  MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
            print(f"  R²:  {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        
        print(f"Model also saved to Modal volume: {result.get('model_volume_path', 'N/A')}")

        # Save the trained model locally
        model_filename = "weather_safety_model_kaggle.pkl"
        joblib.dump(result["model_data"], model_filename)
        print(f"Model saved locally as: {model_filename}")

        # Save feature importance if available
        if result["feature_importance"]:
            import pandas as pd
            feature_importance_df = pd.DataFrame(result["feature_importance"])
            feature_importance_df.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to 'feature_importance.csv'")

        print("Model and artifacts successfully saved to local filesystem and Modal volume!")
    else:
        print(f"Training failed: {result['message']}")

    return result

if __name__ == "__main__":
    # This allows running locally for testing, but will deploy to Modal when using modal run
    run_training()
