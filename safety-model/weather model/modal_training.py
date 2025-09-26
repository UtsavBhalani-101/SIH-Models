"""Modal Training Script for Weather Safety Model (Modal v1.1)

This script runs the weather safety model training on Modal infrastructure.
It has been migrated to Modal v1.1 APIs and includes all necessary dependencies.
"""

import modal
from pathlib import Path
import os
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
    # Adjust sys.path to import from the added project directory
    app_dir = Path("/root/app")
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    # Import the training module
    from training import WeatherSafetyModel
    import joblib

    print("Starting weather safety model training on Modal with GPU support...")
    print(f"GPU configured: T4")
    print(f"CPU cores: 2.0, Memory: 4GB")

    # Change to the app directory
    os.chdir("/root/app")

    # Initialize the model
    weather_model = WeatherSafetyModel()

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

    # Create safety labels
    df_labeled = weather_model.create_safety_labels(df_processed)

    # Prepare features
    X, y = weather_model.prepare_features(df_labeled)

    # Train model
    model, accuracy, feature_importance = weather_model.train_model(X, y)

    # Get the model data for return
    model_data = {
        'model': weather_model.model,
        'scaler': weather_model.scaler,
        'feature_names': weather_model.feature_names,
        'precip_encoder': weather_model.precip_encoder
    }

    # Save feature importance for analysis
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")

    # Save model to Modal volume for persistence
    model_volume_path = "/root/models/weather_safety_model_kaggle.pkl"
    joblib.dump(model_data, model_volume_path)
    print(f"Model saved to Modal volume: {model_volume_path}")

    # Also save locally in the container
    local_model_path = "/root/app/weather_safety_model_kaggle.pkl"
    joblib.dump(model_data, local_model_path)
    print(f"Model saved locally in container: {local_model_path}")

    print("Training completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")

    # Return the model data and metadata
    return {
        "status": "success",
        "message": "Weather safety model trained successfully",
        "accuracy": float(accuracy),
        "model_data": model_data,
        "feature_importance": feature_importance.to_dict('records'),
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
        print(f"Training completed with accuracy: {result['accuracy']:.4f}")
        print(f"Model also saved to Modal volume: {result.get('model_volume_path', 'N/A')}")

        # Save the trained model locally
        model_filename = "weather_safety_model_kaggle.pkl"
        joblib.dump(result["model_data"], model_filename)
        print(f"Model saved locally as: {model_filename}")

        # Save feature importance
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
