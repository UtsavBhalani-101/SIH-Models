# feature_engineering.py (Final and Complete)

import pandas as pd
import numpy as np

def prepare_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction in the same format as training.
    This function serves as a single source of truth for both training and API.
    
    Args:
        input_data: A DataFrame containing the raw weather features.
                    Can be a single row for prediction or a full dataset for training.
    
    Returns:
        A DataFrame with all engineered features.
    """
    df = input_data.copy()

    # --- Derived Features ---
    if 'temperature' in df.columns and 'apparent_temperature' in df.columns:
        df['temp_difference'] = df['apparent_temperature'] - df['temperature']

    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] * 100 - 50)

    if 'wind_speed' in df.columns and 'temperature' in df.columns:
        df['wind_chill'] = np.where(
            df['temperature'] < 10,
            13.12 + 0.6215 * df['temperature'] - 11.37 * (df['wind_speed'] ** 0.16) + 0.3965 * df['temperature'] * (df['wind_speed'] ** 0.16),
            df['temperature']
        )

    # --- Interaction Features ---
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

    if 'wind_speed' in df.columns and 'visibility' in df.columns:
        df['wind_visibility_interaction'] = df['wind_speed'] / (df['visibility'] + 1e-6)

    if 'pressure' in df.columns:
        df['pressure_deviation'] = abs(df['pressure'] - 1013.25)

    return df