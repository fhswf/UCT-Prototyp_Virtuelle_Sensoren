"""Create rolling, time-based, and categorical model features."""

import numpy as np
import pandas as pd


def calculate_ewma(df, columns):
    """Add 15-observation exponentially weighted means for selected columns."""
    for col in columns:
        if col in df.columns:
            df[f"ewma15_{col}"] = df[col].ewm(span=15).mean()
    
    df= df.dropna(subset=[f"ewma15_{col}" for col in columns])
    
    return df

def calculate_rolling_mean(df, columns):
    """Add 15-observation rolling means for selected columns."""
    for col in columns:
        if col in df.columns:
            df[f"ema15_{col}"] = df[col].rolling(15).mean()
    
    df =df.dropna(subset=[f"ema15_{col}" for col in columns])
    
    return df

def messzeit_utc_handling(df, handling):
    """Add configured cyclical features derived from UTC measurement timestamps."""
    if handling.get("extract_month", False):
        if "messzeit_utc" in df.columns:
            # Encode the month (1-12) as a cyclical feature.
            df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.month / 12)
            #df["month"] = pd.to_datetime(df["messzeit_utc"]).dt.month

    if handling.get("extract_time", False):
        if "messzeit_utc" in df.columns:
            # Encode the hour (0-23) as a cyclical feature.
            df["hour_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["messzeit_utc"]).dt.hour / 24)
            #df["hour"] = pd.to_datetime(df["messzeit_utc"]).dt.hour
        
    if handling.get("extract_doy", False):
        if "messzeit_utc" in df.columns:
            doy = pd.to_datetime(df["messzeit_utc"]).dt.dayofyear
            angle = (doy + 11) % 365 / 365 * 2 * np.pi 
            df["doy_cos"] = (-np.cos(angle) + 1) / 2
    return df

def transform_categorical_features(df):
    """Convert categorical terrain-form values to the model's binary encoding."""
    if "GEL_FORM" in df.columns:
        # Convert the terrain type to a binary feature: {1, 2} -> {0, 1}.
        df['GEL_FORM'] = df['GEL_FORM'].apply(lambda x: 1 if x == 2 else 0)
    return df