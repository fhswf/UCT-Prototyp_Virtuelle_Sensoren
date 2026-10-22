"""Handle missing and invalid temperature and neighbor measurements."""

import datetime
import pandas as pd

def nan_handling(df, handling):
    """Apply configured missing-value replacements and drop remaining rows."""
    print(f"Handling NaN values ...")

    if 'temperature_value_DWD001' in df.columns:
        df['temperature_value_DWD001'] = df['temperature_value_DWD001'].replace(-999, pd.NA)
    
    if handling.get("NaN-fanactive_replace_with_0", False) and "sensebox_fan_active" in df.columns:
        df["sensebox_fan_active"] = df["sensebox_fan_active"].fillna(0)

    # Fill missing DWD temperature values.
    dwd_nan_handling = handling.get("dwd_NaN_replacement", "drop")
    if dwd_nan_handling == "neighbors" and "temperature_value_neighbor_1" in df.columns and "temperature_value_neighbor_2" in df.columns:
        print("   Filling DWD NaN values with neighbor measurements ...")
        df["temperature_value_DWD001"] = df["temperature_value_DWD001"].fillna(df['temperature_value_neighbor_1'])
        df["temperature_value_DWD001"] = df["temperature_value_DWD001"].fillna(df['temperature_value_neighbor_2'])
        df["temperature_value_DWD001"] = df["temperature_value_DWD001"].ffill()
    elif isinstance(dwd_nan_handling, (int, float)):
        dwd_nan_handling = float(dwd_nan_handling)
        print(f"   Filling DWD NaN values with specific value: {dwd_nan_handling} ...")
        df["temperature_value_DWD001"] = df["temperature_value_DWD001"].fillna(dwd_nan_handling)
    else:
        print("   Dropping rows with NaN DWD values ...")

    # Fill missing values using neighboring measurements.
    neighbor_handling = handling.get("neighbors", "drop")
    if neighbor_handling == "each_other_else_dwd" and "temperature_value_neighbor_1" in df.columns and "temperature_value_neighbor_2" in df.columns:
        print("   Filling neighbor NaN values with each other and DWD as fallback ...")
        neighbor_cols = [col for col in df.columns if "neighbor" in col]
        replace_with_number = {1:2, 2:1}
        for col in neighbor_cols:
            feature = col.split("_")[0]
            neighbor_number = col.split("_")[-1]
            replace_with_col = f"{feature}_value_neighbor_{replace_with_number[int(neighbor_number)]}"
            df[col] = df[col].fillna(df[replace_with_col])

        # Finally, use DWD data as a fallback for temperature measurements.
        if "temperature_value_DWD001" in df.columns:
            df["temperature_value_neighbor_1"] = df["temperature_value_neighbor_1"].fillna(df['temperature_value_DWD001'])
            df["temperature_value_neighbor_2"] = df["temperature_value_neighbor_2"].fillna(df['temperature_value_DWD001'])
    else:
        print("   Dropping rows with NaN neighbor values ...")
    
    if handling.get("drop_rest", False):
        exclude_list = handling.get("exclude_from_drop", [])
        to_drop = [col for col in df.columns if col not in exclude_list]
        df = df.dropna(subset=to_drop)
    print(f"   Shape: {df.shape}")
    return df