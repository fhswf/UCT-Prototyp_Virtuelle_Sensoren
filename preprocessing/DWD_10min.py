"""Attach DWD station temperature measurements to sensor and raster data."""

import datetime
import pandas as pd

def attach_dwd_data_10min(df, dwd_ids, measured_values):
    """Merge nearby DWD observations and extract measurements."""
    print(f"Merging DWD 10min data ...")
    # Prepare the DWD data.
    dwd_data = df[df["quell_id"].isin(dwd_ids)].copy()
    dwd_data = dwd_data[["quell_id", "messzeit_utc", "measurement_data", "corrected_values"]]

    # Merge data separately for each DWD station.
    for dwd_id in dwd_ids:
        df = pd.merge_asof(
            df.sort_values("messzeit_utc"),
            dwd_data[dwd_data["quell_id"] == dwd_id].sort_values("messzeit_utc"),
            on="messzeit_utc",
            direction="nearest",
            tolerance=pd.Timedelta("20m"),
            suffixes=("", f"_{dwd_id}")
        )

        if 'temperature' in measured_values and not 'main' in measured_values['temperature']:
            # Get measured temperature data from measurement_data.
            measurement_col = f"measurement_data_{dwd_id}"

            # Add air_temperature values to measured_values.
            measured_values["air_temperature"] = ["value"]

            for key, subkeys in measured_values.items():
                for subkey in subkeys:
                    column_name = f"{key}_{subkey}_{dwd_id}"

                    df[column_name] = df.apply(
                        lambda row: (
                            row.get(measurement_col, {}).get(key, {}).get(subkey)
                            if isinstance(row.get(measurement_col), dict) else None
                        ),
                        axis=1
                    )
            
            # Merge temperature and air_temperature into temperature_value.
            if 'temperature' in measured_values and 'air_temperature' in measured_values:
                df[f"temperature_value_{dwd_id}"] = df.apply(
                    lambda row: row[f"temperature_value_{dwd_id}"] if pd.notnull(row[f"temperature_value_{dwd_id}"]) else row[f"air_temperature_value_{dwd_id}"],
                    axis=1
                )

                df = df.drop(columns=[f"air_temperature_value_{dwd_id}"])
        else:
            # Get smoothed temperature data from corrected_values.
            measurement_col = f"corrected_values_{dwd_id}"

            column_name = f"temperature_value_{dwd_id}"

            df[column_name] = df.apply(
                lambda row: (
                    row.get(measurement_col, {}).get("temperature", {}).get("main", {}).get("value")
                    if isinstance(row.get(measurement_col), dict) else None
                ),
                axis=1
            )

    df = df.drop(columns=[f"measurement_data_{dwd_id}" for dwd_id in dwd_ids])
    df = df.drop(columns=[f"corrected_values_{dwd_id}" for dwd_id in dwd_ids])
    df = df.drop(columns=[f"quell_id_{dwd_id}" for dwd_id in dwd_ids])

    print(f"   Shape: {df.shape}")
    return df