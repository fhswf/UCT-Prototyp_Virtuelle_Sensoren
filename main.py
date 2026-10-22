import joblib
import json
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import yaml
import geopandas as gpd
from datetime import timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from build_model import UCTModel
from sqlalchemy import create_engine
from preprocessing.utils import load_dataset, expected_feature_order, features_pass
from value_transfer import transfer_values_from_neighbors, transfer_values_from_neighbors_long

load_dotenv()

DEBUG = False

with open("assets/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Create the database connection.
user, password, host, port = [os.getenv(var) for var in ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]]
dbname = config["data"]["db"]["dbname"]
schema = config["data"]["db"]["schema"]
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
print("Database connection established successfully.")

# Load data for the Stage 2 correction.
df = load_dataset(config, engine)

# Drop geographic features here; they will be joined back in later.
geo_features = config["preprocessing"]["raster_properties"]
df = df.drop(columns=geo_features)

print(f"Dataset loaded successfully with shape: {df.shape}")
print(f"Columns in dataset: {df.columns.tolist()}")
# Load raster IDs from the DataFrame.
raster_ids_in_df = np.array([], dtype=int)
if "raster_id" in df.columns:
    print("Raster ID column found in dataset.")
    raster_ids_in_df = (
        pd.to_numeric(df["raster_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

# Merge Stage 2 data.
cells_to_predict_cfg = config["stage2_specifications"].get("cells_to_predict", None)

if cells_to_predict_cfg is None:
    cfg_ids = []
elif isinstance(cells_to_predict_cfg, (list, tuple, set, np.ndarray, pd.Series)):
    cfg_ids = list(cells_to_predict_cfg)
else:
    cfg_ids = [cells_to_predict_cfg]

cells_series = pd.to_numeric(pd.Series(cfg_ids + raster_ids_in_df.tolist()), errors="coerce").dropna()
cells_to_predict = cells_series.astype(int).unique().tolist() if not cells_series.empty else None

if cells_to_predict_cfg is None:
    query = "SELECT id, area_validity, properties FROM location WHERE id >=100000 AND id < 200000;"
else:
    # cells_to_predict may be a single integer or an iterable of integers.
    if isinstance(cells_to_predict, (list, tuple, set)):
        ids = ",".join(str(int(x)) for x in cells_to_predict)
        query = f"SELECT id, area_validity, properties FROM location WHERE id IN ({ids});"
    else:
        # Handle a single value (integer or numeric string).
        query = f"SELECT id, area_validity, properties FROM location WHERE id = {int(cells_to_predict)};"
raster = gpd.read_postgis(query, engine, geom_col="area_validity")

if "properties" in raster.columns:
    raster = pd.concat([raster.drop(columns=["properties"]), raster["properties"].apply(pd.Series)], axis=1)
    keep_cols = ["id", "area_validity"] + [col for col in raster.columns if col in geo_features]
    raster = raster[keep_cols]
    raster = raster.dropna()

print(f"Raster data loaded successfully with shape: {raster.shape}")

start = pd.to_datetime(config["stage2_specifications"]["start"])
end = pd.to_datetime(config["stage2_specifications"]["end"])
minutes = (end - start).total_seconds() / 60.0


if minutes > 60:

    times = pd.date_range(start, end, freq="1h")

    # Ensure that messzeit_utc is stored as a datetime.
    if "messzeit_utc" in df.columns:
        df["messzeit_utc"] = pd.to_datetime(df["messzeit_utc"])

    print("Start: Erst Median der Messungen berechnen, dann Join mit Raster.")

    # Step 1: Assign each measurement to the nearest full hour.
    # Round messzeit_utc to the nearest hour (within +/- 30 minutes).
    df["messzeit_utc_target"] = df["messzeit_utc"].dt.round("1h")

    # Keep only timestamps within the requested period.
    valid_times = set(times)
    df_in_range = df[df["messzeit_utc_target"].isin(valid_times)].copy()

    # Step 2: Calculate the median for each (raster_id, hour, source_id).
    print("Berechne Median der Rohmessungen...")
    numeric_cols = df_in_range.select_dtypes(include="number").columns.tolist()
    group_keys = ["raster_id", "messzeit_utc_target", "quell_id"]
    # Use only grouping keys that are present in the DataFrame.
    group_keys = [k for k in group_keys if k in df_in_range.columns]
    # Do not aggregate the grouping keys again; that would conflict with reset_index().
    agg_cols = [c for c in numeric_cols if c not in group_keys]

    median_df = (
        df_in_range
        .groupby(group_keys)[agg_cols]
        .median()
        .reset_index()
    )
    median_df = median_df.rename(columns={"messzeit_utc_target": "messzeit_utc"})
    print(f"Median berechnet: {median_df.shape[0]} Zeilen (statt {df_in_range.shape[0]} Rohmessungen)")

    # Step 3: Cross-join the raster with the time series (hourly data keeps this small).
    print("Erstelle Raster-Zeitreihe und joine mit Medianen...")
    raster_times = raster[["id", "area_validity"]].copy()
    raster_times = raster_times.assign(_key=1).merge(
        pd.DataFrame({"messzeit_utc": times, "_key": 1}), on="_key"
    ).drop(columns=["_key"])
    raster_times = raster_times.rename(columns={"id": "raster_id"})
    raster_times["raster_id"] = raster_times["raster_id"].astype(float)

    # Step 4: Perform a simple left join; no time tolerance is needed.
    result = raster_times.merge(
        median_df,
        on=["raster_id", "messzeit_utc"],
        how="left",
    )

    print("Spalten nach merge und Medianbildung:", result.columns.tolist())
    print(result.head())
    print(result.shape)

    result.to_csv("raster_hourly_median.csv", index=False)

    # Fill missing values using neighboring measurements.
    df = transfer_values_from_neighbors_long(
        result,
        config["stage2_specifications"]["transfer_values_method"]
    )

    print("Median aggregation + neighbor transfer completed successfully.")

elif minutes > 10 and minutes <= 60:
    # Create a messzeit_utc entry for every 10-minute interval and raster cell.
    # Then match on raster_id and messzeit_utc using the nearest timestamp within 10 minutes.
    times = pd.date_range(start, end, freq="10min")

    # Ensure that messzeit_utc is stored as a datetime.
    if "messzeit_utc" in df.columns:
        df["messzeit_utc"] = pd.to_datetime(df["messzeit_utc"])

    # Cross-join each raster ID with every 10-minute timestamp.
    raster_times = raster[["id", "area_validity"] + geo_features].copy()
    raster_times = raster_times.assign(_key=1).merge(
        pd.DataFrame({"messzeit_utc": times, "_key": 1}), on="_key").drop(columns=["_key"])
    raster_times = raster_times.rename(columns={"id": "raster_id"})

    # Ensure merge keys have matching data types to avoid int/float merge errors.
    raster_times["raster_id"] = raster_times["raster_id"].astype(float)

    # Sort the data before calling merge_asof.
    raster_times = raster_times.sort_values(["raster_id", "messzeit_utc"]) 
    if ("raster_id" in df.columns) and ("messzeit_utc" in df.columns):
        # Convert raster_id to a numeric float to match the dtype in raster_times.
        df["raster_id"] = pd.to_numeric(df["raster_id"], errors="coerce").astype(float)
        df["messzeit_utc"] = pd.to_datetime(df["messzeit_utc"])

        # Find the nearest measurement within 10 minutes for each raster ID and target time.
        raster_times = raster_times.reset_index().rename(columns={"index": "_left_idx"})

        merged_all = pd.merge(
            raster_times,
            df,
            on="raster_id",
            how="left",
            suffixes=("_target", "_meas"),
        )

        # Calculate the absolute time difference.
        merged_all["time_diff"] = (merged_all["messzeit_utc_target"] - merged_all["messzeit_utc_meas"]).abs()

        tol = pd.Timedelta("10min")

        # Keep only valid matches.
        merged_within = merged_all[merged_all["time_diff"] <= tol].copy()

        if not merged_within.empty:
            idx = merged_within.groupby("_left_idx")["time_diff"].idxmin()
            best_matches = merged_within.loc[idx].copy()
        else:
            best_matches = pd.DataFrame(columns=merged_all.columns)

        # Ensure the downstream merge key exists even when there are no valid matches.
        if ("messzeit_utc" not in best_matches.columns) and ("messzeit_utc_target" in best_matches.columns):
            best_matches = best_matches.rename(columns={"messzeit_utc_target": "messzeit_utc"})

        # Join back to all raster_times, using the target messzeit_utc.
        merged_df = raster_times.merge(
            best_matches,
            on=["raster_id", "messzeit_utc"],
            how="left",
            suffixes=("", "_meas"),
        )

        print(merged_df.head())
        print(merged_df.shape)
        print("Spalten nach merge: ", merged_df.columns)

        #merged_df.to_csv("merged_df.csv", index=False)

        # Drop geo_features_meas columns to avoid duplicates.
        for feature in geo_features:
            if f"{feature}_meas" in merged_df.columns:
                merged_df = merged_df.drop(columns=[f"{feature}_meas"])

        df = transfer_values_from_neighbors_long(merged_df, geo_features=geo_features, method=config["stage2_specifications"]["transfer_values_method"])
        print("Value transfer from neighbors completed successfully.")
else:
    merged_df = pd.merge(raster, df, left_on="id", right_on="raster_id", how="left")
    merged_df = merged_df.rename(columns={"id_x": "raster_id", "id_y": "id"})

    print(merged_df.head())
    print(merged_df.shape)
    print("Spalten nach merge: ", merged_df.columns)

    merged_df.to_csv("merged_df.csv", index=False)

    df = transfer_values_from_neighbors(merged_df, config["stage2_specifications"]["transfer_values_method"])
    print("Value transfer from neighbors completed successfully.")

# Exit if the DataFrame is empty.
if df.empty:
    exit()

# Save the input data as a CSV file.
#df.to_csv("stage2_input.csv", index=False)

# Load the scaler and model.
scaler = joblib.load(config["stage2_specifications"]["file_paths"]["scaler"]) 
model = UCTModel(input_dim=len(expected_feature_order), hidden_units=config["model"]["hidden_units"])
model.build((None, len(expected_feature_order)))
if not DEBUG:
    model.load_weights(config["stage2_specifications"]["file_paths"]["model"])
print("Model and scaler loaded successfully.")

# Scale the features and prepare the model input.
X = df.copy()
if "id" in X.columns:
    X = X.drop(columns=["id"])
X.drop(columns=["quell_id", "messzeit_utc", "stage0", "stage1", "stage2"], inplace=True)

scale_cols = [col for col in expected_feature_order if col not in features_pass]
X_scaled = pd.DataFrame(scaler.transform(X[scale_cols].values), columns=scale_cols, index=X.index)
X_pass = X[features_pass]
# Restore the exact feature order used during model training.
X_input = pd.concat([X_scaled, X_pass], axis=1)[expected_feature_order].values 

# Generate predictions.
y_pred = model.predict(X_input, batch_size=config["stage2_specifications"]["batch_size"])
assert len(y_pred) == len(df), "Number of predictions does not match number of rows in the dataset for Stage2 correction"

# Add predictions to the DataFrame.
df["stage2"] = y_pred

if minutes > 10:
    df = pd.DataFrame(df).groupby("raster_id").agg(
        area_validity=("area_validity", "first"),
        temperature_value=("temperature_value", "mean"),
        stage0=("stage0", "mean"),
        stage2_mean=("stage2", "mean"),
        stage2_min=("stage2", "min"),
        stage2_max=("stage2", "max"),
    ).reset_index()
    print(df.head())

# Build the filename from the selected period and transfer method.
method = config["stage2_specifications"].get("transfer_values_method", "unknown")

# convert start and end to timezone mez
start = start.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Berlin"))
end = end.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Europe/Berlin"))

start_str = start.strftime('%Y%m%dT%H%M')
end_str = end.strftime('%Y%m%dT%H%M')

def _sanitize(s):
    """Replace filename-unsafe characters with underscores."""
    return ''.join(c if (c.isalnum() or c in ('-','_')) else '_' for c in str(s))
method_safe = _sanitize(method)
period = f"{start_str}-{end_str}"
base_name = f"stage2_predictions_{period}_{method_safe}"


# get output path from config, default to current directory if not specified
output_path = config["stage2_specifications"].get("output_path", ".")
if not output_path.endswith("/"):
    output_path += "/"

#create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

#new dictionary per base_name to store
path = os.path.join(output_path, base_name)
os.makedirs(path, exist_ok=True)

# Save the results as a Shapefile.
gdf = gpd.GeoDataFrame(df, geometry="area_validity", crs="EPSG:25832")
gdf.to_file(os.path.join(path, base_name + ".shp"))


# Save the results as a CSV file in output_path
df.to_csv(os.path.join(path, base_name + ".csv"), index=False)