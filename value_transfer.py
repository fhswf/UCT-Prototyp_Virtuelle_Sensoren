"""Fill missing raster measurements using spatial or climatological neighbors."""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from shapely import wkb, wkt
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from scipy.spatial import cKDTree


def transfer_values_from_neighbors_long(df, geo_features, method="zeros", climatological_neighbors=True):
    """Fill missing values by finding valid neighbors separately for each timestamp."""
    gdf = df.copy()
    gdf = gpd.GeoDataFrame(gdf, geometry=gdf["area_validity"], crs="EPSG:25832")

    if isinstance(gdf, gpd.GeoDataFrame):
        geom_col = gdf.geometry.name
    elif "area_validity" in gdf.columns:
        geom_col = "area_validity"
    
    elif "geometry" in gdf.columns:
        geom_col = "geometry"
    else:
        raise ValueError("No geometry column found. Expected 'area_validity' or 'geometry'.")
    gdf.geometry = gdf[geom_col]
    print("Geometry column set to:", geom_col)
    gdf.head()

    cols = ["temperature_value_DWD001", "hour_sin", "hour_cos", "doy_cos"]
    value_cols = ["temperature_value", "humidity_value", "pressure_value", "lux_value", "uv_value"]
    results = []

    # Print progress information to show how long the operation takes.
    anzahl_groups = gdf["messzeit_utc"].nunique()
    print(f"Anzahl Gruppen (messzeit_utc): {anzahl_groups}")
    counter = 0
    for ts, group in gdf.groupby("messzeit_utc"):
        counter += 1
        print(f" Gruppe {counter} von {anzahl_groups} - messzeit_utc: {ts}")
        group = group.copy()

        means = group[cols].mean()              
        group[cols] = group[cols].fillna(means)
        print("Mittelwerte für Spalten mit fehlenden Werten berechnet und angewendet")

        temp = group["temperature_value"]
        valid_temp_mask = temp.notna() & (~np.isclose(temp, 0.0)) & (temp > -20) & (temp < 50)

        candidates = group[valid_temp_mask].copy().reset_index()
        missing = group[~valid_temp_mask].copy()

        if len(candidates) < 2:
            results.append(group)
            continue

        if climatological_neighbors:
            # For each missing raster ID, find the two candidates most similar in climatological features.
            feature_weights = {col: 1.0 for col in geo_features}
            feature_weights['NDVI'] = 2.0
            scaler = MinMaxScaler()
            scaler.fit(group[geo_features])

            dists, idxs = determine_climatological_neighbors(scaler, missing, candidates, geo_features, feature_weights=feature_weights)
        else:
            # Find geographic neighbors.
            cand_coords = np.array([(g.centroid.x, g.centroid.y) for g in candidates.geometry])
            missing_coords = np.array([(g.centroid.x, g.centroid.y) for g in missing.geometry])

            tree = cKDTree(cand_coords)
            dists, idxs = tree.query(missing_coords, k=2)

        for col in value_cols:
            group.loc[missing.index, f"{col}_neighbor_1"] = candidates[col].iloc[idxs[:, 0]].values
            group.loc[missing.index, f"{col}_neighbor_2"] = candidates[col].iloc[idxs[:, 1]].values

        results.append(group)

    gdf = pd.concat(results, ignore_index=True)

    if "_left_idx_meas" in gdf.columns:
        gdf = gdf.drop(columns=["_left_idx_meas"])
    if "area_validity_meas" in gdf.columns:
        gdf = gdf.drop(columns=["area_validity_meas"])

    #gdf = gdf.drop(columns=["temperature_value_DWD001", "temperature_value_DWD001_neighbor_2", "hour_sin", "hour_sin_neighbor_2", "hour_cos", "hour_cos_neighbor_2", "doy_cos", "doy_cos_neighbor_2"])
    #gdf = gdf.rename(columns={"temperature_value_DWD001_neighbor_1": "temperature_value_DWD001", "hour_sin_neighbor_1": "hour_sin", "hour_cos_neighbor_1": "hour_cos", "doy_cos_neighbor_1": "doy_cos"})

    if method == "zeros":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(0)
    elif method == "average_all":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(gdf[col].mean())
    elif method == "average_neighbors":
        for col in value_cols:
            gdf[col] = gdf[[f"{col}_neighbor_1", f"{col}_neighbor_2"]].mean(axis=1)
    elif method == "median_all":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(gdf[col].median())
    else:
        raise ValueError(f"Unknown method: {method}")

    return gdf

def transfer_values_from_neighbors(df, method="zeros"):
    """Fill missing measurements using the two nearest valid raster cells."""
    gdf = df.copy()
    
    # Set these features to the mean of their populated columns.
    cols = ["temperature_value_DWD001", "hour_sin", "hour_cos", "doy_cos"]
    means = gdf[cols].mean()  # Calculate each column's mean, excluding missing values.
    gdf.loc[:, cols] = gdf[cols].fillna(means)
    print("Mittelwerte für Spalten mit fehlenden Werten berechnet und angewendet")

    value_cols = ["temperature_value", "humidity_value", "pressure_value", "lux_value", "uv_value"]
    geom_col = gdf.geometry.name

    # Candidates must have a plausible temperature_value: nonzero and strictly between -20 and 50.
    temp = gdf["temperature_value"]
    valid_temp_mask = temp.notna() & (~np.isclose(temp, 0.0)) & (temp > -20) & (temp < 50)
    candidates = gdf[valid_temp_mask].copy().reset_index(drop=True)
    missing = gdf[~valid_temp_mask].copy()

    # Extract coordinates and build the KDTree.
    cand_coords    = np.array([(g.centroid.x, g.centroid.y) for g in candidates.geometry])
    missing_coords = np.array([(g.centroid.x, g.centroid.y) for g in missing.geometry])

    tree = cKDTree(cand_coords)
    dists, idxs = tree.query(missing_coords, k=2)

    # Write all value columns for both neighbors.
    for col in value_cols:
        gdf.loc[missing.index, f"{col}_neighbor_1"] = candidates[col].iloc[idxs[:, 0]].values
        gdf.loc[missing.index, f"{col}_neighbor_2"] = candidates[col].iloc[idxs[:, 1]].values

    # Print a summary of the neighbor data.
    neighbor_cols = [f"{col}_neighbor_{n}" for n in [1, 2] for col in value_cols]
    print("Fertig:", gdf[neighbor_cols].notna().sum())
    print("Spalten:", gdf.columns.tolist())

    gdf = gdf.drop(columns=["raster_id"])
    # Save the output as a CSV file.
    gdf.to_csv("gdf_with_neighbors.csv", index=False)
    gdf = gpd.GeoDataFrame(gdf, geometry="area_validity", crs="EPSG:25832")
    gdf.to_file("gdf_with_neighbors.shp")


    if method == "zeros":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(0)
    elif method == "average_all":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(gdf[col].mean())
    elif method == "average_neighbors":
        for col in value_cols:
            gdf[col] = gdf[[f"{col}_neighbor_1", f"{col}_neighbor_2"]].mean(axis=1)
    elif method == "median_all":
        for col in value_cols:
            gdf[col] = gdf[col].fillna(gdf[col].median())
    else:
        raise ValueError(f"Unknown method: {method}")

    return gdf

def determine_climatological_neighbors(scaler, missing, candidates, geo_features, feature_weights=None, k=2, metric='euclidean'):
    """Find nearest candidates using scaled and weighted geographic features."""
    # Find climatologically similar neighbors, mapping raster cells to senseBox raster cells.
    # For each missing raster ID, select the two candidates with the most similar climate features.

    # Rescale the features.
    missing_scaled = missing[geo_features].copy()
    missing_scaled = scaler.transform(missing_scaled)

    candidates_scaled = candidates[geo_features].copy()
    candidates_scaled = scaler.transform(candidates_scaled)

    # Apply feature weights (square-rooted for Euclidean distance).
    if feature_weights is None:
        feature_weights = {col: 1.0 for col in geo_features}

    weights = np.array([feature_weights[col] for col in geo_features])
    candidates_scaled = candidates_scaled * np.sqrt(weights)
    missing_scaled = missing_scaled * np.sqrt(weights)

    # Find the nearest neighbors.
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(candidates_scaled)
    dists, idxs = nn.kneighbors(missing_scaled)
    return dists, idxs