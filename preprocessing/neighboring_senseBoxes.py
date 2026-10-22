"""Select, validate, and aggregate geographic and climatological sensor neighbors."""

import datetime
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely import wkb

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def determine_neighboring_senseBoxes(df, number, measured_values_neighbors, plausibility_checks, reduction_method="none", consistency_threshold_neighbors=None, use_climatological_neighbors=False):
    """
    Add valid neighboring senseBox measurements to each row.

    Neighbors are selected by geographic proximity or climatological similarity,
    checked for plausibility, and optionally reduced or filtered.
    """
    
    print("Determining neighboring senseBoxes ...")

    if use_climatological_neighbors:
        feature_weights = {'NDVI': 2.0}
        climatological_neighbors = determine_climatological_neighbors(df, feature_weights=feature_weights)

    # Retrieve values for plausibility checks.
    boundaries = plausibility_checks.get("boundaries", {})
    specific_values = plausibility_checks.get("specific_values", {})

    if isinstance(number, list):
        num_neighbors_considered, num_neighbors_output = number
    else:
        num_neighbors_considered, num_neighbors_output = number, number

    if num_neighbors_considered == 0:
        print("   Number of neighbors is 0, skipping neighbor processing")
        return df
    
    df["geometry"] = df["location"].apply(lambda x: wkb.loads(x, hex=True) if isinstance(x, str) else x)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="25832")

    senseBoxes = gdf[gdf["quell_typ"] == "senseBox"].copy()
    senseBoxes = senseBoxes[["quell_id", "geometry", "from_date", "until_date"]].drop_duplicates()

    def find_geographic_neighbors(row):
        """Find active senseBoxes nearest to the row's location and timestamp."""
        valid_boxes = senseBoxes[
            (senseBoxes["from_date"] <= row["messzeit_utc"]) & 
            (senseBoxes["until_date"] >= row["messzeit_utc"])
        ]

        neighbors = (valid_boxes.distance(row["geometry"]).sort_values().head(num_neighbors_considered + 3)).reset_index()

        # Check whether any neighbor row has a distance of 0.0 (the point itself).
        if 0.0 in neighbors[0].values:
            neighbors = neighbors[neighbors[0] != 0.0]
        neighbors = neighbors.head(num_neighbors_considered)['index'].tolist()
        return [senseBoxes.loc[idx, "quell_id"] for idx in neighbors[:num_neighbors_considered] if idx in senseBoxes.index]

    def find_climatological_neighbors(row):
        """Return active climatological neighbors for this senseBox measurement."""
        quell_id = row["quell_id"]
        candidates = climatological_neighbors[climatological_neighbors["quell_id"] == quell_id]

        if row['quell_typ'] != 'senseBox' or candidates.empty:
            return []
        
        valid_boxes = senseBoxes[
            (senseBoxes["from_date"] <= row["messzeit_utc"]) & 
            (senseBoxes["until_date"] >= row["messzeit_utc"])
        ]

        # Find potential neighbors.
        neighbors = []
        for cand_row in candidates.iterrows():
            if cand_row[1]["from_date"] == row["from_date"] and cand_row[1]["until_date"] == row["until_date"]:
                neighbors = cand_row[1]["neighbor_ids"]

        # Keep only neighbors included in valid_boxes.
        valid_neighbors = valid_boxes["quell_id"].tolist()
        nearest_neighbors = [n for n in neighbors if n in valid_neighbors]
        nearest_neighbors = nearest_neighbors[:num_neighbors_considered]
        return nearest_neighbors

    # Find neighboring sensors.
    if use_climatological_neighbors:
        gdf["neighbors"] = gdf.apply(find_climatological_neighbors, axis=1)
    else:
        gdf["neighbors"] = gdf.apply(find_geographic_neighbors, axis=1)

    # Create individual columns for each neighbor.
    for i in range(num_neighbors_considered):
        gdf[f"neighbor_SB_id_{i+1}"] = gdf["neighbors"].apply(lambda lst: lst[i] if i < len(lst) else None)

    # Save the neighbors to a CSV file.
    # df_neighbors = pd.DataFrame((gdf[['quell_id', 'neighbor_SB_id_1', 'neighbor_SB_id_2']]).astype(str).groupby(['quell_id', 'neighbor_SB_id_1', 'neighbor_SB_id_2']).nunique().index.tolist())
    # df_neighbors.to_csv("df_neighbors.csv")

    # Drop the neighbors column.
    gdf = gdf.drop(columns=["neighbors"])

    # Copy the DataFrame to add neighbor temperatures.
    copy4merge = gdf[["quell_id", "messzeit_utc", "measurement_data"]].copy()

    # Add neighbor measurements to the main DataFrame.
    for i in range(1, num_neighbors_considered + 1):
        gdf = pd.merge_asof(
            gdf.sort_values("messzeit_utc"),
            copy4merge.sort_values("messzeit_utc"),
            on="messzeit_utc",
            direction="nearest",
            tolerance=pd.Timedelta("20m"),
            left_by=f"neighbor_SB_id_{i}",
            right_by="quell_id",
            suffixes=("", f"_neighbor_{i}")
        )
    #print("Neighboring-SenseBoxes merged")
    #print(gdf.head())

    # Extract measured values for neighboring sensors.
    for i in range(1, num_neighbors_considered + 1):
        for key, subkeys in measured_values_neighbors.items():
            for subkey in subkeys:
                column_name = f"{key}_{subkey}_neighbor_{i}"
                gdf[column_name] = gdf.apply(
                    lambda row: row[f"measurement_data_neighbor_{i}"].get(key, {}).get(subkey)
                    if isinstance(row[f"measurement_data_neighbor_{i}"], dict) else None,
                    axis=1
                )
    
    # Check the plausibility of neighbor measurements.
    if boundaries or specific_values:
        print("   Performing plausibility checks for neighboring senseBoxes ...")
        for i in range(1, num_neighbors_considered + 1):
            col = f"temperature_value_neighbor_{i}"

            # Check for values outside physical limits.
            for feature, limits in boundaries.items():
                if col.startswith(feature) and col in gdf.columns:
                    min_val, max_val = limits
                    gdf[col] = gdf[col].where((gdf[col] >= min_val) & (gdf[col] <= max_val), np.nan)

            # Check for specific values.
            for feature, values in specific_values.items():
                if col.startswith(feature) and col in gdf.columns:
                    gdf[col] = gdf[col].where(~gdf[col].isin(values), np.nan)

    if reduction_method == "keep_n_closest_to_median" and num_neighbors_output > 1:
        # Keep the neighbors closest to the median temperature.
        print(f"   Keeping {num_neighbors_output} closest neighbors to median temperature (of {num_neighbors_considered} total candidates) ...")

        # Create a DataFrame containing only neighbor temperature columns.
        neighbor_cols = [f"temperature_value_neighbor_{i}" for i in range(1, num_neighbors_considered + 1)]
        neighbor_data = gdf[neighbor_cols].copy()
        
        # Calculate the median temperature for each row.
        gdf["median_temperature_neighbors"] = neighbor_data.median(axis=1)

        # Drop missing values only for senseBox sources; other source types may have no neighbor values.
        mask_sensebox = gdf["quell_typ"] == "senseBox"
        gdf = gdf[~(mask_sensebox & gdf["median_temperature_neighbors"].isna())]
        
        # Find the neighbors closest to each row's median temperature.
        def find_closest_neighbors(row, num_neighbors_output):
            """Return neighbor labels whose temperatures are closest to the row median."""
            neighbor_temps = [row[f"temperature_value_neighbor_{i}"] for i in range(1, num_neighbors_considered + 1)]
            median_temp = row["median_temperature_neighbors"]
            # Calculate absolute differences from the median temperature.
            diffs = [abs(temp - median_temp) if not pd.isna(temp) else float('inf') for temp in neighbor_temps]
            # Get the indices of the neighbors with the smallest differences.
            sorted_indices = sorted(range(len(diffs)), key=lambda i: diffs[i])
            # Return the IDs of the closest non-missing neighbors.
            closest_ids = []
            for idx in sorted_indices:
                if not pd.isna(neighbor_temps[idx]) and len(closest_ids) < num_neighbors_output:
                    closest_ids.append(f"neighbor_{idx+1}")
            return closest_ids

        gdf["closest_neighbors"] = gdf.apply(lambda row: find_closest_neighbors(row, num_neighbors_output), axis=1)
        
        # Keep only the requested number of closest neighbors.
        for i in range(1, num_neighbors_output + 1):
            for key, subkeys in measured_values_neighbors.items():
                for subkey in subkeys:
                    gdf[f"keep_{key}_{subkey}_neighbor_{i}"] = gdf.apply(
                        lambda row: row[f"{key}_{subkey}_{row['closest_neighbors'][i-1]}"] 
                        if len(row['closest_neighbors']) >= i else np.nan,
                        axis=1
                    )

        # Mark or drop helper columns; some may still be needed for later plausibility checks.
        for i in range(1, num_neighbors_considered + 1):
            for key, subkeys in measured_values_neighbors.items():
                for subkey in subkeys:
                    gdf = gdf.rename(columns={f"{key}_{subkey}_neighbor_{i}": f"{key}_{subkey}_neighbor_{i}__OLD"})
        gdf = gdf.drop(columns=["median_temperature_neighbors", "closest_neighbors"])

        # Rename the retained neighbor columns.
        keep_cols = [col for col in gdf.columns if col.startswith("keep_")]
        for col in keep_cols:
            new_col_name = col.replace("keep_", "")
            gdf = gdf.rename(columns={col: new_col_name})        
        
    elif reduction_method == "agg_mean" and num_neighbors_output == 1:
        # Aggregate all neighbors into a single mean value.
        print(f"   Aggregating {num_neighbors_considered} neighbors to 1 mean value ...")
        for key, subkeys in measured_values_neighbors.items():
            for subkey in subkeys:
                mean_column = f"{key}_{subkey}_mean_neighbors"
                neighbor_columns = [f"{key}_{subkey}_neighbor_{i}" for i in range(1, num_neighbors_considered + 1)]
                
                # Select neighbor columns and exclude zeros, values below -20, and missing values.
                neighbor_data = gdf[neighbor_columns].copy()
                # Replace 0.000 with NaN for temperature values only.
                if key == "temperature":
                    neighbor_data = neighbor_data.replace(0.000, np.nan)
                neighbor_data = neighbor_data.where(neighbor_data >= -20, np.nan) 
                
                # Calculate the mean, ignoring NaN values (skipna=True is the default).
                gdf[mean_column] = neighbor_data.mean(axis=1)
                
                gdf = gdf.drop(columns=neighbor_columns)   
    
    else:
        print(f"   Keeping all {num_neighbors_considered} neighbors without reduction ...")

    # Optionally filter neighbors by the maximum allowed temperature difference.
    if consistency_threshold_neighbors is not None and num_neighbors_output == 2:
        mask_inconsistency = abs(gdf["temperature_value_neighbor_1"] - gdf["temperature_value_neighbor_2"]) >= consistency_threshold_neighbors
        mask_inconsistency = mask_inconsistency & gdf["quell_typ"] == "senseBox"
        print(f"   Removing rows where temperature difference between final two neighbors exceeds consistency threshold of {consistency_threshold_neighbors}°C: {(mask_inconsistency).sum()} rows removed")
        gdf = gdf[~mask_inconsistency]

    # Drop unnecessary columns.
    gdf = gdf.drop(columns=[f"neighbor_SB_id_{i}" for i in range(1, num_neighbors_considered + 1)])
    gdf = gdf.drop(columns=[f"quell_id_neighbor_{i}" for i in range(1, num_neighbors_considered + 1)])
    gdf = gdf.drop(columns=[f"measurement_data_neighbor_{i}" for i in range(1, num_neighbors_considered + 1)])
    gdf = gdf.drop(columns=["geometry"])
    print(f"   Shape: {gdf.shape}")
    return gdf

def determine_climatological_neighbors(df, feature_weights, k_final=10, metric="euclidean"):
    """
    Find climatologically similar senseBoxes and their overlapping active periods.

    Returns a DataFrame containing each senseBox and its filtered neighbor IDs.
    """
    # Extract unique senseBox sources from the measurements DataFrame.
    sensors = df[df["quell_typ"] == "senseBox"][["quell_id", "from_date", "until_date", "raster_properties"]].drop_duplicates(subset=["quell_id", "from_date", "until_date"])

    # Drop rows with missing raster properties.
    sensors = sensors.dropna(subset=["raster_properties"]) 

    # Expand the raster_properties column.
    properties_df = sensors['raster_properties'].apply(pd.Series)
    sensors = pd.concat([sensors.drop(columns=['raster_properties']), properties_df], axis=1)

    # Convert the terrain type to a binary feature: {1, 2} -> {0, 1}.
    sensors['GEL_FORM'] = sensors['GEL_FORM'].apply(lambda x: 1 if x == 2 else 0)

    # Drop selected properties.
    sensors = sensors.drop(columns=['LT_4', 'LT_14', 'PET_4', 'PET_14', 'LSTDS_TM', 'LSTNS_TM', 'KLIMATOP', 'FRISCHL', 'LAGE', 'DISTANZ', 'VEGET_D', 'LSTDS_MA', 'LSTDS_MI', 'LSTNS_MA', 'LSTNS_MI', 'LSTW1_TM', 'LSTW2_TM', 'P_D_WG_F', 'GRU_VOL_S', 'WI_G_8100', 'COOL_O', 'FT_71_00', 'ST_71_00'])

    # Assign feature weights.
    for col in sensors.columns:
        if col not in feature_weights and col not in ['quell_id', 'from_date', 'until_date']:
            feature_weights[col] = 1.0  # Use a weight of 1.0 when none is specified.
    # Rescale the features.
    scaler = MinMaxScaler()
    scale_cols = [col for col in sensors.columns if col not in ['quell_id', 'raster_id', 'center', 'from_date', 'until_date', 'location_id']]   
    sensors_scaled = sensors.copy()
    sensors_scaled[scale_cols] = scaler.fit_transform(sensors_scaled[scale_cols])
    sensors_scaled = sensors_scaled.reset_index(drop=True)

    # Apply feature weights (square-rooted for Euclidean distance).
    weights = np.array([feature_weights[col] for col in scale_cols])
    weighted_features = sensors_scaled[scale_cols].values * np.sqrt(weights)

    # Find the k nearest neighbors, filter by date overlap, and retain the nearest k_final.
    k = 2*k_final  

    # Fit NearestNeighbors with k+1 entries so the point itself can be removed.
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(weighted_features)
    distances, indices = nn.kneighbors(weighted_features)

    ids = sensors_scaled['quell_id'].values

    # Store one row per ID with neighbor IDs and distances sorted from nearest to farthest.
    neighbors_list = []
    for row_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        mask = idx_row != row_idx
        neighbor_ids = [ids[neighbor_idx] for neighbor_idx in idx_row[mask]][:k]
        neighbor_distances = [dist for dist in dist_row[mask]][:k]
        neighbors_list.append({'id': ids[row_idx], 'neighbors': neighbor_ids, 'distances': neighbor_distances})

    neighbors_df = pd.DataFrame(neighbors_list)

    final_neighbors = sensors_scaled[['quell_id', 'from_date', 'until_date']].copy()
    
    # For each senseBox ID, find its nearest candidates, filter by date overlap, and retain k_final neighbors.
    final_neighbors['neighbor_ids'] = None

    for idx, row in final_neighbors.iterrows():
        sensebox_id = row['quell_id']
        sensebox_from_date = row['from_date']
        sensebox_until_date = row['until_date']

        # Get the candidates for this senseBox.
        neighbors_row = neighbors_df[neighbors_df['id'] == sensebox_id]
        neighbor_ids = neighbors_row.iloc[0]['neighbors']
        neighbor_distances = neighbors_row.iloc[0]['distances']

        # Filter neighbors by date overlap.
        filtered_neighbors = []
        for neighbor_id, distance in zip(neighbor_ids, neighbor_distances):
            neighbor_info = sensors_scaled[sensors_scaled['quell_id'] == neighbor_id]
            neighbor_from_date = neighbor_info.iloc[0]['from_date']
            neighbor_until_date = neighbor_info.iloc[0]['until_date']

            # Check for date overlap.
            if (sensebox_from_date <= neighbor_until_date) and (neighbor_from_date <= sensebox_until_date):
                filtered_neighbors.append((neighbor_id, distance))

        # Sort by distance and retain the nearest k_final neighbors.
        filtered_neighbors.sort(key=lambda x: x[1])
        final_neighbors_list = [neighbor_id for neighbor_id, _ in filtered_neighbors[:k_final]]

        # Update final_neighbors with the nearest k_final neighbors.
        final_neighbors.at[idx, 'neighbor_ids'] = final_neighbors_list

    return final_neighbors