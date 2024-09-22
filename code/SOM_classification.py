#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:31:08 2024

@author: mar250
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial.distance import cdist
import pandas as pd

# Open SOM datasets
som_result = xr.open_dataset('som_wind_pattern_ver2.nc')

# Extract SOM wind patterns
som_u_wind = som_result['u_wind'].values  # Shape: (3, 3, 61, 61)
som_v_wind = som_result['v_wind'].values  # Shape: (3, 3, 61, 61)

# Average the SOM wind patterns over latitude and longitude to get a single vector per SOM node
som_u_avg = np.mean(som_u_wind, axis=(2, 3))  # Shape: (3, 3)
som_v_avg = np.mean(som_v_wind, axis=(2, 3))  # Shape: (3, 3)

# Stack u and v components to create a (9, 2) matrix of wind vectors (one for each SOM node)
som_wind_patterns_avg = np.stack([som_u_avg.flatten(), som_v_avg.flatten()], axis=-1)  # Shape: (9, 2)


# Look at ERA5 wind data from 2022
filepath = 'ERA5_wind_2021_2023.nc'
data = xr.open_dataset(filepath)

start_date = '2022-01-01T00:00'
end_date = '2022-12-31T23:59'
target_lat = -77.85
target_lon = 166.77

nearest_data = data.sel(latitude=target_lat, longitude=target_lon, method='nearest')
filtered_data = nearest_data.sel(time=slice(start_date, end_date))

# Resample the data to daily averages, using the 'D' frequency for days one day data
daily_data = filtered_data.resample(time='1D').mean()

wind_u10_era5 = daily_data['u10'].values
wind_v10_era5 = daily_data['v10'].values

# Stacking both u and v component in one matrix
wind_data_2022 = np.stack([wind_u10_era5, wind_v10_era5], axis=-1) # Shape: (365, 2)


# Initialize an array to store the SOM classification for each day
bmu_indices = np.zeros(wind_data_2022.shape[0], dtype=int)

# For each day, calculate the Euclidean distance to each SOM node's average wind vector and find the BMU
for i, wind_vector in enumerate(wind_data_2022):
    distances = cdist([wind_vector], som_wind_patterns_avg)  # Compare to all SOM nodes' avg wind vectors
    bmu_indices[i] = np.argmin(distances)  # Find the BMU (index of the minimum distance) BMU: Best Matching Unit
    
# Convert BMUs to a DataFrame and save to CSV
bmu_df = pd.DataFrame({
    'date': daily_data['time'].values,  # Storing the corresponding dates
    'bmu': bmu_indices
})

# Save to CSV file
bmu_df.to_csv('bmu_classification_2022_ver2.csv', index=False)


