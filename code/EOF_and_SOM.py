# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:09:04 2024

@author: akmal
"""

import xarray as xr
import numpy as np
from eofs.standard import Eof
import matplotlib.pyplot as plt
from minisom import MiniSom
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# List of file paths for your 5 years of data
nc_files = ['ERA5_large_2019.nc', 'ERA5_large_2020.nc', 'ERA5_large_2021.nc', 'ERA5_large_2022.nc',  'ERA5_large_2023.nc',]

# Load and concatenate the datasets along the 'time' dimension
datasets = [xr.open_dataset(f) for f in nc_files]
combined_ds = xr.concat(datasets, dim='time')

# Extract relevant variables (e.g., u-wind and v-wind at 10m)
u_wind = combined_ds['u10']
v_wind = combined_ds['v10']

# Optional: Combine u and v wind into wind speed (if needed)
wind_speed = (u_wind**2 + v_wind**2)**0.5

# Calculate the mean wind speed over the time period
u_mean = u_wind.mean(dim='time')
v_mean = v_wind.mean(dim='time')

# Subtract the mean to get the anomaly
u_anomaly = u_wind - u_mean
v_anomaly = v_wind - v_mean

# Reshape the data into a matrix with dimensions (time, space)
time_len = u_anomaly.shape[0]
lat_len = u_anomaly.shape[1]
lon_len = u_anomaly.shape[2]

# Flatten spatial dimensions into one
u_data_matrix = u_anomaly.values.reshape(time_len, lat_len * lon_len)
v_data_matrix = v_anomaly.values.reshape(time_len, lat_len * lon_len)

# Perform EOF analysis on u and v components separately
solver_u = Eof(u_data_matrix)
solver_v = Eof(v_data_matrix)

# Compute the leading EOFs for u and v wind components
eof_u = solver_u.eofs(neofs=5)
eof_v = solver_v.eofs(neofs=5)

# Calculate the principal components (PCs) - using u wind for SOM training
pcs_u = solver_u.pcs(npcs=5)

# Set up the SOM grid dimensions (e.g., 6x6 grid)
som_x = 3
som_y = 3

# Initialize the SOM
som = MiniSom(som_x, som_y, pcs_u.shape[1], sigma=1.0, learning_rate=0.5)

# Train the SOM using the principal components (PCs) from EOF analysis
som.train_random(pcs_u, num_iteration=10000)

# Visualize the SOM nodes (synoptic wind patterns)
fig, axs = plt.subplots(som_x, som_y, figsize=(15, 15), 
                        subplot_kw={'projection': ccrs.PlateCarree()}, 
                        constrained_layout=True)

# Iterate over the SOM grid and plot each node (synoptic wind pattern)
for i in range(som_x):
    for j in range(som_y):
        # Get the weight vector for the current SOM node
        node_weights = som.get_weights()[i, j]
        
        # Reconstruct the wind patterns (u and v components) using the EOFs
        node_u = np.dot(node_weights, eof_u.reshape(5, lat_len * lon_len)).reshape(lat_len, lon_len)
        node_v = np.dot(node_weights, eof_v.reshape(5, lat_len * lon_len)).reshape(lat_len, lon_len)
        node_wind_speed = np.sqrt(node_u**2 + node_v**2)
        
        # Plot the wind speed as a filled contour
        cs = axs[i, j].contourf(combined_ds['longitude'], combined_ds['latitude'], 
                                node_wind_speed, cmap='Oranges', transform=ccrs.PlateCarree())
        
        # Add the wind vectors using quiver
        step = 5  # Adjust the step size for arrow density
        axs[i, j].quiver(combined_ds['longitude'][::step], combined_ds['latitude'][::step], 
                         node_u[::step, ::step], node_v[::step, ::step], 
                         transform=ccrs.PlateCarree(), color='black', 
                         scale=100)
        
        # Set up the map features (coastlines, borders, etc.)
        axs[i, j].coastlines()
        axs[i, j].add_feature(cfeature.BORDERS, linestyle=':')
        axs[i, j].set_title(f'Node ({i}, {j})', fontsize=10)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

# Add a colorbar for wind speed magnitude
fig.colorbar(cs, ax=axs, orientation='vertical', shrink=0.8, label='Wind Speed (m/s)')

# Set the main title for all subplots
plt.suptitle('SOM Nodes: Synoptic Wind Patterns with Wind Vectors', fontsize=16)

# Show the plot
plt.show()





