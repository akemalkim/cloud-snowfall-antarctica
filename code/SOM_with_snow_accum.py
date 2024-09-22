# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:37:58 2024

@author: akmal
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Read CSV file with BMU index
df = pd.read_csv('bmu_classification_2022_ver2.csv')

# Ensure the 'date' column is in datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Load snow accumulation data
snow_data = xr.open_dataset('./snow_accum_test/snow_accum_2022_ver3.nc')

# Extract dates from the NetCDF dataset and ensure they are in the same format
snow_dates = pd.to_datetime(snow_data['date'].values)

# Find the common dates between the two datasets
common_dates = np.intersect1d(df['date'], snow_dates)

# Convert common_dates to strings in 'YYYY-MM-DD' format
common_dates_str = common_dates.astype('datetime64[D]').astype(str)  # Ensure the format is YYYY-MM-DD

# Filter the CSV dataset to keep only rows with matching dates
filtered_df = df[df['date'].isin(common_dates)]

# Filter the NetCDF dataset to keep only the data for the matching dates
filtered_snow_data = snow_data.sel(date=common_dates_str)

# Taking out the data
snow_data_array = filtered_snow_data['snow_accumulation'].values
bmu_index = filtered_df['bmu'].values

# Merging into a single array
merged_data = np.column_stack((snow_data_array, bmu_index))

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()  # Flatten the grid for easier indexing



# Create separate plots for each BMU node
for bmu_value in range(9):
    ax = axs[bmu_value]

    # Filter data for the current BMU
    node_data = merged_data[merged_data[:, -1] == bmu_value]

    # Calculate cumulative snow accumulation for each type
    aggregates_SA = np.cumsum(node_data[:, 0])  # Cumulative sum of the first column (snow accumulation)
    three_bullet_SA = np.cumsum(node_data[:, 1])
    low_density_sphere_SA = np.cumsum(node_data[:, 2])
    aggregates_spheroid_SA = np.cumsum(node_data[:, 3])
    durville_SA = np.cumsum(node_data[:, 4])

    # Get corresponding dates for x-axis
    dates = filtered_df['date'][filtered_df['bmu'] == bmu_value].values

    # Plot the cumulative snow accumulation on the subplot
    ax.plot(dates, aggregates_SA, label='Aggregates')
    ax.plot(dates, three_bullet_SA, label='Three bullet')
    ax.plot(dates, low_density_sphere_SA, label='Low density sphere')
    ax.plot(dates, aggregates_spheroid_SA, label='Aggregate spheroids')
    ax.plot(dates, durville_SA, label="D'urville value")

    # Set x-ticks to be monthly
    ax.set_xticks(pd.date_range(start=min(dates), end=max(dates), freq='MS'))
    ax.tick_params(axis='x', rotation=45)

    # Customize the subplot
    ax.set_title(f'BMU Node {bmu_value}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Snow Accumulation (mm)')
    ax.grid(True)

# Adjust the layout of the entire figure
plt.tight_layout()

# Add a global legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

# Show the full canvas with all subplots
plt.savefig('snow_accum_classified_with_SOM.png', bbox_inches='tight')
plt.show()


https://openstartechnologies.bamboohr.com/careers/44

