#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:17:02 2024

Run the nc data file amd plot a kde 

@author: mar250
"""

import time
import xarray as xr
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def main():

    start_time = time.time()
    
    filename = '2022_radar_data.nc'
    
    data = xr.open_dataset(filename)
    
    # Convert byte strings to strings, then to pandas datetime objects
    time_data = data['time'].values
    time_str = [t.decode('utf-8') for t in time_data]  # Convert byte strings to regular strings
    time_dates = pd.to_datetime(time_str)  # Convert strings to datetime
    
    # Update the dataset with the new time format
    data['time'] = ('time', time_dates)
    
    
    # Filtered data for a month selection
    start_date = '2022-04-01T00:00'
    end_date = '2022-04-30T00:00'
    
    filtered_data = data.sel(time=slice(start_date, end_date))
    
    "Prepping Ze data"
    # Ze_matrix = filtered_data['Ze'].values
    # Ze_matrix[Ze_matrix<0.0]=np.nan
    
    # # Quality control (omitting data that got less than 6 data point)
    # altitude = filtered_data['altitude'].values
    # new_Ze = Ze_matrix[:, 0:26]  # Chop down all the data about 3km
    # new_altitude = altitude[0:26]
    
    
    # num = 0
    # bad_row = []
    # for i, row in enumerate(new_Ze):
    #     nan_count =  np.isnan(row).sum()
        
    #     if nan_count > 13:
    #         bad_row.append(i)
    
    # for index in bad_row:
    #     new_Ze[index, :] = np.nan
        
    

    # Ze_ave, time_ave = re_averaging_data(new_Ze, 10, 1, filtered_data['time'].values)
    
    # "Try and error region"
    # # for Ze scatter
    # Ze_ave_over_altitude = np.nanmean(new_Ze, axis=0)
    
    # # Flatten and match the get the altitude data to match Ze
    # Ze_flat = Ze_ave.flatten()
    # altitude_flat = np.repeat(new_altitude, Ze_ave.shape[0])
    
    # nan_indices = np.where(np.isnan(Ze_flat))[0]
    
    # # Remove elements at the NaN indices
    # filtered_list_Ze = [item for i, item in enumerate(Ze_flat) if i not in nan_indices]
    # filtered_list_alt = [item for i, item in enumerate(altitude_flat) if i not in nan_indices]
    
    # # Convert back to a NumPy array if needed
    # filtered_Ze = np.array(filtered_list_Ze)
    # filtered_alt = np.array(filtered_list_alt)
    
    
    # A = 0
    # for value in filtered_Ze:
    #     if value < 3:
    #         A += 1
            
    # print(A, '\n')
    
    
    # print(filtered_Ze.shape)
    # print(filtered_alt.shape)
    
    
    "Whole year analysis"
    All_Ze_data = data['Ze'].values
    All_Ze_data[All_Ze_data<0.0]=np.nan
    
    new_Ze = All_Ze_data[:, 3:26]
    altitude = data['altitude'].values
    new_altitude = altitude[3:26]
    
    
    # print(new_altitude.shape)
    num = 0
    bad_row = []
    for i, row in enumerate(new_Ze):
        nan_count =  np.isnan(row).sum()
        
        if nan_count > 14:
            bad_row.append(i)
    
    for index in bad_row:
        new_Ze[index, :] = np.nan
        
    
    all_Ze_ave, time_ave = re_averaging_data(new_Ze, 10, 1, data['time'].values)
    Ze_ave_over_altitude = np.nanmean(all_Ze_ave, axis=0)
    
    Ze_median = np.nanmedian(all_Ze_ave, axis=0)
    print(Ze_ave_over_altitude, '\n')
    print(Ze_median)
    
    Ze_flat = all_Ze_ave.flatten()
    altitude_flat = np.repeat(new_altitude, all_Ze_ave.shape[0])
    
    print(Ze_flat.shape)
    print(altitude_flat.shape)
    
    # Create a boolean mask for non-NaN values
    mask = ~np.isnan(Ze_flat)
    
    # Apply the mask to filter out NaN values
    filtered_Ze = Ze_flat[mask]
    filtered_alt = altitude_flat[mask]
        
    
    A = 0
    for value in filtered_Ze:
        if value < 3:
            A += 1
            
    print(A, '\n')
    
    
    print("Length of Ze_flat:", len(filtered_Ze))
    print("Length of altitude_flat:", len(filtered_alt), '\n')
    
    "Prepping spectral width"
    # raw_width = filtered_data['width_spectral'].values
    # raw_width[Ze_matrix<6.0]=np.nan
    # vel_ave, vel_time_ave = re_averaging_data(raw_width, 10, 5, filtered_data['time'].values)
    
    # new_vel = vel_ave[:, 0:26]
    # vel_ave_over_alt = np.nanmean(new_vel, axis=0)
    # vel_flat = new_vel.flatten()
    
    # print(vel_flat.shape)
    # print(altitude_flat.shape)
    
    "2022 speactral width"
    all_width = data['width_spectral'].values
    all_width[All_Ze_data<6.0]=np.nan
    all_vel_ave, vel_time_ave = re_averaging_data(all_width, 10, 1, data['time'].values)
    
    new_vel = all_vel_ave[:, 0:26]
    vel_ave_over_alt = np.nanmean(new_vel, axis=0)
    vel_flat = new_vel.flatten()
    
    print(vel_flat.shape)
    print(altitude_flat.shape)
    
    "Doppler velocity 2022"
    # all_doppler = data['Velocity'].values
    # all_dop_ave, vel_dop_ave = re_averaging_data(all_doppler, 10, 10, data['time'].values)
    
    # print(all_dop_ave.shape)
    
    # new_dop = all_dop_ave[:, 3:26]
    # dop_ave_over_alt = np.nanmean(new_dop, axis=0)
    # dop_flat = new_dop.flatten()
    
    # dop_alt = np.repeat(new_altitude, new_dop.shape[0])
    
    
    # print(dop_flat.shape)
    # print(dop_alt.shape)
    
    
    "SNR plot in dB"
    all_Sn = data['Sn'].values
    # alt_Sn = data['altitude'].values
    
    
    new_Sn = all_Sn[:, 3:26]
    # new_alt_Sn = alt_Sn[3:26]
    
    Sn_ave, Sn_time_ave = re_averaging_data(new_Sn, 10, 10, data['time'].values)
    Sn_ave_alt = np.nanmean(Sn_ave, axis=0)

    Sn_flat = Sn_ave.flatten()
    Sn_alt = np.repeat(new_altitude, Sn_ave.shape[0])
    print(Sn_flat.shape)
    print(Sn_alt.shape)
    
    
    
    
    # # Create three subplots side by side
    fig, axs = plt.subplots(1, 3, figsize=(14, 8))
       
    # Plot 1 - KDE for Ze variable
    kde1 = sns.kdeplot(x=filtered_Ze, y=filtered_alt, cmap="viridis", fill=True, ax=axs[0])
    axs[0].scatter(Ze_ave_over_altitude, new_altitude, color='k')
    axs[0].plot(Ze_ave_over_altitude, new_altitude, color='k')
    axs[0].set_xlim(0, 25)
    axs[0].set_xlabel('Reflectivity, Ze (dBZ)')
    axs[0].set_ylabel('Altitude (m)')
    axs[0].set_title('Reflectivity')
       
    # Plot 2 - KDE for Spectral width variable
    kde2 = sns.kdeplot(x=vel_flat, y=altitude_flat, cmap="viridis", fill=True, ax=axs[1])
    axs[1].scatter(vel_ave_over_alt, new_altitude, color='k')
    axs[1].plot(vel_ave_over_alt, new_altitude, color='k')
    axs[1].set_xlim(0, 3)
    axs[1].set_xlabel('Spectral width (m/s)')
    axs[1].set_ylabel('Altitude (m)')
    axs[1].set_title('Spectral width')
    
    # Plot 3 - KDE for Spectral width variable
    kde3 = sns.kdeplot(x=Sn_flat, y=Sn_alt, cmap="viridis", fill=True, ax=axs[2])
    axs[2].scatter(Sn_ave_alt, new_altitude, color='k')
    axs[2].plot(Sn_ave_alt, new_altitude, color='k')
    axs[2].set_xlim(100, 110)
    axs[2].set_xlabel('SNR (dBz)')
    axs[2].set_ylabel('Altitude (m)')
    axs[2].set_title('SNR')
    
    # Add a single colorbar for all subplots
    all_collections = kde1.collections + kde2.collections + kde3.collections 
    colorbar = plt.colorbar(all_collections[0], ax=axs, orientation='horizontal', pad=0.1)
    colorbar.set_label('Occurrence (%)')
   
    plt.savefig('./new_analysis/kde_plot_2022_ver5.png')
    plt.show()



    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} min")
    
    
    
    
    
def re_averaging_data(data, interval_seconds, averaging_periods, time_list):
    """
    
    Average data over a specific period of time
    
    Parameters:
        data: 2D array of Ze
        interval_seconds: 10 second (the default interval time for the radar instrument)
        averaging_periods: The period of time you want ur data to be average at in minutes
        
    Return:
        np.array (2D) average data within that time in shape of (number_period, height_level)
    
    """
    
    # Calculate number of data points per averaging periods
    averaging_periods_seconds = averaging_periods * 60
    num_points_per_period = averaging_periods_seconds // interval_seconds
    
    # Calculate number of periods
    num_periods = data.shape[0] // num_points_per_period
    
    # Reshape data to groups it by periods
    reshaped_data = data[:num_periods * num_points_per_period].reshape(num_periods, num_points_per_period, data.shape[1])
    
    # Calculate the mean for each period
    averaged_data = np.nanmean(reshaped_data, axis=1)
    
    # Calculate the average time list
    averaged_datetimes = [time_list[i * num_points_per_period] for i in range(num_periods)]
    
    return averaged_data, averaged_datetimes
    
    
    
main()