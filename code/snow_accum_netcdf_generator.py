#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:14:34 2024

@author: mar250
"""

import sys
import os

import numpy as np
import datetime as datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression

import scipy.signal as sig 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import netCDF4 as nc
import datetime



def main():
    
    root_dir = './2022_data'
   
    process_files(root_dir)
    
    
    
def process_files(directory):
    """
    Process all files in the given directory, extract snow data,
    and accumulate snow for each day.
    """
    all_Ze = np.empty((0, 32))
    all_dates = []

    num_month = 0
    for month_dir in sorted(os.listdir(directory)):
        month_path = os.path.join(directory, month_dir)
        
        num_month += 1
        print(f'Processing month {num_month}')
        
        if os.path.isdir(month_path):
            for day_file in sorted(os.listdir(month_path)):
                file_path = os.path.join(month_path, day_file)
                
                # Read file and process data
                [date_vector, F_matrix, H_matrix, T_matrix, CC, no_times] = read_MRR_raw_spectra(file_path)
                noise_removal = True
                try:
                    [eta_matrix, eta3_matrix, Ze_matrix, SN_matrix, velocity_matrix, width_matrix] = calculate_Ze(F_matrix[:no_times,:,:], T_matrix[:no_times,:], H_matrix[:no_times,:], CC, noise_removal)
                except ValueError:
                    continue
                else:
                    all_Ze = np.concatenate((all_Ze, Ze_matrix), axis=0)
                    all_dates.extend(date_vector[:no_times])

    all_dates = np.array(all_dates)
    all_Ze[all_Ze < 0.0] = np.nan

    # Calculate Snowfall Rates
    delta_t = 1 / 360  # Time interval in hours
    SR1, SR2, SR3, SR4, SR5 = calculate_snowfall_rates(all_Ze)

    # Accumulate Snow Data
    snow_accumulation1 = np.nancumsum(SR1 * delta_t)
    snow_accumulation2 = np.nancumsum(SR2 * delta_t)
    snow_accumulation3 = np.nancumsum(SR3 * delta_t)
    snow_accumulation4 = np.nancumsum(SR4 * delta_t)
    snow_accumulation5 = np.nancumsum(SR5 * delta_t)

    # Daily Snow Accumulation
    daily_accumulation, daily_dates = accumulate_daily_snow(all_dates, snow_accumulation1, snow_accumulation2, snow_accumulation3, snow_accumulation4, snow_accumulation5)

    # Save to NetCDF
    snow_types = ['Aggregates', 'Three-bullet rosettes', 'Low density spheres', 'Aggregate spheroids', 'Durville station']
    save_daily_snow_accumulation('./snow_accum_2022.nc', daily_dates, snow_types, daily_accumulation)

    # Plot results
    plot_results(daily_dates, snow_accumulation1, snow_accumulation2, snow_accumulation3, snow_accumulation4, snow_accumulation5)

    

def calculate_snowfall_rates(all_Ze):
    a1, b1 = 313.29, 1.85
    a2, b2 = 24.04, 1.51
    a3, b3 = 19.66, 1.74
    a4, b4 = 56, 1.2
    a5, b5 = 43.3, 0.88

    SR1 = np.power(np.power(10.0, np.mean(all_Ze[:,2:5], axis=1) / 10.0) / a1 , 1 / b1)
    SR2 = np.power(np.power(10.0, np.mean(all_Ze[:,2:5], axis=1) / 10.0) / a2 , 1 / b2)
    SR3 = np.power(np.power(10.0, np.mean(all_Ze[:,2:5], axis=1) / 10.0) / a3 , 1 / b3)
    SR4 = np.power(np.power(10.0, np.mean(all_Ze[:,2:5], axis=1) / 10.0) / a4 , 1 / b4)
    SR5 = np.power(np.power(10.0, np.mean(all_Ze[:,2:5], axis=1) / 10.0) / a5 , 1 / b5)

    return SR1, SR2, SR3, SR4, SR5


def accumulate_daily_snow(dates, accumulation1, accumulation2, accumulation3, accumulation4, accumulation5):
    """
    Accumulate snow data daily and return the daily accumulation and corresponding dates.
    """
    unique_dates = np.unique([d.date() for d in dates])
    daily_accumulation = np.zeros((len(unique_dates), 5))
    
    for i, date in enumerate(unique_dates):
        mask = np.array([d.date() == date for d in dates])
        daily_accumulation[i, 0] = accumulation1[mask][-1] - accumulation1[mask][0]
        daily_accumulation[i, 1] = accumulation2[mask][-1] - accumulation2[mask][0]
        daily_accumulation[i, 2] = accumulation3[mask][-1] - accumulation3[mask][0]
        daily_accumulation[i, 3] = accumulation4[mask][-1] - accumulation4[mask][0]
        daily_accumulation[i, 4] = accumulation5[mask][-1] - accumulation5[mask][0]

    return daily_accumulation, unique_dates


def save_daily_snow_accumulation(output_file, dates, snow_types, snow_data):
    """
    This function processes daily snow accumulation data from variables and saves
    the data in a NetCDF file with the dimensions of (date, snow_type).
    
    Parameters:
    - output_file: Path to the output NetCDF file.
    - dates: A list of dates corresponding to the snow data (e.g., ['2023-01-01', '2023-01-02', ...]).
    - snow_types: A list of snow types (e.g., ['wet_snow', 'dry_snow', 'mixed_snow']).
    - snow_data: A 2D array or list of lists with daily snow accumulation for each snow type.
                 The shape should be (number_of_dates, number_of_snow_types).
                 For example, snow_data[0][0] represents the snow accumulation for the first date and first snow type.
    """
    
    # Check that the shape of snow_data matches the dates and snow_types
    assert len(snow_data) == len(dates), "Number of rows in snow_data must match the number of dates"
    assert len(snow_data[0]) == len(snow_types), "Number of columns in snow_data must match the number of snow types"
    
    # Open a new NetCDF file for writing
    dataset = nc.Dataset(output_file, 'w', format='NETCDF4')  # Using NETCDF4 format

    # Create dimensions
    date_dim = dataset.createDimension('date', len(dates))  # Number of dates
    snow_type_dim = dataset.createDimension('snow_type', len(snow_types))  # Number of snow types

    # Create variables
    date_var = dataset.createVariable('date', str, ('date',))  # Variable-length string for dates
    snow_type_var = dataset.createVariable('snow_type', str, ('snow_type',))  # Variable-length string for snow types
    snow_accumulation_var = dataset.createVariable('snow_accumulation', 'f4', ('date', 'snow_type'))

    # Assign data to variables
    date_var[:] = dates  # Store dates in the 'date' variable
    snow_type_var[:] = snow_types  # Store snow types in the 'snow_type' variable

    # Assign snow data
    for i, day_data in enumerate(snow_data):
        for j, accumulation in enumerate(day_data):
            snow_accumulation_var[i, j] = accumulation  # Accumulate snow for the day and snow type
    
    # Close the dataset
    dataset.close()

    print(f"NetCDF file {output_file} created successfully.")


def plot_results(dates, accumulation1, accumulation2, accumulation3, accumulation4, accumulation5):
    """
    Plot the snow accumulation results.
    """
    plt.figure(figsize=(15, 6))
    
    plt.plot(dates, accumulation1, label='Aggregates snow', color='red')
    plt.plot(dates, accumulation2, label='Three-bullet rosettes', color='orange')
    plt.plot(dates, accumulation3, label='Low density spheres', color='green')
    plt.plot(dates, accumulation4, label='Aggregate spheroids', color='black')
    plt.plot(dates, accumulation5, label='Durville station', color='purple')

    plt.xlabel('Date')
    plt.ylabel('Snow accumulation (mm)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('Snow_accum_2022.png', dpi=900)
    plt.show()

  

        
def find_widths(tspectra):
    
    centre_index=np.nanargmax(tspectra[32:])+32
#    print(centre_index)
    for i_up in np.arange(centre_index,tspectra.shape[0]):
        if(np.isfinite(tspectra[i_up])):
            iupper=i_up
        else:
            break
    for i_down in np.arange(centre_index,0,-1):
        if(np.isfinite(tspectra[i_down])):
            ilower=i_down
        else:
            break
#    print(centre_index,ilower,iupper)
    tspectra[:ilower]=np.nan
    tspectra[iupper:]=np.nan
    width=iupper-ilower+1
    return centre_index,width,tspectra


def re_averaging_data(data, interval_seconds, averaging_periods, time_list):
    """
    
    Average data over a specific period of time
    
    Parameters:
        data: 2D array of Ze
        interval_seconds: 10 second (the default interval time for the radar instrument)
        averaging_periods: The period of time you want ur data to be average at
        
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
    

def parse_fixed_width_array(line_raw,index):
    return [line_raw[ index[i-1] : index[i] ] for i in range(1,len(index)) ]


def parse_MRR2_M_line_raw(line_raw):
    # this is the metadata line_raw
    if (line_raw[0] == 'M'):
        file_year1=(int(line_raw[4:6])+2000)
        file_month1=(int(line_raw[6:8]))
        file_day1=(int(line_raw[8:10]))
        file_hour1=(int(line_raw[10:12]))
        file_min1=(int(line_raw[12:14]))
        file_sec1=(int(line_raw[14:16]))
        tmp=(line_raw[-33:-23])
#        print(tmp)
        if(tmp[0:2]=='CC'):
            CC=float(tmp[2:])
        else:
            CC=np.nan
        return datetime.datetime(file_year1,file_month1,file_day1,file_hour1,file_min1,file_sec1),CC
    
        
def number_float_elements(array):
    number_elements=0
    for element in array:
        if(isfloat(element)):
            number_elements+= 1
    return number_elements


def number_int_elements(array):
    number_elements=0
    for element in array:
        if(isint(element)):
            number_elements+= 1
    return number_elements



def parse_float_array(rec):    
    return np.array([
        float(x) if x != '' else np.nan
        for x in rec[1:-1]]
    )        
    

def isfloat(str):
    try:
        float(str)
        return True
    except ValueError:
        return False
    

def isint(str):
    try:
        int(str)
        return True
    except ValueError:
        return False
    
def parse_space(array):
    length_spaces=len(array)
    no_spaces=0
    for element in array:
        if(element==' '):
            no_spaces+=1
    if(no_spaces==length_spaces):
        output=True
    else:
        output=False
    return output



def parse_MRR2_D_line_raw(line_raw,delimiter):    
    indice=index_create(0,3,12,9,len(line_raw))  # for raw spectra spacing
    array=parse_fixed_width_array(line_raw,indice)
#    print(array)
    number_elements=number_float_elements(array)
    D_array=np.ones((32))*np.nan
    D_number=int(array[0][1:3])
#    print(line_raw)
#    print(D_number)
    i=0
    for element in array[1:]:
        if(isfloat(element)):
            D_array[i]=element
            i+=1
#            print(i)
        else:
            if (parse_space(element)):
#                print('empty element found')
                D_array[i]=np.nan
                i+=1
#                print(i)
    if(i!=32):        
        number_elements=32
        D_array=np.ones((number_elements))*np.nan        
    else:
        number_elements=i
    return D_array,number_elements,D_number    

def parse_MRR2_H_line_raw(line_raw,delimiter):
    array=line_raw.split(delimiter)   
    number_elements=number_int_elements(array)
#    print(number_elements)
    if(number_elements==32):
        H_array=np.ones((number_elements))
        i=0
        for element in array:
            if(isint(element)):
                H_array[i]=element
                i+=1
    else:
        number_elements=32
        H_array=np.ones((number_elements))
    return H_array,number_elements    


def parse_MRR2_T_line_raw(line_raw,delimiter):
    array=line_raw.split(delimiter)
    
    number_elements=number_float_elements(array)
    if(number_elements==32):
        T_array=np.ones((number_elements))
        i=0
        for element in array:
            if(isfloat(element)):
                T_array[i]=element
                i+=1
    else:
        number_elements=32
        T_array=np.ones((number_elements))
    return T_array,number_elements    



def parse_MRR2_F_line_raw(line_raw,delimiter):
    return parse_MRR2_D_line_raw(line_raw,delimiter)


def index_create(val1,val2,val3,increment,length_line_raw):
    index=[]
    index.append(val1)
    index.append(val2)
    val4=val3+np.arange(0,(length_line_raw-val3),increment)
    for element in val4:
        index.append(element)
    return index



def read_MRR_raw_spectra(filename):
    input_file=open(filename)
    ii=-1
    max_no_times=8641*2
    max_no_heights=32
    no_FFTs=64
    no_heights=np.ones(max_no_times)*np.nan

    date_vector=[]
    F_matrix=np.ones((max_no_times,max_no_heights,no_FFTs))*np.nan
    H_matrix=np.ones((max_no_times,max_no_heights))*np.nan
    T_matrix=np.ones((max_no_times,max_no_heights))*np.nan
        

    for line_raw in input_file:
#        print(line_raw)
        if (line_raw[0] == 'M'):
            [date,CC]=parse_MRR2_M_line_raw(line_raw)
            date_vector.append(date)
            ii+=1
        elif (line_raw[0] == 'H'):
            delimiter=' '
            [H_array,no_heights[ii-1]]=parse_MRR2_H_line_raw(line_raw,delimiter)
#            print(H_array)
            if((len(H_array)==32)):
                H_matrix[ii,:]=H_array
            else:
                ii-=1                
        elif (line_raw[0] == 'T'):
            delimiter=' '
            [T_array,no_T_elements]=parse_MRR2_T_line_raw(line_raw,delimiter)
            T_matrix[ii,:]=T_array
        elif (line_raw[0] == 'F'):
            delimiter=' '
            [F_array,no_F_elements,F_number]=parse_MRR2_F_line_raw(line_raw,delimiter)
            F_matrix[ii,:,F_number]=F_array
#            print(no_F_elements)
        else:
            print('warning')

    return date_vector,F_matrix,H_matrix,T_matrix,CC,ii
    

def calculate_Ze(F_matrix,T_matrix,H_matrix,CC,noise_removal):
# calculate eta
#        eta = (rawSpectra.data.T * np.array(
#            (self.co["mrrCalibConst"] * (heights**2 / deltaH)) / (1e20), dtype=float).T).T

    Vmin= 0
    # nyquist range maximum
    Vmax = 11.9301147
    # nyquist delta
    Vdelta = 0.1893669
    # list with nyquist velocities
    velocities= np.arange(Vmin,Vmax,Vdelta)


    # See Equation 2.1 in MRR Physical Basics version 5.2.0.1
    # Trying to calculate the spectral reflectivity
    # T_matrix is the transfer function
    # F_matrix is the raw spectral power
    deltaH=H_matrix[0,1]-H_matrix[0,0]
    i=0
    for height in H_matrix[1,:]:
        eta_matrix=(F_matrix*CC*((i+1)**2.0*deltaH))/(T_matrix[1,i]*1e20)
        i+=1
    
    
    K2=0.92
    mrrFrequency=24.23e9 
    wavelength=299792458.0/ mrrFrequency
    deltaf=30.52  # frequency resolution of bins
    Vdelta=0.1905
    velocity_matrix=np.ones((eta_matrix.shape[0],eta_matrix.shape[1]))*np.nan
    width_matrix=np.ones((eta_matrix.shape[0],eta_matrix.shape[1]))*np.nan
    eta3_matrix=np.ones((eta_matrix.shape[0],eta_matrix.shape[1],(eta_matrix.shape[2]*3)-1))*np.nan
    if(noise_removal):
        eta_tmp=np.copy(eta_matrix)
        for i in np.arange(0,eta_tmp.shape[0]):
            print(i)
            for j in np.arange(0,eta_tmp.shape[1]):
#               print(eta_matrix[i,j,:])
                [indice,noise]=Hildebrand_noise_identification(eta_tmp[i,j,:])
#                print(eta_matrix[i,j,:])
                spectra=(indice*eta_matrix[i,j,:])-noise
#                print(spectra)
                [triple_spectra,triple_velocity]=tripleandclean(np.squeeze(spectra))
#                print(triple_spectra)
                [centre_index,width,triple_spectra]=find_widths(triple_spectra)

                velocity_matrix[i,j]=triple_velocity[centre_index]
                width_matrix[i,j]=width*Vdelta
                eta3_matrix[i,j,:]=triple_spectra

    Ze = (1e18*K2*wavelength**4*np.nansum(eta3_matrix,axis=2))/(Vmax*np.pi**5)
    Ze_matrix = (10.0*np.log10(Ze))
    SN_matrix=1e18*K2*(wavelength**4*np.nansum(eta_matrix,axis=2)/(noise*np.pi**5))
    SN_matrix = (10.0*np.log10(SN_matrix))

    return eta_matrix,eta3_matrix,Ze_matrix,SN_matrix,velocity_matrix,width_matrix  #,Ze_matrix


def Hildebrand_noise_identification(spectra):
    spectral_mean_old=-999.0
    index=np.ones(spectra.shape[0])*np.nan
    for i in np.arange(0,spectra.shape[0]):
        spectral_mean=np.nanmean(spectra)
        spectral_variance=np.nanvar(spectra)
#        print(spectral_mean,spectral_variance)                
        index[spectra>spectral_mean]=1.0
        spectra[spectra>spectral_mean]=np.nan
        if((spectral_mean**2.0/spectral_variance)>5.8):  # assumed limit 5.8 for 10 second data
            return index,(spectral_mean)
    return index,(spectral_mean)

def tripleandclean(spectra):
    tmp_spectra=np.copy(spectra)
    triple_spectra=np.concatenate([tmp_spectra[1:],tmp_spectra,tmp_spectra])
    Vdelta=0.1887
    Vmax = 11.9301147
    velocity1=np.arange(0.0,Vmax+0.001,Vdelta)
    triple_velocity=np.concatenate([-np.flipud(velocity1[1:])-Vdelta,velocity1,velocity1+Vmax])
    return triple_spectra,triple_velocity



    
main()
