#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:02:31 2024

@author: mar250
"""

import sys

import numpy as np
import datetime as datetime
from datetime import timedelta
from sklearn.linear_model import LinearRegression

import scipy.signal as sig 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def main():
   
    # Read the file and extract the important data
    day_list = ['01'] #, '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
                #'14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
                #'28', '29', '30', '31']
    
    # base_path = '/home/users/mar250/Documents/PHYS480/radar/Radar_data(raw)/202206/06'
    # base_path2 = '/home/users/mar250/atmosdata1/mar250/MRR2_data_SB/MRR Data/RawSpectra/202207'
    
    # The real file path
    filepath = "D:/4th year project/Radar/Project_analysis/RadarRawSpectra/202207/07"
    
    for num in day_list:
        filename = filepath + num + '.raw'
        
        try:
            [date_vector,F_matrix,H_matrix,T_matrix,CC,no_times]=read_MRR_raw_spectra(filename)
            noise_removal=True  # set to False if don't want noise removal
            #no_times=3
            [eta_matrix,eta3_matrix,Ze_matrix,SN_matrix,velocity_matrix,width_matrix]=calculate_Ze(F_matrix[:no_times,:,:],T_matrix[:no_times,:],H_matrix[:no_times,:],CC,noise_removal)


            # Filter out value that is non logical
            velocity_matrix[Ze_matrix<6.0]=np.nan
            width_matrix[Ze_matrix<6.0]=np.nan
            Ze_matrix[Ze_matrix<0.0]=np.nan
            
            "Five minutes interval reflectivity data"
            Z_5min, new_time_list = re_averaging_data(Ze_matrix, 10, 10, date_vector)

        
        except FileNotFoundError:
            continue
        
        
        print(eta_matrix)
        
        
        
        "New set of calculation to calculate Snowfall rate"
        # Aggregates snow
        a1=313.29
        b1=1.85 
        # SR1 = (np.mean(Ze_matrix[:,2:5], axis=1) / a1) ** (1/b1)
        SR1 = np.power(np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0) / a1 , 1/b1) 
        # SR1 = np.power((np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0)), 1/b1 ) / a1 # The original equation
        SR1_a = np.power(np.power(10.0, np.mean(Z_5min[:, 2:5], axis=1) / 10.0) / a1, 1/b1)

        # Three-bullet rosettes
        a2 = 24.04
        b2 = 1.51
        # SR2 = (np.mean(Ze_matrix[:,2:5], axis=1) / a2) ** (1/b2)
        SR2 = np.power(np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0) / a2 , 1/b2)
        SR2_a = np.power(np.power(10.0, np.mean(Z_5min[:,2:5], axis=1) / 10.0) / a2, 1/b2)

        # Low density spheres
        a3 = 19.66
        b3 = 1.74
        # SR3 = (np.mean(Ze_matrix[:,2:5], axis=1) / a3) ** (1/b3)
        SR3 = np.power(np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0) / a3 , 1/b3)
        SR3_a = np.power(np.power(10.0, np.mean(Z_5min[:,2:5], axis=1) / 10.0) / a3, 1/b3)

        # Agregate spheroids
        a4 = 56
        b4 = 1.2
        # SR4 = (np.mean(Ze_matrix[:,2:5], axis=1) / a4) ** (1/b4)
        SR4 = np.power(np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0) / a4 , 1/b4)
        SR4_a = np.power(np.power(10.0, np.mean(Z_5min[:,2:5], axis=1) / 10.0) / a4, 1/b4)
        
        # D'urville coefficient from V. Wiener et. al. (2017)
        a5 = 43.3
        b5 = 0.88
        SR5 = np.power(np.power(10.0, np.mean(Ze_matrix[:,2:5], axis=1) / 10.0) / a5 , 1/b5)
        SR5_a = np.power(np.power(10.0, np.mean(Z_5min[:,2:5], axis=1) / 10.0) / a5, 1/b5)
         
        year = date_vector[1].year
        month = date_vector[1].month
        day = date_vector[1].day

        # Setting up the new xticks
        xticks_dates = [
            datetime.datetime(year, month, day, 0, 0),
            datetime.datetime(year, month, day, 6, 0),
            datetime.datetime(year, month, day, 12, 0),
            datetime.datetime(year, month, day, 18, 0),
            datetime.datetime(year, month, day, 23, 0)
            ]

        xticks_label = ['00:00', '06:00', '12:00', '18:00', '23:00']

        fig = plt.figure(figsize=(210/25.4,200/25.4))
        gs = gridspec.GridSpec(4, 1)
        gs.update(wspace=0.3, hspace=0.4)

        # This plots the reflectivity derive from the raw spectrum value from the radar data
        # H_matrix[0,2:21] this will be a 1D element with 19 element
        # np.squeeze(Ze_matrix[:no_times,2:21]).T that will match up the shape of x and y axis

        plt.subplot(gs[0])
        plt.pcolormesh(date_vector[:no_times], H_matrix[0,2:21], np.squeeze(Ze_matrix[:no_times,2:21]).T, cmap='Blues', vmin=0, vmax=20.0)
        plt.xticks(xticks_dates, xticks_label)
        cbar=plt.colorbar()
        cbar.set_label('Ze (dBZ)', rotation=90,fontsize=10)
        plt.xlabel('UTC Time (hours)')
        plt.ylabel('Altitude (m)')

        # The Doppler velocity plot
        plt.subplot(gs[1])
        plt.pcolormesh(date_vector[:no_times], H_matrix[0,2:21], np.squeeze(velocity_matrix[:no_times,2:21]).T, cmap='seismic', vmin=-6.0, vmax=6.0)
        plt.xticks(xticks_dates, xticks_label)
        cbar=plt.colorbar()
        cbar.set_label('V$_D$ (m/s)', rotation=90,fontsize=10)
        plt.xlabel('Time (hours)')
        plt.ylabel('Altitude (m)')

        # The spectral width plot
        plt.subplot(gs[2])
        plt.pcolormesh(date_vector[:no_times], H_matrix[0,2:21], np.squeeze(width_matrix[:no_times,2:21]).T, cmap='Blues', vmin=0.0, vmax=6.0)
        plt.xticks(xticks_dates, xticks_label)
        cbar=plt.colorbar()
        cbar.set_label('$\sigma_D$ (m/s)', rotation=90,fontsize=10)
        plt.xlabel('UTC Time (hours)')
        plt.ylabel('Altitude (m)')

        # The precipitation plot 
        plt.subplot(gs[3])
        
        # 10 seconds interval
        plt.plot(date_vector[:no_times], SR1[:no_times], label='Aggregates snow', color='blue', alpha=0.3)
        plt.plot(date_vector[:no_times], SR2[:no_times], label='Three-bullet rosettes', color='Orange', alpha=0.3)
        plt.plot(date_vector[:no_times], SR3[:no_times], label='Low density spheres', color='Green', alpha=0.3)
        plt.plot(date_vector[:no_times], SR4[:no_times], label='Agregate spheroids', color='Red', alpha=0.3)
        
        # # 5 minutes interval integration
        # plt.plot(new_time_list, SR1_a, label='Aggregates snow', color='blue', alpha=0.3)
        # plt.plot(new_time_list, SR2_a, label='Three-bullet rosettes', color='Orange', alpha=0.3)
        # plt.plot(new_time_list, SR3_a, label='Low density spheres', color='Green', alpha=0.3)
        # plt.plot(new_time_list, SR4_a, label='Agregate spheroids', color='Red', alpha=0.3)
        # plt.plot(new_time_list, SR5_a, label='Agregate spheroids', color='purple', alpha=0.3)
        
        plt.xticks(xticks_dates, xticks_label)
        # plt.ylim(0.0, 0.1)
        plt.grid('on')
        plt.legend(prop={'size':5}, loc='best')
        plt.xlabel('UTC Time (hours)')
        plt.ylabel('Precipitation (mm/h)')


        plt.colorbar()

        fig.delaxes(fig.axes[7]) 

        plt.tight_layout()

        plt.savefig('./july2022/' + 'MRR2_raw_analysis'+str(month)+'-'+str(day)+'.png',dpi=900)  

        # plt.close()
        plt.show()
        
        print(f'Right now we are at this day {num}. Plz be patient')


        # plt.plot(SR1, np.mean(Ze_matrix[:,2:5], axis=1), label='Aggregates snow', color='blue')
        # plt.plot(SR2, np.mean(Ze_matrix[:,2:5], axis=1), label='Three-bullet rosettes', color='orange')
        # plt.plot(SR3, np.mean(Ze_matrix[:,2:5], axis=1), label='Low density spheres', color='green')
        # plt.plot(SR4, np.mean(Ze_matrix[:,2:5], axis=1), label='Agregate spheroids', color='black')

        # # plt.plot(S_clean, a_model * S_clean**b_model, color='red', label='Fitted curve')
        # plt.xlabel('Snowfall rate (Sr)')
        # plt.ylabel('Reflectivity (Ze)')
        # plt.title('Relationship between Ze and S using different a & b parameter')
        # # plt.savefig('MRR2_snowfall_to_Ze relationship_'+str(year)+'-'+str(month)+'-'+str(day)+'.png',dpi=900)
        # plt.savefig('Relationship_plot.png',dpi=900)
        # plt.legend()
        # plt.show()
        
        
        "Snow accumulation for every five minutes interval"
        delta_t = 1/360
        
        A1 = SR1 * delta_t
        A2 = SR2 * delta_t
        A3 = SR3 * delta_t
        A4 = SR4 * delta_t
        A5 = SR5 * delta_t

        snow_accumulation1 = np.nancumsum(A1)
        snow_accumulation2 = np.nancumsum(A2)
        snow_accumulation3 = np.nancumsum(A3)
        snow_accumulation4 = np.nancumsum(A4)
        snow_accumulation5 = np.nancumsum(A5)
        
        # 10 seconds interval
        plt.plot(date_vector[:no_times], snow_accumulation1[:no_times], label='Aggregates snow', color='red',)
        plt.plot(date_vector[:no_times], snow_accumulation2[:no_times], label='Three-bullet rosettes', color='orange')
        plt.plot(date_vector[:no_times], snow_accumulation3[:no_times], label='Low density spheres', color='green')
        plt.plot(date_vector[:no_times], snow_accumulation4[:no_times], label='Agregate spheroids', color='black')
        plt.plot(date_vector[:no_times], snow_accumulation5[:no_times], label='Durville station', color='purple')

        # # # 5 minute interval
        # plt.plot(new_time_list, snow_accumulation1, label='Aggregates snow', color='red',)
        # plt.plot(new_time_list, snow_accumulation2, label='Three-bullet rosettes', color='orange')
        # plt.plot(new_time_list, snow_accumulation3, label='Low density spheres', color='green')
        # plt.plot(new_time_list, snow_accumulation4, label='Agregate spheroids', color='black')
        # plt.plot(new_time_list, snow_accumulation5, label='D urville coeff', color='purple')

        plt.xticks(xticks_dates, xticks_label)
        # plt.ylim(0.0, 0.1)
        plt.grid('on')
        plt.xlabel('UTC Time (hours)')
        plt.ylabel('snow (mm)')
        plt.legend(loc='best')
        plt.savefig('./july2022/snow_accumulation' + 'MRR2_raw_snow_analysis'+str(year)+'-'+str(month)+'-'+str(day)+'.png',dpi=900)  
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


    
main()    