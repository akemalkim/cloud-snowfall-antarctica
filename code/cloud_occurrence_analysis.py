# -*- coding: utf-8 -*-
"""
Created on Sun May  5 19:28:06 2024

@author: akmal
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob

def main():
    
    # Threshold 1: 5e-7  threshold 2: 1e-6  threshold 3: 2e-6  threshold 4: 4e-6
    # "2023 whole year data path by different threshold and cloud occurence plot"
    # data23_thr1 = xr.open_dataset("D:/4th year project/stats data/2023_stats1.nc")
    # data23_thr2 = xr.open_dataset("D:/4th year project/stats data/2023_stats2.nc")
    # data23_thr3 = xr.open_dataset("D:/4th year project/stats data/2023_stats3.nc")
    # data23_thr4 = xr.open_dataset("D:/4th year project/stats data/2023_stats4.nc")
    
    # # cl = data23_thr1['cl'].values
    # # clt = data23_thr1['cl'].values
    # # print(cl, '\n')
    
    # # print(data23_thr1, '\n')
    # # print(data23_thr1['clt'].values, '\n')
    
    # # print(data23_thr1['clt'])
    
    # plot_graph(data23_thr1['cl'], data23_thr1['zfull']/1000.0, 'b', 'Threshold 5e-7')
    # plot_graph(data23_thr2['cl'], data23_thr2['zfull']/1000.0, 'g', 'Threshold 1e-6')
    # plot_graph(data23_thr3['cl'], data23_thr3['zfull']/1000.0, 'r', 'Threshold 2e-6')
    # plot_graph(data23_thr4['cl'], data23_thr4['zfull']/1000.0, 'k', 'Threshold 4e-6')
    
    # plt.savefig('2023_ver2.png')
    # plt.show()
    
    "2022 whole year data path by different threshold and cloud occurence plot"
    # data22_thr1 = xr.open_dataset("D:/4th year project/stats data/stats2022_1.nc")
    # data22_thr2 = xr.open_dataset("D:/4th year project/stats data/stats2022_2.nc")
    # data22_thr3 = xr.open_dataset("D:/4th year project/stats data/stats2022_3.nc")
    # data22_thr4 = xr.open_dataset("D:/4th year project/stats data/stats2022_4.nc")
    
    # plot_graph(data22_thr1['cl'], data22_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph(data22_thr2['cl'], data22_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph(data22_thr3['cl'], data22_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph(data22_thr4['cl'], data22_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # plt.savefig('2022_ver2.png')
    # plt.show()
    
    "seasonal path and plot"
    # # MAM 2023
    # mam23_thr1 = xr.open_dataset("D:/4th year project/stats data/mam23_1.nc")
    # mam23_thr2 = xr.open_dataset("D:/4th year project/stats data/mam23_2.nc")
    # mam23_thr3 = xr.open_dataset("D:/4th year project/stats data/mam23_3.nc")
    # mam23_thr4 = xr.open_dataset("D:/4th year project/stats data/mam23_4.nc")
    
    
    # plot_graph((mam23_thr1['cl'] / mam23_thr1['clt'])*100.0, mam23_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((mam23_thr2['cl'] / mam23_thr2['clt'])*100.0, mam23_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((mam23_thr3['cl'] / mam23_thr3['clt'])*100.0, mam23_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((mam23_thr4['cl'] / mam23_thr4['clt'])*100.0, mam23_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # plt.savefig('mam23_ver2.png')
    # plt.show()
    
    
    # # MAM 2022
    # mam22_thr1 = xr.open_dataset("D:/4th year project/stats data/mam22_1.nc")
    # mam22_thr2 = xr.open_dataset("D:/4th year project/stats data/mam22_2.nc")
    # mam22_thr3 = xr.open_dataset("D:/4th year project/stats data/mam22_3.nc")
    # mam22_thr4 = xr.open_dataset("D:/4th year project/stats data/mam22_4.nc")
    
    
    # plot_graph((mam22_thr1['cl']), mam22_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((mam22_thr2['cl']), mam22_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((mam22_thr3['cl']), mam22_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((mam22_thr4['cl']), mam22_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # plt.savefig('mam22_ver2.png')
    # plt.show()
    
    # # JJA 2022
    # jja22_thr1 = xr.open_dataset("D:/4th year project/stats data/jja22_1.nc")
    # jja22_thr2 = xr.open_dataset("D:/4th year project/stats data/jja22_2.nc")
    # jja22_thr3 = xr.open_dataset("D:/4th year project/stats data/jja22_3.nc")
    # jja22_thr4 = xr.open_dataset("D:/4th year project/stats data/jja22_4.nc")
    
    
    # plot_graph((jja22_thr1['cl']), jja22_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((jja22_thr2['cl']), jja22_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((jja22_thr3['cl']), jja22_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((jja22_thr4['cl']), jja22_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
   
    # plt.savefig('jja22ver_2.png')
    # plt.show()
    
    # SON 2022
    # son22_thr1 = xr.open_dataset("D:/4th year project/stats data/son22_1.nc")
    # son22_thr2 = xr.open_dataset("D:/4th year project/stats data/son22_2.nc")
    # son22_thr3 = xr.open_dataset("D:/4th year project/stats data/son22_3.nc")
    # son22_thr4 = xr.open_dataset("D:/4th year project/stats data/son22_4.nc")
    
    
    # plot_graph((son22_thr1['cl']), son22_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((son22_thr2['cl']), son22_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((son22_thr3['cl']), son22_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((son22_thr4['cl']), son22_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # # plt.savefig('son22_ver2.png')
    # plt.show()
    
    # # # Model
    # # Model data stats for 2023 (MERRA2)
    # mo23_thr1 = xr.open_dataset("D:/4th year project/stats data/model stats/stats1_2023.nc")
    # mo23_thr2 = xr.open_dataset("D:/4th year project/stats data/model stats/stats2_2023.nc")
    # mo23_thr3 = xr.open_dataset("D:/4th year project/stats data/model stats/stats3_2023.nc")
    # mo23_thr4 = xr.open_dataset("D:/4th year project/stats data/model stats/stats4_2023.nc")
    
    # # print(mo23_thr1, '\n')
    # # print(mo23_thr1['clt'])
    # # print(mo23_thr1['cl'].values, '\n')
    
    # # print(mo23_thr1['cl'].values[0])
    
    # "try to think about how array addition works"
    # cl_thr1 = model_cl_mean_array(mo23_thr1['cl'].values)
    # cl_thr2 = model_cl_mean_array(mo23_thr2['cl'].values)
    # cl_thr3 = model_cl_mean_array(mo23_thr3['cl'].values)
    # cl_thr4 = model_cl_mean_array(mo23_thr4['cl'].values)
    
    # clt_mean1 = np.mean(mo23_thr1['clt'].values)
    # clt_mean2 = np.mean(mo23_thr2['clt'].values)
    # clt_mean3 = np.mean(mo23_thr3['clt'].values)
    # clt_mean4 = np.mean(mo23_thr4['clt'].values)
    
    # plot_graph((cl_thr1), mo23_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((cl_thr2), mo23_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((cl_thr3), mo23_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((cl_thr4), mo23_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # # plt.savefig('2023_model_ver2.png')
    # plt.show()
    
    
    # # Model data stats for 2022 (MERRA2)
    # mo22_thr1 = xr.open_dataset("D:/4th year project/stats data/model stats/stats1_2022.nc")
    # mo22_thr2 = xr.open_dataset("D:/4th year project/stats data/model stats/stats2_2022.nc")
    # mo22_thr3 = xr.open_dataset("D:/4th year project/stats data/model stats/stats3_2022.nc")
    # mo22_thr4 = xr.open_dataset("D:/4th year project/stats data/model stats/stats4_2022.nc")
    
    # cl_thr1a = model_cl_mean_array(mo22_thr1['cl'].values)
    # cl_thr2a = model_cl_mean_array(mo22_thr2['cl'].values)
    # cl_thr3a = model_cl_mean_array(mo22_thr3['cl'].values)
    # cl_thr4a = model_cl_mean_array(mo22_thr4['cl'].values)
    
    # clt_mean1a = np.mean(mo22_thr1['clt'].values)
    # clt_mean2a = np.mean(mo22_thr2['clt'].values)
    # clt_mean3a = np.mean(mo22_thr3['clt'].values)
    # clt_mean4a = np.mean(mo22_thr4['clt'].values)
    
    # plot_graph((cl_thr1a), mo22_thr1['zfull']/1000, 'b', 'Threshold 5e-7')
    # plot_graph((cl_thr2a), mo22_thr2['zfull']/1000, 'g', 'Threshold 1e-6')
    # plot_graph((cl_thr3a), mo22_thr3['zfull']/1000, 'r', 'Threshold 2e-6')
    # plot_graph((cl_thr4a), mo22_thr4['zfull']/1000, 'k', 'Threshold 4e-6')
    
    # plt.savefig('2022_model_ver2.png')
    # plt.show()
    
    
    "geographical analysis"
    
    "2023(MERRA2)"
 
    # mo23_g1 = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/off_2.5/all.nc")
    # mo23_g2 = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/off_5/all.nc")
    # mo23_g3 = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/off_10/all.nc")
    # mo23_g4 = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/off_20/all.nc")
    # mo23_g5 = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_ori/all.nc")
    
    
    # cl_g1 = model_cl_mean_array(mo23_g1['cl'].values)
    # cl_g2 = model_cl_mean_array(mo23_g2['cl'].values)
    # cl_g3 = model_cl_mean_array(mo23_g3['cl'].values)
    # cl_g4 = model_cl_mean_array(mo23_g4['cl'].values)
    # cl_g5 = model_cl_mean_array(mo23_g5['cl'].values)
    
    # # clt_mean1b = np.mean(mo23_g1['clt'].values)
    # # clt_mean2b = np.mean(mo23_g2['clt'].values)
    # # clt_mean3b = np.mean(mo23_g3['clt'].values)
    # # clt_mean4b = np.mean(mo23_g4['clt'].values)
    
    # plot_graph((cl_g1), mo23_g1['zfull']/1000, 'b', 'Longitude offset +2.5')
    # plot_graph((cl_g2), mo23_g2['zfull']/1000, 'g', 'Longitude offset +5')
    # plot_graph((cl_g3), mo23_g3['zfull']/1000, 'r', 'Longitude offset +10')
    # plot_graph((cl_g4), mo23_g4['zfull']/1000, 'k', 'Longitude offset +20')
    # plot_graph((cl_g5), mo23_g5['zfull']/1000, 'y', 'Longitude original')
    
    # plt.savefig('2023model_long_off_ver2.png')
    # plt.show()
    
    "Geographical analysis with latitude offset"
    
    # mo23_g1a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_2.5/all.nc")
    # mo23_g2a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_5/all.nc")
    # mo23_g3a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_-5/all.nc")
    # mo23_g4a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_-2.5/all.nc")
    # mo23_g5a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/stats_ori/all.nc")
    # mo23_g6a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/off_lat_10/all.nc")
    # mo23_g7a = xr.open_dataset("D:/4th year project/stats data/geographical_analysis/latitude/off_-10/all.nc")
    
    # print(mo23_g1a)
    
    # cl_g1a = model_cl_mean_array(mo23_g1a['cl'].values)
    # cl_g2a = model_cl_mean_array(mo23_g2a['cl'].values)
    # cl_g3a = model_cl_mean_array(mo23_g3a['cl'].values)
    # cl_g4a = model_cl_mean_array(mo23_g4a['cl'].values)
    # cl_g5a = model_cl_mean_array(mo23_g5a['cl'].values)
    # cl_g6a = model_cl_mean_array(mo23_g6a['cl'].values)
    # cl_g7a = model_cl_mean_array(mo23_g7a['cl'].values)
    
    # plot_graph((cl_g1a), mo23_g1a['zfull']/1000, 'b', 'Latitude offset +2.5')
    # plot_graph((cl_g2a), mo23_g2a['zfull']/1000, 'g', 'Latitude offset +5')
    # plot_graph((cl_g3a), mo23_g3a['zfull']/1000, 'r', 'Latitude offset -5')
    # plot_graph((cl_g4a), mo23_g4a['zfull']/1000, 'k', 'Latitude offset -2.5')
    # plot_graph((cl_g5a), mo23_g5a['zfull']/1000, 'y', 'original (no offset)')
    # plot_graph((cl_g6a), mo23_g6a['zfull']/1000, 'm', 'latitude offset +10')
    # plot_graph((cl_g7a), mo23_g7a['zfull']/1000, 'c', 'latitude offset -10')
    
    # # plt.savefig('2023m_latitude.png')
    # plt.show()
    
    "Cloud occurrence calculation from alcf lidar files using cloud mask data"
    
    # 2022 (lidar data)
    # cloud_occurrence_thr1, z_alt1 = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data/", '2022')
    # cloud_occurrence_thr2, z_alt2 = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data2/", '2022')
    # cloud_occurrence_thr3, z_alt3 = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data3/", '2022')
    # cloud_occurrence_thr4, z_alt4 = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data4/", '2022')
    
    # plot_graph(cloud_occurrence_thr1*100, z_alt1/1000, 'b', 'Threshold 5e-7')
    # plot_graph(cloud_occurrence_thr2*100, z_alt2/1000, 'g', '2022')
    # plot_graph(cloud_occurrence_thr3*100, z_alt3/1000, 'r', 'Threshold 2e-6')
    # plot_graph(cloud_occurrence_thr4*100, z_alt4/1000, 'k','Threshold 4e-6')
    
    # plt.savefig('2022thr_ver3.png')
    # plt.show()
    
    # # 2023 (lidar data)
    # cloud_occurrence_thr1a, z_alt1a = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data/", '2023')
    # cloud_occurrence_thr2a, z_alt2a = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data2/", '2023')
    # cloud_occurrence_thr3a, z_alt3a = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data3/", '2023')
    # cloud_occurrence_thr4a, z_alt4a = lidar_cloud_occurrence_stats("D:/4th year project/Lidar data/lidar_clocc_analysis", "/lidar_data4/", '2023')
    
    # plot_graph(cloud_occurrence_thr1a*100, z_alt1a/1000, 'b', 'Threshold 5e-7')
    # plot_graph(cloud_occurrence_thr2a*100, z_alt2a/1000, 'b', '2023')
    # plot_graph(cloud_occurrence_thr3a*100, z_alt3a/1000, 'r', 'Threshold 2e-6')
    # plot_graph(cloud_occurrence_thr4a*100, z_alt4a/1000, 'k','Threshold 4e-6')
    
    # plt.savefig('2023thr_ver3.png')
    # plt.show()
    
    # Model data MERRA2 (LOOK BACK AT THE )
    
    # 2022 Merra2 data
    # mo_cloud_occurrence_thr1, mo_z_alt1 = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar/", '2022')
    # mo_cloud_occurrence_thr2, mo_z_alt2 = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar2/", '2022')
    # mo_cloud_occurrence_thr3, mo_z_alt3 = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar3/", '2022')
    # mo_cloud_occurrence_thr4, mo_z_alt4 = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar4/", '2022')
    
    # plot_graph(mo_cloud_occurrence_thr1*100, mo_z_alt1/1000, 'b', 'Threshold 5e-7')
    # plot_graph(mo_cloud_occurrence_thr2*100, mo_z_alt2/1000, 'g', 'Threshold 1e-6')
    # plot_graph(mo_cloud_occurrence_thr3*100, mo_z_alt3/1000, 'r', 'Threshold 2e-6')
    # plot_graph(mo_cloud_occurrence_thr4*100, mo_z_alt4/1000, 'k','Threshold 4e-6')
    
    # plt.show()
    
    # # 2023 Merra2 data
    
    # mo_cloud_occurrence_thr1a, mo_z_alt1a = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar/", '2022')
    # mo_cloud_occurrence_thr2a, mo_z_alt2a = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar2/", '2022')
    # mo_cloud_occurrence_thr3a, mo_z_alt3a = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar3/", '2022')
    # mo_cloud_occurrence_thr4a, mo_z_alt4a = model_cloud_occurrence_stats("D:/4th year project/Lidar data/model_clocc_analysis", "/lidar4/", '2022')
    
    # plot_graph(mo_cloud_occurrence_thr1a*100, mo_z_alt1a/1000, 'b', 'Threshold 5e-7')
    # plot_graph(mo_cloud_occurrence_thr2a*100, mo_z_alt2a/1000, 'g', 'Threshold 1e-6')
    # plot_graph(mo_cloud_occurrence_thr3a*100, mo_z_alt3a/1000, 'r', 'Threshold 2e-6')
    # plot_graph(mo_cloud_occurrence_thr4a*100, mo_z_alt4a/1000, 'k','Threshold 4e-6')
    
    # plt.show()
    
    # # comparison version 2 and 3
    # plot_graph2(data22_thr1['cl'], data22_thr1['zfull']/1000, 'b', '--', 'Threshold 5e-7')
    # plot_graph2(data22_thr2['cl'], data22_thr2['zfull']/1000, 'g', '--', 'Threshold 1e-6')
    # plot_graph2(data22_thr3['cl'], data22_thr3['zfull']/1000, 'r', '--', 'Threshold 2e-6')
    # plot_graph2(data22_thr4['cl'], data22_thr4['zfull']/1000, 'k', '--', 'Threshold 4e-6')
    
    # plot_graph2(cloud_occurrence_thr1*100, z_alt1/1000, 'b', '-', 'Threshold 5e-7')
    # plot_graph2(cloud_occurrence_thr2*100, z_alt2/1000, 'g', '-', 'Threshold 1e-6')
    # plot_graph2(cloud_occurrence_thr3*100, z_alt3/1000, 'r', '-', 'Threshold 2e-6')
    # plot_graph2(cloud_occurrence_thr4*100, z_alt4/1000, 'k', '-', 'Threshold 4e-6')
    
    
    # plt.savefig('2022_comparison2_v2v3.png')
    # plt.show()
    
    ## Version 2 and version 3 ratio (v3/v2)
    
    # ratiov2_v3 = cloud_occurrence_thr3[:150]*100 / data22_thr3['cl']
    # plot_graph2(ratiov2_v3, data22_thr3['zfull']/1000, 'r', '--', 'Threshold 2e-6')
    
    # plt.show()
    
    "Calibrated and uncalibrated comparison (march 2022)"
    # cal_stats = xr.open_dataset("D:/4th year project/calibration/mar2022_calibrated/stats/all.nc")
    # ori_stats = xr.open_dataset("D:/4th year project/calibration/mar2022_original/stats/all.nc")
    
    # plot_graph((cal_stats['cl']), cal_stats['zfull']/1000, 'b', 'Calibrated data')
    # plot_graph((ori_stats['cl']), ori_stats['zfull']/1000, 'g', 'Original data')
    
    # plt.savefig('calibration_comparison.png')
    # plt.show()
    
    
    "All season for 2022"
    
    djf22 = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/djf21_22_stats.nc")
    mam22 = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/mam22_stats.nc")
    jja22 = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/jja22_stats.nc")
    son22 = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/son22_stats.nc")
    
    plot_graph(djf22['cl'], djf22['zfull']/1000, 'b', 'DJF')
    plot_graph(mam22['cl'], mam22['zfull']/1000, 'g', 'MAM')
    plot_graph(jja22['cl'], jja22['zfull']/1000, 'r', 'JJA')
    plot_graph(son22['cl'], son22['zfull']/1000, 'k', 'SON')
    
    # plt.savefig('2022_ver2.png')
    plt.show()
    
    
    "seasonal model data"
    
    djf21_22_model = xr.open_dataset("D:/4th year project/Figure for the 3 page report/model data(stats)/djf21_22.nc")
    mam22_model = xr.open_dataset("D:/4th year project/Figure for the 3 page report/model data(stats)/mam22.nc")
    jja22_model = xr.open_dataset("D:/4th year project/Figure for the 3 page report/model data(stats)/jja22.nc")
    son22_model = xr.open_dataset("D:/4th year project/Figure for the 3 page report/model data(stats)/son22.nc")
    
    cl_djf = model_cl_mean_array(djf21_22_model['cl'].values)
    cl_mam = model_cl_mean_array(mam22_model['cl'].values)
    cl_jja = model_cl_mean_array(jja22_model['cl'].values)
    cl_son = model_cl_mean_array(son22_model['cl'].values)
    
    
    plot_graph(cl_djf, djf21_22_model['zfull']/1000, 'b', 'DJF')
    plot_graph(cl_mam, mam22_model['zfull']/1000, 'g', 'MAM')
    plot_graph(cl_jja, jja22_model['zfull']/1000, 'r', 'JJA')
    plot_graph(cl_son, son22_model['zfull']/1000, 'k', 'SON')
    
    # plt.savefig('model_seasoanal2022.png')
    plt.show()
    
    "Calibrated data (Aug 2022) stats analysis"
    
    # cali_data = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/mam22_stats.nc")
    # uncali_data = xr.open_dataset("D:/4th year project/Figure for the 3 page report/Normal stats data/2022/jja22_stats.nc")
    
    # plot_graph(cali_data['cl'], cali_data['zfull']/1000, 'b', 'Calibrated')
    # plot_graph(uncali_data['cl'], uncali_data['zfull']/1000, 'g', 'Uncalibrated')
    
    # # plt.savefig('2022_ver2.png')
    # plt.show()
    
    
    
    
def model_cloud_occurrence_stats(directory, thr, year):
    """Function that calculate the cloud occurrence"""
    
    # Initialise values for couple of array
    cloud_fraction_raw_occurrence = np.ones((1000,300))*np.nan
    cbh_stats = np.ones((1000,288))*np.nan
    total_number = np.ones(1000)*np.nan
    date_vector = np.ones(1000).astype(dtype='datetime64[ns]')
    
    i = 0
    
    filenames = glob.glob(f"{directory}/{year}/{thr}/*.nc")
    
    print(filenames, len(filenames), '\n')
    
    for filename in filenames:
        day_data = xr.open_dataset(filename, engine='netcdf4')
        date_vector[i] = day_data.time[0].dt.round('d').values
        total_number[i] = day_data.time.shape[0]
        cloud_fraction_raw_occurrence[i,:] = np.sum(np.sum(day_data.cloud_mask.values,axis=0), axis=1) / 10
        # cbh_stats[i,:] = day_data.cbh.values
        
    else:
        i = i - 1
        
    altitude = day_data.zfull
    
    mean_cloud_fraction_occurrence = np.nansum(cloud_fraction_raw_occurrence, axis=0) / np.nansum(total_number, axis=0)

    return mean_cloud_fraction_occurrence, altitude

    
def lidar_cloud_occurrence_stats(directory, thr, year):
    """Function that calculate the cloud occurrence"""
    
    
    # Initialise values for couple of array
    cloud_fraction_raw_occurrence = np.ones((1000,300))*np.nan
    cbh_stats = np.ones((1000,288))*np.nan
    total_number = np.ones(1000)*np.nan
    date_vector = np.ones(1000).astype(dtype='datetime64[ns]')
    
    filenames = glob.glob(f"{directory}/{year}/{thr}/*.nc")
    
    print(filenames, len(filenames), '\n')
    
    i = 0
    for filename in filenames:
        day_data = xr.open_dataset(filename, engine='netcdf4')
        date_vector[i] = day_data.time[0].dt.round('d').values
        total_number[i] = day_data.time.shape[0]
        cloud_fraction_raw_occurrence[i,:] = np.sum(day_data.cloud_mask.values, axis=0)
        cbh_stats[i,:] = day_data.cbh.values
        
    else:
        i = i - 1
        
    altitude = day_data.zfull
    
    mean_cloud_fraction_occurrence = np.nansum(cloud_fraction_raw_occurrence, axis=0) / np.nansum(total_number, axis=0)

    return mean_cloud_fraction_occurrence, altitude
            
    
def model_cl_mean_array(row):
    """Generate the mean of row for the whole column inside the array"""
    
    new_array = np.array([])
    
    i = 0
    for i in range(150):
        row1 = row[i]
        row_mean = np.mean(row1)
        new_array = np.append(new_array, row_mean)
        i += 1
        
    return new_array

def plot_graph(x, y, color, line_title):
    """Generate simple line plot"""
    
    # plt.plot(MERRA2_mean_cloud_fraction_occurrence*100.0,MERRA2_altitude/1000.0,'b',label='MERRA2')
    plt.plot(x, y, color, label=line_title)
    
    plt.ylabel('Altitude (km)')
    plt.xlabel('Cloud occurrence (%)')
    # plt.title(title)
    plt.xlim((0.0, 70.0))
    plt.ylim((0.0, 15.0))
    plt.grid(True)
    plt.legend()

    
def plot_graph2(x, y, color, style, line_title):
    """Generate simple line plot"""
    
    # plt.plot(MERRA2_mean_cloud_fraction_occurrence*100.0,MERRA2_altitude/1000.0,'b',label='MERRA2')
    plt.plot(x, y, color, linestyle=style, label=line_title)
    
    plt.ylabel('Altitude (km)')
    plt.xlabel('Cloud occurrence (%)')
    # plt.title(title)
    plt.xlim((0.0, 70.0))
    plt.ylim((0.0, 15.0))
    plt.grid(True)
    plt.legend()



main()