# Cloud and Snowfall Study in Antarctica

## Project Overview
This project investigates cloud formation and snowfall patterns in Antarctica using advanced lidar and radar data. By leveraging these technologies, we aim to enhance our understanding of atmospheric conditions and precipitation dynamics in this unique and critical region of the Earth.

## Results
Our analysis yielded several key findings:

1. **Lidar Climatology:** Analysis of lidar data provided insights into the vertical distribution of clouds, revealing significant variations in cloud height and thickness over different seasons.

- Attenuated volume backscatter profile

<img src="time_altitude_backscatter_10032022.png" alt="Lidar Climatology" width="600"/>  <!-- Update with your actual figure path -->

- Observational Cloud Occurrence in 2022
<img src="cl_occur_obs.png" alt="Lidar Climatology" width="600"/>

- MERRA2 Cloud Occurrence in 2022
<img src="cl_occ_merra2.png" alt="Lidar Climatology" width="600"/>


2. **Radar Climatology:** Radar data helped quantify precipitation rates, indicating how snowfall varies spatially and temporally across the Antarctic landscape.

- Radar reflectivity, Doppler velocity, spectral width and snow rate
<img src="MRR2_raw_analysis4-7.png" alt="Radar Climatology" width="600"/>  <!-- Update with your actual figure path -->


3. **Snow Accumulation:** The study assessed snow accumulation trends, highlighting periods of significant accumulation and their correlation with prevailing weather patterns.

- Snow accumulation for 2022

<img src="snow_accumulation2022.png" alt="Snow Accumulation" width="600"/>  <!-- Update with your actual figure path -->

- Kernel density estimate plot
<img src="kde_plot_2022_2.png" alt="Snow Accumulation" width="600"/>

5. **Snow Accumulation Based on Synoptic Weather Patterns:** Utilizing Self-Organizing Maps (SOM), we identified distinct synoptic weather patterns that influence snow accumulation, providing a clearer understanding of how large-scale atmospheric conditions impact local snowfall.

- The sypnotic weather pattern from SOM
<img src="synoptic_type_ver3.png" alt="Snow Accumulation Patterns" width="600"/>  <!-- Update with your actual figure path -->

- 

## Methodology
The analysis was conducted using a combination of lidar and radar datasets collected from various locations in Antarctica. The following steps summarize the general approach:

1. **Data Collection:** Gathered lidar and radar data from relevant sources, ensuring a robust dataset for analysis.
2. **Data Processing:** Developed Python scripts to clean, calibrate, and process the data, including filtering noise and interpolating missing values.
3. **Climatological Analysis:** Performed statistical analyses to derive climatological insights from the processed lidar and radar data.
4. **Snow Accumulation Analysis:** Analyzed snow accumulation data in relation to synoptic weather patterns identified through Self-Organizing Maps (SOM), facilitating a deeper understanding of precipitation drivers.

## Installation
To run the analysis code, ensure you have the following Python packages installed:
- NumPy
- Matplotlib
- Pandas
- SciPy

You can install the required packages using pip:
```bash
pip install numpy matplotlib pandas scipy
