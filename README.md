# The 2021 Pacific Northwest Heat Wave: Meteorological Interpretation of Forecast Uncertainties in Data-Driven and Physics-Based Ensembles

This repository provides code for the thesis, which includes code for simulating data-driven models and methods of evaluation of data-driven models. Furthermore, codes for climatology computing are also included.

## The code and description in each folder are as follows:

| Folder             | General Description                | Detailed Description                |
|--------------------|----------------------------|----------------------------|
| code_aimodel_run    |Running Data-Driven Models| - job_fourcastnetv2_horeka.sh.default: job submission script to run models<br>-loop_fourcastnetv2_uc2_era5.sh.default: automate the submission of multiple jobs to a cluster by looping through members and dates<br>- run_fourcastnetv2_horeka.sh.default: run script | 
| code_era5_clim      |Climatology Calculation| - era5_clim_JW.py: calculating the climatology<br>- run_era5_clim.sh: run script automates the process of calculating climatology from ERA5 dataset | 
| code_main/Forecast_evaluation          |Evaluation of forecasts|- ACC_calculation_vis.ipynb: calculating the anomaly correlation coefficient and visualizing<br>- forecast_evolution_vis.ipynb<br>- forecast_skill_horizon_cal_vis.ipynb: calculating the forecast skill horizon and visualizing<br>- good_bad_memebr_classfication.ipynb: classifying the bad forecasts and good forecasts based on the root mean square error. | 
| code_main/Meteorological_analysis          |Meteorological analysis of forecasts|- diurnal cycle.ipynb<br>- function_basic.py<br>- good_bad_member_vis.ipynb: visualizing the forecast filed of good members and bad members of forecasts<br>- vertical_T_FCNV2_V1_EC.ipynb: vertical profile of temperature in FourCastNet version1 and version2<br>- vertical_T_Q_FCNV2_EC.ipynb: vertical profile of temperature and specific humidity in FourCastNet version2<br>- vertical_T_Q_IFS.ipynb: vertical profile of temperature and specific humidity in IFS | 
