# The 2021 Pacific Northwest Heat Wave: Meteorological Interpretation of Forecast Uncertainties in Data-Driven and Physics-Based Ensembles

This repository provides code for my master thesis, which includes code for simulating data-driven models and methods of evaluation of data-driven models. Furthermore, codes for climatology computing are also included.

## The code and description in each folder are as follows:

| Folder             | General Description                | Detailed Description                |
|--------------------|----------------------------|----------------------------|
| code_aimodel_run    |Running Data-Driven Models| - job_fourcastnetv2_horeka.sh.default<br>- loop_fourcastnetv2_uc2_era5.sh.default<br>- run_fourcastnetv2_horeka.sh.default | 
| code_era5_clim      |Climatology Calculation| - era5_clim_JW.py<br>- era5_clim_JW_RH.py<br>- run_era5_clim.sh<br>- run_era5_clim_RH.sh | 
| code_main/Forecast_evaluation          |Evaluation of forecasts|- ACC_calculation_vis.ipynb<br>- forecast_evolution_vis.ipynb<br>- forecast_skill_horizon_cal_vis.ipynb<br>- good_bad_memebr_classfication.ipynb | 
| code_main/Meteorological_analysis          |Meteorological analysis of forecasts|- diurnal cycle.ipynb<br>- function_basic.py<br>- good_bad_member_vis.ipynb<br>- vertical_T_FCNV2_V1_EC.ipynb<br>- vertical_T_Q_FCNV2_EC.ipynb<br>- ertical_T_Q_IFS.ipynb | 
