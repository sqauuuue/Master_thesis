#!/bin/bash

####################################################################
#                                                                  #
# run_era5_clim.sh                                                 #
#                                                                  #
# This script can be used for calculating ERA-5 climatologies from #
# Z-files in IMK-TRO's data archive.                               #
#                                                                  #
# Usage: Change the seven environment variables, save, and then    #
#        run ./run_era5_clim.sh.                                   #
#                                                                  #
# Dependency: era5_clim_JW.py (adapted from Moritz Deinhard)       #
#                                                                  #
# Data output: clim_*30day_centered_on_0629_1979-2020* netcdf file #
#                                                                  #
# Time period: 1979-2020, 15 June - 14 July                        #
# Spatial domain: 180° W-60° W; 25° N - 80° N                      #
# --> can be changed in lines 25-39 in era5_clim_JW.py             #
#                                                                  #
# 20 Sep 2023, Jannik Wilhelm, jannik.wilhelm@kit.edu              #
#                                                                  #
####################################################################

# Change variable name and pressure level here:
# =============================================
VARNAME="RH"       # available variables: "RH": relative humidity
PLEV=10000         # available levels: 10000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 85000, 90000 Pa

# Change path and file names here:
# ================================
PYTHONPATH=/home/kit/imk-tro/gj5173/anaconda3/envs/era5_clim/bin/python   # the path to your python binary of the proper conda environment
LSDFDIR=/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams                                          # the path to the LSDF ERA-5 archive (do not change!)
OUTPUTDIR=/home/kit/imk-tro/gj5173/tmp                                                       # the path, where you want to store the output and log file to
ORISCRIPT=/home/kit/imk-tro/gj5173/era5_clim/era5_clim_JW_RH.py                                       # the path+name to the python script doing the climatology calculations
PARTITION="single"                                                                           # the name of the SLURM partition (queue) where to run the job (set to: "single")


# =======================================================================================
# Do NOT change the following lines, which will modify and run the script era5_clim_JW.py
# =======================================================================================

# Copy the original script to the one that gets actually modified
SCRIPT=${OUTPUTDIR}/era5_clim_JW_RH.py
cp ${ORISCRIPT} ${SCRIPT}

# Modification of the python script
sed -i "1s,[^ ]*[^ ],\#\!${PYTHONPATH},1" ${SCRIPT}
sed -i "3s,[^ ]*[^ ],"--partition=${PARTITION}",2" ${SCRIPT}
sed -i "9s,[^ ]*[^ ],"--job-name=${VARNAME}_${PLEV}_clim",2" ${SCRIPT}
sed -i "11s,[^ ]*[^ ],"--output=${OUTPUTDIR}/${VARNAME}_${PLEV}_clim_$(date +%y%m%d%H%M%S).log",2" ${SCRIPT}
sed -i "31s,[^ ]*[^ ],lsdfdir=\"${LSDFDIR}\",1" ${SCRIPT}
sed -i "32s,[^ ]*[^ ],outdir=\"${OUTPUTDIR}\",1" ${SCRIPT}
sed -i "41s,[^ ]*[^ ],varname=\"${VARNAME}\",1" ${SCRIPT}
sed -i "42s,[^ ]*[^ ],plev=${PLEV},1" ${SCRIPT}
sed -i "86s,[^ ]*[^ ],data_out.${VARNAME}.attrs[\"_FillValue\"]=False,1" ${SCRIPT}
wait

# Running the python script
sbatch ${SCRIPT}
