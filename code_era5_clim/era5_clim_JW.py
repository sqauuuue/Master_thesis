#!/home/hk-project-test-teemleap/ey9908/share/conda/era5_clim/bin/python3.9

#SBATCH --partition=dev_cpuonly
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000mb
#SBATCH --job-name=t850_clim
#SBATCH --constraint=LSDF
#SBATCH --output=/home/hk-project-test-teemleap/ey9908/output/era5_clim/30day_T850_climatology_era5.log

import xarray as xr
import sys
import os
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from datetime import datetime,timedelta

def newtime(time,tinc):
    time=str((datetime.strptime(time, '%Y%m%d_%H')+timedelta(hours=int(tinc))).strftime('%Y%m%d_%H'))
    return time

year_start=1979
year_end=2020

firstdate="0615_00"
lastdate="0714_00"

lsdfdir="/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams"
outdir="/home/hk-project-test-teemleap/ey9908/tmp"

era5dir="%s/ec.era5" % (lsdfdir)

lonmin=-180.
lonmax=-60.
latmin=25.
latmax=80.

varname="T"
plev=85000 #Pa

lons=np.arange(lonmin,lonmax,0.5)
lats=np.arange(latmin,latmax,0.5)

#outdir="%s/test" % (lsdfdir)
filename_out="%s/clim_%s%s_30day_centered_on_0629_%s-%s_%s-%sdegE_%s-%sdegN_by_grid_point" % (outdir,varname,plev,year_start,year_end,str(lonmin),str(lonmax),str(latmin),str(latmax))

index = 0
while year_start != year_end:
        print(year_start,flush=True)

        day="%s%s" % (year_start,firstdate)

        lastday_per_year=newtime("%s%s" % (year_start,lastdate), 24)

        filenames_per_year=[]
        while day != lastday_per_year:

            month=day[4:6]
            datadir="%s/%s/%s" % (era5dir,year_start,month)

            filename="Z%s" % (day)
            data_in=xr.load_dataset("%s/%s" % (datadir,filename)).to_array().loc[dict(variable=varname,plev=plev,lon=lons,lat=lats)].squeeze()

            if index == 0:
                data=data_in
            else:
                data=data+data_in

            day=newtime(day,24)
            index=index+1

        year_start=year_start+1

clim=data/index

data_out=clim.to_dataset(name=varname).drop("variable")
data_out.lon.attrs["_FillValue"]=False
data_out.lat.attrs["_FillValue"]=False
data_out.plev.attrs["_FillValue"]=False
data_out.T.attrs["_FillValue"]=False

data_out.to_netcdf(filename_out)
data_out.close()
print(filename_out)

