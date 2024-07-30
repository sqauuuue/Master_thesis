import numpy as np
import h5py
import matplotlib.pyplot as plt
#import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.patches as patches
import os
import cartopy.mpl.ticker as cticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import re
import matplotlib.colors as mcolors
import datetime
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm


def load_data(path):
    """
        Given a path, read the file and return the contents.

        Arguments:
        path (string) : File path 
    """
    ds_era5 = xr.open_dataset(path+'/era5/2021.h5')
    ds_fcn_era = xr.open_dataset(path+'/fourcastnet/2021_dt_list_ens_backtransformed.nc',engine = 'netcdf4')
    #ds_fcn_era = xr.open_dataset(path,engine=engine_type)
    #ds_fcn_ec =  xr.open_dataset(path,engine=engine_type)
    #ds_ec =  xr.open_dataset(path,engine=engine_type)
    return ds_era5, ds_fcn_era



def variable_extraction_iterated(directory, filename_pattern, variable_name):
    '''
        Give the parameter needed, read file for each day and put them all together into another dataset, read the variable and return the values.
        Arguments:
        directory(string):base directory 
        filename_pattern(string): a raw string literal that is used to create a regular expression pattern
        variable_name(string):the variable you want to choose
    '''
    #directory = '/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams/nk2448/2023_FourCastNet/Yangfan/validation/'
    #filename_pattern = r'fc_cf_\d{8}_\d{2}_sfc\.nc'

    files = sorted([f for f in os.listdir(directory) if re.match(filename_pattern, f)])

    variable_data = []
    for file in files:
        file_path = os.path.join(directory, file)
        ds = xr.open_dataset(file_path)
        variable_data = ds[variable_name]
        data.append(variable_data)

    return variable_data


def concatenate_files(directory, filename_pattern,var_name):
    '''
        Give the parameter needed, read file for each day and put them all together into another dataset.
        Arguments:
        directory(string):base directory 
        filename_pattern(string): a raw string literal that is used to create a regular expression pattern
    '''
    #directory = '/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams/nk2448/2023_FourCastNet/Yangfan/validation/'
    #filename_pattern = r'fc_cf_\d{8}_\d{2}_sfc\.nc'

    files = sorted([f for f in os.listdir(directory) if re.match(filename_pattern, f)])

    datasets = []
    for file in files:
        file_path = os.path.join(directory, file)
        ds = xr.open_dataset(file_path)
        variable = ds[var_name]
        datasets.append(variable)

    return datasets




'''
#fcn_ecmwf
directory_fcn_ec = '/pfs/work7/workspace/scratch/ab6801-fourcastnet/fourcastnet/with_ecmwf_ic/'
filename_pattern_fcn_ec = r'\d{8}_00_backtransformed\.nc'
#fc_cf_20210615_00_sfc.nc
#20210704_00_backtransformed.nc
files_fcn_ec = sorted([f for f in os.listdir(directory_fcn_ec) if re.match(filename_pattern_fcn_ec, f)])
files_fcn_ec

data_fcn_ec = []
for file1 in files_fcn_ec:
    file_path_fcn_ec = os.path.join(directory_fcn_ec, file1)
    # Extract the date from the filename
    date_str = re.search(r'\d{8}', file1).group(0)
    
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    # Read the file and extract the desired variable
    ds_fcn_ec = xr.open_dataset(file_path_fcn_ec)
    variable_data_fcn_ec =  ds_fcn_ec['forecast'][50,:,2,79:159,194:274]
    data_fcn_ec.append(variable_data_fcn_ec)

data_array_fcn_ec =xr.DataArray(data_fcn_ec)    

mean_fcn_ec = np.mean(data_array_fcn_ec, axis=(-2, -1))
mean_fcn_ec

start_index = 56
step_size = 1
num_iterations =15
fcn_ec_values=[]
for i in range(num_iterations):
    fcn_ec_value = mean_fcn_ec[(i+1) * step_size, start_index - i * step_size*4]
    fcn_ec_values.append(fcn_ec_value)
data_array_fcn_ec_values =xr.DataArray(fcn_ec_values)
data_array_fcn_ec_values[14] = data_array_ec_values[14]
#replace the missing data

    
#def contour_spacial():

#def spacial_diff():

#def interpolate():

#def acc_computing():

#def panel_plot():
    
#def contour_diff():

#def box_plot_1ds():

#def box_plot_3ds():
'''