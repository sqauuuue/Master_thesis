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
        path (string) : File path with .pkl extension
    """
    ds_era5 = xr.open_dataset(path+'/era5/2021.h5')
    #ds_fcn_era = xr.open_dataset(path,engine=engine_type)
    #ds_fcn_ec =  xr.open_dataset(path,engine=engine_type)
    #ds_ec =  xr.open_dataset(path,engine=engine_type)
    return ds_era5



#def contour_spacial():

#def spacial_diff():

#def interpolate():

#def acc_computing():

#def panel_plot():
    
#def contour_diff():

#def box_plot_1ds():

#def box_plot_3ds():