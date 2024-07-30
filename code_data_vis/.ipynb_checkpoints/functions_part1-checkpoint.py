import os
import xarray as xr

########################################## read the fcnv2 data ####################################################
def read_fcnv2_forecast(date_str, dataset_type):
    # Define base directories and filename patterns based on dataset type
    '''
    date_str(string):convert the datetime into string first before put it into function;
    dataset_type(string):'era' or 'ec', which means two different initial condition.
    '''
    base_directory = '/pfs/work7/workspace/scratch/gj5173-ws_ai_models/'
    if dataset_type == 'ec':
        filename_pattern = f'fcnv2_fc_ifs_{date_str}_0000_m0.nc'
        dataset_directory = 'ifs'
    elif dataset_type == 'era':
        filename_pattern = f'fcnv2_fc_era5_{date_str}_0000_m0.nc'
        dataset_directory = 'era5'
    else:
        print(f"Invalid dataset type: {dataset_type}")
        return None

    # Construct the full path
    file_path = os.path.join(base_directory, dataset_type, date_str, 'forecasts', filename_pattern)

    # Check if the file exists
    if os.path.exists(file_path):
        # Read the dataset using xarray
        ds = xr.open_dataset(file_path, engine='netcdf4')
        return ds
    else:
        print(f"File not found: {file_path}")
        return None





########################################## read the ecmwf data and choose the certain variable ####################################################
def read_and_choose_variable_ec(directory, filename_pattern, variable_name, level ):
    # Read data and choose the variable
    files = sorted([f for f in os.listdir(directory) if re.match(filename_pattern, f)])
    variable_data_list = []

    for file in files:
        file_path = os.path.join(directory, file)
        ds = xr.open_dataset(file_path)
        if level is not None:
            variable_data = ds[variable_name][:, level, :, :]
        else:
            variable_data = ds[variable_name]

        variable_data_list.append(variable_data)

    ds_variable_na = xr.DataArray(variable_data_list)

    return ds_variable_na
    
''' # Example usage for temperature at 850 hPa
directory_t850 = '/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams/nk2448/2023_FourCastNet/Yangfan/validation/'
filename_pattern_t850 = r'fc_cf_\d{8}_\d{2}_pl\.nc'
variable_name_t850 = 't'
level_t850 = 1  # Adjust the level as needed, set to None if not applicable

ds_ec_t850_na = read_and_choose_variable_ec(directory_t850, filename_pattern_t850, variable_name_t850, level_t850)'''

'''# Example usage for 2m temperature
directory_t2m = '/lsdf/kit/imk-tro/projects/MOD/Gruppe_Grams/nk2448/2023_FourCastNet/Yangfan/validation/'
filename_pattern_t2m = r'fc_cf_\d{8}_\d{2}_sfc\.nc'
variable_name_t2m = 't2m'
level_t2m = None  # No level for surface variables


ds_ec_t2m_na = read_ecmwf_data(directory_t2m, filename_pattern_t2m, variable_name_t2m, level_t2m)'''



########################################## choose the certain forecast in the target day ####################################################
def process_target_forecast(ds, start_index, step_size, num_iterations):
    'choose the certain date and lead time and get the target forecast'
    ds_target_forecast = []
    for i in range(num_iterations):
        ec_value = ds[(i + 1) * step_size, start_index - i * step_size * 4, :, :]
        ds_target_forecast.append(ec_value)

    ds_target_forecast = xr.DataArray(ds_target_forecast)

    return ds_target_forecast
'''#usage example
start_index = 56
step_size = 1
num_iterations = 15
ds=ds_ec_t2m_box
ds_ec_t2m_box_0629 = process_target_forecast(ds, start_index, step_size, num_iterations)'''



############################################################## calculate the batch difference ##################################################

def calculate_batch_difference(v1, v2, batch_size):
    """
    Calculate the point-wise difference between two arrays (v2 - v1) in batches.

    Parameters:
    - v1: The first input array.
    - v2: The second input array.
    - batch_size: The desired batch size for processing.

    Returns:
    - An array containing the point-wise differences between v2 and v1.
    """
    # Get the shape of the input arrays
    rows, cols = v1.shape
    
    # Initialize the difference array
    difference = np.empty((rows, cols))

    for i in range(0, rows, batch_size):
        for j in range(0, cols, batch_size):
            batch_variable1 = v1[i:i + batch_size, j:j + batch_size]
            batch_variable2 = v2[i:i + batch_size, j:j + batch_size]
            
            # Calculate the difference point by point for the current batch
            batch_difference = batch_variable2 - batch_variable1

            # Update the difference array with the current batch_difference
            difference[i:i + batch_size, j:j + batch_size] = batch_difference

    return difference



####################################################### create 9 subplots for normal colorbar ######################################################

def make_normal_9plots(v1,v2,v3,v4,v5,v6,v7,v8,v9,r1title,r2title,r3title,c1title,c2title,c3title,pathdirect):

    fig, axs = plt.subplots(3, 3, figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
    
    vmax = max(v1.max(), v2.max(), v3.max(), v4.max(),v5.max(), v6.max(), v7.max(), v8.max(),v9.max())
    vmin = min(v1.min(), v2.min(), v3.min(), v4.min(),v5.min(), v6.min(), v7.min(), v8.min(),v9.min())
    plot_kwargs = dict(cmap = "coolwarm", vmin=vmin, vmax=vmax,levels = 20)                                         
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    lat = np.linspace(25, 80, 220)
    lon = np.linspace(-180, -60, 480)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = lon_grid, lat_grid
    #add red star sign
    red_point_lon = -121.50
    red_point_lat = 50.25
    #add patches to show the box range
    rect = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_2 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_3 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_4 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_5 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_6 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_7 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_8 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')



    ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax1.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    contour1 = ax1.contourf(x,y,v1 ,**plot_kwargs,extend ='both')
    ax1.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax1.add_patch(rect)
    ax1.set_title(c1title)
    #ax1.set_xlabel('Longitude')
    ax1.set_ylabel(r1title)



    ax2.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax2.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax2.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.set_title(c2title)
    contour2 = ax2.contourf(x,y,v2,**plot_kwargs)
    ax2.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax2.add_patch(rect_1)
    #cbar = plt.colorbar(contour, ax=ax[0], label='Temperature/K', orientation='horizontal', shrink=0.5)
    #ax2.set_title('ECMWF',loc='left')
    #ax2.set_title('lead time = +42h',loc='right')
    #ax2.set_xlabel('Longitude')
    #ax2.set_ylabel('Latitude')



    ax3.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax3.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax3.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    ax3.set_title(c3title)
    contour3 = ax3.contourf(x,y,v3,**plot_kwargs)
    ax3.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax3.add_patch(rect_2)
    #fourcastnet(ERA5):FIRST:TIME,SECOND:LEAD TIME,THIRD:P2RTURB MEMBERS
    #cbar = plt.colorbar(contour, ax=ax[0], label='Temperature/K', orientation='horizontal', shrink=0.5)
    #phony_dim_3: 21phony_dim_0: 21phony_dim_1: 51phony_dim_2: 57phony_dim_4: 220phony_dim_5: 480
    #ax3.set_title('Forecastnet(ERA5)',loc='left')
    #ax3.set_title('lead time = +42h',loc='right')
    #ax3.set_xlabel('Longitude')
    #ax3.set_ylabel('Latitude')



    ax4.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    ax4.add_feature(cfeature.LAND, facecolor='lightgray')
    ax4.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax4.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax4.contourf(x,y,v4,**plot_kwargs)
    ax4.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax4.add_patch(rect_3)
    #era5: only first dimension needs to be changed, 2021.06.14:00:00-656
    #cbar = plt.colorbar(contour, ax=ax[0], label='Temperature/K orientation='horizontal', shrink=0.5)
    #ax4.set_title('ERA5',loc='left')
    #ax4.set_xlabel('Longitude')
    ax4.set_ylabel(r2title)
    plt.tight_layout()



    ax5.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax5.add_feature(cfeature.COASTLINE)
    ax5.add_feature(cfeature.BORDERS, linestyle=':')
    ax5.add_feature(cfeature.LAND, facecolor='lightgray')
    ax5.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax5.contourf(x,y,v5,**plot_kwargs)
    ax5.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax5.add_patch(rect_4)
    #era5: only first dimension needs to be changed, 2021.06.14:00:00-656
    #cbar = plt.colorbar(contour, ax=ax[0], label='Temperature/K orientation='horizontal', shrink=0.5)
    #ax4.set_title('ERA5',loc='left')
    #ax4.set_xlabel('Longitude')
    #ax4.set_ylabel('Latitude')
    plt.tight_layout()

    ax6.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax6.add_feature(cfeature.COASTLINE)
    ax6.add_feature(cfeature.BORDERS, linestyle=':')
    ax6.add_feature(cfeature.LAND, facecolor='lightgray')
    ax6.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax6.contourf(x,y,v6,**plot_kwargs)
    ax6.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax6.add_patch(rect_5)

    ax7.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax7.add_feature(cfeature.COASTLINE)
    ax7.add_feature(cfeature.BORDERS, linestyle=':')
    ax7.add_feature(cfeature.LAND, facecolor='lightgray')
    ax7.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax7.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax7.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax7.xaxis.set_major_formatter(lon_formatter)
    ax7.yaxis.set_major_formatter(lat_formatter)
    ax7.set_ylabel(r3title)
    contour4 = ax7.contourf(x,y,v7,**plot_kwargs)
    ax7.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax7.add_patch(rect_6)

    ax8.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax8.add_feature(cfeature.COASTLINE)
    ax8.add_feature(cfeature.BORDERS, linestyle=':')
    ax8.add_feature(cfeature.LAND, facecolor='lightgray')
    ax8.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax8.set_xticks(np.arange(leftlon+20,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax8.xaxis.set_major_formatter(lon_formatter)
    ax8.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax8.contourf(x,y,v8,**plot_kwargs)
    ax8.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax8.add_patch(rect_7)


    ax9.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax9.add_feature(cfeature.COASTLINE)
    ax9.add_feature(cfeature.BORDERS, linestyle=':')
    ax9.add_feature(cfeature.LAND, facecolor='lightgray')
    ax9.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax9.set_xticks(np.arange(leftlon+20,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax9.xaxis.set_major_formatter(lon_formatter)
    ax9.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax9.contourf(x,y,v9,**plot_kwargs)
    ax9.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax9.add_patch(rect_8)


    cbar = fig.colorbar(contour1, ax=axs, extend = 'both',label='(K)', orientation='horizontal',shrink=0.5,pad=0.05)
    plt.savefig(pathdirect, bbox_inches='tight',dpi=100)

#################################################### create 9 subplots with colorbar(0 in the middle) ############################################

def make_middle_9plots(v1,v2,v3,v4,v5,v6,v7,v8,v9,r1title,r2tile,r3title,c1title,c2title,c3title,pathdirect):
    
    fig, axs = plt.subplots(3, 3, figsize=(14, 13), subplot_kw={'projection': ccrs.PlateCarree()})
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = axs
    lat = np.linspace(25, 80, 220)
    lon = np.linspace(-180, -60, 480)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = lon_grid, lat_grid
    cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue', 
                                                     colors =[(0, 0, 1), 
                                                              (1, 1., 1), 
                                                              (1, 0, 0)],
                                                     N=22,
                                                     )
    
    vmax = max(v1.max(), v2.max(), v3.max(), v4.max(),v5.max(), v6.max(), v7.max(), v8.max(),v9.max())
    vmin = min(v1.min(), v2.min(), v3.min(), v4.min(),v5.min(), v6.min(), v7.min(), v8.min(),v9.min())
    
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plot_kwargs = dict(cmap=cmap, norm = norm,levels = 20)
    
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #add star sign
    red_point_lon = -121.50
    red_point_lat = 50.25
    
    #add patch to show the box area 
    rect = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_1 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_2 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_3 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_4 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_5 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_6 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_7 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    rect_8 = patches.Rectangle((-131.5,40.25),20,20, linewidth=1, edgecolor='black', facecolor='none')
    
    
    
    
    
    ax1.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax1.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    contour1 = ax1.contourf(x,y,v1,**plot_kwargs,extend = 'both')
    ax1.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax1.add_patch(rect)
    ax1.set_title(c1title)
    ax1.set_ylabel(r1title)
    
    
    
    ax2.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax2.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax2.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.set_title(c2title)
    contour2 = ax2.contourf(x,y,v2,**plot_kwargs)
    ax2.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax2.add_patch(rect_1)
    
    
    
    ax3.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax3.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax3.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    ax3.set_title(c3title)
    ax3.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    contour3 = ax3.contourf(x,y,v3,**plot_kwargs)
    ax3.add_patch(rect_2)
    
    
    
    
    ax4.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    ax4.add_feature(cfeature.LAND, facecolor='lightgray')
    ax4.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax4.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax4.contourf(x,y,v4,**plot_kwargs)
    ax4.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax4.add_patch(rect_3)
    ax4.set_ylabel(r2title)
    
    
    
    
    
    ax5.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax5.add_feature(cfeature.COASTLINE)
    ax5.add_feature(cfeature.BORDERS, linestyle=':')
    ax5.add_feature(cfeature.LAND, facecolor='lightgray')
    ax5.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax5.contourf(x,y,v5,**plot_kwargs)
    ax5.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax5.add_patch(rect_4)
    
    
    ax6.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax6.add_feature(cfeature.COASTLINE)
    ax6.add_feature(cfeature.BORDERS, linestyle=':')
    ax6.add_feature(cfeature.LAND, facecolor='lightgray')
    ax6.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    #ax4.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax6.contourf(x,y,v6,**plot_kwargs)
    ax6.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax6.add_patch(rect_5)
    
    ax7.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax7.add_feature(cfeature.COASTLINE)
    ax7.add_feature(cfeature.BORDERS, linestyle=':')
    ax7.add_feature(cfeature.LAND, facecolor='lightgray')
    ax7.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax7.set_xticks(np.arange(leftlon,rightlon+10,20), crs=ccrs.PlateCarree())
    ax7.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax7.xaxis.set_major_formatter(lon_formatter)
    ax7.yaxis.set_major_formatter(lat_formatter)
    ax7.set_ylabel(r3title)
    contour4 = ax7.contourf(x,y,v7,**plot_kwargs)
    ax7.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax7.add_patch(rect_6)
    
    ax8.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax8.add_feature(cfeature.COASTLINE)
    ax8.add_feature(cfeature.BORDERS, linestyle=':')
    ax8.add_feature(cfeature.LAND, facecolor='lightgray')
    ax8.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax8.set_xticks(np.arange(leftlon+20,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax8.xaxis.set_major_formatter(lon_formatter)
    ax8.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax8.contourf(x,y,v8,**plot_kwargs)
    ax8.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax8.add_patch(rect_7)
    
    
    ax9.set_extent([leftlon, rightlon, lowerlat, upperlat], crs=ccrs.PlateCarree())
    ax9.add_feature(cfeature.COASTLINE)
    ax9.add_feature(cfeature.BORDERS, linestyle=':')
    ax9.add_feature(cfeature.LAND, facecolor='lightgray')
    ax9.add_feature(cfeature.OCEAN, facecolor='w')
    leftlon, rightlon, lowerlat, upperlat = (-180,-60,25,80)#define map extent
    ax9.set_xticks(np.arange(leftlon+20,rightlon+10,20), crs=ccrs.PlateCarree())
    #ax5.set_yticks(np.arange(lowerlat,upperlat,20), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    lon_formatter = cticker.LongitudeFormatter()
    ax9.xaxis.set_major_formatter(lon_formatter)
    ax9.yaxis.set_major_formatter(lat_formatter)
    contour4 = ax9.contourf(x,y,v9,**plot_kwargs)
    ax9.plot(red_point_lon, red_point_lat, 'r*', markersize=7)
    ax9.add_patch(rect_8)
    
    #norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    #norm = mcolors.TwoSlopeNorm(vcenter=0)
    #pc = plt.pcolormesh(axs, norm=norm, cmap=cmap)
    #cb = plt.colorbar(pc,orientation='horizontal',shrink=0.5,label='Temperature/K',extend='both')
    #cbar.set_ticks()
    #cb.add_lines(CS) 
    #im = ax9.pcolormesh(x, y, difference_fcn_era_7, cmap=cmap, norm=norm)
    #fig.colorbar(im, ax=axs)
    #tick_positions = [-abs(vmin), -abs(vmin)/2, 0, abs(vmax)/2, abs(vmax)]
    #tick_labels = [f'{val:.1f}' for val in tick_positions]
    plt.tight_layout()
    cbar = plt.colorbar(contour1, ax=axs, orientation='horizontal', cmap=cmap, norm=norm, shrink=0.5,label='(K)',pad = 0.05,extend = 'both')
    plt.savefig(pathdirect, bbox_inches='tight',dpi=100)
    #cbar.set_ticklabels(tick_labels)
    #plt.colorbar()
    #cbar = fig.colorbar(contour1, ax=axs, extend = 'both',label='Temperature/K', orientation='vertical',shrink=0.5)
























