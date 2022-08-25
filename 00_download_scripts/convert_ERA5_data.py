import xarray as xr
import glob
import numpy as np
from glob import glob
import os

def sort_and_zero_data(data_,i_sort_lat,i_sort_lon):

    tmp_1 = data_[:,:,i_sort_lon]
    tmp_2 = tmp_1[:,i_sort_lat,:]

    mask = np.isnan(tmp_2)
    tmp_2[mask] = 0.
    return tmp_2
    
if os.environ['USER'] == 'kaandorp': # desktop
    download_folder = '/Users/kaandorp/Data/Temp/kaand004/ERA5'
elif os.environ['USER'] == 'kaand004': #lorenz
    download_folder = '/storage/shared/oceanparcels/output_data/data_Mikael/ERA5/waves'    
        
    
files = sorted(glob(os.path.join(download_folder,'ERA5_global_waves_monthly_201[5-7]*')))

for file_ in files:
    
    data_ = xr.open_dataset(file_)
    
    lons = data_['longitude'].data
    lons[lons>180] -= 360
    i_sort_lon = np.argsort(lons)
    lats = data_['latitude'].data
    i_sort_lat = np.argsort(lats)
                     
    var_mwp = sort_and_zero_data(data_['mwp'].data,i_sort_lat,i_sort_lon)
    var_pp1d = sort_and_zero_data(data_['pp1d'].data,i_sort_lat,i_sort_lon)
    var_ust = sort_and_zero_data(data_['ust'].data,i_sort_lat,i_sort_lon)
    var_vst = sort_and_zero_data(data_['vst'].data,i_sort_lat,i_sort_lon)
    
    lons_sorted = lons[i_sort_lon]
    lats_sorted = lats[i_sort_lat]
    
    ds = xr.Dataset(
    {"mwp": (("time", "lat", "lon"), var_mwp ),
      "pp1d": (("time", "lat", "lon"), var_pp1d ),
      "ust": (("time", "lat", "lon"), var_ust ),
      "vst": (("time", "lat", "lon"), var_vst ),
      "explanation": 'wave data (mean wave period, peak wave period, Stokes drift) converted from ERA5 data to have lon from -180 to 180, lats sorted from south to north, and zeros instead of NaN'},
    coords={
        "lon": lons_sorted,
        "lat": lats_sorted,
        "time": data_['time'].data,
    },
    )   
   
    encoding_ = {"mwp": {"dtype": "float32"},
                 "pp1d": {"dtype": "float32"},
                "ust": {"dtype": "float32"},
                 "vst": {"dtype": "float32"}}

    str_split = '201'
    split_ = file_.split(str_split)
    file_out = split_[0] + 'converted_' + str_split + split_[1]
    ds.to_netcdf(file_out)  
    
    print(file_)
    
    
#%% wind data
if os.environ['USER'] == 'kaandorp': # desktop
    download_folder = '/Users/kaandorp/Data/Temp/kaand004/ERA5'
elif os.environ['USER'] == 'kaand004': #lorenz
    download_folder = '/storage/shared/oceanparcels/output_data/data_Mikael/ERA5/wind'    
        
    
files = sorted(glob(os.path.join(download_folder,'ERA5_global_wind_monthly_201[5-9]*')))

for file_ in files:
    
    data_ = xr.open_dataset(file_)
    
    lons = data_['longitude'].data
    lons[lons>180] -= 360
    i_sort_lon = np.argsort(lons)
    lats = data_['latitude'].data
    i_sort_lat = np.argsort(lats)
                     
    var_u10 = sort_and_zero_data(data_['u10'].data,i_sort_lat,i_sort_lon)
    var_v10 = sort_and_zero_data(data_['v10'].data,i_sort_lat,i_sort_lon)

    lons_sorted = lons[i_sort_lon]
    lats_sorted = lats[i_sort_lat]
    
    ds = xr.Dataset(
    {"u10": (("time", "lat", "lon"), var_u10 ),
      "v10": (("time", "lat", "lon"), var_v10 ),
      "explanation": 'wind data (u10,v10) converted from ERA5 data to have lon from -180 to 180, lats sorted from south to north, and zeros instead of NaN'},
    coords={
        "lon": lons_sorted,
        "lat": lats_sorted,
        "time": data_['time'].data,
    },
    )   
   
    encoding_ = {"u10": {"dtype": "float32"},
                 "v10": {"dtype": "float32"}}

    str_split = '201'
    split_ = file_.split(str_split)
    file_out = split_[0] + 'converted_' + str_split + split_[1]
    ds.to_netcdf(file_out)  
    
    print(file_)