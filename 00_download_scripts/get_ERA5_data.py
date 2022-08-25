#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:57:57 2020

@author: kaandorp
"""
import os
import datetime
import cdsapi
import pandas as pd

if os.environ['USER'] == 'kaandorp': # desktop
    download_folder = '/Users/kaandorp/Data/Temp/kaand004/ERA5'
elif os.environ['USER'] == 'kaand004': #lorenz
    download_folder = '/storage/shared/oceanparcels/output_data/data_Mikael/ERA5'    
        
c = cdsapi.Client()

def download_ERA5_data_daily(date_,variables):
    
    file_str = os.path.join(download_folder,'ERA5_global_wind_waves_' + str(date_.date()) + '.nc')
    
    if not os.path.exists(file_str):
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': str(date_.year),
                'month': '%2.2i' %date_.month,
                'day': '%2.2i' %date_.day,
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
            },
            file_str)  

def download_ERA5_data(date_,name_,variables):
    
    file_str = os.path.join(download_folder, name_ + str(date_.date()) + '.nc')
    
    if not os.path.exists(file_str):
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': str(date_.year),
                'month': '%2.2i' %date_.month,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
            },
            file_str)  
            
            



# variables = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_wave_period','peak_wave_period',
#              'u_component_stokes_drift', 'v_component_stokes_drift']
variables = ['mean_wave_period','peak_wave_period','u_component_stokes_drift', 'v_component_stokes_drift']

dates = pd.date_range(datetime.date(2019,1,1),datetime.date(2019,12,1),freq='MS')

for date_ in dates:
    
    download_ERA5_data(date_,'ERA5_global_waves_monthly_',variables)


