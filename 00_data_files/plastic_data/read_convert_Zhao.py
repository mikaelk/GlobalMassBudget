#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:11:10 2022

@author: kaandorp
"""
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import math
from h3.unstable import vect

data = pd.read_excel('/Users/kaandorp/Data/PlasticData/Zhao_Zettler_2022.xlsx')

df_name = df_name = 'converted_Zhao2022_depth.csv' 

lons = pd.to_numeric(data.loc[:,'Lon'],errors='coerce')
mask_data = ~np.isnan(lons)
# mask_data[row_end:] = False
lons = lons[mask_data]
lats = pd.to_numeric(data.loc[:,'Lat'],errors='coerce')[mask_data]
dates = pd.to_datetime(data.loc[:,'Date'],errors='coerce')[mask_data]
# wind_mag = pd.to_numeric(data.loc[:,'Sea state [Beaufort]'],errors='coerce')[mask_data]
years = np.array([date_.year for date_ in dates])
months = np.array([date_.month for date_ in dates])
days = np.array([date_.day for date_ in dates])
depth = pd.to_numeric(data.loc[:,'Depth'],errors='coerce')[mask_data]
volume = pd.to_numeric(data.loc[:,'Volume'],errors='coerce')[mask_data]
n = pd.to_numeric(data.loc[:,'pieces'],errors='coerce')[mask_data]
n_s = n/volume

mass_c = pd.to_numeric(data.loc[:,'microgram m-3'],errors='coerce')[mask_data] / 1e6
num_c = pd.to_numeric(data.loc[:,'n m-3'],errors='coerce')[mask_data]
mean_mass = np.nanmean(mass_c / num_c)


df_tmp1 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,'Volume':volume,
                        'ParentEventID':'Zhao_2022'}).reset_index(drop=True)
df_tmp2 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,'Volume':volume,
                        'ParentEventID':'Zhao_2022'}).reset_index(drop=True)

h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_tmp1['h%i' % res_] = vect.geo_to_h3(df_tmp1['decimalLatitude'],df_tmp1['decimalLongitude'],res_) #- index_remove_depth[np.digitize(depth,depth_layers)-1]
    df_tmp2['h%i' % res_] = vect.geo_to_h3(df_tmp2['decimalLatitude'],df_tmp2['decimalLongitude'],res_) #- index_remove_depth[np.digitize(depth,depth_layers)-1]

size_classes = [[0.3e-3,5e-3]]
for i2,size_ in enumerate(size_classes):
        # data_use['measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = ""
    # n_s = data.loc[:,col_names_n[i2]][mask_data]
    mask_0_nan = (n_s.isna()) | (n_s == 0)
    
    n_s[mask_0_nan] = 1./volume[mask_0_nan]
    detection_limit = np.zeros(len(n_s),dtype=int)
    detection_limit[mask_0_nan] = 1
    # n_no_detlim[~mask_0_nan] += n_s[~mask_0_nan].values.astype(float)
    
    m_s = n_s*mean_mass
    # med_mass = np.median(m_s[~mask_0_nan] / n_s[~mask_0_nan])
    # print('Median mass (%f-%fm): %f g' % (size_[0],size_[1],med_mass))
    
    # m_s[mask_0_nan] = med_mass/volume[mask_0_nan]
    # m_no_detlim[~mask_0_nan] += m_s[~mask_0_nan].values.astype(float)
    
    df_tmp1['measurementValue_vol_%5.5fm_%5.5fm' % (size_[0],size_[1])] = n_s.values
    df_tmp1['measurementValue_vol_%5.5fm_%5.5fm_detlim.' % (size_[0],size_[1])] = detection_limit
    
    df_tmp2['measurementValue_vol_%5.5fm_%5.5fm' % (size_[0],size_[1])] = m_s.values
    df_tmp2['measurementValue_vol_%5.5fm_%5.5fm_detlim.' % (size_[0],size_[1])] = detection_limit
    
df_tmp1['measurementUnit_vol'] = 'nm-3'
df_tmp2['measurementUnit_vol'] = 'gm-3'

data_use = pd.concat([df_tmp1,df_tmp2])
data_use.to_csv(df_name)


fig,ax = plt.subplots(1,2)
# mask_date = np.zeros(len(df_tmp1),dtype=bool)
# mask_date = (df_tmp1['eventDate'] == date_)
# mask_date[i_r[i1]:i_r[i1+1]] = True
ax[0].semilogx(df_tmp1.loc[(detection_limit==0),'measurementValue_vol_0.00030m_0.00500m'],-df_tmp1.loc[(detection_limit==0),'Depth'],'ko')
ax[1].semilogx(df_tmp2.loc[(detection_limit==0),'measurementValue_vol_0.00030m_0.00500m'],-df_tmp2.loc[(detection_limit==0),'Depth'],'ko',label='particles measured')
ax[0].semilogx(df_tmp1.loc[(detection_limit==1),'measurementValue_vol_0.00030m_0.00500m'],-df_tmp1.loc[(detection_limit==1),'Depth'],'ro')
ax[1].semilogx(df_tmp2.loc[(detection_limit==1),'measurementValue_vol_0.00030m_0.00500m'],-df_tmp2.loc[(detection_limit==1),'Depth'],'ro',label='detection limit')
ax[0].set_ylabel('-Depth')
ax[0].set_xlabel('n m-3')
ax[1].set_xlabel('g m-3')    
fig.suptitle('Zhao et al. (2022)')
fig.tight_layout()
fig.legend()
