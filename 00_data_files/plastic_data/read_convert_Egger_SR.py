#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:28:21 2022

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

def calculate_rise_velocity_Dietrich(l,rho_p=1010,rho_sw=1025):
    w_b_arr = []
    d_star_arr = []
    for r_tot in l:
        # rho_sw = 1029
        g = 9.81
        sw_kin_visc = 1e-6
        
        dn = r_tot  # equivalent spherical diameter [m], calculated from Dietrich (1982) from A = pi/4 * dn**2
        delta_rho = (rho_p - rho_sw) / rho_sw  # normalised difference in density between total plastic+bf and seawater[-]
        dstar = (abs(rho_p - rho_sw) * g * dn ** 3.) / (rho_sw * sw_kin_visc ** 2.)  # [-]
    
        if dstar > 5e9:
            w_star = 265000
        elif dstar < 0.05:
            w_star = (dstar ** 2.) * 1.71E-4
        else:
            w_star = 10. ** (-3.76715 + (1.92944 * math.log10(dstar)) - (0.09815 * math.log10(dstar) ** 2.) - (
                        0.00575 * math.log10(dstar) ** 3.) + (0.00056 * math.log10(dstar) ** 4.))
        
        if delta_rho > 0:  # sinks
            w_b = -(g * sw_kin_visc * w_star * delta_rho) ** (1. / 3.)
        else:  # rises
            a_del_rho = delta_rho * -1.
            w_b = (g * sw_kin_visc * w_star * a_del_rho) ** (1. / 3.)  # m s-1    
    
        w_b_arr.append(w_b)
        d_star_arr.append(dstar)
        
    return np.array(w_b_arr), np.array(d_star_arr)
def u_fric_air_Thorpe(U_10): #Thorpe 2003
    C_d = 1e-3*(0.75+0.067*U_10)
    return np.sqrt(C_d*U_10**2)

def ua_to_uw(u_a,ratio_rho_aw):
    return(np.sqrt(ratio_rho_aw*u_a**2))
    
def u_fric_water_Thorpe(U_10):
    u_fric_a = u_fric_air_Thorpe(U_10)
    return ua_to_uw(u_fric_a,1.2e-3)

def H_s_Thorpe(U_10):
    return 0.96*(1/9.81)*35**(3/2)*u_fric_air_Thorpe(U_10)**2

def correction_factor(wb,d,U_10):
    if U_10 == 0:
        f = 1.
    else:
        k = 0.4
        H_s = H_s_Thorpe(U_10)
        A_0 = 1.5*u_fric_water_Thorpe(U_10)*k*H_s
        f = 1/(1 - np.exp(-d * wb * (1/(A_0)) )) 
    return f

def beaufort_to_ms(B):
    return 0.836*B**(3/2)

data = pd.read_excel('/Users/kaandorp/Data/PlasticData/Egger2020/Egger2020_SR_SI.xlsx')

row_end = 18
lons = pd.to_numeric(data.loc[:,'LON'],errors='coerce')
mask_data = ~np.isnan(lons)
mask_data[row_end:] = False
lons = lons[mask_data]
lats = pd.to_numeric(data.loc[:,'LAT'],errors='coerce')[mask_data]
dates = pd.to_datetime(data.loc[:,'Date'],errors='coerce')[mask_data]
wind_mag = pd.to_numeric(data.loc[:,'Sea state [Beaufort]'],errors='coerce')[mask_data]
years = np.array([date_.year for date_ in dates])
months = np.array([date_.month for date_ in dates])
days = np.array([date_.day for date_ in dates])
depth = 0.15*np.ones(len(lons))


size_classes = [[5e-4,1.5e-3],[1.5e-3,5e-3],[5e-3,1.5e-2],[1.5e-2,5e-2]]
size_midpoints = np.array([0.5*(i1+i2) for (i1,i2) in size_classes])
w_b, _ = calculate_rise_velocity_Dietrich(size_midpoints,rho_p=1010,rho_sw=1025)

col_tot_n0 = 'Total [#/km2]'
col_tot_n1 = 'Unnamed: 23'
col_tot_n2 = 'Unnamed: 24'
col_tot_n3 = 'Unnamed: 25'
N_raw = {}
N_raw[str(size_classes[0])] = pd.to_numeric(data.loc[mask_data,col_tot_n0],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[1])] = pd.to_numeric(data.loc[mask_data,col_tot_n1],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[2])] = pd.to_numeric(data.loc[mask_data,col_tot_n2],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[3])] = pd.to_numeric(data.loc[mask_data,col_tot_n3],errors='coerce').fillna(0).values / 1e6

col_tot_m0 = 'Total [g/km2]'
col_tot_m1 = 'Unnamed: 45'
col_tot_m2 = 'Unnamed: 46'
col_tot_m3 = 'Unnamed: 47'
M_raw = {}
M_raw[str(size_classes[0])] = pd.to_numeric(data.loc[mask_data,col_tot_m0],errors='coerce').fillna(0).values / 1e6 # km2 -> m2
M_raw[str(size_classes[1])] = pd.to_numeric(data.loc[mask_data,col_tot_m1],errors='coerce').fillna(0).values / 1e6
M_raw[str(size_classes[2])] = pd.to_numeric(data.loc[mask_data,col_tot_m2],errors='coerce').fillna(0).values / 1e6
M_raw[str(size_classes[3])] = pd.to_numeric(data.loc[mask_data,col_tot_m3],errors='coerce').fillna(0).values / 1e6

upper_cell_heights = [1., 5.] #transition matrix height of the upper layer

df_tmp1 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,
                        'ParentEventID':'Egger_2020_SR'}).reset_index(drop=True)
df_tmp2 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,
                        'ParentEventID':'Egger_2020_SR'}).reset_index(drop=True)

h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_tmp1['h%i' % res_] = vect.geo_to_h3(df_tmp1['decimalLatitude'],df_tmp1['decimalLongitude'],res_)
    df_tmp2['h%i' % res_] = vect.geo_to_h3(df_tmp2['decimalLatitude'],df_tmp2['decimalLongitude'],res_)
df_tmp1['wind_mag'] = beaufort_to_ms(wind_mag.values)
df_tmp2['wind_mag'] = beaufort_to_ms(wind_mag.values)


col_names = []
for height_ in upper_cell_heights:
    for i2,size_ in enumerate(size_classes):
        # data_use['measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = ""
        df_tmp1['measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = ""
        df_tmp2['measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = ""
        col_names.append('measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1]))
        
for height_ in upper_cell_heights:
    
    for i2,size_ in enumerate(size_classes):

        N_tot = []
        M_tot = []
        for i1 in range(len(lons)):

            f_Nm = correction_factor(w_b[i2], height_, beaufort_to_ms(wind_mag.iloc[i1]) )
            f_net = correction_factor(w_b[i2], depth[i1], beaufort_to_ms(wind_mag.iloc[i1]))
            
            N_Nm = N_raw[str(size_)][i1] * (f_net / f_Nm) #get the upper N meters total concentration
            M_Nm =  M_raw[str(size_)][i1] * (f_net / f_Nm)
            
            concentration_Nm = N_Nm / height_ #assuming uniform distribution in the upper 5meters, concentration in m-3 
            concentration_Mm = M_Nm / height_
        
            df_tmp1.loc[i1,'measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = concentration_Nm
            df_tmp2.loc[i1,'measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = concentration_Mm
    
df_tmp1.loc[:,'measurementValue_vol_1m_%5.5fm_%5.5fm' % (size_classes[0][0],size_classes[-1][-1])] = df_tmp1.loc[:,col_names[0:4]].sum(axis=1)
df_tmp1.loc[:,'measurementValue_vol_5m_%5.5fm_%5.5fm' % (size_classes[0][0],size_classes[-1][-1])] = df_tmp1.loc[:,col_names[4:8]].sum(axis=1)
df_tmp2.loc[:,'measurementValue_vol_1m_%5.5fm_%5.5fm' % (size_classes[0][0],size_classes[-1][-1])] = df_tmp2.loc[:,col_names[0:4]].sum(axis=1)
df_tmp2.loc[:,'measurementValue_vol_5m_%5.5fm_%5.5fm' % (size_classes[0][0],size_classes[-1][-1])] = df_tmp2.loc[:,col_names[4:8]].sum(axis=1)


df_tmp1['measurementUnit_vol'] = 'nm-3'
df_tmp2['measurementUnit_vol'] = 'gm-3'

data_use = pd.concat([df_tmp1,df_tmp2])
# data_use.to_csv('converted_Egger2020SR_surface.csv')
    
#%%
# depth_layers = np.array([0,5,50,500,np.inf])
# index_remove_depth = np.array([int(0),int(1e17),int(2e17),int(3e17),int(4e17)])

# col_names_n = ['Total [#/km2]','Unnamed: 23','Unnamed: 24','Unnamed: 25']
# col_names_m = ['Total [g/km2]','Unnamed: 45','Unnamed: 46','Unnamed: 47']

#col_names changed to (hard plastic [H]) fragments:
col_names_n = ['0.05-0.15cm [#/km2]','0.15-0.5cm [#/km2]','0.5-1.5cm [#/km2]','1.5-5cm [#/km2]']
col_names_m = ['0.05-0.15cm [g/km2]','0.15-0.5cm [g/km2]','0.5-1.5cm [g/km2]','1.5-5cm [g/km2]']

df_name = 'converted_Egger2020SR_depth_frag.csv' 
# for d_ in depth_layers[:-1]:
#     df_name = df_name + '_%im' % d_
# df_name = df_name + '.csv'

row_start = 42
# row_end 
dates = pd.to_datetime(data.loc[:,'Date'],errors='coerce')#[mask_data]
mask_data = ~np.isnat(dates)
mask_data[:row_start] = False
dates = dates[mask_data]
volume = pd.to_numeric(data.loc[:,'Distance [km]'],errors='coerce')[mask_data]
depth = pd.to_numeric(data.loc[:,'Sea state [Beaufort]'],errors='coerce')[mask_data]

lon_mean = []
lat_mean = []
for date_ in dates:
    mask_date = data_use['eventDate'] == date_
    if mask_date.sum() == 0:
        mask_date[np.argmin(np.abs(data_use['eventDate']-date_))] = True #if no exact date match present, take the closest one
        
    lon_mean.append(data_use.loc[mask_date,'decimalLongitude'].mean())
    lat_mean.append(data_use.loc[mask_date,'decimalLatitude'].mean())

years = np.array([date_.year for date_ in dates])
months = np.array([date_.month for date_ in dates])
days = np.array([date_.day for date_ in dates])

df_tmp1 = pd.DataFrame(data={'decimalLatitude':lat_mean,'decimalLongitude':lon_mean,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,'Volume':volume,
                        'ParentEventID':'Egger_2020_SR'}).reset_index(drop=True)
df_tmp2 = pd.DataFrame(data={'decimalLatitude':lat_mean,'decimalLongitude':lon_mean,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,'Volume':volume,
                        'ParentEventID':'Egger_2020_SR'}).reset_index(drop=True)

h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_tmp1['h%i' % res_] = vect.geo_to_h3(df_tmp1['decimalLatitude'],df_tmp1['decimalLongitude'],res_) #- index_remove_depth[np.digitize(depth,depth_layers)-1]
    df_tmp2['h%i' % res_] = vect.geo_to_h3(df_tmp2['decimalLatitude'],df_tmp2['decimalLongitude'],res_) #- index_remove_depth[np.digitize(depth,depth_layers)-1]

col_names = []
m_no_detlim = np.zeros(mask_data.sum())
n_no_detlim = np.zeros(mask_data.sum())
for i2,size_ in enumerate(size_classes):
        # data_use['measurementValue_vol_%im_%5.5fm_%5.5fm' % (height_,size_[0],size_[1])] = ""
    n_s = data.loc[:,col_names_n[i2]][mask_data]
    mask_0_nan = (n_s.isna()) | (n_s == 0)
    
    n_s[mask_0_nan] = 1./volume[mask_0_nan]
    detection_limit = np.zeros(len(n_s),dtype=int)
    detection_limit[mask_0_nan] = 1
    n_no_detlim[~mask_0_nan] += n_s[~mask_0_nan].values.astype(float)
    
    m_s = data.loc[:,col_names_m[i2]][mask_data]
    med_mass = np.median(m_s[~mask_0_nan] / n_s[~mask_0_nan])
    print('Median mass (%f-%fm): %f g' % (size_[0],size_[1],med_mass))
    
    m_s[mask_0_nan] = med_mass/volume[mask_0_nan]
    m_no_detlim[~mask_0_nan] += m_s[~mask_0_nan].values.astype(float)
    
    df_tmp1['measurementValue_vol_%5.5fm_%5.5fm' % (size_[0],size_[1])] = n_s.values
    df_tmp1['measurementValue_vol_%5.5fm_%5.5fm_detlim.' % (size_[0],size_[1])] = detection_limit
    
    df_tmp2['measurementValue_vol_%5.5fm_%5.5fm' % (size_[0],size_[1])] = m_s.values
    df_tmp2['measurementValue_vol_%5.5fm_%5.5fm_detlim.' % (size_[0],size_[1])] = detection_limit
  
    col_names.append('measurementValue_vol_%5.5fm_%5.5fm' % (size_[0],size_[1]))
    

df_tmp1.loc[:,'measurementValue_vol_%5.5fm_%5.5fm_detlim' % (size_classes[0][0],size_classes[-1][-1])] = np.nansum(df_tmp1.loc[:,col_names],axis=1)
df_tmp2.loc[:,'measurementValue_vol_%5.5fm_%5.5fm_detlim' % (size_classes[0][0],size_classes[-1][-1])] = np.nansum(df_tmp2.loc[:,col_names],axis=1)
df_tmp1.loc[:,'measurementValue_vol_%5.5fm_%5.5fm_nodetlim' % (size_classes[0][0],size_classes[-1][-1])] = n_no_detlim
df_tmp2.loc[:,'measurementValue_vol_%5.5fm_%5.5fm_nodetlim' % (size_classes[0][0],size_classes[-1][-1])] = m_no_detlim

df_tmp1['measurementUnit_vol'] = 'nm-3'
df_tmp2['measurementUnit_vol'] = 'gm-3'

data_use = pd.concat([df_tmp1,df_tmp2])
data_use.to_csv(df_name)

#%%
i_r = [0,7,15,30,38,45]
for i1 in range(len(i_r)-1):
    fig,ax = plt.subplots(1,2)
    mask_date = np.zeros(len(df_tmp1),dtype=bool)
    # mask_date = (df_tmp1['eventDate'] == date_)
    mask_date[i_r[i1]:i_r[i1+1]] = True
    ax[0].semilogx(df_tmp1.loc[mask_date,'measurementValue_vol_0.00050m_0.05000m_nodetlim'],-df_tmp1.loc[mask_date,'Depth'],'ko')
    ax[1].semilogx(df_tmp2.loc[mask_date,'measurementValue_vol_0.00050m_0.05000m_nodetlim'],-df_tmp2.loc[mask_date,'Depth'],'ko')
    ax[0].set_ylabel('-Depth')
    ax[0].set_xlabel('n m-3')
    ax[1].set_xlabel('g m-3')    
    fig.suptitle('Station %i, no detection limit set' % i1)
    fig.tight_layout()
    
    fig2,ax2 = plt.subplots(1,2)
    ax2[0].semilogx(df_tmp1.loc[mask_date,'measurementValue_vol_0.00050m_0.05000m_detlim'],-df_tmp1.loc[mask_date,'Depth'],'ko')
    ax2[1].semilogx(df_tmp2.loc[mask_date,'measurementValue_vol_0.00050m_0.05000m_detlim'],-df_tmp2.loc[mask_date,'Depth'],'ko')
    ax2[0].set_ylabel('-Depth')
    ax2[0].set_xlabel('n m-3')
    ax2[1].set_xlabel('g m-3')       
    fig2.suptitle('Station %i, detection limit set' % i1)
    fig2.tight_layout()
# N0_tot = np.array(N0_tot)    
# N1_tot = np.array(N1_tot)    
# N2_tot = np.array(N2_tot)    
# N3_tot = np.array(N3_tot)    
# N_tot = N0_tot+N1_tot+N2_tot+N3_tot

# M0_tot = np.array(M0_tot)    
# M1_tot = np.array(M1_tot)    
# M2_tot = np.array(M2_tot)    
# M3_tot = np.array(M3_tot)    
# M_tot = M0_tot+M1_tot+M2_tot+M3_tot

# N_tot = N0+N1+N2+N3

# lons[mask_data]

# df_out = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
#                        'ParentEventID','measurementValue','measurementUnit'])

# df_out[]