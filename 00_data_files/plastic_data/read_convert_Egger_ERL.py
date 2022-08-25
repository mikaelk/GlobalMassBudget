#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:19:28 2022

@author: kaandorp
"""
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

data = pd.read_excel('/Users/kaandorp/Data/PlasticData/Egger2020/Egger2020_EnvResLett_SI.xlsx')

# row_end = 18
lons = pd.to_numeric(data.loc[:,'LON'],errors='coerce')
mask_data = ~np.isnan(lons)
# mask_data[row_end:] = False
lons = lons[mask_data]
lats = pd.to_numeric(data.loc[:,'LAT'],errors='coerce')[mask_data]
dates = pd.to_datetime(data.loc[:,'Date'],errors='coerce')[mask_data]
wind_mag = pd.to_numeric(data.loc[:,'Sea state [Beaufort]'],errors='coerce')[mask_data]
years = np.array([date_.year for date_ in dates])
months = np.array([date_.month for date_ in dates])
days = np.array([date_.day for date_ in dates])
depth = 0.15*np.ones(len(lons))
mask_alaska = lats > 46
depth[mask_alaska] = 0.4

size_classes = [[5e-4,1.5e-3],[1.5e-3,5e-3],[5e-3,1.5e-2],[1.5e-2,5e-2]]
size_midpoints = np.array([0.5*(i1+i2) for (i1,i2) in size_classes])
w_b, _ = calculate_rise_velocity_Dietrich(size_midpoints,rho_p=1010,rho_sw=1025)
#%%
col_tot_n0 = 'Total [#/km2]'
col_tot_n1 = 'Unnamed: 22'
col_tot_n2 = 'Unnamed: 23'
col_tot_n3 = 'Unnamed: 24'
N_raw = {}
N_raw[str(size_classes[0])] = pd.to_numeric(data.loc[mask_data,col_tot_n0],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[1])] = pd.to_numeric(data.loc[mask_data,col_tot_n1],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[2])] = pd.to_numeric(data.loc[mask_data,col_tot_n2],errors='coerce').fillna(0).values / 1e6
N_raw[str(size_classes[3])] = pd.to_numeric(data.loc[mask_data,col_tot_n3],errors='coerce').fillna(0).values / 1e6

col_tot_m0 = 'Total [g/km2]'
col_tot_m1 = 'Unnamed: 44'
col_tot_m2 = 'Unnamed: 45'
col_tot_m3 = 'Unnamed: 46'
M_raw = {}
M_raw[str(size_classes[0])] = pd.to_numeric(data.loc[mask_data,col_tot_m0],errors='coerce').fillna(0).values / 1e6 # km2 -> m2
M_raw[str(size_classes[1])] = pd.to_numeric(data.loc[mask_data,col_tot_m1],errors='coerce').fillna(0).values / 1e6
M_raw[str(size_classes[2])] = pd.to_numeric(data.loc[mask_data,col_tot_m2],errors='coerce').fillna(0).values / 1e6
M_raw[str(size_classes[3])] = pd.to_numeric(data.loc[mask_data,col_tot_m3],errors='coerce').fillna(0).values / 1e6


upper_cell_heights = [1., 5.] #transition matrix height of the upper layer

df_tmp1 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,
                        'ParentEventID':'Egger_2020_ERL'}).reset_index(drop=True)
df_tmp2 = pd.DataFrame(data={'decimalLatitude':lats,'decimalLongitude':lons,'eventDate':dates,
                              'Year':years,'Month':months,'Day':days,'Depth':depth,
                        'ParentEventID':'Egger_2020_ERL'}).reset_index(drop=True)

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
#%%
# data_use = pd.concat([df_tmp1,df_tmp2])
# data_use.to_csv('converted_Egger2020_ERL_surface.csv')
from scipy.optimize import curve_fit

def lin_fn(x,a,b):
    return a*x+b

l0 = 0.4096 #TODO: change back to 0.1024
k_arr = np.arange(0,12)
l_arr = l0 / (2**k_arr)
l_mid = 10**(.5*(np.log10(l_arr[1:])+np.log10(l_arr[:-1])))

l_Egger = np.unique(np.array(size_classes))
l_Egger_mid = 10**(.5*(np.log10(l_Egger[1:])+np.log10(l_Egger[:-1])))

plt.figure()
plt.loglog(l_mid,np.ones(len(l_mid)),'ko')
plt.loglog(np.unique(np.array(size_classes)),np.ones(5),'ro')

# plt.loglog(l_arr,1*np.ones(len(l_arr)),'bo')
N_norm = N_raw['[0.015, 0.05]']
N_norm = 10**(np.log10(N_norm[N_norm>0]).mean())
n_arr = []
for key_ in N_raw.keys():
    
    N_sizeclass = 10**(np.log10(N_raw[key_][N_raw[key_]>0]).mean())
    N_sizeclass /= N_norm
    
    dx = eval(key_)[1] - eval(key_)[0]
    N_sizeclass_norm = N_sizeclass / dx
    print(N_sizeclass_norm)

    n_arr.append(N_sizeclass_norm)

plt.figure()
plt.loglog(l_Egger_mid,n_arr,'o-')

coeff,_ = curve_fit(lin_fn, np.log10(l_Egger_mid), np.log10(n_arr))
print(coeff)

M_norm = M_raw['[0.015, 0.05]']
M_norm = 10**(np.log10(M_norm[M_norm>0]).mean())
M_arr = []
for key_ in M_raw.keys():
    
    M_sizeclass = 10**(np.log10(M_raw[key_][M_raw[key_]>0]).mean())
    M_sizeclass /= M_norm
    
    dx = eval(key_)[1] - eval(key_)[0]
    M_sizeclass_norm = M_sizeclass / dx
    print(M_sizeclass_norm)

    M_arr.append(M_sizeclass_norm)

plt.figure()
plt.loglog(l_Egger_mid,M_arr,'o-')

coeff,_ = curve_fit(lin_fn, np.log10(l_Egger_mid), np.log10(M_arr))
print(coeff)



# integrate linear function with given slopes above
# size classes
l_int = [[5.65685425e-04, 1.13137085e-03],
         [1.13137085e-03,4.52548340e-03],
         [4.52548340e-03,1.81019336e-02],
         [1.81019336e-02,3.62038672e-02]]

correction_factor_n = []
correction_factor_m = []
for interval_E,interval_M in zip(size_classes,l_int):
    
    print(interval_E)

    n1 = 0
    m1 = -2.3
    m2 = -0.4
    
    int_E = (np.exp(n1)/(m1+1))*(interval_E[1]**(m1+1) - interval_E[0]**(m1+1))
    int_M = (np.exp(n1)/(m1+1))*(interval_M[1]**(m1+1) - interval_M[0]**(m1+1))
    
    correction_factor_n.append( int_E / int_M)

    int_E = (np.exp(n1)/(m2+1))*(interval_E[1]**(m2+1) - interval_E[0]**(m2+1))
    int_M = (np.exp(n1)/(m2+1))*(interval_M[1]**(m2+1) - interval_M[0]**(m2+1))
        
    correction_factor_m.append(int_E / int_M)
    
    