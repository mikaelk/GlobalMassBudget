#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:08:37 2022

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
import datetime


detections_1 = pd.read_csv('/Users/kaandorp/Data/PlasticData/deVries_2021/NPM3_Objects/NPM3_detections_portside.csv')
detections_2 = pd.read_csv('/Users/kaandorp/Data/PlasticData/deVries_2021/NPM3_Objects/NPM3_detections_starboard.csv')
detections_tot = pd.concat((detections_1,detections_2))

region_data = pd.read_csv('/Users/kaandorp/Data/PlasticData/deVries_2021/transects.csv')

detections_tot['a_mean'] = .5*(detections_tot['amaj'] + detections_tot['amin'])
# for checking the transect lengths:    
# for i in range(11):
    
#     str_ = region_data['geometry'][i]
#     lon = float(str_.split('(')[1].split(' ')[0])
#     lat = float(str_.split('(')[1].split(' ')[1].split(')')[0])
    
#     plt.text(lon,lat,'%i'%i)

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate',
             'Year','Month','Day',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])


# bnds_lon = [-145,-142,-139,-136,-133]
bnds_t = [43801.6,43802.2,43803.4,43804.5,43805.5,43808.]
transect_lengths = [135389,  95871, 195918, 183411,  61981+67611]

# areas = (80*region_data['Length']/1e6).values

concentration_paper = (2*region_data['NUMPOINTS']) / (80*region_data['Length']/1e6)

l_arr = np.array([0.4096*2**k for k in range(4)])
lengths = 10**(.5*(np.log10(l_arr[1:]) + np.log10(l_arr[:-1]))) #[0.4096,0.8192,1.6384]

plt.figure()

n_tot_hist = np.zeros(3)

c=0
for l_,r_, length_ in zip(bnds_t[:-1],bnds_t[1:],transect_lengths):
    
    
    mask_time = (detections_tot['timestamp'] >= l_) & (detections_tot['timestamp'] < r_)
    
    lon_mean = detections_tot['longitude'][mask_time].mean()
    lat_mean = detections_tot['latitude'][mask_time].mean()
    date_mean = datetime.date(1900,1,1) + datetime.timedelta(days=detections_tot['timestamp'][mask_time].mean())
    
    for i1,(l_l,l_u) in enumerate(zip(lengths[:-1],lengths[1:])):
        
        mask_length = (detections_tot['a_mean'] >= l_l) & (detections_tot['a_mean'] < l_u )
    
        mask_tot = mask_time & mask_length
        concentration = (mask_tot.sum() ) / (80*length_)
    
        print(l_l,mask_tot.sum(),concentration)
 
    
        df_tmp.loc[c,'decimalLatitude'] = lat_mean
        df_tmp.loc[c,'decimalLongitude'] = lon_mean
        df_tmp.loc[c,'eventDate'] = pd.Timestamp(date_mean)
        df_tmp.loc[c,'Year'] = date_mean.year
        df_tmp.loc[c,'Month'] = date_mean.month
        df_tmp.loc[c,'Day'] = date_mean.day
        df_tmp.loc[c,'Depth'] = 0
        df_tmp.loc[c,'ParentEventID'] = 'deVries2021'
        df_tmp.loc[c,'measurementValue'] = concentration
        df_tmp.loc[c,'measurementUnit'] = 'n m-2'
        df_tmp.loc[c,'l_min'] = l_l
        df_tmp.loc[c,'l_max'] = l_u
        
        
        n_tot_hist[i1] += mask_tot.sum()
        c += 1   
 
    # mask_tot = (detections_tot['timestamp'] > l_) & (detections_tot['timestamp'] <= r_) & (detections_tot['amaj'] > .5)
    # mask_2 = (detections_2['timestamp'] > l_) & (detections_2['timestamp'] <= r_) & (detections_2['amaj'] > .5)
    
    # concentration = (mask_tot.sum() ) / (80*length_/1e6)
    
    

    plt.plot(detections_tot['longitude'][mask_tot],detections_tot['latitude'][mask_tot],'o')
    # plt.plot(detections_2['longitude'][mask_2],detections_2['latitude'][mask_2],'o')
    plt.plot(lon_mean, lat_mean,'ko')
    
    
    
h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_tmp['h%i' % res_] = vect.geo_to_h3(df_tmp['decimalLatitude'],df_tmp['decimalLongitude'],res_)
    
upper_cell_heights= np.array([1,5])
for height_ in upper_cell_heights:
    df_tmp['measurementValue_vol_%im' % (height_)] = df_tmp['measurementValue'] / height_
df_tmp['measurementUnit_vol'] = 'n m-3'   



l_arr = np.array([0.1048*2**k for k in range(8)])
lengths_all = 10**(.5*(np.log10(l_arr[1:]) + np.log10(l_arr[:-1]))) 

plt.figure()
plt.hist(np.log10(detections_tot['a_mean']),bins=np.log10(lengths_all) )
plt.xticks(ticks=np.log10(lengths_all),labels=lengths_all)
plt.xlabel('Particle size [m]')
plt.ylabel('Frequency')    
    
# plt.figure()
# plt.bar((.5* (np.log10(lengths[1:])+ (np.log10(lengths[:-1])))),n_tot_hist, width=.2 )
# # plt.xticks(ticks=np.log10(lengths_all),labels=lengths_all)
# plt.xlabel('Particle size [m]')
# plt.ylabel('Frequency')    
#%%
df_tmp.to_csv('converted_surface_macro.csv')
    