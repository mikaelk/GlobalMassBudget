#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:36:00 2022
convert MDMAP/OSPAR data, aggregrate on a given h3 grid
@author: kaandorp
"""

from glob import glob
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import math
from h3.unstable import vect
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature, LAND


df_total = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','percent_fishing','l_min','l_max'])


data = pd.read_csv('MDMAP_Export_20220301.csv',parse_dates=['survey_date'])

str_aqua = 'bait|fish|oyster|buoy|aquaculture|rope-and-nets'
str_exclude = 'cigarettes|cigar-tips'

mask_contain_plastic = (data.keys().str.contains('plastic') & ~data.keys().str.contains(str_exclude) )

mask_fishing_related = data.keys().str.contains(str_aqua) & mask_contain_plastic

keys_plastic = data.keys()[mask_contain_plastic]
keys_fishing_related = data.keys()[mask_fishing_related]

# exclude the total reported count
mask_total = data.keys().str.contains('total_plastic_items')

data_plastic_total = data.loc[:,(mask_contain_plastic & ~mask_total)].sum(axis=1)
data_fishing_total = data.loc[:,(mask_fishing_related & ~mask_total)].sum(axis=1)

mask_use = (data_plastic_total > 0)
fraction_fisheries = (data_fishing_total[mask_use]/data_plastic_total[mask_use])
fraction_fisheries[data_plastic_total[mask_use] < 10] = np.nan
concentration_MDMAP = data_plastic_total[mask_use] / 20 # 4 transects of 5 meters (=20m) -> to items per meter

lons = data.loc[mask_use,data.keys().str.contains('_lon')].mean(axis=1)
lats = data.loc[mask_use,data.keys().str.contains('_lat')].mean(axis=1)
dates = data.loc[mask_use,'survey_date']

df_agg = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','fishing_fraction','l_min','l_max','N_meas','h3'])

c = 0
h3_index = vect.geo_to_h3(lats,lons,3)
h3_unique = np.unique(h3_index)  
for h3_ in h3_unique:
    mask_h3 = (h3_index == h3_) & (concentration_MDMAP > 0)
    year_at_h3 = dates[mask_h3].dt.year
    month_at_h3 = dates[mask_h3].dt.month
    
    for year_ in np.unique(year_at_h3):
        for month_ in np.unique(month_at_h3):
            mask_total = mask_h3 & (dates.dt.year == year_) & (dates.dt.month == month_)
            
            if mask_total.sum() > 0:
                mean_meas = 10**(np.log10(concentration_MDMAP[mask_total]).mean())
                mean_fis_frac = np.nanmean(fraction_fisheries[mask_total])
                df_agg.loc[c,'decimalLatitude'] = lats[mask_total].mean()
                df_agg.loc[c,'decimalLongitude'] = lons[mask_total].mean()
                df_agg.loc[c,'eventDate'] = pd.Timestamp(year_, month_, 15)
                df_agg.loc[c,'Year'] = year_
                df_agg.loc[c,'Month'] = month_
                df_agg.loc[c,'Day'] = 15
                df_agg.loc[c,'Depth'] = 0
                df_agg.loc[c,'ParentEventID'] = 'MDMAP'
                df_agg.loc[c,'measurementValue'] = mean_meas
                df_agg.loc[c,'measurementUnit'] = 'n m-1'   
                df_agg.loc[c,'fishing_fraction'] = mean_fis_frac 
                df_agg.loc[c,'l_min'] = 0.025  
                df_agg.loc[c,'N_meas'] = mask_total.sum()
                df_agg.loc[c,'h3'] = np.uint(h3_)
                # df_agg.loc[c,'l_max'] = 0
                c += 1
# df_agg['h3']=df_agg['h3'].astype(np.uint64)                
# fig = plt.figure(figsize=(13, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(LAND, zorder=0,edgecolor='black')
# ax.scatter(lons,lats,data_plastic_total[mask_use])
percentage_fisheries = fraction_fisheries*100
fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lons,lats,percentage_fisheries,c=percentage_fisheries)
ax.set_title('MDMAP, fishing related items percentage')
fig.colorbar(scat)


fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lons[mask_use],lats[mask_use],s=concentration_MDMAP*10,c=concentration_MDMAP)
ax.set_title('MDMAP, concentration')
fig.colorbar(scat)


df_total = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','fishing_fraction','l_min','l_max'])

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','fishing_fraction','l_min','l_max'])

df_tmp['decimalLatitude'] = lats
df_tmp['decimalLongitude'] = lons
df_tmp['eventDate'] = dates
df_tmp['Year'] = np.array([date_.year for date_ in dates])
df_tmp['Month'] = np.array([date_.month for date_ in dates])
df_tmp['Day'] = np.array([date_.day for date_ in dates])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'MDMAP'
df_tmp['measurementValue'] = concentration_MDMAP
df_tmp['measurementUnit'] = 'n m-1'
df_tmp['fishing_fraction'] = fraction_fisheries
df_tmp['l_min'] = 0.025

df_total = pd.concat((df_total,df_tmp))

#%%
def to_decimal_lonlat(vals):
    return_vals = []
    for i1,val_ in enumerate(vals):
        if not type(val_) == str:
            return_vals.append(np.nan)
        else:
            #erroneous characters:
            val_ = val_.replace('”','\"')
            val_ = val_.replace('’’','\"')
            val_ = val_.replace('″','\"')
            val_ = val_.replace('′','\'')
            try:
                if '\'' in val_.split('°')[1]: #value is notated in deg,min,sec format
                    # print(i1,val_)
                    deg = float(val_.split('°')[0])
                    minute = float(val_.split('°')[1].split('\'')[0])
                    sec = float(val_.split('°')[1].split('\'')[1].split('\"')[0].replace(',','.'))
                    
                    type_ = val_.split('°')[1].split('\'')[1].split('\"')[1]
                    value = deg + minute/60 + sec/3600
                    
                    if type_ == 'S' or type_ == 'W':
                        value *= -1
                    return_vals.append(value)
                else: #value is notated in deg, min format (min=decimal number)
                    deg = float(val_.split('°')[0])
                    minute = float(val_.split('°')[1].split('"')[0])
                    type_ = val_.split('°')[1].split('"')[1]
                    value = deg + minute/60
                    if type_ == 'S' or type_ == 'W':
                        value *= -1
                    return_vals.append(value)         
            except:
                print('unable to parse: %s' % val_)
                return_vals.append(np.nan)
    return np.array(return_vals)
        

data_OSPAR = pd.read_csv('OSPAR_1km.csv',parse_dates=['Survey date'])
# str_exclude = ''
str_aqua = 'Buoy|Fish|Rope|String|Net|rope'

mask_contain_plastic = (data_OSPAR.keys().str.contains('Plastic'))
mask_fishing_related = data_OSPAR.keys().str.contains(str_aqua) & mask_contain_plastic

data_plastic_total = data_OSPAR.loc[:,(mask_contain_plastic)].sum(axis=1)
data_fishing_total = data_OSPAR.loc[:,(mask_fishing_related)].sum(axis=1)
mask_outlier = data_plastic_total > np.quantile(data_plastic_total,.99)

lat_100m = .5*(to_decimal_lonlat(data_OSPAR['100m Start N/S'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR['100m End N/S'].str.replace('\'\'','\"')))
lon_100m = .5*(to_decimal_lonlat(data_OSPAR['100m Start E/W'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR['100m End E/W'].str.replace('\'\'','\"')))
lat_1km = .5*(to_decimal_lonlat(data_OSPAR['1km Start N/S'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR['1km End N/S'].str.replace('\'\'','\"')))
lon_1km = .5*(to_decimal_lonlat(data_OSPAR['1km Start E/W'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR['1km End E/W'].str.replace('\'\'','\"')))
dates = data_OSPAR['Survey date']


# c = 0

                
                
mask_100m = ~np.isnan(lat_100m)
mask_1km = ~np.isnan(lat_1km)

mask_use = (data_plastic_total > 0) & mask_1km & ~mask_outlier
fraction_fisheries = (data_fishing_total[mask_use]/data_plastic_total[mask_use])
fraction_fisheries[data_plastic_total[mask_use] < 10] = np.nan
concentration_OSPAR1km = data_plastic_total[mask_use] / 1000 # to items per meter


percentage_fisheries = fraction_fisheries*100
fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lon_1km[mask_use],lat_1km[mask_use],percentage_fisheries,c=percentage_fisheries)
ax.set_title('OSPAR 1km, fishing related items percentage')
fig.colorbar(scat)

fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lon_1km[mask_use],lat_1km[mask_use],s=concentration_OSPAR1km*1000,c=concentration_OSPAR1km)
ax.set_title('OSPAR 1km, concentration')
fig.colorbar(scat)

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','fishing_fraction','l_min','l_max'])

df_tmp['decimalLatitude'] = lat_100m[mask_use]
df_tmp['decimalLongitude'] = lon_100m[mask_use]
df_tmp['eventDate'] = dates[mask_use].values
df_tmp['Year'] = np.array([date_.year for date_ in dates[mask_use]])
df_tmp['Month'] = np.array([date_.month for date_ in dates[mask_use]])
df_tmp['Day'] = np.array([date_.day for date_ in dates[mask_use]])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'OSPAR_1km'
df_tmp['measurementValue'] = concentration_OSPAR1km.values
df_tmp['measurementUnit'] = 'n m-1'
df_tmp['fishing_fraction'] = fraction_fisheries.values
df_tmp['l_min'] = 0.50

df_total = pd.concat((df_total,df_tmp))
# ax.plot(lon_100m,lat_100m,'ro')

lons = lon_1km[mask_use]
lats = lat_1km[mask_use]
dates = dates[mask_use]
h3_index = vect.geo_to_h3(lats,lons,3)
h3_unique = np.unique(h3_index)  
for h3_ in h3_unique:
    mask_h3 = (h3_index == h3_) & (concentration_OSPAR1km > 0)
    year_at_h3 = dates[mask_h3].dt.year
    month_at_h3 = dates[mask_h3].dt.month
    
    for year_ in np.unique(year_at_h3):
        for month_ in np.unique(month_at_h3):
            mask_total = mask_h3 & (dates.dt.year == year_) & (dates.dt.month == month_)
            
            if mask_total.sum() > 0:
                mean_meas = 10**(np.log10(concentration_OSPAR1km[mask_total]).mean())
                mean_fis_frac = np.nanmean(fraction_fisheries[mask_total])
                df_agg.loc[c,'decimalLatitude'] = lats[mask_total].mean()
                df_agg.loc[c,'decimalLongitude'] = lons[mask_total].mean()
                df_agg.loc[c,'eventDate'] = pd.Timestamp(year_, month_, 15)
                df_agg.loc[c,'Year'] = year_
                df_agg.loc[c,'Month'] = month_
                df_agg.loc[c,'Day'] = 15
                df_agg.loc[c,'Depth'] = 0
                df_agg.loc[c,'ParentEventID'] = 'OSPAR_1km'
                df_agg.loc[c,'measurementValue'] = mean_meas
                df_agg.loc[c,'measurementUnit'] = 'n m-1'   
                df_agg.loc[c,'fishing_fraction'] = mean_fis_frac 
                df_agg.loc[c,'l_min'] = 0.50 
                df_agg.loc[c,'N_meas'] = mask_total.sum()
                df_agg.loc[c,'h3'] = np.uint(h3_)
                # df_agg.loc[c,'l_max'] = 0
                c += 1
#%%

def read_frames(files):
    for i1,file_ in enumerate(files):
        if i1 == 0:
            data = pd.read_csv(file_,parse_dates=['Survey date'])
        else:
            data = pd.concat((data,pd.read_csv(file_)))
    return data
    
files_100m = glob('OSPAR_100*')    
    
data_OSPAR100m = read_frames(files_100m)

str_exclude = 'cigarettes|cigar-tips'
str_aqua = 'Buoy|Fish|Rope|String|Net|rope|Lobster|Oyster|Mussel|line|float'.lower()
mask_contain_plastic = (data_OSPAR100m.keys().str.lower().str.contains('plastic') & ~data_OSPAR100m.keys().str.lower().str.contains(str_exclude))
mask_fishing_related = data_OSPAR100m.keys().str.lower().str.contains(str_aqua) & mask_contain_plastic

data_plastic_total = data_OSPAR100m.loc[:,(mask_contain_plastic)].sum(axis=1)
data_fishing_total = data_OSPAR100m.loc[:,(mask_fishing_related)].sum(axis=1)
mask_outlier = data_plastic_total > np.quantile(data_plastic_total,.99)

lat_100m = .5*(to_decimal_lonlat(data_OSPAR100m['100m Start N/S'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR100m['100m End N/S'].str.replace('\'\'','\"')))
lon_100m = .5*(to_decimal_lonlat(data_OSPAR100m['100m Start E/W'].str.replace('\'\'','\"')) + to_decimal_lonlat(data_OSPAR100m['100m End E/W'].str.replace('\'\'','\"')))
dates = data_OSPAR100m['Survey date'].astype(np.datetime64)

mask_use = (data_plastic_total > 0) & ~np.isnan(lat_100m) & ~mask_outlier
fraction_fisheries = (data_fishing_total[mask_use]/data_plastic_total[mask_use])
fraction_fisheries[data_plastic_total[mask_use] < 10] = np.nan
concentration_OSPAR100m = data_plastic_total[mask_use] / 100 # to items per meter

percentage_fisheries = fraction_fisheries*100
fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lon_100m[mask_use],lat_100m[mask_use],percentage_fisheries,c=percentage_fisheries)
ax.set_title('OSPAR100m, fishing related items percentage')
fig.colorbar(scat)

fig = plt.figure(figsize=(13, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(LAND, zorder=0,edgecolor='black')
scat=ax.scatter(lon_100m[mask_use],lat_100m[mask_use],s=concentration_OSPAR100m*10,c=concentration_OSPAR100m)
ax.set_title('OSPAR 100m, concentration')
fig.colorbar(scat)

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','fishing_fraction','l_min','l_max'])

df_tmp['decimalLatitude'] = lat_100m[mask_use]
df_tmp['decimalLongitude'] = lon_100m[mask_use]
df_tmp['eventDate'] = dates[mask_use].values
df_tmp['Year'] = np.array([date_.year for date_ in dates[mask_use]])
df_tmp['Month'] = np.array([date_.month for date_ in dates[mask_use]])
df_tmp['Day'] = np.array([date_.day for date_ in dates[mask_use]])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'OSPAR_100m'
df_tmp['measurementValue'] = concentration_OSPAR100m.values
df_tmp['measurementUnit'] = 'n m-1'
df_tmp['fishing_fraction'] = fraction_fisheries.values
df_tmp['l_min'] = 0.005

df_total = pd.concat((df_total,df_tmp))



lons = lon_100m[mask_use]
lats = lat_100m[mask_use]
dates = dates[mask_use]
h3_index = vect.geo_to_h3(lats,lons,3)
h3_unique = np.unique(h3_index)  
for h3_ in h3_unique:
    mask_h3 = (h3_index == h3_) & (concentration_OSPAR100m > 0)
    year_at_h3 = dates[mask_h3].dt.year
    month_at_h3 = dates[mask_h3].dt.month
    
    for year_ in np.unique(year_at_h3):
        for month_ in np.unique(month_at_h3):
            mask_total = mask_h3 & (dates.dt.year == year_) & (dates.dt.month == month_)
            
            if mask_total.sum() > 0:
                mean_meas = 10**(np.log10(concentration_OSPAR100m[mask_total]).mean())
                mean_fis_frac = np.nanmean(fraction_fisheries[mask_total])
                df_agg.loc[c,'decimalLatitude'] = lats[mask_total].mean()
                df_agg.loc[c,'decimalLongitude'] = lons[mask_total].mean()
                df_agg.loc[c,'eventDate'] = pd.Timestamp(year_, month_, 15)
                df_agg.loc[c,'Year'] = year_
                df_agg.loc[c,'Month'] = month_
                df_agg.loc[c,'Day'] = 15
                df_agg.loc[c,'Depth'] = 0
                df_agg.loc[c,'ParentEventID'] = 'OSPAR_100m'
                df_agg.loc[c,'measurementValue'] = mean_meas
                df_agg.loc[c,'measurementUnit'] = 'n m-1'   
                df_agg.loc[c,'fishing_fraction'] = mean_fis_frac 
                df_agg.loc[c,'l_min'] = 0.005 
                df_agg.loc[c,'N_meas'] = mask_total.sum()
                df_agg.loc[c,'h3'] = np.uint(h3_)
                # df_agg.loc[c,'l_max'] = 0
                c += 1
                
#%%
h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_total['h%i' % res_] = vect.geo_to_h3(df_total['decimalLatitude'],df_total['decimalLongitude'],res_)

#%%
df_total.to_csv('converted_MDMAP_OSPAR.csv')
df_agg.to_csv('converted_aggregrated_MDMAP_OSPAR.csv')
