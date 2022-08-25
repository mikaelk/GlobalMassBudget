#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:36:00 2022

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

files = glob('/Users/kaandorp/Data/PlasticData/data_beaches/measurements*')

df_total = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','l_min','l_max'])

#Edyvane
data = pd.read_excel(files[0])

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','l_min','l_max'])
df_tmp['decimalLatitude'] = data['Latitude']
df_tmp['decimalLongitude'] = data['Longitude']
df_tmp['eventDate'] = data['Date of survey']
df_tmp['Year'] = np.array([date_.year for date_ in data['Date of survey']])
df_tmp['Month'] = np.array([date_.month for date_ in data['Date of survey']])
df_tmp['Day'] = np.array([date_.day for date_ in data['Date of survey']])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Edyvane2004'
df_tmp['measurementValue'] = data['Plastic Litter kg/km']
df_tmp['measurementUnit'] = 'g m-1'
df_tmp['l_min'] = 0.02

df_total = pd.concat((df_total,df_tmp))


#Polasek2017
# data = pd.read_excel(files[1])

# df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
#                        'ParentEventID','measurementValue','measurementUnit','l_min','l_max'])
# df_tmp['decimalLatitude'] = data['End latitude']
# df_tmp['decimalLongitude'] = data['End longitude']
# df_tmp['eventDate'] = pd.Timestamp('2015-06-15')
# df_tmp['Year'] = 2015
# df_tmp['Month'] = 6
# df_tmp['Day'] = 15
# df_tmp['Depth'] = 0
# df_tmp['ParentEventID'] = 'Polasek2017'
# df_tmp['measurementValue'] = data['Debris density (kg/km)']
# df_tmp['measurementUnit'] = 'g m-1'
# df_tmp['l_min'] = 0.01

# df_total = pd.concat((df_total,df_tmp))


#Lee2015
data = pd.read_excel(files[2])

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','Year','Month','Day','Depth',
                       'ParentEventID','measurementValue','measurementUnit','l_min','l_max'])
df_tmp['decimalLatitude'] = data['Latitude']
df_tmp['decimalLongitude'] = data['Longitude']
df_tmp['eventDate'] = data['Collection date (month/year)']
df_tmp['Year'] = np.array([date_.year for date_ in df_tmp['eventDate']])
df_tmp['Month'] = np.array([date_.month for date_ in df_tmp['eventDate']])
df_tmp['Day'] = np.array([date_.day for date_ in df_tmp['eventDate']])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Lee2015'
df_tmp['measurementValue'] = data['Plastics collected (kg/1000\xa0m)']
df_tmp['measurementUnit'] = 'g m-1'
df_tmp['l_min'] = 0.025

df_total = pd.concat((df_total,df_tmp))

h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_total['h%i' % res_] = vect.geo_to_h3(df_total['decimalLatitude'],df_total['decimalLongitude'],res_)

df_BBCT = pd.read_csv('converted_beach_BBCT.csv',index_col=0,dtype={'h0':np.uint64,'h1':np.uint64,'h2':np.uint64,'h3':np.uint64})
df_BBCT = df_BBCT[df_BBCT['measurementValue'] > 0]

df_total = pd.concat((df_BBCT,df_total))
df_total.to_csv('converted_beach_inst.csv')

#%%
#Hong2014

data = pd.read_excel(files[3])

df_total = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])
df_tmp['decimalLatitude'] = data['Latitude']
df_tmp['decimalLongitude'] = data['Longitude']
df_tmp['eventDate'] = data['from']
df_tmp['eventDate_end'] = data['to']
df_tmp['Year_start'] = np.array([date_.year for date_ in df_tmp['eventDate']])
df_tmp['Month_start'] = np.array([date_.month for date_ in df_tmp['eventDate']])
df_tmp['Day_start'] = np.array([date_.day for date_ in df_tmp['eventDate']])
df_tmp['Year_end'] = np.array([date_.year for date_ in df_tmp['eventDate_end']])
df_tmp['Month_end'] = np.array([date_.month for date_ in df_tmp['eventDate_end']])
df_tmp['Day_end'] = np.array([date_.day for date_ in df_tmp['eventDate_end']])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Hong2014'
df_tmp['measurementValue'] = data['mean mass']*10 #from kg/100m to kg/km (or g/m)
df_tmp['CV'] = data['CV_mass']
df_tmp['measurementUnit'] = 'g m-1'
df_tmp['l_min'] = 0.025

df_total = pd.concat((df_total,df_tmp))

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])
df_tmp['decimalLatitude'] = data['Latitude']
df_tmp['decimalLongitude'] = data['Longitude']
df_tmp['eventDate'] = data['from']
df_tmp['eventDate_end'] = data['to']
df_tmp['Year_start'] = np.array([date_.year for date_ in df_tmp['eventDate']])
df_tmp['Month_start'] = np.array([date_.month for date_ in df_tmp['eventDate']])
df_tmp['Day_start'] = np.array([date_.day for date_ in df_tmp['eventDate']])
df_tmp['Year_end'] = np.array([date_.year for date_ in df_tmp['eventDate_end']])
df_tmp['Month_end'] = np.array([date_.month for date_ in df_tmp['eventDate_end']])
df_tmp['Day_end'] = np.array([date_.day for date_ in df_tmp['eventDate_end']])
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Hong2014'
df_tmp['measurementValue'] = data['mean num'] / 100 #from n/100m to n/m 
df_tmp['CV'] = data['CV_num']
df_tmp['measurementUnit'] = 'n m-1'
df_tmp['l_min'] = 0.025

df_total = pd.concat((df_total,df_tmp))



#%%
data = pd.read_excel(files[4],header=2)

# df_total = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
#              'Year_start','Month_start','Day_start',
#              'Year_end','Month_end','Day_end',
#              'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])
df_tmp['decimalLatitude'] = -data['Latitude (S)']
df_tmp['decimalLongitude'] = data['Longitude (E)']
df_tmp['eventDate'] = pd.Timestamp(2015, 5, 1)
df_tmp['eventDate_end'] = pd.Timestamp(2015, 9, 1)
df_tmp['Year_start'] = 2015
df_tmp['Month_start'] = 5
df_tmp['Day_start'] = 1
df_tmp['Year_end'] = 2015
df_tmp['Month_end'] = 9
df_tmp['Day_end'] = 1
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Ryan2018'
df_tmp['measurementValue'] = np.nansum(data.iloc[:,3:10],axis=1)*2 / 20 #from #/.5m to #/m, factor 20 for 95% burial of small plastics
df_tmp['CV'] = ''
df_tmp['measurementUnit'] = 'n m-1'
df_tmp['l_min'] = 0.002
df_tmp['l_max'] = 0.025

df_total = pd.concat((df_total,df_tmp))

df_tmp = pd.DataFrame(columns=['decimalLatitude','decimalLongitude','eventDate','eventDate_end',
             'Year_start','Month_start','Day_start',
             'Year_end','Month_end','Day_end',
             'Depth', 'ParentEventID','measurementValue','CV','measurementUnit','l_min','l_max'])

df_tmp['decimalLatitude'] = -data['Latitude (S)']
df_tmp['decimalLongitude'] = data['Longitude (E)']
df_tmp['eventDate'] = pd.Timestamp(2015, 5, 1)
df_tmp['eventDate_end'] = pd.Timestamp(2015, 9, 1)
df_tmp['Year_start'] = 2015
df_tmp['Month_start'] = 5
df_tmp['Day_start'] = 1
df_tmp['Year_end'] = 2015
df_tmp['Month_end'] = 9
df_tmp['Day_end'] = 1
df_tmp['Depth'] = 0
df_tmp['ParentEventID'] = 'Ryan2018'
df_tmp['measurementValue'] = np.nansum(data.iloc[:,10:],axis=1)*2 / (20*1000) #from g/.5m to g/m, factor 20 for 95% burial of small plastics
df_tmp['CV'] = ''
df_tmp['measurementUnit'] = 'g m-1'
df_tmp['l_min'] = 0.002
df_tmp['l_max'] = 0.025

df_total = pd.concat((df_total,df_tmp))


h3_resolutions = [0,1,2,3]
for res_ in h3_resolutions:
    df_total['h%i' % res_] = vect.geo_to_h3(df_total['decimalLatitude'],df_total['decimalLongitude'],res_)

# df_ospar = pd.read_csv('converted_beach_ospar.csv',index_col=0,dtype={'h0':np.uint64,'h1':np.uint64,'h2':np.uint64,'h3':np.uint64})
# df_ospar['measurementUnit'] = 'n m-1'
# df_ospar['CV'] /= 100

# df_total = pd.concat((df_ospar,df_total))
df_total.to_csv('converted_beach_mean_test2.csv')
