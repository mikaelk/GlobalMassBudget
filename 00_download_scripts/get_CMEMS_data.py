#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:43:06 2021

@author: kaandorp
"""

import pandas as pd
import os

time_array = pd.date_range('2013-01-01 00:00:00','2020-01-01 00:00:00')
str_cmd = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id NWSHELF_REANALYSIS_WAV_004_015-TDS --product-id MetO-NWS-WAV-RAN --longitude-min -16 --longitude-max 13 --latitude-min 46 --latitude-max 62.7432 --date-min "%s 00:00:00" --date-max "%s 21:00:00" --variable VSDX --variable VSDY --out-dir /data/oceanparcels/input_data/CMEMS/NWSHELF_REANALYSIS_WAV/ --out-name NWSHELF_REANALYSIS_WAV_%s.nc --user mkaandorp --pwd MikaelCMEMS2018'

for time_ in time_array[::-1]:
    
    str_date = str(time_.date())
    cmd = str_cmd % (str_date,str_date,str_date)
    
    os.system(cmd)