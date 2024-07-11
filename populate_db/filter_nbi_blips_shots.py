#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:48:33 2024

@author: tomble
"""
import numpy as np, json, time
import urllib.error
from w7xdia.nbi import get_total_nbi_power_for_program
from w7x_preparation_pystrahl.utilities.geometry import get_reference_equilibrium

import logging
logging.disable(logging.CRITICAL) 

from w7xdia.thomson import get_data_from_archive
from w7x_preparation_pystrahl.profiles.profile_fitting import(
    _fetch_thomson_data_raw,
    _get_thomson_points
    )

# lists to catogrize discharges
#nbi_available = []
#nbi_length = []
#nbi_continuous = []
#thomson_available = []
nbi_blips_shots = []
blimp_nbi_length = []

# final dict which is saved as json with the lists above
good_shots = {}

nbi_length_threshold = 0.5 #s, i.e. 500ms
p_threshold = 1.0

def get_quality_shots():
    """
    Method to filter shots of interest

    Returns
    -------
    Saves JSON file with sorted lists

    """
    # load json containing all experimental days and shots
    with open(r"/home/IPP-HGW/tomble/Desktop/_signal.json", "r") as file_obj:
        raw_data = json.load(file_obj)
          
    # returns array with all unique experimental days
    unique_days = np.unique(raw_data["values"])
        
    # for each day in the unique days loop to get quality shots
    for day in unique_days:

        # count how many occurences we have of the same date to determine number of shots
        shots = raw_data['values'].count(day)
        
        # now we loop over all shots in unique day
        for shot_number in range(1, shots+1):

            # transform shot_number to right format
            shot_name = f"{day}.{str(shot_number).zfill(3)}"
        
            try:
                # request thomson data for official branch
                thomson_data = _fetch_thomson_data_raw(
                        shot_name,         
                        get_reference_equilibrium(shot_name)[0],
                        branch=None
                        )         
                #thomson_available.append(shot_name)
                
                # request nbi data from archive 
                shot_time, shot_power = get_total_nbi_power_for_program(
                    shot_name, 
                    timeout=10, 
                    useLastVersion=True, 
                    version=None, 
                    )
            
                if len(shot_time) > 0:
                    
                    # reduce data to positive time points only
                    shot_power = shot_power[shot_time > 0]               
                    shot_time = shot_time[shot_time > 0]
                    
                    # check whether there is nbi, list not empty and data is not due to fluctuations
                    if len(shot_power) > 0 and max(shot_power) > p_threshold:
                        # append to according list
                        #nbi_available.append(shot_name)
                        
                        threshold_crossings = np.diff(shot_power > p_threshold, prepend=False)
                        t_stop = shot_time[threshold_crossings][1::2]
                        t_start = shot_time[threshold_crossings][0::2]
                        t_length = (t_stop - t_start)[0]

                        # append to according list
                        #nbi_continuous.append(shot_name)
                        #nbi_length.append(t_length)
                        
                        # check for nbi blimps - add them to according list if shorter than threshold
                        if t_length < nbi_length_threshold: 
                            nbi_blips_shots.append(shot_name)
                            blimp_nbi_length.append(t_length)

                        else:
                            # no continuous nbi
                            continue
                    else:
                        # no nbi
                        continue                
                else: 
                    # no time
                    continue
                
            except:
                # no thomson data
                continue       
            
    # after shots have been charecrtrized to according list, add the lists to a dictionary and save to json
    #for shot_type in ["nbi_available", "nbi_length", "nbi_continuous", "thomson_available"]:
    for shot_type in ["nbi_blips_shots", "blimp_nbi_length"]:
        good_shots[shot_type] = globals()[shot_type]

    # write final dict with all lists to json file
    with open('blips_shot_database.json', 'w') as file:
        json.dump(good_shots, file)

get_quality_shots()


