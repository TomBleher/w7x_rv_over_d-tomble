#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:44:43 2024

@author: tomble
"""

import numpy as np, json
from w7xdia.nbi import get_total_nbi_power_for_program
from w7x_preparation_pystrahl.utilities.geometry import get_reference_equilibrium

import logging
logging.disable(logging.CRITICAL) 

from w7x_preparation_pystrahl.profiles.profile_fitting import(
    _fetch_thomson_data_raw,
    )

# lists to catogrize discharges
nbi_available = []
nbi_length = []
nbi_continuous = []
thomson_available = []

# final dict which is saved as json with the lists above
good_shots = {}

nbi_length_threshold = 0.5 #s, i.e. 500ms
p_threshold = 1.0

def get_quality_shots():

    # all experimental days according to the logbook
    unique_days = [
        "20181018",   "20181017",   "20181016",   "20181011",
        "20181010",   "20181009",   "20181004",   "20181002",
        "20180927",   "20180925",   "20180920",   "20180919",
        "20180918",   "20180912",   "20180911",   "20180906",
        "20180905",   "20180904",   "20180829",   "20180828",
        "20180823",   "20180822",   "20180821",   "20180816",
        "20180814",   "20180809",   "20180808",   "20180807",
        "20180801",   "20180731",   "20180726",   "20180725",
        "20180724",   "20180719",   "20180718",
        ]
        
    # for each day in the unique days loop to get quality shots
    for day in unique_days:
        
        # now we loop over all shots in unique day
        for shot_number in range(1, 100+1):

            # transform shot_number to right format
            shot_name = f"{day}.{str(shot_number).zfill(3)}"
        
            try:
                # request thomson data for official branch
                thomson_data = _fetch_thomson_data_raw(
                        shot_name,         
                        get_reference_equilibrium(shot_name)[0],
                        branch=None
                        )         
                thomson_available.append(shot_name)
                
                # request nbi data from archive 
                shot_time, shot_power = get_total_nbi_power_for_program(
                    shot_name, 
                    timeout=10, 
                    useLastVersion=True, 
                    version=None, 
                    )
            
                if len(shot_time) > 0:
                    print("meow")
                    # reduce data to positive time points only
                    shot_power = shot_power[shot_time > 0]               
                    shot_time = shot_time[shot_time > 0]
                    
                    # check whether there is nbi, list not empty and data is not due to fluctuations
                    if len(shot_power) > 0 and max(shot_power) > p_threshold:
                        # append to according list
                        nbi_available.append(shot_name)
                        
                        threshold_crossings = np.diff(shot_power > p_threshold, prepend=False)
                        t_stop = shot_time[threshold_crossings][1::2]
                        t_start = shot_time[threshold_crossings][0::2]
                        t_length = (t_stop - t_start)[0]
                        
                        # check for nbi blimps 
                        if t_length > nbi_length_threshold: 
                            # append to according list
                            nbi_continuous.append(shot_name)
                            nbi_length.append(t_length)

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
    for shot_type in ["nbi_available", "nbi_length", "nbi_continuous", "thomson_available"]:
        good_shots[shot_type] = globals()[shot_type]

    # write final dict with all lists to json file
    with open('OP1.2b_shot_database.json', 'w') as file:
        json.dump(good_shots, file)

get_quality_shots()



