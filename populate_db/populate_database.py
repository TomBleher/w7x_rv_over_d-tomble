#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:38:12 2024

@author: tomble
"""
import json
import os
import logging
import shutil
from subprocess import Popen, PIPE, DEVNULL
import matplotlib.pyplot as plt

"""
Routine to populate the database by running all shots gathered from filter_good_shots.py.
The code reads the json of the filtered shots (or any list provided) and runs the full fitting 
routine for the file. After the current shot has finished and was saved; the code will proceed to the next
until all shots were ran. Errors are handled with the try except block and are recorded to the log file specified
at the top.

While shots in the main_shot_database (continuous nbi shots from op2.1) have been tested and lasers to exclude
were determined, for using this routine to run nbi blips or shots of different operation, change the main loop
to loop over the desired shots, and comment the laser exclude writing to the created file.
"""


# set up logging to catch errors (error.log will be automatically configured in the directory of the file)
logging.basicConfig(filename='rerundb_mtanh.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

shot_laser_excluder = {
    "20221201.033": [2],
    "20221201.036": [2],
    "20221201.045": [],
    "20221201.053": [2],
    "20221207.054": [3],
    "20230214.041": [2],
    "20230214.056": [2],
    "20230216.024": [],
    "20230216.026": [],
    "20230216.028": [],
    "20230216.036": [],
    "20230216.058": [2],
    "20230216.061": [2,3],
    "20230216.068": [2,3],
    "20230216.071": [2],
    "20230216.074": [2],
    "20230216.077": [],
    "20230223.027": [2],
    "20230314.022": [],
    "20230314.026": [],
    "20230316.020": [],
    "20230316.023": [2],
    "20230316.028": [2],
    "20230316.039": [2,3],
    "20230316.066": [2,3],
    "20230316.069": [],
    "20230316.076": [2,3],
    "20230323.032": [],
    "20230323.034": [1,3],
    "20230323.047": [],
    "20230323.060": [],
    "20230126.056": [],
    "20230126.058": [],
    "20230126.059": [],
    "20230323.057": [],
    "20230316.061": [],
    "20230316.034": [],
    "20230307.061": [],
    "20230307.064": [],
    "20230214.061": [],
    "20230126.070": [],
    "20230316.051": [],
    "20230216.063": [],
    "20230126.062": [],
    "20230323.052": [],
    "20230316.064": [],
    "20230214.045": [],
    "20230214.047": [],
    "20230314.059": [],
    "20230314.024": [2,3],
    "20221201.049": [3],
    "20230223.013": [],
    "20230216.032": [],
}

json_path = "/home/IPP-HGW/tomble/Desktop/Code"
example_path = "/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Examples/Program_runfiles"

# read json file with filtered continuous nbi shot database
with open(os.path.join(json_path, "main_shot_database.json"), "r") as file:
    data = json.load(file)

# Extract continues NBI list and loop over all shots in it 
for shot_name in list(data.get("nbi_continuous", [])):

    # lay path foundation for file copying
    src_file = os.path.join(example_path, "example.py")
    dest_file = os.path.join(example_path, f"{shot_name}.py")

    # if it's among the bad shots
    #if shot_name in ["20230316.051", "20230216.063", "20230126.062", "20230323.052"]:
        #continue
    # if the file exists continue
    if os.path.exists(dest_file):
        continue
    # if not, create the file
    else:
        shutil.copy(src_file, dest_file)
            
    try:
        
        # modify the example file
        with open(dest_file, "r+") as file:
            content = file.read()
            content = content.replace('shot_number = ""', f'shot_number = "{shot_name}"')
            content = content.replace('overwrite = False', 'overwrite = True')
            content = content.replace('all_los = True', 'all_los = False')
            content = content.replace('exclude_lasers=[]', f'exclude_lasers={shot_laser_excluder[shot_name]}')
            file.seek(0)
            file.write(content)
            file.truncate()
            
        # run the file for the shot with error handling
        process = Popen(["python", dest_file], stdout=DEVNULL, stderr=PIPE)

        # wait while file is running
        process.wait()
        
        # get the console and console error of the subprocess
        stderr = process.communicate()[1]

        # after shot is ran overwrite back to False
        with open(dest_file, "r+") as file:
            content = file.read()
            content = content.replace('overwrite = True', 'overwrite = False')
            file.seek(0)
            file.write(content)
            file.truncate()

        # close all figures and clear console 
        plt.close()
        os.system('clear')
        
        #HACK: the logging does not work as intended and logs valid shots which ran, the real errored shots 
        # and the bugged logged shots are different such that the false logs include the console (loading of bars, etc..)
        # while the real errors only include the traceback
        
        # if we had an error raise it to be logged
        if stderr:
          error = stderr.decode()
          error = error.rpartition("Traceback (most recent call last):\n")[-1]
          raise Exception(error)
            
    # except the raised error and log it
    except Exception as e:

        # log the shot name along with the traceback info
        logging.error(f"error processing shot '{shot_name}':", exc_info=True)
   
        # this shot errored, we logged it, now we can continue to the next shot
        continue
        