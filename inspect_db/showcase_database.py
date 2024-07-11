#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:44:14 2024

@author: tomble
"""

# line integrated denisty
from w7xdia.interferometer import get_line_integrated_density

# ecrh power
from w7xdia.ecrh import read_ecrh_total_power

# nbi power 
from w7xdia.nbi import get_total_nbi_power_for_program

import json, os, matplotlib.pyplot as plt, numpy as np

import logging 
logging.disable(logging.CRITICAL)

from w7x_preparation_pystrahl.utilities.geometry import (
    get_magnetic_configuration,
)

from cxrs_pyfidasim.utilities.general import _format_plot_handle

# for multiple shots plotsW
shots = ["20230214.056", "20230316.023", "20221201.036"]

# set thresholds for under which power plots will ignore data
p_nbi_threshold = 1.0
p_ecrh_threshold = 1.0

def check_heating_for_multi_time_plot(
        t_start: float,
        t_stop: float,
        shot_name: str,
        )-> [str, str]:
        """
        Method to get the heating for a shot's single time interval
    
        Parameters
        ----------
        t_start : float
            start of time interval to be checked
        t_stop : float
            end of time interval to be checked
        shot_name : str

        Returns
        -------
        (str)
            returns the heating method as str
    
        """
        # get the ecrh power per shot
        nbi_shot_time, nbi_shot_power = get_total_nbi_power_for_program(shot_name)
        ecrh_shot_time, ecrh_shot_power = read_ecrh_total_power(shot_name)
    
        nbi_shot_power = nbi_shot_power[nbi_shot_time > 0]               
        nbi_shot_time = nbi_shot_time[nbi_shot_time > 0]

        ecrh_shot_power = ecrh_shot_power[ecrh_shot_time > 0]               
        ecrh_shot_time = ecrh_shot_time[ecrh_shot_time > 0]
            
        # get the average NBI power in this window
        nbi_indices = (nbi_shot_time > t_start) & (nbi_shot_time < t_stop)
        nbi_power_window = np.mean(nbi_shot_power[nbi_indices])
        
        ecrh_indices = (ecrh_shot_time > t_start) & (ecrh_shot_time < t_stop)
        ecrh_power_window =  np.mean(ecrh_shot_power[ecrh_indices])
        
        # both nbi and ecrh
        if nbi_power_window > p_nbi_threshold and ecrh_power_window > p_nbi_threshold:
            heating = "NBI+ECRH"

        # pure ecrh 
        elif ecrh_power_window > p_ecrh_threshold: 
            heating = "ECRH"

        # pure nbi heated
        elif nbi_power_window > p_nbi_threshold:
            heating = "NBI"
    
        return heating
    

def take_items_equally_spaced(lst, num_items):
    """
    Method to retrieve items from a list at equally spaced intervals.

    Parameters
    ----------
    lst : list
        The input list from which items are to be retrieved
    num_items : int
        The number of items to be retrieved from the list

    Returns
    -------
    list
        list containing the selected items at equally spaced intervals
    """
    
    if num_items <= 1:
        return lst[:1]
    interval = max(len(lst) // (num_items - 1), 1)
    return [lst[i] for i in range(0, len(lst), interval)]

def get_time_handles_per_param_shot(
        parameter: str,
        shot_name: str,
        *,
        heating = None
        )-> list:
    """
    Method to get all the time handles for a particular shot

    Parameters
    ----------
    shot_name : str
        the name of the shot we are interested in 

    Returns
    -------
    time_handles : list
        all the time handles for the shot
    """
    
    # general directory with json files
    json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"
    
    # get the name of the json file for the shot
    json_filename = f"{shot_name}_impurity_profiles.json"
    
    # find path for single file
    json_file_path = os.path.join(json_path, json_filename)

    # load json file for shot name 
    with open(json_file_path, "r") as file_obj:
        file_data = json.load(file_obj)
    
    time_handles = list(file_data[parameter]["parlog"].keys())
    
    if heating is None:
        return time_handles
    
    # if heating was passed - return the handles of the heating phase
    else:

        heating_indices = np.full_like(time_handles, fill_value=False, dtype=bool)

        # get all handles into t_start t_stop lists and check heating for interval
        for idx, handle in enumerate(time_handles):
            
            t_start = float(handle.split("_")[0].split("=")[1])
            t_stop = float(handle.split("_")[1])
            
            if check_heating_for_multi_time_plot(t_start, t_stop, shot_name) == heating:
                heating_indices[idx] = True
                
        return np.array(time_handles)[heating_indices]
        
# for multiple time points
def plot_time_trace_per_param(
        parameter: str,
        shot_name: str,
        time_points: list
        )-> None:
    """
    Method to overplot multiple time points of the electron 
    density in a single discharge

    Parameters
    ----------
    param : str
        parameter for which time traces will be plotted
    shot_name : str
        desired shot to plot
    time_points : list
    
        specify the time points to overplot, the time points are in form of the handle
        t=t1_t2 where for t_1 and t_2 either use the get_handle function or
        change to form of f"t={t_start_ind:.4f}_{t_stop_ind:.4f}" such that "t=1.0000_1.1000"
        
        there is an option to pass "all" which
        in turn will plot all the time points available
    
    Returns
    -------
    None
    
    """
    
    heating_linestyle_dict = {
        "NBI": "-",
        "ECRH": "-.",
        "NBI+ECRH": "--"
        }
     
    # used to set ylim in the plot, initialized at 0, each time point new max is taken until we arrive at 
    # maximum of all time points
    max_n_e = 0 
    
    # general directory with json files
    json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"
    
    # get the name of the json file for the shot
    json_filename = f"{shot_name}_impurity_profiles.json"
    
    # find path for single file
    json_file_path = os.path.join(json_path, json_filename)

    # load json file for shot name 
    with open(json_file_path, "r") as file_obj:
        file_data = json.load(file_obj)
    
    time_handles = list(file_data[parameter]["parlog"].keys())
    time_indices = range(0, len(time_handles))

    # start filter for passed time points
    time_indices_mask = np.full_like(list(file_data[parameter]["parlog"].keys()), fill_value=False, dtype=bool)
    
    for mask_idx, _time_handle in enumerate(list(file_data[parameter]["parlog"].keys())):
        if _time_handle in time_points:
            time_indices_mask[mask_idx] = True

    # get colors for the time traces using linspace on a matplotlib colormap
    colors = plt.cm.viridis(np.linspace(0, 0.87, len(np.array(time_indices)[time_indices_mask])))
    # loop over relevant time points and plot in single plot
    for iter_idx, time_idx in enumerate(np.array(time_indices)[time_indices_mask]):
        
        # get the current handle for the time_idx
        time_handle = time_handles[time_idx]
        
        # translate t_handle to t_start and t_stop to pass to plotting function
        t_start = float(time_handle.split("_")[0].split("=")[1])
        t_stop = float(time_handle.split("_")[1])
        
        short_time_handle = f"t = {t_start:.1f}_{t_stop:.1f}"
        
        heating = check_heating_for_multi_time_plot(t_start, t_stop, shot_name)
        
        # plot the time point with the respective line style and label
        plt.plot(file_data["rho"][time_idx],
                 file_data[parameter]["fit"][time_idx],
                 color = colors[iter_idx],
                 linestyle=heating_linestyle_dict[heating],
                 label=f"{short_time_handle}, {heating}")    
    
        # for each time point find the biggest y value check for all time points
        # and finally set the y limit as the maximum among all time points
        if np.max(file_data[parameter]["fit"][time_idx]) > max_n_e:
            max_n_e = np.max(file_data[parameter]["fit"][time_idx])

    ncol = len( np.array(time_indices)[time_indices_mask])//4
    if ncol != 0:
        ncol=ncol
    else:
        ncol=1
    plt.legend(fontsize="7", ncol=ncol)
    plt.xlim(0,1.2)
    plt.ylim(0,None)#1.5e18)#max_n_e*1.2)
    
    plt.title(f"Time evolution of {parameter} for {shot_name}")
    if parameter in ["n_C6", "n_e"]:
        plt.ylabel(_format_plot_handle(parameter) + " in m" + r"$^{-3}$")
    elif parameter in ["T_e", "T_i"]:
        plt.ylabel(_format_plot_handle(parameter) + " in " + r"$keV$")
    plt.xlabel(r"$\rho$")
        
    # finally shot the plot
    plt.show()

def plot_multi_shot_integrated_density_power(
        shots: list,
        *,
        time_handles = None,
        overplot = None,
        )-> None:
        """
        This method is used to plot the power and integrated density overview
        plots for multiple shots

        Parameters
        ----------
        shots : list
            list of shots for which the power and integrated density will be plotted
    
        time_handles : list
        
            specify the time points to overplot, the time points are in form of the handle
            t=t1_t2 where for t_1 and t_2 either use the get_handle function or
            change to form of f"t={t_start_ind:.4f}_{t_stop_ind:.4f}" such that "t=1.0000_1.1000"
        
        overplot : bool
            if passed multiple shots, do you wish all to be overplotted in one figure or 
            for each to have a seperate figure
        Returns
        -------
        None
    
        """
        # if str was passed for shot then correct it to be a list
        if type(shots) is str:
            shots = [shots]
        
        if overplot is not None:
            #get the power and density plots in one plot
            fig, axs = plt.subplots(2, sharex=True)            
        
        if time_handles is not None:
            # get colors for the time traces using linspace on a matplotlib colormap
            colors = plt.cm.viridis(np.linspace(0, 0.87, len(time_handles)))
            
        elif time_handles is None and len(shots)>1:
            #colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
            colors = ["black", "tab:orange", "darkred"]
            
        # loop over shots passed to function
        for idx, shot in enumerate(shots):
            
            if overplot is None:
                # get the power and density plots in one plot
                fig, axs = plt.subplots(2, sharex=True)
            
            magnetic_config = get_magnetic_configuration(shot)
            
            # get the line integrated density per shot
            line_integrated_density_time, line_integrated_density = get_line_integrated_density(shot)

            # get the ecrh power per shot
            nbi_shot_time, nbi_shot_power = get_total_nbi_power_for_program(shot)
            ecrh_shot_time, ecrh_shot_power = read_ecrh_total_power(shot)

            # filter out negative time
            line_integrated_density = line_integrated_density[line_integrated_density_time > 0]    
            line_integrated_density_time = line_integrated_density_time[line_integrated_density_time > 0]    
            
            nbi_shot_power = nbi_shot_power[nbi_shot_time > 0]               
            nbi_shot_time = nbi_shot_time[nbi_shot_time > 0]
        
            ecrh_shot_power = ecrh_shot_power[ecrh_shot_time > 0]               
            ecrh_shot_time = ecrh_shot_time[ecrh_shot_time > 0]
            
            nbi_shot_power = nbi_shot_power[::12]
            nbi_shot_time = nbi_shot_time[::12]
        
            ecrh_shot_power = ecrh_shot_power[::500]         
            ecrh_shot_time = ecrh_shot_time[::500]
   
            if len(shots) > 1:
                if idx == 0:            
                    axs[0].plot(nbi_shot_time, nbi_shot_power, color=colors[idx], linestyle="-", label=r"$P_{NBI}$")
                    axs[0].plot(ecrh_shot_time, ecrh_shot_power , color=colors[idx], linestyle=":", label=r"$P_{ECRH}$")
                else:           
                    axs[0].plot(nbi_shot_time, nbi_shot_power, color=colors[idx], linestyle="-")
                    axs[0].plot(ecrh_shot_time, ecrh_shot_power , color=colors[idx], linestyle=":")
    
            else:
                # if we only have one shot no need to put it on the plot
                axs[0].plot(nbi_shot_time, nbi_shot_power, color="tab:green", label=r"$P_{NBI}$")
                axs[0].plot(ecrh_shot_time, ecrh_shot_power, color="tab:blue", label=r"$P_{ECRH}$")                
            
            if len(shots) > 1:
                # second plot is the line density
                axs[1].plot(line_integrated_density_time, line_integrated_density, color=colors[idx], label=f"{shot}, {magnetic_config}")
            
            else:
                # second plot is the line density
                axs[1].plot(line_integrated_density_time, line_integrated_density, label=f"{shot}")

            # only one shot was passed and the request to highlight the times is passed - plot them
            if len(shots) == 1 and time_handles is not None:
                for iter_idx, handle in enumerate(time_handles):
                      
                    # get mean of t_start t_stop from handle
                    t_start = float(handle.split("_")[0].split("=")[1])
                    t_stop = float(handle.split("_")[1])
                    
                    short_time_handle = f"t = {t_start:.1f}_{t_stop:.1f}"
                    t_mean = round((t_start+t_stop)/2, 2) # take mean and round to two decimal points
                    
                    # plot vertical line on both plots
                    axs[0].axvline(x=t_mean, color=colors[iter_idx], linestyle="--", label=f"{short_time_handle}", lw=2)
                    axs[1].axvline(x=t_mean, color=colors[iter_idx], linestyle="--", label=f"{short_time_handle}", lw=2)
                                        
            elif len(shots) != 1 and time_handles:
                print("Only one shot must be passed for time highlights")
            
            if time_handles is not None:
                ncol = len(time_handles)//4
                if ncol != 0:
                    ncol=ncol
                else:
                    ncol=1
                axs[0].legend(fancybox=True, framealpha=0.5, loc="upper right", fontsize="6", ncol=ncol)
                axs[1].legend(fancybox=True, framealpha=0.5, loc="upper right", fontsize="6", ncol=ncol)
                        
            axs[0].set_ylim(0, np.max(np.concatenate([np.array(nbi_shot_power), np.array(ecrh_shot_power)]))*1.1)
            if len(shots)==1:
                axs[0].text(1 - 0.01, 0.02, f"{magnetic_config}", fontsize=8, ha='right', va='bottom', transform=axs[0].transAxes)

            axs[1].set_ylim(0,np.max(line_integrated_density)*1.2)

            # x axis is shared for both        
            axs[1].set_xlabel(r"$t$ in $[s]$")
            
            # upper pannel is power
            axs[0].set_ylabel(r"$P$ $[MW]$")
            
            #plt.text(1,1, f"{shot_name}")
            # lower pannel in density
            axs[1].set_ylabel(r"$\int$$n_{e}$$dl$ in $10^{19}$ $[m^{-2}]$")
            #plt.suptitle(f"{shot}")
            axs[1].legend(loc='upper right', fontsize=6)
            axs[0].legend(loc='upper right', fontsize=6)
            plt.xlim(0,max(line_integrated_density_time)*1.2)
            
            if len(shots)>1:
                leg = axs[0].get_legend()
                leg.legendHandles[0].set_color('black')
                leg.legendHandles[1].set_color('black')

        plt.show()

def plot_imp_electron_density_ratio(
        shot_name: str,
        locations: list,
        impurity: str
        )-> None:
    """
    Method to get the concentration (=ratio of imp to electron density at rho location) time plot

    Parameters
    ----------
    locations : list
        rho locations for which the concetration will be takes
    impurity : str
        impurity which will be divided by n_e

    Returns
    -------
    None

    """
    if parameter == "n_e":
        print("Can not compare n_e to n_e")
        raise ValueError
    
    # general directory with json files
    json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"
    
    # get the name of the json file for the shot
    json_filename = f"{shot_name}_impurity_profiles.json"
    
    # find path for single file
    json_file_path = os.path.join(json_path, json_filename)

    # load json file for shot name 
    with open(json_file_path, "r") as file_obj:
        file_data = json.load(file_obj)    

    # loop over passed list and for each requested location plot a figure
    for loc_idx, location in enumerate(locations):
        
        imp_n_e_ratios = []
        times = []
        
        # loop over all times
        for time_idx, handle in enumerate(list(file_data[parameter]["parlog"].keys())):
            
            # take the mean of the handles and add to list which will be used for plotting
            t_start = float(handle.split("_")[0].split("=")[1])
            t_stop = float(handle.split("_")[1])
            t_mean = round((t_start+t_stop)/2, 2)
            times.append(t_mean)

            # find the actual rho index in json closest to passed rho
            json_rho_idx = np.abs(np.array(file_data["rho"][time_idx]) - location).argmin()
            
            # find the actual rho index in json closest to passed rho
            json_rho = file_data["rho"][time_idx][json_rho_idx]
            
            # for each location in passed locations calculate the ration and plot
            imp_n_e_ratio = (np.array(file_data[impurity]["fit"])[time_idx, json_rho_idx]/
                             np.array(file_data["n_e"]["fit"])[time_idx, json_rho_idx])
            
            # add to respective ratio list
            imp_n_e_ratios.append(imp_n_e_ratio)

        plt.plot(times, np.array(imp_n_e_ratios)*100, label=f"$\\rho={location}$") # *100 for precent
    
    plt.text(1 - 0.01, 0.02, f"{shot_name}", fontsize=8, ha='right', va='bottom', transform=plt.gca().transAxes)
    plt.xlabel(r"$t$ " "in seconds")
    plt.ylabel(_format_plot_handle("n_C6") + "/" + _format_plot_handle("n_e") + " in " + r"%")
    plt.ylim(bottom = 0)
    plt.legend()
    plt.show()

shot_name = "20230316.066"
parameter = "n_C6"

time_points = take_items_equally_spaced(get_time_handles_per_param_shot(parameter, shot_name, heating="NBI+ECRH"),5) 
#time_points = get_time_handles_per_param_shot(parameter, shot_name, heating=None)#[::6]

plot_time_trace_per_param(parameter, shot_name, time_points)
plot_multi_shot_integrated_density_power([shot_name], time_handles=None, overplot=None)#, time_handles=time_points)
plot_imp_electron_density_ratio(shot_name, [0.1, 0.4, 0.6, 0.8], parameter)