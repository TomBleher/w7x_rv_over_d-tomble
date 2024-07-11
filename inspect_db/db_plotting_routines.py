#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:05:01 2024

@author: tomble
"""

import corner
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import cxrs_pyfidasim

from cxrs_pyfidasim.utilities.archive import (
    _fetch_profile_dict_from_archive,
    _unpack_dict,
    _write_profile_dict_to_archive,
)
from cxrs_pyfidasim.utilities.general import _format_plot_handle

from w7x_preparation_pystrahl.utilities.geometry import (
    get_magnetic_configuration,
    get_minor_radius,
)
from w7x_preparation_pystrahl.utilities.plots import plot_profile_overview

# load path to files in cxrs_pyfidasim
file_path = os.path.dirname(cxrs_pyfidasim.__file__)
file_path = os.path.abspath(os.path.join(file_path, os.pardir)) + "/Data"

from cxrs_pyfidasim.utilities.archive import _pack_dict

from w7x_preparation_pystrahl.profiles.profile_fitting import make_tanh_fit, make_mtanh_fit
from cxrs_pyfidasim.utilities.math import _find_data_norm

from w7x_preparation_pystrahl.utilities.plots import plot_profile_overview_ind, plot_impurity_overview_ind
from w7x_preparation_pystrahl.utilities.time import _create_pit_handle


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###################################################################################
##                                                                               ##           
##  All the functions in this file could be easily ran through run_db_plotting   ##
##                                                                               ##
###################################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

tanh_params = [
    "inner_amp",
    "inner_curve_loc",
    "inner_curve_slope",
    "outer_amp",
    "outer_curve_loc",
    "outer_curve_slope",
]

colors = [
    "tab:blue", "tab:green", "tab:red", "magenta", "yellow", "orange", "purple", "pink", "olive",
    "navy", "teal", "maroon", "sienna", "lime", "orchid", "turquoise", "salmon", "indigo", "cyan", "violet",
    "tomato", "steelblue", "coral", "dodgerblue", "goldenrod", "greenyellow",
    "hotpink", "lawngreen", "lightblue", "lightcoral", "lightgreen", "lightsalmon", "lightskyblue",
    "lightsteelblue", "mediumblue", "mediumorchid", "mediumseagreen", "mediumslateblue", "mediumspringgreen",
    "mediumturquoise", "mediumvioletred", "midnightblue", "palegreen", "paleturquoise", "palevioletred",
    "peachpuff", "plum", "seagreen", "skyblue", "slateblue", "springgreen", "tan", "thistle", "wheat", "yellowgreen"
]

colors_config = {"AHM": "green", "EIM": "black", "KJM": "blue", "FLM": "red", "FNM": "pink", "DAM": "lime"}

# path to output data
path = os.path.dirname(cxrs_pyfidasim.__file__)
path = os.path.abspath(os.path.join(path, os.pardir)) + "/Data/" 

# read the data and fit the tanh function
def read_data(
        additional_info: dict,
        fit_parameter: str
        )-> dict:
    """
    Method to read all data from jsons and organize into dict where each key 
    consists of the parameter name and all the available data for it across 
    different shots and different time points
 
    Parameters
    ----------
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x}
                }
    Returns
    -------
    dict
        dictionary containing all data for all shots and time points, dictionary 
        is used as the basis for all the functions in this file
    """
    
    # initialize dict in which the datatbase data will be locally stored
    data = {}
    
    # initialize array of errors for the mtanh fitting below 
    n_e_errors = []
    n_C6_errors = []
    n_C6 = []
    
    file_list = os.listdir(f"../{additional_info['species']}/")

    json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"

    # loop over all json files which contain errors to be passed to mtanh fitting later
    for json_idx, json_filename in enumerate(sorted(os.listdir(json_path))):
        
        json_file_path = os.path.join(json_path, json_filename)
        
        # open the file
        with open(json_file_path, "r") as file_obj:
            imp_file_data = json.load(file_obj)
        
        for time_idx, time_handle in enumerate(list(imp_file_data["n_e"]["parlog"].keys())):
            
            n_e_error = imp_file_data["n_e"]["error"][time_idx]
            
            carbon_density = imp_file_data["n_C6"]["fit"][time_idx]
            n_C6_error = imp_file_data["n_C6"]["error"][time_idx]
            
            n_C6.append(carbon_density)
            n_C6_errors.append(n_C6_error)
            n_e_errors.append(n_e_error)
        
    # loop over all shots
    for file_idx, file in enumerate(sorted([file for file in file_list if file.endswith('.json')])):

        # open the file
        with open(f"../{additional_info['species']}/{file}", "r") as file_obj:
            file_data = json.load(file_obj)
 
            # identify the latest version
            version = f"V{len(file_data.keys()) - 1}"
 
            # identify the rho radius we are interested in
            rho_idx = np.argmin(
                abs(np.array(file_data[version]["rho"][0]) - additional_info["rho"])
            )
 
            # add keys that contain the according lists
            file_data[version]["shots"] = np.unique(file.split(".json")[0])[0]
            file_data[version]["magnetic_configurations"] = get_magnetic_configuration(
                file_data[version]["shots"]
            )
            
            # add additional parameters which aren't included in the main json
            file_data[version]["n_C6"] = n_C6
            file_data[version]["n_e_errors"] = n_e_errors
            
            # loop over all points in time and add data points
            for entry in list(file_data[version].keys()):
    
                # we take care of rho via the above index
                if entry in ("Time stamp"):
                    continue
 
                # if not happended earlier, set up list for the entries
                if entry not in data:
                    data[entry] = []
 
                # loop through all points in time for set rho_idx
                for pit_idx in range(0, len(np.array(file_data[version]["rho"])[:, 0])):
                    # extract the nbi fraction here as this is only a single data point
 
                    # add the mag_config and shot for each time point to a list
                    # INFO parameters which don't change during the shot
                    if entry in (
                        "magnetic_configurations",
                        "shots",
                        "vmec_id",
                        "num_of_shots",
                        "num_of_time_points",
                        "fit_type"
                    ):
                        data[entry] = np.append(data[entry], file_data[version][entry])
 
                    # case where we have 1D data, i.e. we just take the
                    # data from the matching time points
                    # INFO global parameters which change per time point
                    elif entry in (
                        "f_nbi", 
                        "tau_e", 
                        "f_rad", 
                        "t_points",
                        "density_peaking_ratio",
                        "eta_e",
                        "eta_imp",
                        "total_carbon_density",
                        "line_integrated_density"
                    ):
                        
                        data[entry] = np.append(data[entry], file_data[version][entry][pit_idx])
                    # case where we have 2D data (times)x(radius)
                    # here we only want to take the specific rho value
                    # INFO radially dependent parameters which also change in time
                    else:
                        data[entry] = np.append(data[entry], file_data[version][entry][pit_idx][rho_idx])
 
            r_minor = get_minor_radius(file_data[version]["vmec_id"])
                
            # loop over all time points
            for idx_time, time in enumerate(file_data[version]["t_points"]):
                
                # perform fitting on data
                # better to use _perform_profile_fit as this does the normalization
                # internally
                
                # find the normalization factor 
                norm = _find_data_norm(np.abs(file_data[version][fit_parameter][idx_time])) 
                
                parameter_values, fit_summary = make_mtanh_fit(
                    np.array(file_data[version]["rho"][idx_time]) * r_minor,
                    np.array(file_data[version]["rho"][idx_time]) * r_minor, # pass r_eff
                    np.array(file_data[version][fit_parameter][idx_time]) / norm, # pass norm data
                    np.array(file_data[version]["n_e_errors"][idx_time]) / norm, # pass norm errors
                    fit_parameter,
                    return_fit_results=True # request function to return parameter values
                    )

                # for one time point in one shot we get the solution for the parameters
                # initiate an according key in the main data dict to keep track of the value
                # of the parameter over time
                for parameter in list(parameter_values.keys()):
                    if f"{parameter}_history" not in list(data.keys()):
                        data[f"{parameter}_history"] = [float(parameter_values[parameter])] # create key to hold param values over time
                    else: # a key was already made to hold the parameter's value
                        data[f"{parameter}_history"].append(float(parameter_values[parameter]))

                # add each fit for time point to list
                for param in list(fit_summary.keys()):
                    if param not in ["error", "outer_plateau", "result"]:
                        # if this parameter wasnt previously written, create an array
                        if param not in data:
                            data[param] = []
                        # add the data to the array
                        data[param] = np.append(data[param], fit_summary[param])                
 
    # cast everything to numpy
    for entry in data:
        data[entry] = np.array(data[entry])
 
    # add election and ion etas to data dictionary
    data['eta_e'] = np.divide(data['-dT_e_drho/T_e'], data['-dn_e_drho/n_e'])
    data['eta_imp'] = np.divide(data['-dT_i_drho/T_i'], data['-dn_imp_drho/n_imp'])
    
    return data

def prime_factorization(n: int) -> list:
    """
    Method to find factors of a number n; n=4 -> factors = [2,2]
    This is used in the calculate_rows_and_columns for the plot_database_overview()
    function to find the number of rows and columns for plotting 

    Parameters
    ----------
    n : int
        number which we ask to find prime factors of

    Returns
    -------
    list
        list with prime factors

    """
    
    factors = []
    if n < 2:
        return factors
    
    divisor = 2
    while n >= 2:
        if n % divisor == 0:
            factors.append(divisor)
            n /= divisor
        else:
            divisor += 1
    return factors

def calculate_rows_and_columns(length: int) -> tuple[int,int]:
    """
    Method to dynamically pass the numbers of rows and columns based on the length 
    of parameters desired to be plotted. This is used in plot_database_overview() 
    to get the final plot

    Parameters
    ----------
    length : int
        how many parameters does the user pass to list_colorsbars

    Returns
    -------
    tuple[int,int]
        tuple of [columns, rows]

    """
    if length < 2:
        return 1, length
    
    factors = prime_factorization(length)
    num_factors = len(factors)
    if num_factors == 1:
        return 1, length
    
    # find the factor closest to the square root of the length
    closest_factor = min(factors, key=lambda x: abs(x - int(length ** 0.5)))
    num_rows = closest_factor
    num_cols = length // num_rows
    
    # ensure num_cols is the longer dimension
    if num_cols > num_rows:
        num_rows, num_cols = num_cols, num_rows
    
    return num_cols, num_rows

def filter_data_by_dict(data: dict, filter_dict: dict) -> np.ndarray:
    """
    Method to filter by blindly stating ranges with dictionary of 
    form:
        filter_dict = { 
            "eta_imp": [0, 2.0],
            }
        
    anything outside this range will not be plotted; if nothing is passed all is 
    plotted

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()
    filter_dict : dict
        dictionary of formar above

    Returns
    -------
    dict_mask : np.ndarray of bool type
        used to then mask data with True values plotted

    """
    # total mask is initiated with all values True to assume no filter is needed
    dict_mask = np.full_like(data["vmec_id"], fill_value=True, dtype=bool)
        
    # loop through every parameter and its according filter range    
    for param in filter_dict.keys():
                
        # initiate every paramtere with Trues, assume ales gut
        dict_mask_parameter = np.full_like(data["vmec_id"], fill_value=True, dtype=bool)

        # check which of the data needs filtering for this parameter
        for idx_time, (time_point, shot_name) in enumerate(zip(data["t_points"], data["shots"])):
            # if parameter value is outside the range of filter defined above
            if not np.logical_and((data[param][idx_time] > filter_dict[param][0]), (data[param][idx_time] < filter_dict[param][1])):
                dict_mask_parameter[idx_time] = False        
                # print(f"Shot number: {shot_name}, Filtered data PIT: {time_point}")
                
        # update the total mask
        dict_mask = np.logical_and(dict_mask, dict_mask_parameter)
 
    return dict_mask

def filter_data_by_mag_config(
        data: dict, 
        mag_config: str
        ) -> np.ndarray:
    
    mag_config_mask = np.full_like(data["magnetic_configurations"], fill_value=False, dtype=bool)
    
    for idx, configuration in enumerate(data["magnetic_configurations"]):
        if configuration == mag_config:
            mag_config_mask[idx] = True
    
    return mag_config_mask 


def filter_data_by_figures(data: dict) -> np.ndarray:
    """
    Manual filtering that works by dragging the good fits into a dedicated folder
    so that only the figures which are labeled by shot and time point (=point in db)
    are plotted. To populate the figures please use populate_figure_db() then drag 
    the good fits to "Valid_figures" after manual inspection

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()

    Returns
    -------
    dict_mask : np.ndarray of bool type
        used to then mask data with True values plotted

    """
    # Define paths to figure path and valid figures (that passed manual inspection process) path
    figures_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/w7x_rv_over_d/Carbon/Figures"
    valid_figures_path = os.path.join(figures_path, "Valid_figures")

    valid_figures = [file for file in os.listdir(valid_figures_path) if file.endswith('.png')]
    
    # Initiate mask with all False
    figure_mask = np.full(len(data["shots"]), fill_value=False, dtype=bool)
    
    # Loop over all shots
    for shot_name in np.unique(data["shots"]):
        
        # Load the shot corresponding JSON file 
        with open(f"{file_path}/output_fitted_profiles/{shot_name}_impurity_profiles.json", "r") as file:
            shot_profile_dict = json.load(file)

        # Iterate over all time points 
        for idx, handle in enumerate(shot_profile_dict["n_e"]["parlog"].keys()):
            
            # Extract start and end times from the handle
            start_time, end_time = handle.split('_')[0].split('=')[-1], handle.split('_')[1]
            formatted_handle = f"t={start_time}_{end_time}"
            
            # Check if shot and handle are in valid folder, if so, set True
            if f'{shot_name}_{formatted_handle}_Carbon.png' in valid_figures:
                figure_mask[idx] = True       

    return figure_mask

def plot_database_overview(
        plot_x: str, 
        plot_y: str, 
        list_colorbars: list, 
        additional_info: dict, 
        data: dict,
        mask: np.ndarray,
        ) -> None:   
    """
    This is the main plotting function. It works both for color-coding plots and
    day, magnetic configuration, shot, sorted plots. It is built on the list_colorbars
    list which can handle both color-coding and special plots at the same time such that:
        list_colorbars = ["n_e", config_resolved]

    Parameters
    ----------
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    list_colorbars : list
        main list through which we can specify what figures to plot
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None
    
    """
    num_rows, num_cols = calculate_rows_and_columns(len(list_colorbars))
    
    # dynamically generate plot screen
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows), sharey=True, sharex=True)
    #fig.suptitle(f"$\\rho$ = {additional_info['rho']}")
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]  

    for idx, (param, ax) in enumerate(zip(list_colorbars, axes)):
                    
        if param in ["shot_resolved"]:
            plot_shot_resolved(ax, plot_x, plot_y, additional_info, data, mask)
            labelpad = 0  # adjust labelpad for shot resolved

        elif param in ["partial_shot_resolved"]:
            plot_partial_shot_resolved(ax, plot_x, plot_y, additional_info, data, mask)
            labelpad = 0  # adjust labelpad for shot resolved

        elif param in ["config_resolved"]:
            plot_config_resolved(ax, plot_x, plot_y, additional_info, data, mask)
            labelpad = 0  # adjust labelpad for shot resolved

        elif param in ["day_resolved"]:
            plot_day_resolved(ax, plot_x, plot_y, additional_info, data, mask)
            labelpad = 0  # adjust labelpad for shot resolved
        else:
            plot_colorbar(ax, plot_x, plot_y, param, data, mask)
            labelpad = 60  # adjust labelpad for other parameters

        if idx % num_cols == 0:
            ax.set_ylabel(additional_info['labels']['label_y'], labelpad=labelpad)
        
        if idx >= (num_rows - 1) * num_cols:  # check if the current subplot is in the last row
            ax.set_xlabel(additional_info['labels']['label_x'])
        
    for ax in axes[len(list_colorbars):]:
        ax.axis('off')
    
    plt.savefig(f'shot_resolved.png', dpi=800)
    plt.tight_layout()
    plt.show()
        

def plot_tanh_fit( #HACK: this data is already available in data dict, replace reading in func
        data: dict, # to use the already read data
        additional_info: dict, 
        mask: np.ndarray,
        fit_type: str,
        fit_parameter: str,
        ) -> None:
    """
    To inspect the validity of the tanh this function plots all the tanh fits of 
    all the shots and their time points

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    
    n_C6_errors = []
    n_C6 = []
    n_e_errors = []
    
    json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"

    # loop over all json files which contain errors to be passed to mtanh fitting later
    for json_idx, json_filename in enumerate(sorted(os.listdir(json_path))):
        
        json_file_path = os.path.join(json_path, json_filename)
        
        # open the file
        with open(json_file_path, "r") as file_obj:
            imp_file_data = json.load(file_obj)
        
        for time_idx, time_handle in enumerate(list(imp_file_data["n_e"]["parlog"].keys())):
            
            carbon_density = imp_file_data["n_C6"]["fit"][time_idx]
            n_C6_error = imp_file_data["n_C6"]["error"][time_idx]
            n_e_error = imp_file_data["n_e"]["error"][time_idx]
            
            n_e_errors.append(n_e_error)
            n_C6.append(carbon_density)
            n_C6_errors.append(n_C6_error)        
        
    for file_idx, file in enumerate(os.listdir(f"../{additional_info['species']}/")):
        
        # open the file
        with open(f"../{additional_info['species']}/{file}", "r") as file_obj:
            file_data = json.load(file_obj)

            version = f"V{len(file_data.keys()) - 1}"
            
        r_minor = get_minor_radius(file_data[version]["vmec_id"])
        
        file_data[version]["n_C6"] = n_C6
        file_data[version]["n_C6_errors"] = n_C6_errors
        file_data[version]["n_e_errors"] = n_e_errors
        
        for idx_time, time in enumerate(file_data[version]["t_points"]):         
            if mask[idx_time]:  # apply the mask
            
                norm = _find_data_norm(np.abs(file_data[version][fit_parameter][idx_time])) 
                
                fit_summary = make_mtanh_fit(
                    np.array(file_data[version]["rho"][idx_time]) * r_minor,
                    np.array(file_data[version]["rho"][idx_time]) * r_minor,
                    np.array(file_data[version][fit_parameter][idx_time]) / norm,
                    np.array(file_data[version][f"{fit_parameter}_errors"])/norm,
                    fit_parameter,
                    )
                
                fig, ax = plt.subplots()
                ax.set_title(file.split(".json")[0])
                ax.plot(np.array(file_data[version]["rho"][idx_time]) * r_minor,fit_summary['result'],label='tanh_fit')
                ax.plot(np.array(file_data[version]["rho"][idx_time]) * r_minor, np.array(file_data[version][fit_parameter][idx_time]) / norm, label='data')
                ax.plot(np.array(file_data[version]["rho"][idx_time]) * r_minor,fit_summary['outer_plateau'],label='outer_plateau')
                ax.set_xlabel(r"$r_{eff}$")
                ax.set_ylabel(r"$n_e$ in $ 10^{19}$ m$^{-3}$")
                ax.legend()
                ax.set_ylim(0,20)
                ax.set_xlim(-0.02,0.65)
                
def plot_colorbar(
        ax, 
        plot_x: str, 
        plot_y: str, 
        parameter: str, 
        data: dict, 
        mask: np.ndarray
        ) -> None:
    """
    Sub-plotting function to plot the color-coded parameters

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    parameter : str
         parameter based on which color-coding will be plotted
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    color = "plasma"
    
    if parameter in ["density_peaking_ratio", "line_integrated_density", "total_carbon_density"]:
        parameter_handle = str(parameter).replace("_", " ")
   
    elif parameter in ['a_1_history','a_2_history','a_3_history','b_1_history','b_2_history','b_3_history', 'b_4_history']:
        
        translation_dict = {
            "a_1": "Inner amplitude",
            "a_2": "Inner location",
            "a_3": "Inner gradient",
            "b_1": "Outer amplitude",
            "b_2": "Outer location",
            "b_3": "Outer gradient",
            "b_4": "Outer hollowness",            
            }
    
        parameter_handle = f"${{{parameter.split('_history')[0]}}}$ | {translation_dict[parameter.split('_history')[0]]}"

    elif parameter in ["eta_imp", "eta_e", "f_rad", "tau_e", "n_e", "f_nbi"]:
        parameter_handle = _format_plot_handle(parameter)
        
    filtered_data_x = np.array(data[plot_x])[mask]
    filtered_data_y = np.array(data[plot_y])[mask]
    
    if len(filtered_data_x) > 0:
        im = ax.scatter(
            filtered_data_x,
            filtered_data_y,
            c=np.array(data[parameter])[mask],
            vmin=np.nanmin(data[parameter][mask]),
            vmax=np.nanmax(data[parameter][mask]),
            cmap=color,
            s=5,
        )
    
        cbar = plt.colorbar(
            im,
            label=parameter_handle,
            ax=ax,
            location="left",
        )
    
        ticklabs = cbar.ax.get_yticklabels()
        ax.tick_params(axis='both', which='major', labelsize=6)
        cbar.ax.set_yticklabels(ticklabs, fontsize=6)

def plot_shot_resolved(
        ax, 
        plot_x: str, 
        plot_y: str, 
        additional_info: dict, 
        data: dict, 
        mask: np.ndarray
        ) -> None:
    """
    Sub-plotting function to sort the database based on the shot which is indicated
    by the legend

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }    data : dict
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    # get all shots that we have data from
    shots = np.unique(data["shots"])
    
    markers = [
        ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H',
        '+', 'x', 'D', 'd', 'P', 'X']
    
    # find the index of the data points which belong to the individual shots
    for idx, shot in enumerate(shots):
        
        marker = markers[idx % len(markers)]
        shot_indices = np.array(data["shots"]) == shot
        mask_indices = np.array(mask)[shot_indices]
        
        # get filtered data
        filtered_data_x = (np.array(data[plot_x])[shot_indices])[mask_indices]
        filtered_data_y = (np.array(data[plot_y])[shot_indices])[mask_indices]
        
        if len(filtered_data_x) > 0:
            ax.scatter(
                       filtered_data_x,
                       filtered_data_y,
                       color=colors[idx % len(colors)],
                       label=shot.split("/")[0].split(".json")[0],
                       marker=marker,
                       s=5,
                    )

    ax.set_xlabel(additional_info["labels"]["label_x"])
    ax.set_ylabel(additional_info["labels"]["label_y"])
    ax.set_ylim(-0.5, 15)
    ax.set_xlim(-0.2, 3)
    ax.legend(ncol=2, prop={'size': 3})


def plot_partial_shot_resolved(
        ax, 
        plot_x: str, 
        plot_y: str, 
        additional_info: dict, 
        data: dict, 
        mask: np.ndarray
        ) -> None:
    """
    Sub-plotting function to plot the database zoomed on the pre-kink area; this 
    is basically plot_shot_resolved(), but zoomed - yes this function is rather 
    funny (and not useful)
    
    P.S 
        Sorry function

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()
        
    Returns
    -------
    None

    """
    # get all shots that we have data from
    shots = np.unique(data["shots"])
    
    markers = [
        ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H',
        '+', 'x', 'D', 'd', 'P', 'X']
        
    # find the index of the data points which belong to the individual shots
    for idx, shot in enumerate(shots):
        
        marker = markers[idx % len(markers)]
        shot_indices = np.array(data["shots"]) == shot
        mask_indices = np.array(mask)[shot_indices]
        
        # get filtered data
        filtered_partial_data_x = (np.array(data[plot_x])[shot_indices])[mask_indices]
        filtered_partial_data_y = (np.array(data[plot_y])[shot_indices])[mask_indices]
        
        filtered_partial_data_x = (np.array(data[plot_x])[shot_indices])[mask_indices]
        filtered_partial_data_y = (np.array(data[plot_y])[shot_indices])[mask_indices]
        
        filtered_partial_data_y = filtered_partial_data_y[filtered_partial_data_x < 0.5]                       
        filtered_partial_data_x = filtered_partial_data_x[filtered_partial_data_x < 0.5] 
        
        filtered_partial_data_x = filtered_partial_data_x[filtered_partial_data_y < 0.025]
        filtered_partial_data_y = filtered_partial_data_y[filtered_partial_data_y < 0.025]                       

        if len(filtered_partial_data_x) > 0:
            ax.scatter(
                       filtered_partial_data_x,
                       filtered_partial_data_y,
                       color=colors[idx % len(colors)],
                       label=shot.split("/")[0].split(".json")[0],
                       marker=marker,
                       s=5,
                    )
            
    ax.set_xlim(-0.3,  0.5)
    ax.set_ylim(-0.3, 0.025)
    
    ax.set_xlabel(additional_info["labels"]["label_x"])
    ax.set_ylabel(additional_info["labels"]["label_y"])
    ax.legend(ncol=2, prop={'size': 3})


# TODO change the indexing to the "new" implementation used for the shot resolved
def plot_config_resolved(
        ax, 
        plot_x: str, 
        plot_y: str, 
        additional_info: dict, 
        data: dict, 
        mask: np.ndarray
        ) -> None:
    """
    Sub-plotting function that is used to sort the database based on the different
    magnetic configurations for each shot

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    # get all shots that we have data from
    magnetic_configurations, magnetic_config_count = np.unique(data["magnetic_configurations"], return_counts=True)
        
    # find the index of the data points which belong to the individual shots
    for idx, magnetic_configuration in enumerate(magnetic_configurations):
        mag_config_indices = np.array(data["magnetic_configurations"]) == magnetic_configuration   
        mask_indices = np.array(mask)[mag_config_indices]
        
        # get filtered data
        filtered_data_x = (np.array(data[plot_x])[mag_config_indices])[mask_indices]
        filtered_data_y = (np.array(data[plot_y])[mag_config_indices])[mask_indices]
        
        if len(filtered_data_x) > 0:
            ax.scatter(
                filtered_data_x,
                filtered_data_y,
                color = colors_config[magnetic_configuration[:3]],
                label = f"{magnetic_configuration} ({magnetic_config_count[idx]})",
                s=3,
            )
            
    ax.set_xlabel(additional_info["labels"]["label_x"])
    ax.set_ylabel(additional_info["labels"]["label_y"])
    ax.legend(prop={'size': 3})

def plot_day_resolved(
        ax, 
        plot_x: str, 
        plot_y: str, 
        additional_info: dict, 
        data: dict, 
        mask: np.ndarray
        ) -> None:
    """
    Sub-plotting function used to sort the database based on the experimental day 
    on which the shot took place

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    # get all shots that we have data from
    all_days = []
    
    for idx, shot in enumerate(data["shots"]): 
        all_days.append(data["shots"][idx].split(".")[0])
    
    days = np.unique(all_days)
    
    markers = [
        ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H',
        '+', 'x', 'D', 'd', 'P', 'X']
    
    # find the index of the data points which belong to the individual shots
    for idx, day in enumerate(days):
        
        marker = markers[idx % len(markers)]
        day_indices = np.array(all_days) == day
        mask_indices = np.array(mask)[day_indices]
        
        # get filtered data
        filtered_data_x = (np.array(data[plot_x])[day_indices])[mask_indices]
        filtered_data_y = (np.array(data[plot_y])[day_indices])[mask_indices]
        
        if len(filtered_data_x) > 0:
            ax.scatter(
                       filtered_data_x,
                       filtered_data_y,
                       color=colors[idx % len(colors)],
                       label=day,
                       marker=marker,
                       s=5,
                    )

    ax.set_xlabel(additional_info["labels"]["label_x"])
    ax.set_ylabel(additional_info["labels"]["label_y"])
    ax.legend(ncol=2, prop={'size': 3})



# from the reading and fitting performed above plot the result
def plot_tanh_param_time(
        data: dict, 
        mask: np.ndarray,
        fit_type: str
        ) -> None:
    """
    Function used to plot the different parameters of the tanh against time

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()
    fit_type:
        mtanh/tanh

    Returns
    -------
    None

    """
    
    if fit_type in ["mtanh"]:
        parameters = ['a_1','a_2','a_3','b_1','b_2','b_3', 'b_4']        
        
        parameter_history = ['a_1_history','a_2_history','a_3_history','b_1_history','b_2_history','b_3_history', 'b_4_history']        
        
        translation_dict = {
            "a_1": "Inner amp",
            "a_2": "Inner loc",
            "a_3": "Inner slope",
            "b_1": "Outer amp",
            "b_2": "Outer loc",
            "b_3": "Outer slope",
            "b_4": "Outer hollowness",            
            }
        
        
    elif fit_type in ["tanh"]:
        parameters = ['inner_amp', 'inner_curve_loc', 'inner_curve_slope',
                         'outer_amp', 'outer_curve_loc', 'outer_curve_slope']  
        
        parameter_history = ['inner_amp_history', 'inner_curve_loc_history', 'inner_curve_slope_history',
                         'outer_amp_history', 'outer_curve_loc_history', 'outer_curve_slope_history']  
        
    else:
        print("Only available fit types are tanh and mtanh")
        raise ValueError

    # get all shots that we have data from
    shots = np.unique(data["shots"])
                         
    # loop over every shot
    for idx, shot in enumerate(shots):
        shot_indices = np.array(data["shots"]) == shot
        mask_indices = np.array(mask)[shot_indices]
        
        if fit_type == "tanh":
            # for every new shot create plot screen
            fig, ax = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
            ax = ax.flatten()
            plt.suptitle(f"tanh fitting for {shot}")
        
        elif fit_type =="mtanh":
            # for every new shot create plot screen
            fig, ax = plt.subplots(2, 4, figsize=(8, 4), constrained_layout=True)
            ax = ax.flatten()
            plt.suptitle(f"tanh fitting for {shot}")            
            
        # loop over every parameter and axis
        for param_name, param_history in zip(parameters, parameter_history):
            
            axis_idx = parameters.index(param_name)

            # plot for each individual shot
            ax[axis_idx].plot(np.array(data["t_points"])[shot_indices][mask_indices], np.array(data[param_history])[shot_indices][mask_indices])
            ax[axis_idx].set_ylabel(translation_dict[param_name]) # change for tanh
            ax[axis_idx].set_xlabel(r'$t$ in [s]')
            ax[axis_idx].tick_params(axis="both", which="major", labelsize=7)
                
        plt.show()

def plot_heatmap(
        data: dict, 
        additional_info: dict,
        mask: np.ndarray, 
        ) -> None:
    """
    Function used to plot a figure containing all the parameters correlated against
    one another; Parameter correlation is colored by a heatmap according to their 
    correlation

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    
    # import seaborn here because it does annoying stuff with matplotlib plots if
    # imported above
    import seaborn as sns
    
    # Shape data to be plotted in 
    for entry in list(data.keys()):
        print(entry)
        if entry in ["magnetic_configurations", "vmec_id", "shots", "rho", "t_points", 
                  "outer_curve_slope", "outer_curve_loc", 
                  "fit_type", "error", "leastsq_total", "result", "outer_plateau"
                  ]:
            data.pop(entry)

    # apply mask
    if mask is not None:
        data_masked = {key: value[mask] for key, value in data.items()}
    else:
        data_masked = data

    sns.set_theme(style="ticks")

    parameter_keys = list(data_masked.keys())

    flatten_data = np.vstack([data_masked[key] for key in parameter_keys])
    corr = np.corrcoef(flatten_data)

    np.fill_diagonal(corr, np.nan)

    tri_mask = np.triu(np.ones_like(corr, dtype=bool))
    
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap='coolwarm',
        square=True,
        mask=tri_mask,
    )

    ax.set_xticklabels(
        parameter_keys,  
        rotation=45,
        horizontalalignment='right',
        fontsize=7
    )
    
    ax.set_yticks(np.arange(len(parameter_keys)) + 0.5)
    
    ax.set_yticklabels(
        parameter_keys, 
        rotation=0,
        fontsize=7  
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5)

    #plt.suptitle(f"$\\rho$ = {additional_info['rho']}")

    pos_corr = np.where((corr > 0.7) & (tri_mask == True))
    neg_corr = np.where((corr < -0.7) & (tri_mask == True))

    # collect highly correlated pairs
    pos_pairs = [(parameter_keys[i], parameter_keys[j], corr[i, j]) for i, j in zip(*pos_corr)]
    neg_pairs = [(parameter_keys[i], parameter_keys[j], corr[i, j]) for i, j in zip(*neg_corr)]

    # Sort the lists
    pos_pairs.sort(key=lambda x: x[2], reverse=True)
    neg_pairs.sort(key=lambda x: x[2])

    exclude_parameters = ['inner_amp', 'inner_curve_loc', 'inner_curve_slope', 'outer_amp', 'outer_curve_slope', 'outer_curve_loc']
   
    print("Positive Correlations:")
    for pair in pos_pairs:
        if pair[0] not in exclude_parameters and pair[1] not in exclude_parameters:
            print(f"{pair[0]}, {pair[1]}: {pair[2]}")
    
    print("\nNegative Correlations:")
    for pair in neg_pairs:
        if pair[0] not in exclude_parameters and pair[1] not in exclude_parameters:
            print(f"{pair[0]}, {pair[1]}: {pair[2]}")
                
def plot_corner(
        data: dict, 
        save_path: str, 
        mask: np.ndarray, 
        dpi: int
        ) -> None:
    """
    Function used to plot the corner plot. It is a large image so we save it for 
    better viewing and quality

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()
    save_path : str
        where should the saved image go to 
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()
    dpi : int
        basically quality, no need to get fancy with definition; set it for around dpi = 300

    Returns
    -------
    None

    """
    # apply mask
    if mask is not None:
        data_masked = {key: value[mask] for key, value in data.items()}
    else:
        data_masked = data
            
    # shape data to be plotted in 
    for entry in ["magnetic_configurations", "vmec_id", "shots", "rho", "t_points", 'inner_amp', 'inner_curve_loc', 'inner_curve_slope', 'outer_amp', 'outer_curve_slope', 'outer_curve_loc']:
        data.pop(entry)   
        
    parameter_keys = list(data.keys())
    
    # apply the mask to the data
    flatten_data = np.vstack([data_masked[key] for key in parameter_keys]).T            
               
    corner.corner(
        flatten_data,
        labels=parameter_keys,
    )
    plt.savefig(save_path, dpi=dpi)


def find_kink(
        plot_x: str, 
        plot_y: str, 
        data: dict, 
        additional_info: dict,
        mask: np.ndarray, 
        ) -> None:
    """
    The "Kink" is what we call the vertical line which seperates the current flat
    data from the database to the sloped data in the database. In the regular plotting
    for x,y specified below this translates to peaked/flat impurity profiles which are
    in turn neoclassicaly and anomalously dominated

    Parameters
    ----------
    plot_x : str
        what should we plot on horizontal axis; usually set to "-dn_e_drho/n_e"s
    plot_y : str
        what should we plot on the vertical axis; usually set to "-dn_imp_drho/n_imp"
    data : dict
        main database data dictionary gained from read_data()
    additional_info : dict
        dictionary of the following form:
            additional_info = {
                'rho': rho,
                'species': species,
                'labels': {'label_y': label_y, 'label_x': label_x},
                }
    mask : np.ndarray
        boolean mask numpy array from either filter_by_figure() or filter_by_dict()

    Returns
    -------
    None

    """
    # arrays to store kink and correlations for each iteration
    kink_values = []
    left_correlations = []
    right_correlations = []

    # remove unused data from dict
    keys_to_remove = []
    for key in data.keys():
        if key not in [plot_x, plot_y]:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        data.pop(key)

    # apply filters if there are any
    if mask is not None:
        data = {key: value[mask] for key, value in data.items()}
    else:
        data = data
    
    kink_step_size = 0.01
    cloud_cut_x = 1
    cloud_cut_y = 4
    
    # for each kink loc  
    for kink_loc in np.arange(0.2, 0.95, kink_step_size):
    
        # initialize kink_mask with Trues
        kink_mask = np.ones_like(data[plot_x], dtype=bool)
        
        # paint right group as False
        for idx, value in enumerate(data[plot_x]):
            if value > kink_loc:
                kink_mask[idx] = False     
                
        cloud_mask = np.ones_like(data[plot_x], dtype=bool)

        # paint right cloud as False
        for idx, (value_x, value_y) in enumerate(zip(data[plot_x], data[plot_y])):
            if value_x > cloud_cut_x or value_y > cloud_cut_y:
                cloud_mask[idx] = False     

        right_mask = np.logical_and(cloud_mask, ~kink_mask)

        flatten_left_plot_data = np.vstack([data[key][kink_mask] for key in data])
        flatten_right_plot_data = np.vstack([data[key][right_mask] for key in data])
    
        # get the covariance matrix for left and right groups to kink
        x_y_corr_left = np.corrcoef(flatten_left_plot_data) 
        x_y_corr_right = np.corrcoef(flatten_right_plot_data)
        
        # get the x,y correlation from matrix above
        x_y_corr_left_value = x_y_corr_left[0, 1]
        x_y_corr_right_value = x_y_corr_right[0, 1]
        
        kink_values.append(kink_loc)
        left_correlations.append(x_y_corr_left_value)
        right_correlations.append(x_y_corr_right_value)           
    
    # plot kink vs correlation plot
    plt.ylim(-1,1)
    plt.plot(kink_values, right_correlations, label='Right Group Correlation')
    plt.plot(kink_values, left_correlations, label='Left Group Correlation')
    plt.title(f"kink step size: {kink_step_size}")
    plt.xlabel('Kink Location')
    plt.ylabel('Correlation')
 
    plt.legend(loc="lower left")
    plt.show()    

def plot_fitted_imp_main_profiles_single(
        shot_number: str,
        t_start: float,
        t_stop: float,
        impurity: str,
        ):
    
    # load path to files in cxrs_pyfidasim
    file_path = os.path.dirname(cxrs_pyfidasim.__file__)
    file_path = os.path.abspath(os.path.join(file_path, os.pardir)) + "/Data/"
    
    """
    Method to plot the main and impurity profiles in one figure. Function takes one
    time interval and could be looped to result in multi-time plots.

    Parameters
    ----------
    shot_number : str
        number of shot to be plotted
    t_start : float
        starting time for plotting interval
    t_stop : float
        stopping time for plotting interval
    impurity : str
        impurity to be plotted in impurity fit
    Returns
    -------
    None.

    """
    
    # load path to the output folder for the figures
    figure_path = f"/home/IPP-HGW/tomble/Desktop/Code/git/w7x_rv_over_d/{impurity}/Figures"
    
    # load json file 
    with open(f"{file_path}/output_fitted_profiles/{shot_number}_impurity_profiles.json", "r") as file:
        profile_dict = _pack_dict(json.load(file))

    try:
        profile_dict["n_e"]["parlog"].pop("chanDescs")
    except KeyError:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Adjust figsize as needed
    handle = _create_pit_handle(t_start, t_stop)
    
    # plot main plasma profile and impurity overview side by side

    t_idx = np.where(np.array(list(profile_dict["T_e"]["parlog"].keys())) == handle)[0][0]

    plot_profile_overview_ind(
        profile_dict, "gp", shot_number, t_idx, t_start, t_stop, ax=axes[0]
    )

    # TODO change marker and add legend
    plot_impurity_overview_ind(
        profile_dict, 
        "tanh", 
        shot_number, 
        t_idx,
        t_start,
        t_stop,
        impurity,
        ax=axes[1]
        ) 

    plt.tight_layout()

    # save the figure
    plt.savefig(f'{figure_path}/{shot_number}_{handle}_{impurity}.png', dpi=300)

    
def populate_figure_db(
        data: dict
        ) -> None:
    """
    Populate the figure database based on which filter_by_figure() works. This 
    plots the impurity and main profiles side to side for each shot and its respective
    time points. The file name is detailed and could be used for other uses when broken apart

    Parameters
    ----------
    data : dict
        main database data dictionary gained from read_data()

    Returns
    -------
    None

    """
    # initiate shot list from directory
    shots = np.zeros_like(os.listdir(f"{file_path}/output_fitted_profiles"))
    
    # loop over directory to get list all the shots
    for file_idx, file_name in enumerate(os.listdir(f"{file_path}/output_fitted_profiles")):
        shots[file_idx] = file_name.split('_')[0]
    
    # loop over all shots and add to input dictionary
    for shot_number in shots:
                
        # load the shot corresponding JSON file 
        with open(f"{file_path}/output_fitted_profiles/{shot_number}_impurity_profiles.json", "r") as file:
            shot_profile_dict = _pack_dict(json.load(file))
            
        # get all the time points for the shot (should be the same time points for the impurity)

        try:
            shot_profile_dict["n_e"]["parlog"].pop("chanDescs")
        except KeyError:
            pass
        
        for handle in shot_profile_dict["n_e"]["parlog"].keys():

            # translate t_handle to t_start and t_stop to pass to plotting function
            t_start = float(handle.split("_")[0].split("=")[1])
            t_stop = float(handle.split("_")[1])

            # keep the time points in a string format for the figure mask 
            t_start_str = handle.split("_")[0].split("=")[1]
            t_stop_str = handle.split("_")[1]
            
            # get the figure file path for every shot
            figure_shot = f"{shot_number}_t={t_start_str}_{t_stop_str}_Carbon.png" # HACK: hardcoded impurity
                
            # define paths to figure path and valid figures (that passed the manual inspection process) path
            figures_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/w7x_rv_over_d/Carbon/Figures"
            valid_figures_path = f"{figures_path}/Valid_figures"
            
            # check if figure is in the valid figures folder - if so, move on to the next shot or time point in shot
            if figure_shot in os.listdir(figures_path):
                continue
            
            # if shot is not in the valid figures folder, overplot it
            else:
                # plot all data points for current shot
                plot_fitted_imp_main_profiles_single(
                    shot_number,
                    t_start,
                    t_stop,
                    "Carbon", # HACK: hardcoded impurity
                    )
            
        # to enable remote running close figures after each shot to keep program running
        plt.close()

        
