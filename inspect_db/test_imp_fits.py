#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:13:45 2024

@author: tomble
"""
from cxrs_pyfidasim.utilities.general import _format_plot_handle

import numpy as np
import os 
import json
import matplotlib.pyplot as plt
from w7x_preparation_pystrahl.profiles.profile_fitting import _perform_profile_fit
from w7x_preparation_pystrahl.utilities.fits import make_mtanh_fit

from cxrs_pyfidasim.utilities.math import _find_data_norm
from w7x_preparation_pystrahl.utilities.geometry import (
    get_reference_equilibrium,
    get_inflated_paradigm_vmec_id,
    get_minor_radius,
)
json_path = r"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles"

def plot_all_db_impurity_fits(
    impurity: str,
    fit_type: str
    )-> None:
    """
    Method to check all the impurity fittings in the database

    Parameters
    ----------
    impurity : str
        impurity handle: n_c6, etc..
    fit_type : str
        what algorithm is used to fit the data: "tanh"/"mtanh"/"gp"/"lowess"

    Returns
    -------
    None

    """
    
    # loop over all json files in dictionary 
    for json_idx, json_filename in enumerate(os.listdir(json_path)):
        
        json_file_path = os.path.join(json_path, json_filename)
        
        # open the file
        with open(json_file_path, "r") as file_obj:
            file_data = json.load(file_obj)
        
        # get the shot name
        shot_name = json_filename.split("_")[0]
        
        # pass r_eff rather than rho
        a = 0.5
        
        # loop over all time points and extract data
        for time_handle in list(file_data[impurity]["parlog"].keys()):
    
            norm = _find_data_norm(file_data[impurity]["parlog"][time_handle]["values"])
            vmec_id = get_reference_equilibrium(shot_name)[0]
            r_lcfs = get_minor_radius(vmec_id)

            # call the fitting function
            fit, error = _perform_profile_fit(
                fit_type,
                np.linspace(0, 0.5, 100),
                np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
                np.array(file_data[impurity]["parlog"][time_handle]["values"]),
                np.array(file_data[impurity]["parlog"][time_handle]["errors"]),
                r_lcfs,
                parameter = impurity,
                norm=norm,
            )
        
            # plot impurity data
            plt.errorbar(
                np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
                np.array(file_data[impurity]["parlog"][time_handle]["values"]),
                yerr=file_data[impurity]["parlog"][time_handle]["errors"],
                color='black', ls="",
                marker='_',  
                label=impurity
            )
                
            # plot mtanh fit
            plt.plot(np.linspace(0, 0.5, 100),  # linear array in rho
                     fit, label=fit_type)
                    
            #plt.ylim(-0.5e18, 4.5e18)
            plt.xlabel(r"$r_{{eff}}$")
            plt.ylim(0,None)
            plt.ylabel(_format_plot_handle(impurity))
            plt.xlim(0,0.5)
            plt.title(f"{fit_type} fit for {shot_name} at {time_handle}")
            plt.legend()
            plt.show()

def plot_imp_single_shot_time(
        shot_name: str,
        impurity: str,
        time_handle: str,
        fit_type: str
        )-> None:
    """
    PLot a single shot at a single time point

    Parameters
    ----------
    shot_name : str
        shot number in yyyy.mm.dd
    impurity : str
        impurity handle: n_c6, etc..
    time_handle : str
        in 4f: form 
    fit_type : str
        what algorithm is used to fit the data: "tanh"/"mtanh"/"gp"/"lowess"

    Returns
    -------
    None

    """
    with open(f"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles/{shot_name}_impurity_profiles.json", "r") as file_obj:
        file_data = json.load(file_obj)
    
    # pass r_eff rather than rho
    a = 0.5
    
    norm = _find_data_norm(file_data[impurity]["parlog"][time_handle]["values"])
    vmec_id = get_reference_equilibrium(shot_name)[0]
    r_lcfs = get_minor_radius(vmec_id)
    
    fit_summary = make_mtanh_fit(
        np.linspace(0, 0.5, 100),
        np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
        np.array(file_data[impurity]["parlog"][time_handle]["values"]) / norm,
        np.array(file_data[impurity]["parlog"][time_handle]["errors"]) / norm,
        fit_param = impurity,
        )

    fit, error = _perform_profile_fit(
        fit_type,
        np.linspace(0, 0.5, 100),
        np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
        np.array(file_data[impurity]["parlog"][time_handle]["values"]),
        np.array(file_data[impurity]["parlog"][time_handle]["errors"]),
        r_lcfs,
        parameter = impurity,
        norm=norm,
    )

    # plot impurity data
    plt.errorbar(
        np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
        np.array(file_data[impurity]["parlog"][time_handle]["values"]),
        yerr=file_data[impurity]["parlog"][time_handle]["errors"],
        color='black', ls="",
        marker='_',  
        label=_format_plot_handle(impurity)
    )
    
    # plot mtanh fit
    plt.plot(np.linspace(0, 0.5, 100), # linear array in rho
             fit_summary["result"]*norm, linestyle="-", color="tab:blue", label="mtanh tot")

    plt.plot(np.linspace(0, 0.5, 100), # linear array in rho
             fit_summary["outer_plateau"]*norm, linestyle="-", color="tab:green", label="mtanh inner")
            
    #plt.ylim(-0.5e18, 4.5e18)
    plt.xlabel(r"$r_{{eff}}$")
    plt.ylim(0,None)
    plt.ylabel(_format_plot_handle(impurity))
    plt.xlim(0,0.5)
    plt.title(f"{fit_type} fit for {shot_name} at {time_handle}")
    #plt.title(f"{shot_name} at {time_handle}")
    plt.legend()
    plt.show()

def plot_imp_single_shot(
        shot_name: str,
        impurity: str,
        fit_type: str
        )-> None:
    """
    Plot all the impurity fits for a single shot

    Parameters
    ----------
    shot_name : str
        shot number in yyyy.mm.dd
    impurity : str
        impurity handle: n_c6, etc..
    fit_type : str
        what algorithm is used to fit the data: "tanh"/"mtanh"/"gp"/"lowess"
    
    Returns
    -------
    None

    """
    with open(f"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles/{shot_name}_impurity_profiles.json", "r") as file_obj:
        file_data = json.load(file_obj)
    
    
    handles = list(file_data[impurity]["parlog"].keys())
    
    # pass r_eff rather than rho
    a = 0.5
    
    norm = _find_data_norm(file_data[impurity]["parlog"][handles[0]]["values"])
    vmec_id = get_reference_equilibrium(shot_name)[0]
    r_lcfs = get_minor_radius(vmec_id)
    
    
    for time_handle in handles:
        
        # HACK overwrite some error which are so tiny that they throw off the fit
        error_data = np.array(file_data[impurity]["parlog"][time_handle]["errors"])
        error_data[error_data < 2e16] = 2e16
        
        fit, error = _perform_profile_fit(
            fit_type,
            np.linspace(0, 0.5, 100),
            np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
            np.array(file_data[impurity]["parlog"][time_handle]["values"]),
            error_data,
            r_lcfs,
            parameter = impurity,
            norm=norm,
        )
        
        # plot impurity data
        plt.errorbar(
            np.array(file_data[impurity]["parlog"][time_handle]["rho"])*a,
            np.array(file_data[impurity]["parlog"][time_handle]["values"]),
            yerr=error_data,
            color='black', ls="",
            marker='_',  
            label=_format_plot_handle(impurity)
        )
            
        # plot mtanh fit
        plt.plot(np.linspace(0, 0.5, 100), # linear array in rho
                 fit, linestyle="-", color="tab:blue", label="mtanh")
                
        #plt.ylim(-0.5e18, 4.5e18)
        plt.xlabel(r"$r_{{eff}}$")
        plt.ylim(0,None)
        plt.ylabel(_format_plot_handle(impurity))
        plt.xlim(0,0.5)
        plt.title(f"{fit_type} fit for {shot_name} at {time_handle}")
        plt.legend()
        plt.show()

def export_impurity_data_to_txt(
        impurity: str,
        time_handle: str,
        shot_name: str,
        )-> None: 
    """
    Method to export impurity data to txt. This was used to put paste the data into
    an external program like desmos and easily get an intution for adjusting the ranges
    for the fitting for the mtanh and tanh. Data is returned in "fit_data.txt" and could then
    be pasted easily to desmos.

    Parameters
    ----------
    impurity : str
        impurity handle: "n_C6" etc..
    time_handle : str
        time_handle in proper format of {t_start_ind:.4f}_{t_stop_ind:.4f}
    shot_name : str
        shot number of which single data of single time point will be exported

    Returns
    -------
    txt file written with pairs of (x,y)

    """
    
    # open the file
    with open(f"/home/IPP-HGW/tomble/Desktop/Code/git/cxrs_fidasim/Data/output_fitted_profiles/{shot_name}_impurity_profiles.json", "r") as file_obj:
        file_data = json.load(file_obj)
    
    norm = _find_data_norm(file_data[impurity]["parlog"][time_handle]["values"])
    
    values = np.array(file_data[impurity]["parlog"][time_handle]["values"])/norm
    rho_values = np.array(file_data[impurity]["parlog"][time_handle]["rho"])*0.5
    
    pairs_list = []
    
    # switch to desmos format
    for rho, value in zip(values, rho_values):
        current_pair = f"({value}, {rho})"
        pairs_list.append(current_pair)
    
    pairs_str = str(pairs_list).replace("'","")
    
    with open("fit_data.txt", "a+") as f:
        f.write(pairs_str)

export_impurity_data_to_txt("n_C6", "t=5.3000_5.4000", "20230314.059")
plot_all_db_impurity_fits("n_C6", "mtanh")
    

