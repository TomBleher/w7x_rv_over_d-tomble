#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:11:39 2024

@author: tomble
"""
import copy
from db_plotting_routines import(
    read_data, 
    plot_tanh_param_time,
    filter_data_by_dict,
    filter_data_by_mag_config,
    filter_data_by_figures,
    plot_tanh_fit,
    plot_heatmap,
    plot_database_overview,
    plot_corner,
    find_kink,
    populate_figure_db,
    plot_fitted_imp_main_profiles_single
    )

# inputs
species = "Carbon"
rho_list = [0.3]
plot_y = "-dn_imp_drho/n_imp"
label_y = r"-$\frac{1}{n_{imp}}\frac{\partial n_{imp}}{\partial \rho}$"
#list_colorbars = ["eta_imp", "eta_e"]
#list_colorbars = ["f_rad", "tau_e", "n_e", "f_nbi"]
#list_colorbars = ["shot_resolved", "config_resolved"]
list_colorbars = ['a_1_history','a_2_history','a_3_history', 'b_1_history','b_2_history','b_3_history', 'b_4_history']   
#list_colorbars = ["density_peaking_ratio", "line_integrated_density", "total_carbon_density"]
    
# parameters to plot against
if True:
    plot_x = "-dn_e_drho/n_e"
    label_x = r"-$\frac{1}{n_{e}}\frac{\partial n_{e}}{\partial \rho}$"

elif True:
    plot_x = "-dT_e_drho/T_e"
    label_x = r"-$\frac{1}{T_{e}}\frac{\partial T_{e}}{\partial \rho}$"
else:
    plot_x = "-dT_i_drho/T_i"
    label_x = r"-$\frac{1}{T_{i}}\frac{\partial T_{i}}{\partial \rho}$"

# colorcode options
resolve_per_shot = False
resolve_per_config = False

# loop over rho values in list above
for rhos, rho in zip(range(0, len(rho_list)), rho_list):
    
    additional_info = {
        'rho': rho,
        'species': species,
        'labels': {'label_y': label_y, 'label_x': label_x},
    }
    
    # domain for the dictionary filter
    filter_dict = { 
        #"density_peaking_ratio": [0, 50],
        #"a_1_history": [0, 15]
        }
    
    # inspect fits behind data points, can take multiple shots and data points at once
    data_point_lookup = {
        "shot_number": "20230316.066",
        "t_start": 3.4,
        "t_stop": 3.5,
        "impurity": "Carbon",
        }
    
    # read json file locally 
    data = read_data(
        additional_info, 
        fit_parameter="n_C6", 
        fit_type="mtanh"
        )

    # apply the mask (by figure magnetic configuration or dictionary)
    #mask = filter_data_by_figures(data)
    #mask = filter_data_by_mag_config(data, "AHM+252")#"KJM+252" "AHM+252"
    mask = filter_data_by_dict(data, filter_dict)
    
    if True:
        # take list_colorbars and plot figures
        plot_database_overview(
            plot_x, 
            plot_y, 
            list_colorbars, 
            additional_info, 
            data, 
            mask,
            )
    
    if False:
        # plot parameter-time plots
        plot_tanh_param_time(
            data,
            mask,
            fit_type="mtanh",
        )
    
    if False:
        # plot fit of tanh to data
        plot_tanh_fit(
            data, 
            additional_info, 
            mask,
            fit_type="mtanh",
            fit_parameter = "n_C6"
        )
        
    if False:
        # correlations between variables
        plot_heatmap(
            copy.deepcopy(data),
            additional_info,
            mask,
            )
    
    if False:
        # plot cornor figure and save
        plot_corner(
            copy.deepcopy(data),
            r'/home/IPP-HGW/tomble/Desktop/corner', 
            mask,
            350,
            )

    if False:
        # plot seperated data correlation vs split location
        find_kink(
            plot_x, 
            plot_y,
            copy.deepcopy(data),
            additional_info,
            mask,
            )

    if False:
        # fill the directory with images for every shot PIT in database (= every point in plot)
        populate_figure_db(data)
    
    if False:
        # inspect for individual time points, use dictionary above
        plot_fitted_imp_main_profiles_single(
            data_point_lookup["shot_number"],
            data_point_lookup["t_start"],
            data_point_lookup["t_stop"],
            species,
            )
   