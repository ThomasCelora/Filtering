#!/bin/bash

import sys
# import os
sys.path.append('../master_files/')
import pickle
import configparser
import json

from FileReaders import *
from MicroModels import *
from MesoModels import *
from Visualization import *

if __name__ == '__main__':

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # LOADING MESO AND MICRO MODELS 
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']

    MesoModelLoadFile = pickle_directory + meso_pickled_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    micro_model = meso_model.micro_model
    
    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    time_meso = meso_model.domain_vars['T'][1] #Change this if meso_grid is not set up using three central micro_slices
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!")
    else: 
        print("Comparing data at same time-slice, hurray!")
    

    # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D([11.97, 8.36])

    # FINALLY, PLOTTING
    models = [micro_model, meso_model]
    vars = [['bar_vel', 'bar_vel', 'bar_vel'],['U', 'U', 'U']]
    components_indices= [[(0,),(1,),(2,)], [(0,), (1,), (2,)]]
    fig = visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices=components_indices, diff_plot=True, rel_diff=True)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    box_len = str(config['Figures_name_param']['box_length'])
    filename="/Meso_obs_" + box_len + "dx_ET_"+ time_for_filename +".pdf"
    plt.savefig(saving_directory + filename, format="pdf")