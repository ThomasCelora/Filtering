import sys
# import os
sys.path.append('../master_files/')
import pickle
import configparser
import json
import time
import math

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # #CORRELATION PLOTS: RESIDUALS VS CORRESPONDING CLOSURE INGREDIENTS
    # ##################################################################
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO MODEL 
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle:
        meso_model = pickle.load(filehandle)

    dict_to_add = {'vort_sq' : r'$\omega_{ab}\omega^{ab}$'}
    meso_model.upgrade_labels_dict(dict_to_add)

    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON?
    var_strs = json.loads(config['Visualize_correlations']['vars']) 
    vars = []
    for i in range(len(var_strs)):
        vars.append(meso_model.meso_vars[var_strs[i]])
    print(f'Producing scatter plot for the vars: {var_strs}\n')

    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    regression_ranges = json.loads(config['Ranges_for_analysis']['ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE, AND PREPROCESSING
    preprocess_data = json.loads(config['Visualize_correlations']['preprocess_data']) 
    extractions = int(config['Visualize_correlations']['extractions'])

    # PRE-PROCESSING DATA
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(vars, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)
    if extractions != 0: 
        new_data = statistical_tool.extract_randomly(new_data, extractions)
    vars =new_data

    # FINALLY, THE CORRELATION PLOT
    labels = []
    for var_str in var_strs:    
        label = var_str
        if hasattr(meso_model, 'labels_var_dict') and var_str in meso_model.labels_var_dict.keys():
            label = meso_model.labels_var_dict[var_str] 
        if preprocess_data['log_abs'][0] == 1: 
            label = r"$\log($" + label + r"$)$"
        labels.append(label)
    
    if len(var_strs) ==2:
        statistical_tool.visualize_correlation(vars[0], vars[1], xlabel=labels[0], ylabel=labels[1])
    else: 
        statistical_tool.visualize_many_correlations(vars, labels)
    
    saving_directory = config['Directories']['figures_dir']
    filename = '/Correlation'
    for i in range(len(var_strs)):
        filename += "_" + var_strs[i] 
    filename += ".pdf"
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished producing correlation plot for {var_strs}, saved as {filename}\n')

