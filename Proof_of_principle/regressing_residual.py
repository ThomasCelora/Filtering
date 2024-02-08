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
    # # RUN THE REGRESSION ROUTINE GIVEN DEPENDENT DATA AND EXPLANATORY 
    # # VARS. TAKE THE LOG SO TO LOOK FOR REGRESSION IN LOGSPACE.
    # # TO THINK: HOW TO DEAL WITH POSITIVE AND NEGATIVE QUANTITIES? 
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

    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    # Which data you want to run the routine on?
    dep_var_str = config['Regression_settings']['dependent_var']
    regressors_strs = json.loads(config['Regression_settings']['regressors'])
    add_intercept = not not int(config['Regression_settings']['add_intercept'])
    regressors = []
    for i in range(len(regressors_strs)):
        temp = meso_model.meso_vars[regressors_strs[i]]
        regressors.append(np.log10(temp))
    dep_var = np.log10(meso_model.meso_vars[dep_var_str])
    print(f'Dependent var: {dep_var_str}, Explanatory vars: {regressors_strs}\n')

    # Which ranges? (later: add tools to randomly extract from within ranges?)
    regression_ranges = json.loads(config['Regression_settings']['regression_ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # Performing the regression
    statistical_tool = CoefficientsAnalysis()  
    model_points = meso_model.domain_vars['Points']
    coeffs, std_errors = statistical_tool.scalar_regression(dep_var, regressors, ranges=ranges, model_points=model_points,
                                                        add_intercept=add_intercept)
    print('regression coeffs: {}\n'.format(coeffs))
    print('Corresponding std errors: {}\n'.format(std_errors))
    
    # Bulding the regressed model for scatterplot
    regressors_strs = [None] + regressors_strs
    regressors = [np.ones(regressors[0].shape)] + regressors
    if not add_intercept: 
        coeffs = [0] + coeffs
    
    dep_var_model = np.zeros(dep_var.shape)
    for i in range(len(regressors)):
        dep_var_model += np.multiply(coeffs[i], regressors[i]) 
    
    # Plotting dependent var VS its model: to visually check if the regression is decent
    correlation_ranges = json.loads(config['Figure_settings']['correlation_ranges'])
    x_range = correlation_ranges['x_range']
    y_range = correlation_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    ylabel = 'log({})'.format(dep_var_str)
    if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
        ylabel = r"$\log($" + meso_model.labels_var_dict[dep_var_str] + ")"

    xlabel = ""
    if coeffs[0] != 0: 
        sign, val = int(np.sign(coeffs[0])), str(round(np.abs(coeffs[0]),3))
        xlabel = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)

    for i in range(1, len(coeffs)):
        sign, val = int(np.sign(coeffs[i])), str(round(np.abs(coeffs[i]),3))
        xlabel += r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
        text = r"$\log($"
        if hasattr(meso_model, 'labels_var_dict') and regressors_strs[i] in meso_model.labels_var_dict.keys():
            text += meso_model.labels_var_dict[regressors_strs[i]] + ")"
        else:
            text += regressors_strs[i] + ")"
        xlabel +=text        

    statistical_tool.visualize_correlation(dep_var_model, dep_var , xlabel, ylabel, ranges, model_points)
    saving_directory = config['Directories']['figures_dir']
    filename = '/{}_vs_{}'.format(dep_var_str, json.loads(config['Regression_settings']['regressors']))
    if add_intercept:
        filename += "_intercept"
    filename += ".pdf"
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished regression and scatter plot for {dep_var_str}, saved as {filename}\n\n')

        
