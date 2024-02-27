import sys
# import os
sys.path.append('../master_files/')
import pickle
import configparser
import json
import time
import math
from scipy import stats

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # # RUN THE REGRESSION ROUTINE GIVEN DEPENDENT DATA AND EXPLANATORY 
    # # VARS. DATA IS PRE-PROCESSED SO TO DEAL WITH POSITIVE AND NEGATIVE 
    # # DATA (EXTRACTING POS OR NEG PART APPROPRIATELY)
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
    dep_var_str = config['Regression_settings']['dependent_var']
    dep_var = meso_model.meso_vars[dep_var_str]
    regressors_strs = json.loads(config['Regression_settings']['regressors'])
    regressors = []
    for i in range(len(regressors_strs)):
        temp = meso_model.meso_vars[regressors_strs[i]]
        regressors.append(temp)
    print(f'Dependent var: {dep_var_str}, Explanatory vars: {regressors_strs}\n')

    
    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    regression_ranges = json.loads(config['Ranges_for_analysis']['ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['Regression_settings']['preprocess_data']) 
    add_intercept = not not int(config['Regression_settings']['add_intercept'])
    extractions = int(config['Regression_settings']['extractions'])

    # PRE-PROCESSING and REGRESSING
    data = [dep_var]
    for i in range(len(regressors)):
        data.append(regressors[i])
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(data, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)
    # if extractions != 0: 
    #     new_data = statistical_tool.extract_randomly(new_data, extractions)
    
    dep_var = new_data[0]
    regressors = []
    for i in range(1,len(new_data)):
        regressors.append(new_data[i])

    coeffs, std_errors = statistical_tool.scalar_regression(dep_var, regressors, add_intercept=add_intercept)
    print('regression coeffs: {}\n'.format(coeffs))
    print('Corresponding std errors: {}\n'.format(std_errors)) 
    
    # BUILDING DATA FOR REGRESSED MODEL
    regressors_strs = [None] + regressors_strs
    regressors = [np.ones(regressors[0].shape)] + regressors
    if not add_intercept: 
        coeffs = [0] + coeffs
    dep_var_model = np.zeros(dep_var.shape)
    for i in range(len(regressors)):
        dep_var_model += np.multiply(coeffs[i], regressors[i]) 

    if extractions != 0: 
        data_for_scatter = [dep_var, dep_var_model]
        new_data = statistical_tool.extract_randomly(data_for_scatter, extractions)
        dep_var, dep_var_model = new_data[0], new_data[1]

    
    # FINALLY: PLOTTING
    ylabel = dep_var_str
    if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
        ylabel = meso_model.labels_var_dict[dep_var_str] 
    if preprocess_data['log_abs'][0] == 1: 
        ylabel = r"$\log($" + ylabel + r"$)$"

    xlabel = ""
    if coeffs[0] != 0: 
        sign, val = int(np.sign(coeffs[0])), str(round(np.abs(coeffs[0]),3))
        xlabel = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
    for i in range(1, len(coeffs)):
        sign, val = int(np.sign(coeffs[i])), str(round(np.abs(coeffs[i]),3))
        xlabel += r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
        var_name = regressors_strs[i]
        if hasattr(meso_model, 'labels_var_dict') and regressors_strs[i] in meso_model.labels_var_dict.keys():
            var_name = meso_model.labels_var_dict[regressors_strs[i]] 
        if preprocess_data['log_abs'][i] == 1: 
            var_name = r"$\log($" + var_name + r"$)$"
        xlabel +=var_name        

    statistical_tool.visualize_correlation(dep_var_model, dep_var , xlabel, ylabel)
    saving_directory = config['Directories']['figures_dir']
    filename = f'/Regress_{dep_var_str}_vs'
    for i in range(1,len(regressors_strs)):
        filename += f'_{regressors_strs[i]}'
    # if add_intercept:
    #     filename += "_intercept"
    filename += ".pdf"
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished regression and scatter plot for {dep_var_str}, saved as {filename}\n\n')


