import sys
# import os
sys.path.append('../../master_files/')
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
    # # VARS. DATA IS PRE-PROCESSED 
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
    regression_ranges = json.loads(config['Regression_settings']['ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Regression_settings']['num_T_slices'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['Regression_settings']['preprocess_data']) 
    add_intercept = not not int(config['Regression_settings']['add_intercept'])
    extractions = int(config['Regression_settings']['extractions'])
    centralize = int(config['Regression_settings']['centralize'])
    test_percentage = float(config['Regression_settings']['test_percentage'])

    # BUILDING THE WEIGHTS
    weighing_func_str = config['Regression_settings']['weighing_func']
    if weighing_func_str == 'Q2':
        meso_model.weights_Q2()
        weights = meso_model.meso_vars['weights']
        print(f'Finished building weights using {meso_model.weights_Q2.__name__}\n')
    
    elif weighing_func_str == 'Q1_skew':
        meso_model.weights_Q1_skew()
        weights = meso_model.meso_vars['weights']
        print(f'Finished building weights using {meso_model.weights_Q1_skew.__name__}\n')
    
    elif weighing_func_str == 'Q1_non_neg':
        meso_model.weights_Q1_non_neg()
        weights = meso_model.meso_vars['weights']
        print(f'Finished building weights using {meso_model.weights_Q1_non_neg.__name__}\n')
    
    elif  weighing_func_str == "residual_weights":
        residual_str = config['Regression_settings']['residual_str']
        meso_model.residual_weights(residual_str)
        weights = meso_model.meso_vars['weights']
        print(f'Finished building weights using {meso_model.residual_weights.__name__}\n')
    
    elif  weighing_func_str == "denominator_weights":
        residual_str = config['Regression_settings']['denominator_str']
        meso_model.denominator_weights(residual_str)
        weights = meso_model.meso_vars['weights']
        print(f'Finished building weights using {meso_model.denominator_weights.__name__}\n')
    
    else:
        print(f'The string for building weights {weighing_func_str} does not match any of the implemented routines.\n')
        weights = None

    # PRE-PROCESSING: Trimming, pre-processing and splitting intro train + test set 
    data = [dep_var]
    for i in range(len(regressors)):
        data.append(regressors[i])
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']

    if weights is not None:
        new_data = statistical_tool.trim_dataset(data + [weights], ranges, model_points)
        weights = new_data[-1]
        del new_data[-1]
        new_data, weights = statistical_tool.preprocess_data(new_data, preprocess_data, weights=weights)
        
        if centralize: 
            new_data, means = statistical_tool.centralize_dataset(new_data)
            dep_var_mean = means[0]
            regressors_means = means[1:]
        
        training_data, test_data = statistical_tool.split_train_test(new_data + [weights], test_percentage=test_percentage)
        dep_var_train = training_data[0]
        dep_var_test = test_data[0]
        del training_data[0]
        del test_data[0]
        weights_train = training_data[-1]
        weights_test = test_data[-1]
        del training_data[-1]
        del test_data[-1]
        regressors_train = training_data
        regressors_test = test_data


    else:
        new_data = statistical_tool.trim_dataset(data, ranges, model_points)
        new_data = statistical_tool.preprocess_data(new_data, preprocess_data)
        
        if centralize: 
            new_data, means = statistical_tool.centralize_dataset(new_data)
            dep_var_mean = means[0]
            regressors_means = means[1:]
        
        training_data, test_data = statistical_tool.split_train_test(new_data, test_percentage=test_percentage)
        dep_var_train = training_data[0]
        dep_var_test = test_data[0]
        del training_data[0]
        del test_data[0]
        regressors_train = training_data
        regressors_test = test_data

        weights_train = weights_test = None


    coeffs, std_errors = statistical_tool.scalar_regression(dep_var_train, regressors_train, add_intercept=add_intercept, weights=weights_train)
    print('regression coeffs: {}\n'.format(coeffs))
    print('Corresponding std errors: {}\n'.format(std_errors)) 


    # BUILDING TEST-DATA PREDICTIONs GIVEN THE REGRESSED MODEL   
    dep_var_model = np.zeros(dep_var_test.shape)
    if centralize:
        for i in range(len(regressors_test)):
            dep_var_model += np.multiply(coeffs[i], regressors_test[i]) 

    else:
        if add_intercept:
            dep_var_model += coeffs[0]
            for i in range(len(regressors)):
                dep_var_model += np.multiply(coeffs[i+1], regressors_test[i]) 
        elif not add_intercept: 
            for i in range(len(regressors)):
                dep_var_model += np.multiply(coeffs[i], regressors_test[i]) 

    # EXTRACTIONS ARE NOT NEEDED IF YOU SAVE THE FIG AS PNG
    if extractions != 0: 
        data_for_scatter = [dep_var_test, dep_var_model]
        new_data = statistical_tool.extract_randomly(data_for_scatter + [weights_test], extractions)
        if weights is not None:
            weights_test = new_data[-1]
            del new_data[-1]

        dep_var_test, dep_var_model = new_data[0], new_data[1]

    
    # FINALLY: PLOTTING
    ylabel = dep_var_str
    if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
        ylabel = meso_model.labels_var_dict[dep_var_str] 
    if preprocess_data['log_abs'][0] == 1: 
        ylabel = r"$\log($" + ylabel + r"$)$"  
    # statistical_tool.visualize_correlation(dep_var_model, dep_var_test , xlabel=r"$regression$ $model$", ylabel=ylabel, weights = weights)
    fig = statistical_tool.compare_distributions(dep_var_model, dep_var_test, xlabel=r'$regression$ $model$', ylabel=ylabel)

    # Building the annotation box with the specifics of the model
    if centralize: 
        text_for_box = r"$Model$ $coeff.s$, $means:$" + "\n"
        add_text = dep_var_str + " :  , "  
        if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
            add_text = meso_model.labels_var_dict[dep_var_str] + " :  , "
        sign, val = int(np.sign(dep_var_mean)), '%.3f' %np.abs(dep_var_mean)
        coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
        add_text += coeff_for_text_box  
        text_for_box += add_text    

        for i in range(len(regressors_strs)):
            add_text = "\n" + regressors_strs[i] + " :  "
            if hasattr(meso_model, 'labels_var_dict') and regressors_strs[i] in meso_model.labels_var_dict.keys():
                add_text = "\n" + meso_model.labels_var_dict[regressors_strs[i]] + " :  "

            sign, val = int(np.sign(coeffs[i])), '%.3f' %np.abs(coeffs[i]) 
            coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
            add_text += coeff_for_text_box 
            
            sign, val = int(np.sign(regressors_means[i])), '%.3f' %np.abs(regressors_means[i]) 
            coeff_for_text_box = r", $+{}$".format(val) if sign == 1 else r", $-{}$".format(val)  
            add_text += coeff_for_text_box 

            text_for_box += add_text


    else: 
        text_for_box = r"$Model$ $coeff.s:$" + "\n"
        if add_intercept:
            sign, val = int(np.sign(coeffs[0])), '%.3f' %np.abs(coeffs[0]) 
            text_for_box += r'$offset$' +' :  ' 
            coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
            text_for_box += coeff_for_text_box 
        else:
            coeffs = [0] + coeffs

        for i in range(len(regressors_strs)):
            sign, val = int(np.sign(coeffs[i+1])), '%.3f' %np.abs(coeffs[i+1]) 
            coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)

            add_text = "\n" + regressors_strs[i] + " :  "
            if hasattr(meso_model, 'labels_var_dict') and regressors_strs[i] in meso_model.labels_var_dict.keys():
                add_text = "\n" + meso_model.labels_var_dict[regressors_strs[i]] + " :  "
            
            add_text += coeff_for_text_box 
            text_for_box += add_text

    bbox_args = dict(boxstyle="round", fc="0.95")
    plt.annotate(text=text_for_box, xy = (0.99,0.1), xycoords='figure fraction', bbox=bbox_args, ha="right", va="bottom", fontsize = 8)


    # Saving the figure
    saving_directory = config['Directories']['figures_dir']
    filename = f'/Regress_{dep_var_str}_vs'
    for i in range(1,len(regressors_strs)):
        filename += f'_{regressors_strs[i]}'
    format = str(config['Regression_settings']['format_fig'])
    filename += "." + format
    dpi = None
    if format == 'png':
        dpi = 400
    plt.savefig(saving_directory + filename, format=format, dpi=dpi)
    print(f'Finished regression and scatter plot for {dep_var_str}, saved as {filename}\n\n')


