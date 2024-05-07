import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
import math
from scipy import stats
from itertools import product
import multiprocessing as mp
from sklearn.metrics import mean_absolute_error

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # # GIVEN A VAR TO MODEL AND A LIST OF REGRESSORS, PERFORM THE REGRESSION 
    # # WITH ANY POSSIBLE COMBINATION OF REGRESSORS TAKEN FROM THE INPUT LIST
    # # FIND THE ONE THAT BEST DESCRIBE THE QUANITY TO BE MODELLED, AND PRODUCE
    # # A SCATTER PLOT FOR IT
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

    # # RE-COMPUTING DERIVATIVES AND STUFF FOR MODELLING COEFFICIENTS
    # # adding labels to dictionary for better figures
    # entry_dic = {'D_n_tilde' : r'$\nabla_{a}\tilde{n}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'D_eps_tilde' : r'$\nabla_{a}\tilde{\varepsilon}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'n_tilde_dot' : r'$\dot{\tilde{n}}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'T_tilde_dot' : r'$\dot{\tilde{T}}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'sD_T_tilde': r'$D_{a}\tilde{T}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'sD_n_tilde': r'$D_{a}\tilde{n}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'sD_n_tilde_sq' : r'$D_{a}\tilde{n}D^{a}\tilde{n}$'}
    # meso_model.update_labels_dict(entry_dic)
    # entry_dic = {'dot_Dn_Theta' : r'$D_{a}\tilde{n}\Theta^{a}$'}
    # meso_model.update_labels_dict(entry_dic)

    # Nt = meso_model.domain_vars['Nt']
    # Nx = meso_model.domain_vars['Nx']
    # Ny = meso_model.domain_vars['Ny']

    # meso_model.nonlocal_vars_strs = ['u_tilde', 'T_tilde', 'n_tilde', 'eps_tilde'] 
    # meso_model.deriv_vars.update({'D_n_tilde' : np.zeros((Nt,Nx,Ny,3))})
    # meso_model.deriv_vars.update({'D_eps_tilde' : np.zeros((Nt,Nx,Ny,3))})

    # n_cpus = int(config['Find_best_fit_settings']['n_cpus'])
    # start_time = time.perf_counter()
    # meso_model.calculate_derivatives()
    # time_taken = time.perf_counter() - start_time
    # print('Finished computing derivatives (serial), time taken: {}\n'.format(time_taken), flush=True)

    # start_time = time.perf_counter()
    # meso_model.closure_ingredients_parallel(n_cpus)
    # time_taken = time.perf_counter() - start_time
    # print('Finished computing the closure ingredients in parallel, time taken: {}\n'.format(time_taken), flush=True)

    # start_time = time.perf_counter()
    # meso_model.EL_style_closure_parallel(n_cpus)
    # time_taken = time.perf_counter() - start_time
    # print('Finished computing the EL_style closure in parallel, time taken: {}\n'.format(time_taken), flush=True)

    # start_time = time.perf_counter()
    # meso_model.modelling_coefficients_parallel(n_cpus)
    # time_taken = time.perf_counter() - start_time
    # print('Finished computing quantities to model extracted coefficients, time taken: {}\n'.format(time_taken), flush=True)

    # pickle_directory = config['Directories']['pickled_files_dir']
    # filename = config['Filenames']['meso_pickled_filename']
    # MesoModelPickleDumpFile = pickle_directory + filename
    # with open(MesoModelPickleDumpFile, 'wb') as filehandle:
    #     pickle.dump(meso_model, filehandle)


    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON?
    dep_var_str = config['Find_best_fit_settings']['var_to_model']
    dep_var = meso_model.meso_vars[dep_var_str]
    regressors_strs = json.loads(config['Find_best_fit_settings']['regressors_strs'])
    regressors = []
    for i in range(len(regressors_strs)):
        temp = meso_model.meso_vars[regressors_strs[i]]
        regressors.append(temp)
    print(f'Dependent var: {dep_var_str},\n Explanatory vars: {regressors_strs}\n')

    
    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    regression_ranges = json.loads(config['Find_best_fit_settings']['ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Find_best_fit_settings']['num_T_slices'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['Find_best_fit_settings']['preprocess_data']) 
    add_intercept = not not int(config['Find_best_fit_settings']['add_intercept'])
    centralize = int(config['Find_best_fit_settings']['centralize'])
    test_percentage = float(config['Find_best_fit_settings']['test_percentage'])

    data = [dep_var]
    for i in range(len(regressors)):
        data.append(regressors[i])
    
    # PRE-PROCESSING: Trimming, pre-processing and splitting intro train + test set 
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(data, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)

    if centralize: 
        new_data, means = statistical_tool.centralize_dataset(new_data)
        dep_var_mean = means[0]
        regressors_means = means[1:]

    training_data, test_data = statistical_tool.split_train_test(new_data, test_percentage=test_percentage)

    dep_var_train = training_data[0]
    dep_var_test = test_data[0]
    regressors_train = training_data[1:]
    regressors_test = test_data[1:]


    # REGRESSING IN PARALLEL: consider all possible combination of input regressors' list
    # ALSO: EVALUATE GOODNESS OF FIT 
    num_regressors = len(regressors_train)
    bool_regressors_combs = [seq for seq in product((True, False), repeat=num_regressors)][0:-1]
    print(f'Number of tested combinations of regressors: {len(bool_regressors_combs)}\n')
    
    def parall_regress_task(comb_regressors):
        """
        """
        # DOING THE REGRESSION GIVEN SOME COMBINATION OF INPUT REGRESSORS
        actual_regressors = []
        actual_regressors_strs = []
        for i in range(num_regressors):
            if comb_regressors[i] == True:
                actual_regressors.append(regressors_train[i])
                actual_regressors_strs.append(regressors_strs[i])
        
        coeffs, _ = statistical_tool.scalar_regression(dep_var_train, actual_regressors, add_intercept=add_intercept)

        # BUILDING TEST-DATA PREDICTIONs GIVEN THE REGRESSED MODEL   
        actual_regressors = []
        for i in range(num_regressors):
            if comb_regressors[i] == True:
                actual_regressors.append(regressors_test[i])

        dep_var_model = np.zeros(dep_var_test.shape)
        if centralize:
            for i in range(len(actual_regressors)):
                dep_var_model += np.multiply(coeffs[i], actual_regressors[i]) 

        else:
            if add_intercept:
                dep_var_model += coeffs[0]
                for i in range(len(actual_regressors)):
                    dep_var_model += np.multiply(coeffs[i+1], actual_regressors[i]) 
            elif not add_intercept: 
                for i in range(len(actual_regressors)):
                    dep_var_model += np.multiply(coeffs[i], actual_regressors[i]) 
   
        # r, _ = stats.pearsonr(dep_var_test, dep_var_model)
        # return r, coeffs , comb_regressors
    
        # mean_error = mean_absolute_error(dep_var_test, dep_var_model)
        # return mean_error, coeffs , comb_regressors

        w = statistical_tool.wasserstein_distance(dep_var_test, dep_var_model, sample_points=300)            
        return w, coeffs, comb_regressors
                    
    
    n_cpus = int(config['Find_best_fit_settings']['n_cpus'])
    # pearsons = []
    # mean_errors = []
    wassersteins = []
    fitted_coeffs = []
    regressors_combinations = []
    with mp.Pool(processes=n_cpus) as pool:
        print('Performing all possible regression of {} in parallel with {} processes\n'.format(dep_var_str, pool._processes), flush=True)
        for result in pool.map(parall_regress_task, bool_regressors_combs): 
            # pearsons.append(result[0])
            # mean_errors.append(result[0])
            wassersteins.append(result[0])
            fitted_coeffs.append(result[1])
            regressors_combinations.append(result[2])


    # FINDING BEST MODEL, RE-BUILDING DATA PREDICITON FOR TEST SET
    # rs_squared = np.power(pearsons, 2)
    # max_r_index = np.argmax(rs_squared)
    # best_coeffs = fitted_coeffs[max_r_index]
    # best_regressors_combination = regressors_combinations[max_r_index]
    # ordering_idx = np.argsort(rs_squared)
    # print('Printing max r-squared and the corresponding regression combination:')

    min_w_index = np.argmin(wassersteins)
    best_coeffs = fitted_coeffs[min_w_index]
    ordering_idx = np.argsort(wassersteins)
    

    # for i in range(-1,-6,-1):
    for i in range(0,6,1):
        index = ordering_idx[i]
        which_regressors = []
        for j in range(len(regressors_combinations[index])):
            if regressors_combinations[index][j] == True:
                which_regressors.append(regressors_strs[j])
        # print(f'r^2: {rs_squared[index]}, regressors: {which_regressors}')
        print(f'wasserstein: {wassersteins[index]}, regressors: {which_regressors}')

    print(f'\nBest coefficients: {best_coeffs}')

    # # RECONSTRUCTING THE BEST MODEL FOR TEST DATA
    # actual_regressors = []
    # actual_regressors_strs = []
    # if centralize:
    #     actual_regressors_means = []
    # for i in range(len(regressors_strs)):
    #     if best_regressors_combination[i] == True:
    #         actual_regressors.append(regressors_test[i])
    #         actual_regressors_strs.append(regressors_strs[i])
    #         if centralize: 
    #             actual_regressors_means.append(regressors_means[i])

    # dep_var_model = np.zeros(dep_var_test.shape)
    # if centralize:
    #     for i in range(len(actual_regressors)):
    #         dep_var_model += np.multiply(best_coeffs[i], actual_regressors[i]) 

    # else:
    #     if add_intercept:
    #         dep_var_model += best_coeffs[0]
    #         for i in range(len(actual_regressors)):
    #             dep_var_model += np.multiply(best_coeffs[i+1], actual_regressors[i]) 
    #     elif not add_intercept: 
    #         for i in range(len(actual_regressors)):
    #             dep_var_model += np.multiply(best_coeffs[i], actual_regressors[i]) 


    # # FINALLY, PLOTTING THE BEST MODEL PREDICTIONS VS TEST DATA 
    # ylabel = dep_var_str
    # if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
    #     ylabel = meso_model.labels_var_dict[dep_var_str] 
    # if preprocess_data['log_abs'][0] == 1: 
    #     ylabel = r"$\log($" + ylabel + r"$)$"  
    # statistical_tool.visualize_correlation(dep_var_model, dep_var_test, xlabel=r"$regression$ $model$", ylabel=ylabel)

    # # Building the annotation box with the specifics of the model
    # if centralize: 
    #     text_for_box = r"$Model$ $coeff.s$, $means:$" + "\n"
    #     add_text = dep_var_str + " :  , "  
    #     if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
    #         add_text = meso_model.labels_var_dict[dep_var_str] + " :  , "
    #     sign, val = int(np.sign(dep_var_mean)), '%.3f' %np.abs(dep_var_mean)
    #     coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
    #     add_text += coeff_for_text_box  
    #     text_for_box += add_text    

    #     for i in range(len(actual_regressors_strs)):
    #         add_text = "\n" + actual_regressors_strs[i] + " :  "
    #         if hasattr(meso_model, 'labels_var_dict') and actual_regressors_strs[i] in meso_model.labels_var_dict.keys():
    #             add_text = "\n" + meso_model.labels_var_dict[actual_regressors_strs[i]] + " :  "

    #         sign, val = int(np.sign(best_coeffs[i])), '%.3f' %np.abs(best_coeffs[i]) 
    #         coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
    #         add_text += coeff_for_text_box 
            
    #         sign, val = int(np.sign(actual_regressors_means[i])), '%.3f' %np.abs(actual_regressors_means[i]) 
    #         coeff_for_text_box = r", $+{}$".format(val) if sign == 1 else r", $-{}$".format(val)  
    #         add_text += coeff_for_text_box 

    #         text_for_box += add_text


    # else: 
    #     text_for_box = r"$Model$ $coeff.s:$" + "\n"
    #     if add_intercept:
    #         sign, val = int(np.sign(best_coeffs[0])), '%.3f' %np.abs(best_coeffs[0]) 
    #         text_for_box += r'$offset$' +' :  ' 
    #         coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
    #         text_for_box += coeff_for_text_box 
    #     else:
    #         best_coeffs = [0] + best_coeffs

    #     for i in range(len(actual_regressors_strs)):
    #         sign, val = int(np.sign(best_coeffs[i+1])), '%.3f' %np.abs(best_coeffs[i+1]) 
    #         coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)

    #         add_text = "\n" + actual_regressors_strs[i] + " :  "
    #         if hasattr(meso_model, 'labels_var_dict') and actual_regressors_strs[i] in meso_model.labels_var_dict.keys():
    #             add_text = "\n" + meso_model.labels_var_dict[actual_regressors_strs[i]] + " :  "
            
    #         add_text += coeff_for_text_box 
    #         text_for_box += add_text

    # bbox_args = dict(boxstyle="round", fc="0.95")
    # plt.annotate(text=text_for_box, xy = (0.99,0.1), xycoords='figure fraction', bbox=bbox_args, ha="right", va="bottom", fontsize = 8)


    # # Saving the figure
    # saving_directory = config['Directories']['figures_dir']
    # if centralize: 
    #     filename = f'/CentredBestFit_{dep_var_str}_vs'
    # else:
    #     filename = f'/BestFit_{dep_var_str}_vs'
    
    # for i in range(0,len(regressors_strs)):
    #     filename += f'_{regressors_strs[i]}'
    # format = str(config['Regression_settings']['format_fig'])
    # filename += "." + format
    # dpi = None
    # if format == 'png':
    #     dpi = 400
    # plt.savefig(saving_directory + filename, format=format, dpi=dpi)
    # print(f'Finished regression and scatter plot for {dep_var_str}, saved as {filename}\n\n')