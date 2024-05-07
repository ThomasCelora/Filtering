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

    ################################################
    # # REGRESSING THE COEFFICIENT
    ################################################  
    statistical_tool = CoefficientsAnalysis() 

    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON?
    coeff_str = config['Regress+residual_check_settings']['coeff_str']
    coeff = meso_model.meso_vars[coeff_str]
    coeff_regressors_strs = json.loads(config['Regress+residual_check_settings']['coeff_regressors_strs'])
    regressors = []
    for i in range(len(coeff_regressors_strs)):
        temp = meso_model.meso_vars[coeff_regressors_strs[i]]
        regressors.append(temp)
    print(f'Dependent var: {coeff_str}, Explanatory vars: {coeff_regressors_strs}\n')

    residual_str = config['Regress+residual_check_settings']['residual_str']
    closure_ingr_str = config['Regress+residual_check_settings']['closure_ingr_str']
    residual = meso_model.meso_vars[residual_str]
    closure_ingr = meso_model.meso_vars[closure_ingr_str]
    print(f'Residuals and closure ingredient for model check: {residual_str}, {closure_ingr_str}\n')

    # PREPARING THE DATA 
    preprocess_data = json.loads(config['Regress+residual_check_settings']['preprocess_data']) 
    add_intercept = not not int(config['Regress+residual_check_settings']['add_intercept'])
    centralize = int(config['Regress+residual_check_settings']['centralize'])
    test_percentage = float(config['Regress+residual_check_settings']['test_percentage'])
    
    # first step: trimming and pre-processing
    regression_ranges = json.loads(config['Regress+residual_check_settings']['regression_ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    idxs_time_slices = json.loads(config['Regress+residual_check_settings']['idxs_time_slices'])
    times = [meso_model.domain_vars['T'][i] for i in idxs_time_slices]
    t_range = [np.amin(times), np.amax(times)]
    ranges = [t_range, x_range, y_range]

    data = [coeff]
    for i in range(len(regressors)):
        data.append(regressors[i])
    data.append(residual)
    data.append(closure_ingr)
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(data, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)

    # next step: centralizing the coefficients data
    if centralize: 
        new_data, means = statistical_tool.centralize_dataset(new_data)
        coeff_mean = means[0]
        del means[0]
        closure_ingr_mean = means[-1]
        del means[-1]
        residual_means = means[-1]
        del means[-1]
        regressors_means = means

    # second step: splitting into train and test
    if test_percentage==0.:
        coeff_train = new_data[0]
        coeff_test = np.copy(new_data[0])
        del new_data[0]
        closure_ingr_train = new_data[-1]
        closure_ingr_test = np.copy(new_data[-1])
        del new_data[-1]
        residual_train = new_data[-1]
        residual_test = np.copy(new_data[-1])
        del new_data[-1]
        regressors_train = new_data
        regressors_test = []
        for elem in regressors_train:
            regressors_test.append(np.copy(elem))
    else: 
        training_data, test_data = statistical_tool.split_train_test(new_data, test_percentage=test_percentage)
        coeff_train = training_data[0]
        coeff_test = test_data[0]
        del training_data[0]
        del test_data[0]
        closure_ingr_train = training_data[-1]
        closure_ingr_test = test_data[-1]
        del training_data[-1]
        del test_data[-1]
        residual_train = training_data[-1]
        residual_test = test_data[-1]
        del training_data[-1]
        del test_data[-1]
        regressors_train = training_data
        regressors_test = test_data

    # # fourth step: centralizing the coefficients data
    # if centralize: 
    #     tot_coeff = [np.stack((coeff_train, coeff_test))]
    #     tot_regress = [np.stack((regressors_test[i], regressors_train[i])) for i in range(len(regressors))]

    #     _, means = statistical_tool.centralize_dataset(tot_coeff + tot_regress)
    #     coeff_mean = means[0]
    #     regressors_means = means[1:]

    #     coeff_train -= coeff_mean
    #     for i in range(len(regressors_means)):
    #         regressors_train[i] -= regressors_means[i]

    coeffs, std_errors = statistical_tool.scalar_regression(coeff_train, regressors_train, add_intercept=add_intercept)
    print('regression coeffs: {}\n'.format(coeffs))
    # print('Corresponding std errors: {}\n'.format(std_errors)) 

    ######################################################
    # # RECONSTRUCTING PREDICTIONS FOR COEFF AND RESIDUAL
    ###################################################### 
    # Note: the reconstruction depends on the order with which you pre-process above

    # building predictions for the coefficient
    coeff_model = np.zeros(coeff_test.shape)
    if centralize: 
        coeff_model += coeff_mean
        coeff_test += coeff_mean
        for i in range(len(regressors_test)):
            # regressors_test[i] += regressors_means[i] 
            coeff_model += np.multiply(coeffs[i], regressors_test[i]) 
    else:
        if add_intercept:
            coeff_model += coeffs[0]
            for i in range(len(regressors)):
                coeff_model += np.multiply(coeffs[i+1], regressors_test[i]) 
        elif not add_intercept: 
            for i in range(len(regressors)):
                coeff_model += np.multiply(coeffs[i], regressors_test[i]) 

    # preprocessing and building model predictions for the residual    
    residual_model = np.zeros(residual_test.shape)
    residual_model = coeff_model + closure_ingr_test 
    if centralize: 
        residual_model += closure_ingr_mean
        residual_test += residual_means

    # FINALLY: PLOTTING
    plt.rc("font", family="serif")
    plt.rc("mathtext",fontset="cm")
    plt.rc('font', size=10)
    
    fig, axes = plt.subplots(1,3, figsize=[13,4])
    axes = axes.flatten()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
        warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')

        coeff_model = coeff_model.flatten()
        coeff_test = coeff_test.flatten()
        residual_model = residual_model.flatten()
        residual_test = residual_test.flatten()

        sns.set_theme(style="dark")
        sns.scatterplot(x=coeff_model, y=coeff_test, s=4, color=".15", ax=axes[0])
        sns.histplot(x=coeff_model, y=coeff_test, bins=50, ax=axes[0], pthresh=.1, cmap="mako")
        sns.kdeplot(x=coeff_model, y=coeff_test, levels=5, ax=axes[0], color="w", linewidths=1)

        sns.histplot(coeff_model, stat='density', kde=True, color='steelblue', ax=axes[1], label='model')
        sns.histplot(coeff_test, stat='density', kde=True, color='firebrick', ax=axes[1], label='sim. data')

        sns.histplot(residual_model, stat='density', kde=True, color='steelblue', ax=axes[2], label='model')
        sns.histplot(residual_test, stat='density', kde=True, color='firebrick', ax=axes[2], label='sim. data')
        
    print('finished plot, now making it nice\n') 

    # adding labels
    coeff_label = coeff_str
    if hasattr(meso_model, 'labels_var_dict') and coeff_str in meso_model.labels_var_dict.keys():
        coeff_label = meso_model.labels_var_dict[coeff_str] 
    if preprocess_data['log_abs'][0] == 1: 
        coeff_label = r"$\log($" + coeff_label + r"$)$"  

    residual_label = residual_str
    if hasattr(meso_model, 'labels_var_dict') and residual_str in meso_model.labels_var_dict.keys():
        residual_label = meso_model.labels_var_dict[residual_str] 
    # comment the following line if you don't take the log of the residual
    residual_label = r"$\frac{1}{2}\log($" + residual_label + r"$)$" 
    # residual_label = r"$\log($" + residual_label + r"$)$" 

    axes[0].set_xlabel('Regression model', fontsize=12)
    axes[0].set_ylabel(coeff_label, fontsize=12)
    axes[1].set_xlabel(coeff_label, fontsize=12)
    axes[1].set_ylabel('pdf', fontsize=12)
    axes[2].set_xlabel(residual_label, fontsize=12)
    axes[2].set_ylabel('pdf', fontsize=12)

    fig.tight_layout()

    # Building the annotation box with the specifics of the model
    if centralize: 
        text_for_box = r'$Means,$ $model$ $coeffs$:' + '\n'
        add_text = coeff_str + " : "  
        if hasattr(meso_model, 'labels_var_dict') and coeff_str in meso_model.labels_var_dict.keys():
            add_text = meso_model.labels_var_dict[coeff_str] + " : "
        sign, val = int(np.sign(coeff_mean)), '%.3f' %np.abs(coeff_mean)
        coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
        add_text += coeff_for_text_box  
        text_for_box += add_text    

        for i in range(len(coeff_regressors_strs)):
            add_text = "\n" + coeff_regressors_strs[i] + " :  "
            if hasattr(meso_model, 'labels_var_dict') and coeff_regressors_strs[i] in meso_model.labels_var_dict.keys():
                add_text = "\n" + meso_model.labels_var_dict[coeff_regressors_strs[i]] + " : "
            
            sign, val = int(np.sign(regressors_means[i])), '%.3f' %np.abs(regressors_means[i]) 
            coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r", $-{}$".format(val)  
            add_text += coeff_for_text_box 

            sign, val = int(np.sign(coeffs[i])), '%.3f' %np.abs(coeffs[i]) 
            coeff_for_text_box = r", $+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
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

        for i in range(len(coeff_regressors_strs)):
            sign, val = int(np.sign(coeffs[i+1])), '%.3f' %np.abs(coeffs[i+1]) 
            coeff_for_text_box = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)

            add_text = "\n" + coeff_regressors_strs[i] + " :  "
            if hasattr(meso_model, 'labels_var_dict') and coeff_regressors_strs[i] in meso_model.labels_var_dict.keys():
                add_text = "\n" + meso_model.labels_var_dict[coeff_regressors_strs[i]] + " :  "
            
            add_text += coeff_for_text_box 
            text_for_box += add_text


    bbox_args = dict(boxstyle="round", fc="0.95")
    plt.annotate(text=text_for_box, xy = (0.34,0.2), xycoords='figure fraction', bbox=bbox_args, ha="right", va="bottom", fontsize = 9)


    # Adding legend to the distribution comparison panel
    axes[1].legend(loc = 'best', prop={'size': 10})
    h, _ = axes[1].get_legend_handles_labels() 
    labels = ['Regression model', 'sim. data']
    axes[1].legend(h, labels, loc = 'best', prop={'size': 10, 'family' : 'serif'})

    axes[2].legend(loc = 'best', prop={'size': 10})
    h, _ = axes[2].get_legend_handles_labels() 
    labels = ['Regression model', 'sim. data']
    axes[2].legend(h, labels, loc = 'best', prop={'size': 10, 'family' : 'serif'})

    # Saving the figure
    print(f'Finished plot, now saving...\n\n')
    saving_directory = config['Directories']['figures_dir']   
    filename = '/Regress_and_check'
    format = 'png'
    filename += "." + format
    dpi = 300
    plt.savefig(saving_directory + filename, format=format, dpi=dpi)
    


