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
    # # USING PCA TO LOOK FOR CORRELATIONS: FIND THE FIRST FEW PRINCIPAL 
    # # COMPONENTS IN A DATASET, AND PRINT THEIR CORRELATION PLOT WITH 
    # # A RESIDUAL/DISSIPATIVE COEFFICIENT 
    # ##################################################################
    
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

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    # additional_entry = {'acc_mag': r'$|a|$'}
    # meso_model.upgrade_labels_dict(additional_entry) 

    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON?
    dep_var_str = config['PCA_settings']['dependent_var']
    explanatory_vars_strs = json.loads(config['PCA_settings']['explanatory_vars'])
    explanatory_vars = []
    for i in range(len(explanatory_vars_strs)):
        temp = meso_model.meso_vars[explanatory_vars_strs[i]]
        explanatory_vars.append(temp)
    dep_var = meso_model.meso_vars[dep_var_str]
    print(f'Dependent var: {dep_var_str}, Explanatory vars: {explanatory_vars_strs}\n')
    

    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    PCA_ranges = json.loads(config['Ranges_for_analysis']['ranges'])
    x_range = PCA_ranges['x_range']
    y_range = PCA_ranges['y_range']
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['PCA_settings']['preprocess_data']) 
    extractions = int(config['PCA_settings']['extractions'])

    # PRE-PROCESSING 
    data = [dep_var]
    for i in range(len(explanatory_vars)):
        data.append(explanatory_vars[i])
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(data, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)
    if extractions != 0: 
        new_data = statistical_tool.extract_randomly(new_data, extractions)
    
    dep_var = new_data[0]
    explanatory_vars = []
    for i in range(1,len(new_data)):
        explanatory_vars.append(new_data[i])

    # NOW DO THE PCA ANALYSIS
    pcs_num = int(config['PCA_settings']['pcs_num'])
    highest_pcs_decomp, scores = statistical_tool.PCA_find_regressors(dep_var, explanatory_vars, pcs_num=pcs_num)

    for i in range(len(highest_pcs_decomp)):
        print(f'{i}-th highest PCs decomposition: \n{highest_pcs_decomp[i]}\nCorresponding score: {scores[i]}\n') 

    # NOW, USE THE PCs DECOMPOSITION TO BUILD A LINEAR MODEL FOR DEP_VAR 
    # THIS IS USED FOR PRODUCING A CORRELATION PLOT TO VISUALLY GRASP HOW GOOD THE MODEL IS  
    # ONLY THE PC WITH HIGHEST SCORE ON DEP_VAR IS USED, THE OTHERS PROVIDE AUXILIARY INFO?
        

    # BUILD THE LINEAR MODEL FROM PCA
    best_pc_decomp = highest_pcs_decomp[0]
    coeffs = []
    temp = 0 
    for j in range(0, len(explanatory_vars)):
        coeff = -best_pc_decomp[j+1] / best_pc_decomp[0] 
        print(f'{explanatory_vars_strs[j]}-rescaled coefficient: {coeff}\n')
        coeffs.append(coeff)
        temp += np.multiply(coeff, explanatory_vars[j])
    dep_var_model = temp

    # FINALLY, PLOTTING
    print('\nProducing correlation plot using info from PCA')
    x = dep_var_model
    y = dep_var
    
    ylabel = dep_var_str
    if hasattr(meso_model, 'labels_var_dict') and dep_var_str in meso_model.labels_var_dict.keys():
            ylabel = meso_model.labels_var_dict[dep_var_str] 
    if preprocess_data['log_abs'][0] ==1: 
        ylabel = r"$\log($" + ylabel + r"$)$"

    xlabel = ""
    for i in range(len(explanatory_vars_strs)):
        sign, val = int(np.sign(coeffs[i])), str(round(np.abs(coeffs[i]),3))
        coeff_for_label = r"$+{}$".format(val) if sign == 1 else r"$-{}$".format(val)
        text = explanatory_vars_strs[i]
        if hasattr(meso_model, 'labels_var_dict') and explanatory_vars_strs[i] in meso_model.labels_var_dict.keys():
            text = meso_model.labels_var_dict[explanatory_vars_strs[i]] 
        if preprocess_data['log_abs'][i+1]==1:
            text = r"$\log($" + text + r"$)$"
        xlabel += coeff_for_label  + text

    model_points = meso_model.domain_vars['Points']
    g=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    saving_directory = config['Directories']['figures_dir']
    filename = f'/PCA_{dep_var_str}_vs'
    for i in range(len(explanatory_vars_strs)):
        filename += f'_{explanatory_vars_strs[i]}'
    filename += ".pdf"
    plt.savefig(saving_directory + filename, format='pdf')
    print(f'Finished correlation plot for {dep_var_str}, saved as {filename}\n\n')

    