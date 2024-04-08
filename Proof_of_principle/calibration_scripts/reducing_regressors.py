import sys
# import os
sys.path.append('../../master_files/')
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

    # ###################################################################
    # # USING PCA TO REDUCE A LARGE LIST OF REGRESSOR TO A SMALLER SUBSET
    # ###################################################################
    
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


    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON 
    regressors_strs = json.loads(config['PCA_settings']['regressors_2_reduce'])
    regressors = []
    for i in range(len(regressors_strs)):
        temp = meso_model.meso_vars[regressors_strs[i]]
        regressors.append(temp)
    print('Trying to reduce the following list to a smaller subset: {}\n'.format(regressors_strs))

    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    ranges = json.loads(config['PCA_settings']['ranges'])
    x_range = ranges['x_range']
    y_range = ranges['y_range']
    num_slices_meso = int(config['PCA_settings']['num_T_slices'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['PCA_settings']['preprocess_data']) 
    extractions = int(config['PCA_settings']['extractions'])

    # PRE-PROCESSING and REGRESSING
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(regressors, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)
    # if extractions != 0: 
    #     new_data = statistical_tool.extract_randomly(new_data, extractions)

    # PERFORMING PCA TO EXTRACT PRINCIPAL COMPONENTS
    var_wanted = float(config['PCA_settings']['variance_wanted'])
    comp_decomp = statistical_tool.PCA_find_regressors_subset(new_data, var_wanted=var_wanted)

    saving_directory = config['Directories']['figures_dir']
    with open(saving_directory + '/Reducing_regressors.txt', 'w') as filehandle:
        filehandle.write("===================================\nFeatures in the dataset:\n")
        filehandle.write(regressors_strs[0])
        for i in range(1, len(regressors_strs)):
            filehandle.write(", " + regressors_strs[i])
        filehandle.write("\n===================================\n")
        for i in range(len(comp_decomp)):
            filehandle.write(f'The {i}-th component decomposition in terms of original vars is\n\n{comp_decomp[i]}\n\n')
