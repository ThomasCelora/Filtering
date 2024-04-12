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

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO MODEL
    pickle_directory = config['Directories']['pickled_files_dir']

    print('================================================')
    print(f'Starting job on data from {pickle_directory}')
    print('================================================\n\n')

    meso_filename_8  = '/rHD2d_cg=fw=bl=8dx.pickle'
    meso_filename_4 = '/rHD2d_cg=fw=bl=4dx.pickle'
    meso_filename_2 = '/rHD2d_cg=fw=bl=2dx.pickle'

    MesoModelLoadFile = pickle_directory + meso_filename_2
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso2 = pickle.load(filehandle)

    MesoModelLoadFile = pickle_directory + meso_filename_4
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso4 = pickle.load(filehandle)

    MesoModelLoadFile = pickle_directory + meso_filename_8
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso8 = pickle.load(filehandle)

    eta2 = np.abs(meso2.meso_vars['eta'])
    eta4 = np.abs(meso4.meso_vars['eta'])
    eta8 = np.abs(meso8.meso_vars['eta'])

    etas = [eta2, eta4, eta8]
    filter_sizes = [2,4,8]

    log_etas = [np.log10(elem) for elem in etas]
    means = [np.mean(elem) for elem in log_etas]
    print(f'The log-mean values of eta are: {means[0]}, {means[1]}, {means[2]}')
    
    etas_rescaled = [etas[i]/(filter_sizes[i]**2) for i in range(len(etas))]
    log_etas_rescaled = [np.log10(elem) for elem in etas_rescaled]

    shear_sq2 = meso2.meso_vars['shear_sq']
    shear_sq4 = meso4.meso_vars['shear_sq']
    shear_sq8 = meso8.meso_vars['shear_sq']

    pi_res_sq2 = meso2.meso_vars['pi_res_sq']
    pi_res_sq4 = meso4.meso_vars['pi_res_sq']
    pi_res_sq8 = meso8.meso_vars['pi_res_sq']

    shear_sqs = [np.log10(elem) for elem in [shear_sq2, shear_sq4, shear_sq8]]
    pi_res_sqs = [np.log10(elem) for elem in [pi_res_sq2, pi_res_sq4, pi_res_sq8]]

    means = [np.mean(elem) for elem in pi_res_sqs]
    print(f'The log-mean value of pi_res_sq are: {means[0]}, {means[1]}, {means[2]}')


    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(1,3, figsize=[12,4])
    axes = axes.flatten()

    stat = 'density'    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
        warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
        
        sns.histplot(shear_sqs[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[0], label=r'$L=2dx$')
        sns.histplot(shear_sqs[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[0], label=r'$L=4dx$')
        sns.histplot(shear_sqs[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[0], label=r'$L=8dx$')

        sns.histplot(pi_res_sqs[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[1], label=r'$L=2dx$')
        sns.histplot(pi_res_sqs[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[1], label=r'$L=4dx$')
        sns.histplot(pi_res_sqs[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[1], label=r'$L=8dx$')

        sns.histplot(log_etas_rescaled[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[2], label=r'$L=2dx$')
        sns.histplot(log_etas_rescaled[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[2], label=r'$L=4dx$')
        sns.histplot(log_etas_rescaled[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[2], label=r'$L=8dx$')

        
    axes[0].legend(loc = 'best', prop={'size': 10})
    axes[1].legend(loc = 'best', prop={'size': 10})
    axes[2].legend(loc = 'best', prop={'size': 10})

    axes[0].set_ylabel('pdf', fontsize=12)
    axes[1].set_ylabel('pdf', fontsize=12)
    axes[2].set_ylabel('pdf', fontsize=12)


    xlabel = meso2.labels_var_dict['shear_sq']
    xlabel = r'$\log($' + xlabel + r'$)$'
    axes[0].set_xlabel(xlabel, fontsize=12)

    xlabel = meso2.labels_var_dict['pi_res_sq']
    xlabel = r'$\log($' + xlabel + r'$)$'
    axes[1].set_xlabel(xlabel, fontsize=12)

    axes[2].set_xlabel(r'$\eta/\tilde{L}^2,\qquad \tilde{L} = L/dx$', fontsize=12)

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'L_dep_eta'
    format = 'png'
    dpi = 400
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)


    
    

    

