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

    # # READING INFO ON RESIDUALS FROM CONFIG FILE
    # coeff_str = config['Fs_residual_dependence']['coeff']
    # residual_str = config['Fs_residual_dependence']['residual']
    # EL_force_str = config['Fs_residual_dependence']['EL_force']

    # print(f'coeff_str: {coeff_str}\nresidual_str: {residual_str}\nEL_force_str: {EL_force_str}\n')

    # coeff2 = np.abs(meso2.meso_vars[coeff_str])
    # coeff4 = np.abs(meso4.meso_vars[coeff_str])
    # coeff8 = np.abs(meso8.meso_vars[coeff_str])

    # coeffs = [coeff2, coeff4, coeff8]
    # filter_sizes = [2,4,8]

    # log_coeffs = [np.log10(elem) for elem in coeffs]
    # means = [np.mean(elem) for elem in log_coeffs]
    # print(f'The log-mean values of {coeff_str} are: {means[0]}, {means[1]}, {means[2]}')
    
    # coeffs_rescaled = [coeffs[i]/(filter_sizes[i]**2) for i in range(len(coeffs))]
    # log_coeffs_rescaled = [np.log10(elem) for elem in coeffs_rescaled]

    # EL_force2 = np.abs(meso2.meso_vars[EL_force_str])
    # EL_force4 = np.abs(meso4.meso_vars[EL_force_str])
    # EL_force8 = np.abs(meso8.meso_vars[EL_force_str])

    # residual2 = meso2.meso_vars[residual_str]
    # residual4 = meso4.meso_vars[residual_str]
    # residual8 = meso8.meso_vars[residual_str]

    # log_residuals = [np.log10(elem) for elem in [residual2, residual4, residual8]]
    # log_EL_forces = [np.log10(elem) for elem in [EL_force2, EL_force4, EL_force8]]

    # means = [np.mean(elem) for elem in log_residuals]
    # print(f'The log-mean value of pi_res_sq are: {means[0]}, {means[1]}, {means[2]}')


    # plt.rc("font",family="serif")
    # plt.rc("mathtext",fontset="cm")
    # fig, axes = plt.subplots(1,3, figsize=[12,4])
    # axes = axes.flatten()

    # stat = 'density'    
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
    #     warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
        
    #     sns.histplot(log_EL_forces[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[0], label=r'$L=2dx$')
    #     sns.histplot(log_EL_forces[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[0], label=r'$L=4dx$')
    #     sns.histplot(log_EL_forces[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[0], label=r'$L=8dx$')

    #     sns.histplot(log_residuals[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[1], label=r'$L=2dx$')
    #     sns.histplot(log_residuals[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[1], label=r'$L=4dx$')
    #     sns.histplot(log_residuals[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[1], label=r'$L=8dx$')

    #     sns.histplot(log_coeffs_rescaled[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[2], label=r'$L=2dx$')
    #     sns.histplot(log_coeffs_rescaled[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[2], label=r'$L=4dx$')
    #     sns.histplot(log_coeffs_rescaled[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[2], label=r'$L=8dx$')

        
    # axes[0].legend(loc = 'best', prop={'size': 10})
    # axes[1].legend(loc = 'best', prop={'size': 10})
    # axes[2].legend(loc = 'best', prop={'size': 10})

    # axes[0].set_ylabel('pdf', fontsize=12)
    # axes[1].set_ylabel('pdf', fontsize=12)
    # axes[2].set_ylabel('pdf', fontsize=12)

    # xlabel = meso2.labels_var_dict[EL_force_str]
    # xlabel = r'$\log(|$' + xlabel + r'$|)$'
    # axes[0].set_xlabel(xlabel, fontsize=12)

    # xlabel = meso2.labels_var_dict[residual_str]
    # xlabel = r'$\log($' + xlabel + r'$)$'
    # axes[1].set_xlabel(xlabel, fontsize=12)

    # xlabel = meso2.labels_var_dict[coeff_str]
    # xlabel = xlabel + r'$/\tilde{L}^2,\qquad \tilde{L} = L/dx$'
    # axes[2].set_xlabel(xlabel, fontsize=12)

    # fig.tight_layout()

    # fig_directory = config['Directories']['figures_dir'] 
    # filename = 'fs_residual_dependence'
    # format = 'png'
    # dpi = 400
    # filename += "." + format
    # plt.savefig(fig_directory + filename, format=format, dpi=dpi)

    # READING INFO ON RESIDUALS FROM CONFIG FILE
    coeff_str = config['Fs_residual_dependence']['coeff']
    residual_str = config['Fs_residual_dependence']['residual']
    EL_force_str = config['Fs_residual_dependence']['EL_force']

    print(f'coeff_str: {coeff_str}\nresidual_str: {residual_str}\nEL_force_str: {EL_force_str}\n')

    zeta2 = np.abs(meso2.meso_vars['zeta'])
    zeta4 = np.abs(meso4.meso_vars['zeta'])
    zeta8 = np.abs(meso8.meso_vars['zeta'])
    zetas = [zeta2, zeta4, zeta8]

    kappa2 = np.abs(meso2.meso_vars['kappa'])
    kappa4 = np.abs(meso4.meso_vars['kappa'])
    kappa8 = np.abs(meso8.meso_vars['kappa'])
    kappas = [kappa2, kappa4, kappa8]

    filter_sizes = [2,4,8]

    zetas_rescaled = [zetas[i]/(filter_sizes[i]**2) for i in range(len(zetas))]
    kappas_rescaled = [kappas[i]/(filter_sizes[i]**2) for i in range(len(kappas))]

    log_zetas_rescaled = [np.log10(elem) for elem in zetas_rescaled]
    log_kappas_rescaled = [np.log10(elem) for elem in kappas_rescaled]

    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(1,2, figsize=[8,4])
    axes = axes.flatten()

    stat = 'density'    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
        warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
        
        sns.histplot(log_zetas_rescaled[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[0], label=r'$L=2dx$')
        sns.histplot(log_zetas_rescaled[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[0], label=r'$L=4dx$')
        sns.histplot(log_zetas_rescaled[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[0], label=r'$L=8dx$')

        sns.histplot(log_kappas_rescaled[0].flatten(), stat=stat, kde=True, color='firebrick', ax=axes[1], label=r'$L=2dx$')
        sns.histplot(log_kappas_rescaled[1].flatten(), stat=stat, kde=True, color='steelblue', ax=axes[1], label=r'$L=4dx$')
        sns.histplot(log_kappas_rescaled[2].flatten(), stat=stat, kde=True, color='olive', ax=axes[1], label=r'$L=8dx$')

        
    axes[0].legend(loc = 'best', prop={'size': 10})
    axes[1].legend(loc = 'best', prop={'size': 10})

    axes[0].set_ylabel('pdf', fontsize=12)
    axes[1].set_ylabel('pdf', fontsize=12)

    xlabel = meso2.labels_var_dict['zeta']
    xlabel = xlabel + r'$/\tilde{L}^2,\qquad \tilde{L} = L/dx$'
    axes[0].set_xlabel(xlabel, fontsize=12)

    xlabel = meso2.labels_var_dict['kappa']
    xlabel = xlabel + r'$/\tilde{L}^2,\qquad \tilde{L} = L/dx$'
    axes[1].set_xlabel(xlabel, fontsize=12)

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'fs_residual_dependence'
    format = 'png'
    dpi = 400
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)


    
    

    

