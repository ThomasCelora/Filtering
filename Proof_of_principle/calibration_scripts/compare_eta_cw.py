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

    meso_filename  = config['Filenames']['meso_pickled_filename']

    MesoModelLoadFile = pickle_directory + meso_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    n_cpus = int(config['compare_eta_cw']['n_cpus'])
    cw_dict = meso_model.EL_componentwise_parallel(n_cpus, store=False)
    print('Finished computing the coefficients componentwise\n')
    print(f'Keys in the dictionary: {cw_dict.keys()}')

    # meso_model.EL_style_closure_parallel(n_cpus)
    # print('Finished recomputing the coefficients covariantly, now saving\n')
    # pickle_directory = config['Directories']['pickled_files_dir']
    # filename = config['Filenames']['meso_pickled_filename']
    # MesoModelPickleDumpFile = pickle_directory + filename
    # with open(MesoModelPickleDumpFile, 'wb') as filehandle:
    #     pickle.dump(meso_model, filehandle)

    eta_cw = cw_dict['eta_cw']
    print(f'eta_cw.shape: {eta_cw.shape}\n')
    
    eta_cw_shape = eta_cw[0,0,0].shape
    Nt = meso_model.domain_vars['Nt']
    Nx = meso_model.domain_vars['Nx']
    Ny = meso_model.domain_vars['Ny']
    new_shape = tuple(list(eta_cw_shape) + [Nt, Nx, Ny])
    print(f'new_shape: {new_shape}\n')
    eta_cw = eta_cw.reshape(new_shape)

    components = json.loads(config['compare_eta_cw']['components'])
    components = [tuple(comp) for comp in components]
    preprocess_data = json.loads(config['compare_eta_cw']['preprocess_data'])
    statistical_tool = CoefficientsAnalysis()

    eta_comps = [eta_cw[comp] for comp in components]
    eta_comps = statistical_tool.preprocess_data(eta_comps, preprocess_data)

    eta_quadratic = meso_model.meso_vars['eta']
    eta_quadratic = np.log10(np.abs(eta_quadratic))

    print('Finished preparing data for plottting distributions\n')

    # # ####################################################
    # # PLOTTING THE COMPONENTS DISTRIBUTIONS ALL AT ONCE 
    # # ####################################################
    # plt.rc("font",family="serif")
    # plt.rc("mathtext",fontset="cm")
    # fig, ax = plt.subplots(1,1, figsize=[6,4])

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
    #     warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
    #     for i in range(len(eta_comps)):
    #         label = 'comp = ' + str(components[i])
    #         sns.histplot(eta_comps[i].flatten(), ax=ax, stat='density', kde=True, label=label)

    #     label = 'squaring'
    #     sns.histplot(eta_quadratic.flatten(), ax=ax, stat='density', color='black', kde=True, label=label)
        
    #     ax.set_ylabel('pdf', fontsize=10)
    #     xlabel = r'$\log(|\eta|)$'
    #     ax.set_xlabel(xlabel, fontsize=10)
    #     ax.legend(loc = 'best', prop={'size': 10})


    # fig.tight_layout()

    # fig_directory = config['Directories']['figures_dir'] 
    # filename = 'eta_cw_distrib'
    # format = 'png'
    # dpi = 400
    # filename += "." + format
    # plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    # plt.close()

    # # ##################################################################################
    # SINGLE PLOT WITH 6 PANELS AND HISTOGRAMS OF POSITIVE AND NEGATIVE VALUES SEPARATELY
    # # ##################################################################################
    components = [(0,0), (0,1), (0,2), (1,1), (1,2)]
    eta_comps = [np.array(eta_cw[comp]) for comp in components]

    eta_quadratic = meso_model.meso_vars['eta']
    eta_quad_pos = ma.masked_where(eta_quadratic<0, eta_quadratic, copy=True).compressed()
    eta_quad_neg = ma.masked_where(eta_quadratic>0, eta_quadratic, copy=True).compressed()
    eta_quad_pos = np.log10(np.abs(eta_quad_pos))
    eta_quad_neg = np.log10(np.abs(eta_quad_neg))
    eta_quad_counts = [len(eta_quad_pos), len(eta_quad_neg)]

    tot_counts = []
    for i in range(len(eta_comps)):
        eta_pos = ma.masked_where(eta_comps[i]<0, eta_comps[i], copy=True)
        eta_neg = ma.masked_where(eta_comps[i]>0, eta_comps[i], copy=True)
        eta_pos = eta_pos.compressed()
        eta_neg = eta_neg.compressed()
        eta_pos = np.log10(eta_pos)
        eta_neg = np.log10(np.abs(eta_neg))
        eta_comps[i] = [eta_pos, eta_neg]

        counts = [len(eta_pos), len(eta_neg)]
        tot_counts.append(counts)
        tot_number = len((eta_cw[components[i]]).flatten())
        print(f'len(eta_comps[i].flatten()): {tot_number}') 
        

    # Computing the min and max to set a common range in the plot
    min, max = np.amin(eta_quad_pos), np.amax(eta_quad_pos)
    if  np.amin(eta_quad_neg) < np.amin(eta_quad_pos):
        min = np.amin(eta_quad_neg)
    if np.amax(eta_quad_neg) > np.amax(eta_quad_pos):
        max = np.amax(eta_quad_neg)
    
    for i in range(len(eta_comps)):
        m1, M1 = np.amin(eta_comps[i][0]), np.amax(eta_comps[i][0])
        m2, M2 = np.amin(eta_comps[i][1]), np.amax(eta_comps[i][1])
        m, M = np.amin([m1,m2]), np.amax([M1,M2])
        if m < min:
            min = m 
        if M > max:
            max = M 
    if max >0: 
        max = 0

    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(2,3, figsize=[12,8])
    axes = axes.flatten()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
        warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')

        for i in range(5):
            # sns.histplot(eta_comps[i][0], ax=axes[i], stat='density', kde=True, label=r'$\eta_{+}$')
            # sns.histplot(eta_comps[i][1], ax=axes[i], stat='density', kde=True, label=r'$\eta_{-}$')
            label = r'$\eta_{+}$' + ', #=' + str(tot_counts[i][0])
            sns.histplot(eta_comps[i][0], ax=axes[i], label=label)
            label = r'$\eta_{-}$' + ', #=' + str(tot_counts[i][1])
            sns.histplot(eta_comps[i][1], ax=axes[i], label=label)

            xlabel = r'$\log(|\eta|)$, comp=' + str(components[i])
            axes[i].set_xlabel(xlabel, fontsize=10)
            axes[i].set_ylabel('counts', fontsize=10)
            axes[i].set_xlim([min, max])

            axes[i].legend(loc = 'best', prop={'size': 10})

        label = r'$\eta_{+}$' + ', #=' + str(eta_quad_counts[0])
        sns.histplot(eta_quad_pos, ax=axes[5], label=label)
        label = r'$\eta_{-}$' + ', #=' + str(eta_quad_counts[1])
        sns.histplot(eta_quad_neg, ax=axes[5], label=label)

        xlabel = r'$\log(|\eta|)$' + ', squaring'
        axes[5].set_xlabel(xlabel, fontsize=10)
        axes[5].set_ylabel('counts', fontsize=10)
        axes[5].legend(loc = 'best', prop={'size': 10})
        axes[5].set_xlim([min, max])
        

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'pos_neg_histos'
    format = 'png'
    dpi = 400
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    plt.close()

    # #################################################
    # # PLOTTING THE SIGNS
    # #################################################
    # for i in range(len(axes)):
    #     quantity = eta_cw[components[i]]
    #     quantity = np.sign(quantity)
    #     im = axes[i].imshow(quantity, origin='lower', cmap='Spectral_r')
    #     divider = make_axes_locatable(axes[i])
    #     cax = divider.append_axes('right', size='5%', pad=0.05)
    #     fig.colorbar(im, cax=cax, orientation='vertical')

    #     xlabel = r'$|\eta|/\eta$, comp=' + str(components[i])
    #     axes[i].set_xlabel(xlabel, fontsize=12)

    # fig.tight_layout()

    # fig_directory = config['Directories']['figures_dir'] 
    # filename = 'eta_cw_signs'
    # format = 'png'
    # dpi = 400
    # filename += "." + format
    # plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    # plt.close()

    

    
    

    

