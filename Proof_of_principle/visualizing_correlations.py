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

    # ##############################################################
    # #PLOT KEY QUANTITIES RELEVANT TO CALIBRATE CLOSURE 
    # ##############################################################
    
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

    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    # #UNCOMMENT IF YOU NEED TO RE-COMPUTE, SAY, THE CLOSURE COEFFICIENTS
    # ########################################
    # n_cpus = int(config['Meso_model_settings']['n_cpus'])
    # meso_model.EL_style_closure_parallel(n_cpus)
    # print('Finished re-decomposing the meso_model! Now re-saving it')
    # with open(MesoModelLoadFile, 'wb') as filehandle:
    #     pickle.dump(meso_model, filehandle)
    # print('Finished re-computing the dissipative coefficients and related quantities! Now re-saving it')

    
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    # extent = [y_range[0], y_range[-1], x_range[0], x_range[-1]]
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    saving_directory = config['Directories']['figures_dir']

    visualizer = Plotter_2D() 

    vars_strs = ['zeta', 'exp_tilde', 'Pi_res']
    norms = ['mysymlog', 'mysymlog', 'log']
    cmaps = ['seismic', 'seismic', None]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Bulk_viscosity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished bulk viscosity')

    vars_strs = ['eta', 'shear_sq', 'pi_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['seismic', None, None]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Shear_viscosity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished shear viscosity')

    vars_strs = ['kappa', 'Theta_sq', 'q_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['seismic', None, None]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Heat_conductivity.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished heat conductivity')
    
    
    
    