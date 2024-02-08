import sys
# import os
sys.path.append('../master_files/')
import pickle
import configparser
import json
import time

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##############################################################
    # #PLOT KEY QUANTITIES TO COMPARE MICRO AND MESO MODELS
    # #AND THOSE RELEVANT TO CALIBRATE CLOSURE 
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

    print('=========================================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('=========================================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)
    micro_model = meso_model.micro_model

    print('Finished reading pickled data')

    # # If you need to recompute, e.g. the closure coefficients
    # n_cpus = int(config['Meso_model_settings']['n_cpus'])
    # meso_model.EL_style_closure_parallel(n_cpus)
    # print('Finished re-decomposing the meso_model! Now re-saving it')
    # with open(MesoModelLoadFile, 'wb') as filehandle:
    #     pickle.dump(meso_model, filehandle)
    # print('Finished re-computing the dissipative coefficients and related quantities! Now re-saving it')

    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    num_slices_meso = int(config['Models_settings']['mesogrid_T_slices_num'])
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!")
    else: 
        print("Comparing data at same time-slice, hurray!")

    # # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D([11.97, 8.36])
    diff_plot_settings =json.loads(config['Plot_settings']['diff_plot_settings']) 
    diff_method = diff_plot_settings['method']
    interp_dims = diff_plot_settings['interp_dims']

    # PLOTTING MICRO VS FILTERED BC AND SET
    # #####################################
    start_time = time.perf_counter()
    vars = [['BC', 'SET'], ['BC', 'SET']]
    models = [micro_model, meso_model]
    components = [[(0,), (0,0)], [(0,), (0,0)]]
    fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices = components, 
                                               interp_dims = interp_dims, method = diff_method, diff_plot=False, rel_diff=False)
    fig.tight_layout()
    filename = "/microVSmeso_scaling.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')

    vars = [['BC', 'SET'], ['BC', 'SET',]]
    models = [micro_model, meso_model]
    components = [[(2,), (0,2)], [(2,), (0,2)]]
    fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices = components, 
                                               interp_dims = interp_dims, method = diff_method, diff_plot=False, rel_diff=False)
    fig.tight_layout()
    filename = "/microVSmeso_notscaling.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')

    time_taken = time.perf_counter() - start_time
    print(f'Finished plotting model comparison: time taken (X2) ={time_taken}')

    # # PLOTTING THE DECOMPOSED SET 
    # #############################
    vars_strs = ['pi_res', 'pi_res', 'pi_res', 'pi_res', 'pi_res', 'pi_res']
    norms = ['mysymlog', 'mysymlog', 'mysymlog', 'mysymlog', 'mysymlog', 'mysymlog']
    cmaps = ['seismic', 'seismic', 'seismic', 'seismic', 'seismic', 'seismic']
    components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/DecomposedSET_1.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')

    vars_strs = ['q_res', 'q_res', 'q_res', 'Pi_res', 'p_tilde', 'p_filt']
    norms = ['mysymlog', 'mysymlog', 'mysymlog', None, None, None]
    cmaps = ['seismic', 'seismic', 'seismic', None, None, None]
    components = [(0,), (1,), (2,), (), (), ()]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/DecomposedSET_2.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting decomposition of SET')

    # # PLOTTING THE DERIVATIVES OF FAVRE VEL AND TEMPERATURE
    ######################################################### 
    favre_vel_components = [0,1,2]
    for i in range(len(favre_vel_components)):
        components = [tuple([favre_vel_components[i]])]
        for j in range(3):
            components.append(tuple([j,favre_vel_components[i]]))
        vars_strs = ['u_tilde', 'D_u_tilde', 'D_u_tilde', 'D_u_tilde']
        fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components)
        fig.tight_layout()
        time_for_filename = str(round(time_meso,2))
        filename = "/D_favre_comp={}.pdf".format(favre_vel_components[i])
        plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting derivatives of favre velocity', flush=True)

    vars_strs = ['T_tilde', 'D_T_tilde', 'D_T_tilde', 'D_T_tilde']
    components = [(), (0,), (1,), (2,)]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/D_T.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting temperature derivatives', flush=True)

    # # PLOTTING SHEAR, ACCELERATION, EXPANSION AND TEMPERATURE DERIVATIVES
    ########################################################################
    vars_strs = ['shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde']
    components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Shear_comps.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting shear', flush=True)

    vars_strs = ['acc_tilde', 'acc_tilde', 'acc_tilde', 'Theta_tilde', 'Theta_tilde', 'Theta_tilde']
    components = [(0,), (1,), (2,), (0,), (1,), (2,)]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Acc+Tderivs.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting DaT', flush=True)

    vars_strs = ['exp_tilde']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Expansion.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting expansion', flush=True)

    
