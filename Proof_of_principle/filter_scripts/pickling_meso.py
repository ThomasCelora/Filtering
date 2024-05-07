import sys
import os
sys.path.append('../../master_files/')
import pickle
import time 
import configparser
import json

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':

    ####################################################################################################
    # # MAIN SCRIPT OF PIPELINE: SET UP THE MESO MODEL FROM SIM DATA (MESO-GRID + OBSERVERS + FILTER)
    # # AND DECOMPOSE THE RESIDUALS AS WELL AS COMPUTE DERIVATIVES AND QUANTITIES FOR MODELLING THEM 
    ####################################################################################################
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.", flush=True)
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # SETTING UP THE MICROMODEL
    start_time = time.perf_counter()
    hdf5_directory = config['Directories']['hdf5_dir']
    print(f'Starting job with data from {hdf5_directory}', flush=True)
    print('==============================================\n')
    filenames = hdf5_directory  
    snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
    fewer_snaps_required = snapshots_opts['fewer_snaps_required']
    smaller_list = snapshots_opts['smaller_list']
    FileReader = METHOD_HDF5(filenames,fewer_snaps_required, smaller_list)
    num_snaps = FileReader.num_files
    micro_model = IdealHD_2D()
    FileReader.read_in_data_HDF5_missing_xy(micro_model)
    micro_model.setup_structures()
    time_taken = time.perf_counter() - start_time
    print(f'Finished reading micro data from hdf5, structures also set up. Time taken: {time_taken}', flush=True)

    # SETTING UP THE MESO MODEL 
    start_time = time.perf_counter()
    meso_grid = json.loads(config['Meso_model_settings']['meso_grid'])
    filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    coarse_factor = meso_grid['coarse_grain_factor']
    coarse_time = meso_grid['coarse_grain_time']
    central_slice_num = int(num_snaps/2)
    num_T_slices = meso_grid['num_T_slices']
    furthest_slice_number = int((num_T_slices-1)/2)
    if coarse_time:
        furthest_slice_number = int(coarse_factor * furthest_slice_number)

    t_range = [micro_model.domain_vars['t'][central_slice_num-furthest_slice_number], micro_model.domain_vars['t'][central_slice_num+furthest_slice_number]]
    x_range = meso_grid['x_range']
    y_range = meso_grid['y_range']

    ts = micro_model.domain_vars['t'][:]
    print(f'ts: {ts}\n')

    box_len_ratio = float(filtering_options['box_len_ratio'])
    filter_width_ratio =  float(filtering_options['filter_width_ratio'])

    box_len = box_len_ratio * micro_model.domain_vars['dx']
    width = filter_width_ratio * micro_model.domain_vars['dx']
    find_obs = FindObs_root_parallel(micro_model, box_len)
    filter = box_filter_parallel(micro_model, width)
    meso_model = resHD2D(micro_model, find_obs, filter) 
    meso_model.setup_meso_grid([t_range, x_range, y_range], coarse_factor = coarse_factor, coarse_time = coarse_time)

    time_taken = time.perf_counter() - start_time
    print('Grid is set up, time taken: {}\n'.format(time_taken), flush=True)
    num_points = meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny'] * meso_model.domain_vars['Nt']
    print('Number of points: {}\n'.format(num_points), flush=True)

    # FINDING THE OBSERVERS
    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.find_observers_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Observers found in parallel: time taken= {}\n'.format(time_taken), flush=True)


    # # UNCOMMENT IF MESO_MODEL EXISTS PICKLED EXIST ALREADY AND YOU WANT TO RE-RUN SOME OF THE LATER ROUTINES:
    # # LOADING MESO MODEL 
    # pickle_directory = config['Directories']['pickled_files_dir']
    # meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    # MesoModelLoadFile = pickle_directory + meso_pickled_filename
    # n_cpus = int(config['Meso_model_settings']['n_cpus'])

    # print('=========================================================================')
    # print(f'Starting job on data from {MesoModelLoadFile}')
    # print('=========================================================================\n\n')
    # with open(MesoModelLoadFile, 'rb') as filehandle: 
    #     meso_model = pickle.load(filehandle) 


    # # This is to adjust the filter-size by retaining the observers computed above
    # filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])
    # filter_width_ratio =  filtering_options['filter_width_ratio']
    # micro_model = meso_model.micro_model
    # width = filter_width_ratio * micro_model.domain_vars['dx']
    # filter = box_filter_parallel(micro_model, width)
    # meso_model.set_filter(filter)

    
    # FILTERING
    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Parallel filtering stage ended (fw= {}): time taken= {}\n'.format(int(filter_width_ratio), time_taken), flush=True)


    # DECOMPOSING AND CALCULATING THE CLOSURE INGREDIENTS
    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished decomposing meso structures in parallel, time taken: {}\n'.format(time_taken), flush=True)


    start_time = time.perf_counter()
    meso_model.calculate_derivatives()
    time_taken = time.perf_counter() - start_time
    print('Finished computing derivatives (serial), time taken: {}\n'.format(time_taken), flush=True)

    start_time = time.perf_counter()
    meso_model.closure_ingredients_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished computing the closure ingredients in parallel, time taken: {}\n'.format(time_taken), flush=True)

    start_time = time.perf_counter()
    meso_model.EL_style_closure_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished computing the EL_style closure in parallel, time taken: {}\n'.format(time_taken), flush=True)

    start_time = time.perf_counter()
    meso_model.modelling_coefficients_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished computing quantities to model extracted coefficients, time taken: {}\n'.format(time_taken), flush=True)

    # start_time = time.perf_counter()
    # meso_model.build_weights_Q1()
    # time_taken = time.perf_counter() - start_time
    # print('Finished computing weights in serial, time taken: {}\n'.format(time_taken), flush=True)

    # PICKLING THE CLASS INSTANCE FOR FUTURE USE 
    pickle_directory = config['Directories']['pickled_files_dir']
    filename = config['Filenames']['meso_pickled_filename']
    MesoModelPickleDumpFile = pickle_directory + filename
    with open(MesoModelPickleDumpFile, 'wb') as filehandle:
        pickle.dump(meso_model, filehandle)
    



