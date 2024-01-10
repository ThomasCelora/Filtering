import sys
import os
sys.path.append('../master_files/')
import pickle
import time 
import configparser
import json

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    hdf5_directory = config['Directories']['hdf5_dir']
    filenames = hdf5_directory 
    FileReader = METHOD_HDF5(filenames)
    num_snaps = FileReader.num_files
    micro_model = IdealHD_2D()
    FileReader.read_in_data_HDF5_missing_xy(micro_model)
    micro_model.setup_structures()
    print('Finished reading micro data from hdf5, structures also set up.')

    # SETTING UP THE MESO MODEL 
    CPU_start_time = time.perf_counter()
    central_slice_num = int(num_snaps/2)
    t_range = [micro_model.domain_vars['t'][central_slice_num-1], micro_model.domain_vars['t'][central_slice_num+1]]

    meso_grid = json.loads(config['Meso_model_settings']['meso_grid'])
    filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    x_range = meso_grid['x_range']
    y_range = meso_grid['y_range']
    box_len_ratio = float(filtering_options['box_len_ratio'])
    filter_width_ratio =  filtering_options['filter_width_ratio']

    box_len = box_len_ratio * micro_model.domain_vars['dx']
    width = filter_width_ratio * micro_model.domain_vars['dx']
    find_obs = FindObs_root_parallel(micro_model, box_len)
    filter = box_filter_parallel(micro_model, width)
    meso_model = resHD2D(micro_model, find_obs, filter) 
    meso_model.setup_meso_grid([t_range, x_range, y_range])

    time_taken = time.perf_counter() - CPU_start_time
    print('Grid is set up, time taken: {}'.format(time_taken))
    num_points = meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny'] * meso_model.domain_vars['Nt']
    print('Number of points: {}'.format(num_points))

    # FINDING THE OBSERVERS AND FILTERING, THEN PICKLE SAVE
    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.find_observers_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Observers found in parallel: time taken= {}'.format(time_taken))

    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Parallel filtering stage ended (fw= {}): time taken= {}\n'.format(int(filter_width_ratio), time_taken))

    pickled_directory = config['Directories']['pickled_files_dir']
    filename = config['Pickled_filenames']['meso_pickled'] + "_FW_" +str(int(filter_width_ratio)) + "dx" + ".pickle"
    MesoModelPickleDumpFile = pickled_directory + filename
    with open(MesoModelPickleDumpFile, 'wb') as filehandle:
        pickle.dump(meso_model, filehandle)
    



