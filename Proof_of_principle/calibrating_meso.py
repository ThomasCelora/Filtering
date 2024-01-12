import sys
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

    # LOADING THE FILTERED MESO MODEL
    directory = config['Directories']['pickled_files_dir']
    meso_setup_filename = config['Filenames']['meso_setup_pickle_name']
    MesoModelLoadFile = directory + meso_setup_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)


    Nt = meso_model.domain_vars['Nt']
    Nx = meso_model.domain_vars['Nx']
    Ny = meso_model.domain_vars['Ny']

    num_points = Nt * Nx * Ny
    print(f'Number of meso gridpoints: {num_points}')

    # DECOMPOSING AND COMPUTING CLOSURE INGREDIENTS
    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished decomposing meso structures in parallel, time taken: {}'.format(time_taken))

    start_time = time.perf_counter()
    meso_model.calculate_derivatives()
    time_taken = time.perf_counter() - start_time
    print('Finished computing derivatives (serial), time taken: {}'.format(time_taken))

    start_time = time.perf_counter()
    meso_model.closure_ingredients_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished computing the closure ingredients in parallel, time taken: {}'.format(time_taken))

    # NOW SAVING
    filename = str(config['Filenames']['meso_decomposed_pickle_name'])
    MesoModelDumpFile = directory + filename
    with open(MesoModelDumpFile, 'wb') as filehandle: 
        pickle.dump(meso_model, filehandle)


