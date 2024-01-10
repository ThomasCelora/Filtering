import sys
# import os
sys.path.append('../master_files/')
import configparser
import json
import pickle

from FileReaders import *
from MicroModels import *
from Visualization import *

if __name__ == '__main__':

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # LOADING MICRO DATA FROM HDF5 OR PICKLE
    micro_from_hdf5 = True

    if micro_from_hdf5:
        hdf5_directory = config['Directories']['hdf5_dir']
        filenames = hdf5_directory 
        FileReader = METHOD_HDF5(filenames)
        num_snaps = FileReader.num_files
        micro_model = IdealHD_2D()
        FileReader.read_in_data_HDF5_missing_xy(micro_model)
        micro_model.setup_structures()
        print('Finished reading micro data from hdf5, structures also set up.')

    else: 
        pickle_directory = config['Directories']['pickled_files_dir']
        micro_pickled_filename = config['Filenames']['micro_pickled_filename']
        MicroModelLoadFile = pickle_directory + micro_pickled_filename
        with open(MicroModelLoadFile, 'rb') as filehandle: 
            micro_model = pickle.load(filehandle)

    # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']

    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    plot_time = micro_model.domain_vars['t'][central_slice_num]

    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D([11.97, 8.36])

    # FINALLY, PLOTTING
    # Plotting the baryon current
    vars = ['BC', 'BC', 'BC', 'n', 'W', 'vx']
    components = [(0,), (1,), (2,), (), (), ()]
    fig=visualizer.plot_vars(micro_model, vars, plot_time, x_range, y_range, components_indices = components)
    fig.tight_layout()
    time_for_filename = str(round(plot_time,2))
    filename = "/micro_ET_" + time_for_filename + "_BC.pdf"
    plt.savefig(saving_directory + filename, format = "pdf")

    # Plotting the stress energy tensor
    vars = ['SET', 'SET', 'SET', 'SET', 'SET', 'SET']
    components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    fig=visualizer.plot_vars(micro_model, vars, plot_time, x_range, y_range, components_indices = components)
    fig.tight_layout()
    time_for_filename = str(round(plot_time,2))
    filename = "/micro_ET_" + time_for_filename + "_SET.pdf"
    plt.savefig(saving_directory + filename, format = "pdf")

    # plotting primitive quantities
    vars = ['W', 'vx', 'vy', 'n', 'p', 'e']
    fig=visualizer.plot_vars(micro_model, vars, plot_time, x_range, y_range)
    fig.tight_layout()
    time_for_filename = str(round(plot_time,2))
    filename = "/micro_ET_" + time_for_filename + "_prims.pdf"
    plt.savefig(saving_directory + filename, format = "pdf")

