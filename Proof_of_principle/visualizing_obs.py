#!/bin/bash

import sys
# import os
sys.path.append('/home/tc2m23/Filtering/master_files/')
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
import pickle

from FileReaders import *
from MicroModels import *
from MesoModels import *
from Visualization import *

if __name__ == '__main__':

    # Reading micro & meso models (with obs) from pickle
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
    ET=str(sys.argv[1])
    MesoModelLoadFile = directory + "observers_2dx_ET_" + ET+ ".pickle"
    MicroModelLoadFile = directory + "IdealHD_2D_ET_" + ET+"_micro.pickle"

    with open(MicroModelLoadFile, 'rb') as filehandle: 
        micro_model = pickle.load(filehandle)

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    num_snaps = 11
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    time_meso = meso_model.domain_vars['T'][0]
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!")
    else: 
        print("Comparing data at same time-slice, hurray!")
    saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/Figures/400X400/"
    # saving_directory = "./"

    models = [micro_model, meso_model]
    vars = [['bar_vel', 'bar_vel', 'bar_vel'],['U', 'U', 'U']]
    components_indices= [[(0,),(1,),(2,)], [(0,), (1,), (2,)]]
    ranges = [0.04, 0.97]
    visualizer = Plotter_2D([11.97, 8.36])
    fig = visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges, ranges, components_indices=components_indices, diff_plot=True, rel_diff=True)
    fig.tight_layout()
    filename="Meso_observers_2dx_ET_"+ET+".svg"
    plt.savefig(saving_directory + filename, format="svg")