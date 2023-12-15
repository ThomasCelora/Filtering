import sys
import os
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
# sys.path.append('/home/hidra2/celora/Filtering/master_files/')
sys.path.append('/home/tc2m23/Filtering/master_files/')
import pickle
import time 

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':
    
    #########################################################
    # CODE FOR TESTING SPEED UP OF PARALLEL MESO ROUTINES
    #########################################################
    # TESTING ON HIDRA/IRIDIS
    ET = str(sys.argv[1])
    # x_ranges = [0.2,0.4]
    # y_ranges = [0.2, 0.4]
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_3.5_step0.5/20dx/pickled_files/800X800/"
    MesoModelPickleLoadFile = directory + "resHD_2D_ET_" + ET + "_FW_8dx.pickle"

    # directory = "/home/hidra2/celora/Data/KHIrandom/ideal_HD/ET_1_3.5_0.5/20dx/pickled_files/800X800_smaller/"
    # MesoModelPickleLoadFile = directory + "rHD_2D_ET_"+ET+"_FW_8dx_x="+ str(x_ranges)+ "_y="+str(y_ranges)+".pickle"
    

    with open(MesoModelPickleLoadFile, 'rb') as filehandle: 
        meso_model=pickle.load(filehandle)
    
    num_points = meso_model.domain_vars['Nt'] * meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny']
    print('Working with {} points\n'.format(num_points))
   
    n_cpus=40

    # decomposing structures in serial
    start_time = time.perf_counter()
    meso_model.decompose_structures()
    serial_time = time.perf_counter() - start_time
    print('Meso structures decomposed in serial, time taken: {}\n'.format(serial_time))

    # decomposing in parallel and comparing
    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus=n_cpus)
    parallel_time = time.perf_counter() - start_time
    print('Meso structures decomposed in parallel, time taken: {}\n'.format(parallel_time))
    print('Speed-up factor: {}'.format(serial_time/parallel_time))

    # compute derivatives serial
    start_time = time.perf_counter()
    meso_model.calculate_derivatives()
    serial_time = time.perf_counter() - start_time
    print('Calculated derivatives in serial, time taken: {}\n'.format(serial_time))

    # computing the closure ingredients - serial
    start_time = time.perf_counter()
    meso_model.closure_ingredients()
    serial_time = time.perf_counter() - start_time
    print('Closure ingredients calculated in serial, time taken: {}\n'.format(serial_time))

    # computing the closure ingredients in parallel and comparing
    start_time = time.perf_counter()
    meso_model.closure_ingredients_parallel(n_cpus = n_cpus)
    parallel_time = time.perf_counter() - start_time
    print('Closure ingredients calculated in parallel, time taken: {}\n'.format(parallel_time))
    print('Speed-up factor: {}'.format(serial_time/parallel_time))

    # computing the coefficients - serial
    start_time = time.perf_counter()
    meso_model.EL_style_closure()
    serial_time = time.perf_counter() - start_time
    print('Dissipative coefficients calculated in serial, time taken: {}\n'.format(serial_time))

    # computing the closure ingredients in parallel and comparing
    start_time = time.perf_counter()
    meso_model.closure_ingredients_parallel(n_cpus = n_cpus)
    parallel_time = time.perf_counter() - start_time
    print('Dissipative coefficients calculated in parallel, time taken: {}\n'.format(parallel_time))
    print('Speed-up factor: {}'.format(serial_time/parallel_time))



    ##########################################################
    # CODE FOR TESTING THAT THE TASKS DO WHAT THEY'RE MEANT TO  
    ##########################################################
    # werk