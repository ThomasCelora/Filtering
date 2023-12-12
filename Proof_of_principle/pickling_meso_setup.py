import sys
# import os
sys.path.append('/home/tc2m23/Filtering/master_files/')
import pickle
import time 

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':
   
  # READING MICRO DATA FORM PICKLE
  CPU_start_time = time.perf_counter()
  directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_3.5_step0.5/20dx/pickled_files/800X800/"
  ET=str(sys.argv[1])
  MicroModelPickleLoadFile = directory + "HD_2D_ET_"+ET+"_micro.pickle"

  with open(MicroModelPickleLoadFile, 'rb') as filehandle: 
      micro_model = pickle.load(filehandle)
  time_taken = time.perf_counter() - CPU_start_time
  print('Time taken to read file from pickle: {}'.format(time_taken))

  # SETTING UP THE MESO MODEL 
  CPU_start_time = time.perf_counter()
  num_snaps = 21
  central_slice_num = int(num_snaps/2)
  t_bdrs = [micro_model.domain_vars['t'][central_slice_num-1], micro_model.domain_vars['t'][central_slice_num+1]]
  x_bdrs = [0.02, 0.98]
  y_bdrs = [0.02, 0.98]
  # t_bdrs = [micro_model.domain_vars['t'][central_slice_num], micro_model.domain_vars['t'][central_slice_num]]
  # x_bdrs = [0.2, 0.25]
  # y_bdrs = [0.2, 0.25]

  box_len_ratio = 4.
  box_len = box_len_ratio * micro_model.domain_vars['dx']
  # setting the width is needed for constructing the filter, and is changed later
  width_ratio = 2.
  width = width_ratio * micro_model.domain_vars['dx']
  find_obs = FindObs_root_parallel(micro_model, box_len)
  filter = box_filter_parallel(micro_model, width)
  meso_model = resHD2D(micro_model, find_obs, filter)
  meso_model.setup_meso_grid([t_bdrs, x_bdrs, y_bdrs])

  num_points = meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny'] * meso_model.domain_vars['Nt']
  print('Number of points: {}'.format(num_points))

  time_taken = time.perf_counter() - CPU_start_time
  print('Grid is set up, time taken: {}'.format(time_taken))

  # FINDING THE OBSERVERS
  start_time = time.perf_counter()
  meso_model.find_observers_parallel()
  time_taken = time.perf_counter() - start_time
  print('Observers found in parallel: time taken= {}'.format(time_taken))


  # FILTERING WITH DIFFERENT WIDTHS
  filter_width_ratios = [2., 4., 8.]
  # filter_width_ratios = [2.]
  filter = meso_model.filter
  saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_3.5_step0.5/20dx/pickled_files/800X800/"

  for i in range(len(filter_width_ratios)):
    # filtering in parallel
    start_time = time.perf_counter()
    filter.set_filter_width( micro_model.domain_vars['dx'] * filter_width_ratios[i] )
    meso_model.filter_micro_vars_parallel()
    time_taken = time.perf_counter() - start_time
    print('Parallel filtering stage ended (fw= {}): time taken= {}\n'.format(int(filter_width_ratios[i]), time_taken))

    # Now saving the output
    filename = "resHD_2D_ET_" + ET + "_FW_{}dx.pickle".format(int(filter_width_ratios[i]))
    MesoModelPickleDumpFile = saving_directory+filename
    with open(MesoModelPickleDumpFile, 'wb') as filehandle:
       pickle.dump(meso_model, filehandle)
    



