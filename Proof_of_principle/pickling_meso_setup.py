import sys
# import os
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
# sys.path.append('/home/tc2m23/Filtering/master_files/')
import pickle
import time 

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':
   
  # READING MICRO DATA FORM PICKLE
  CPU_start_time = time.perf_counter()
  # directory = "/scratch/tc2m23/NewData4Filtering/hydro/ideal/ET_2_4_6_8/10_dx_after/pickle_files/400X400/"
  directory = "/Users/thomas/Dropbox/Work/projects/Filtering/Data/ET_2_4_6_8/10_dx_after/pickle_files/80X80/"
  MicroModelPickleLoadFile = directory + "IdealHD_2D_ET_02_micro.pickle"

  with open(MicroModelPickleLoadFile, 'rb') as filehandle: 
      micro_model = pickle.load(filehandle)
  time_taken = time.perf_counter() - CPU_start_time
  print('Time taken to read file from pickle: {}'.format(time_taken))
  # print("Micro_model domain_vars keys: {}".format(micro_model.domain_vars.keys()))

  # FINDING THE OBSERVERS, FILTERING 
  CPU_start_time = time.perf_counter()
  num_snaps = 21
  width_ratio_dx = 0.1
  central_slice_num = int(num_snaps/2)
  t_bdrs = [micro_model.domain_vars['t'][central_slice_num], micro_model.domain_vars['t'][central_slice_num]]
  width = width_ratio_dx * micro_model.domain_vars['dx']

  # xm, xM = np.min(micro_model.domain_vars['x']), np.max(micro_model.domain_vars['x'])
  # ym, yM = np.min(micro_model.domain_vars['y']), np.max(micro_model.domain_vars['y'])
  # print('Micro_model grid extent - careful when setting up meso-grid: \nX: {},  {} \nY: {},  {}'.format(xm, xM, ym, yM))

  find_obs = FindObs_drift_root(micro_model, width)
  filter = spatial_box_filter(micro_model, width)
  meso_model = resHD2D(micro_model, find_obs, filter)
  # meso_model.setup_meso_grid([t_bdrs,[0.01,0.99],[0.01,0.99]])
  meso_model.setup_meso_grid([t_bdrs,[0.05,0.95],[0.05,0.95]])
  time_taken = time.perf_counter() - CPU_start_time
  print('Grid is set up, time taken: {}'.format(time_taken))

  CPU_start_time = time.perf_counter()
  meso_model.find_observers()
  time_taken = time.perf_counter() - CPU_start_time
  print('Observers found, time taken: {}'.format(time_taken))

  CPU_start_time = time.perf_counter()
  meso_model.filter_micro_variables()
  time_taken = time.perf_counter() - CPU_start_time
  print('Filter stage ended, time taken: {}'.format(time_taken))


  # PICKLING THE FILTERED DATA CLASS
  # directory = "/scratch/tc2m23/NewData4Filtering/hydro/ideal/ET_2_4_6_8/10_dx_after/pickle_files/400X400/"
  directory = "/Users/thomas/Dropbox/Work/projects/Filtering/Data/ET_2_4_6_8/10_dx_after/pickle_files/80X80/"
  # MesoModelPickleDumpFile = directory + "resHD_2D_ET_02_meso.pickle"
  MesoModelPickleDumpFile = directory + "resHD_2D_ET_02_all_grid.pickle"
  with open(MesoModelPickleDumpFile, 'wb') as filehandle: 
    pickle.dump(meso_model, filehandle)
