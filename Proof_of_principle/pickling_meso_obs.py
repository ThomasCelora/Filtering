import sys
# import os
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
sys.path.append('/home/tc2m23/Filtering/master_files/')
import pickle
import time 

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *


if __name__ == '__main__':
   
  # READING THE MICRO-MODEL FROM PICKLE
  CPU_start_time = time.perf_counter()
  directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
  ET=str(sys.argv[1])
  MicroModelPickleLoadFile = directory + "IdealHD_2D_ET_"+ET+"_micro.pickle"

  with open(MicroModelPickleLoadFile, 'rb') as filehandle: 
      micro_model = pickle.load(filehandle)


  # SETTING UP THE MESO-MODEL 
  num_snaps = 11
  box_len_ratio = 2.
  central_slice_num = int(num_snaps/2)
  t_bdrs = [micro_model.domain_vars['t'][central_slice_num], micro_model.domain_vars['t'][central_slice_num+1]]
  box_len = box_len_ratio * micro_model.domain_vars['dx']

  find_obs = FindObs_drift_root(micro_model, box_len)
  filter = spatial_box_filter(micro_model, box_len) #THIS WILL BE CHANGED IN A LATER SCRIPT
  meso_model = resHD2D(micro_model, find_obs, filter)
  grid_bdrs = [0.03, 0.97]
  meso_model.setup_meso_grid([t_bdrs, grid_bdrs, grid_bdrs])
  time_taken = time.perf_counter() - CPU_start_time
  print('Read from micro, grid is set up, time taken: {}'.format(time_taken))

  # ROOT-FINDING THE OBSERVERS 
  CPU_start_time = time.perf_counter()
  # t, x, y = meso_model.domain_vars['T'][0], meso_model.domain_vars['X'][10], meso_model.domain_vars['Y'][10]
  # point = [t,x,y]
  # meso_model.find_obs.find_observer(point)
  meso_model.find_observers()
  time_taken = time.perf_counter() - CPU_start_time
  print('Observers found, time taken: {}'.format(time_taken))

  directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
  MesoModelPickleDumpFile = directory + "observers_2dx_ET_" + ET + ".pickle"
  with open(MesoModelPickleDumpFile, 'wb') as filehandle: 
    pickle.dump(meso_model, filehandle)
