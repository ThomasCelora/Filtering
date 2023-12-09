#!/bin/bash

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
    
  ET=sys.argv[1]
  CPU_start_time = time.perf_counter()
  directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
  MesoModelPickleLoadFile = directory + "observers_2dx_ET_" + ET + ".pickle"
  with open(MesoModelPickleLoadFile, 'rb') as filehandle: 
    meso_model = pickle.load(filehandle)
  time_taken = time.perf_counter()- CPU_start_time
  print('Time taken to read data from pickle: {}'.format(time_taken))

  # filter_widths_ratio = [8]
  filter_widths_ratio = [2, 4, 8]
  meso_dx = meso_model.domain_vars['Dx']
  filter = meso_model.filter
  saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
  # saving_directory = "./"

  for i in range(len(filter_widths_ratio)):    
      CPU_start_time = time.perf_counter()
      filter.set_filter_width(meso_dx * filter_widths_ratio[i])

      # HOW LONG IT TAKES TO FILTER A VAR AT A POINT? 
      # vars = ['BC', 'SET', 'p']
      # for var in vars:
      #   h,l,j = 0, 200, 200
      #   T, X, Y = meso_model.domain_vars['T'][h], meso_model.domain_vars['X'][l], meso_model.domain_vars['Y'][j]
      #   point = [T, X, Y]
      #   obs = meso_model.filter_vars['U'][h,l,j]
      #   filter.filter_var_point(var, point, obs)
      #   print('Filter var {}'.format(var))

      meso_model.filter_micro_variables()
      filename = "resHD_2D_ET_" + ET + "_FW_{}dx.pickle".format(int(filter_widths_ratio[i]))
      MesoModelPickleDumpFile = saving_directory+filename
      with open(MesoModelPickleDumpFile, 'wb') as filehandle:
        pickle.dump(meso_model, filehandle)
      time_taken = time.perf_counter() - CPU_start_time
      print('Time taken to filter data (fw: {}): {}'.format(int(filter_widths_ratio[i]), time_taken))


      


