import sys
# import os
sys.path.append('/home/tc2m23/Filtering/master_files/')
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
import pickle
import time

from FileReaders import *
from MicroModels import *

if __name__ == '__main__':

    # READING DATA FORM CHECKPOINT, SETTING UP MICOR AND PICKLING
    CPU_start_time = time.perf_counter()
    # directory = "/Users/thomas/Dropbox/Work/projects/Filtering/Data/ET_2_4_6_8/10_dx_after/METHOD_output/80X80/ET_02"
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/METHOD_output/400X400/"
    ET=str(sys.argv[1])
    names = directory + "ET_" + ET
    print(names)

    FileReader = METHOD_HDF5(names)
    micro_model = IdealHD_2D()
    FileReader.read_in_data_HDF5_missing_xy(micro_model)
    micro_model.setup_structures()
    time_taken = int((time.perf_counter() - CPU_start_time) * 100)/100.
    print('Time to set up micro model (ET: {}): {}'.format(ET, time_taken))

    # Pickle save 
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
    CPU_start_time = time.perf_counter()
    MicroModelPickleDumpFile = directory + "IdealHD_2D_ET_" + ET + "_micro.pickle"
    with open(MicroModelPickleDumpFile, 'wb') as filehandle: 
        pickle.dump(micro_model, filehandle)
    time_taken = int((time.perf_counter() - CPU_start_time) * 100)/100.
    print('Time needed to pickle class: {}'.format(time_taken))

    # READING AND PICKLING ALL CHECKPOINTS - FOR VISUALIZATION 
    # CPU_start_time = time.perf_counter()
    # # directory = "/scratch/tc2m23/NewData4Filtering/hydro/ideal/ET_2_4_6_8/10_dx_after/METHOD_output/400X400/"
    # directory = "/Users/thomas/Dropbox/Work/projects/Filtering/Data/ET_2_4_6_8/10_dx_after/METHOD_output/80X80/"
    # FileReader = METHOD_HDF5(directory)
    # micro_model = IdealHD_2D()
    # FileReader.read_in_data_HDF5_missing_xy(micro_model)
    # micro_model.setup_structures()
    # time_taken = int((time.perf_counter() - CPU_start_time) * 100)/100.
    # print('Time to set up micro model (up to 8ET): {}'.format(time_taken))

    # # directory = "/scratch/tc2m23/NewData4Filtering/hydro/ideal/ET_2_4_6_8/10_dx_after/pickle_files/400X400/"
    # directory = "/Users/thomas/Dropbox/Work/projects/Filtering/Data/ET_2_4_6_8/10_dx_after/pickle_files/80X80/"
    # CPU_start_time = time.perf_counter()
    # pickle_save = True
    # MicroModelPickleDumpFile = directory  + "IdealHD_2D_ET_ALL_micro.pickle"
    # if pickle_save: 
    #     with open(MicroModelPickleDumpFile, 'wb') as filehandle: 
    #         pickle.dump(micro_model, filehandle)
    # time_taken = int((time.perf_counter() - CPU_start_time) * 100)/100.
    # print('Time needed to pickle class: {}'.format(time_taken))

