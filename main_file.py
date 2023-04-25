# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:45:31 2023

@author: marcu
"""

from multiprocessing import Process, Pool
from FileReaders import *
from Filters import *
from MicroModels import *
from MesoModels import *



if __name__ == '__main__':
    

    CPU_start_time = time.process_time()

    FileReader = METHOD_HDF5('./Data/Testing/')
    # micro_model = IdealMHD_2D()
    micro_model = IdealHydro_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()

    Filter = Favre_observers(micro_model,box_len=0.001)
    
    # tetrad = filter.get_tetrad_from_vxvy([0.5,0.73])
    # print(type(tetrad[0]),' ',type(tetrad[1]),' ',type(tetrad[2]),'\n', tetrad[0],' ',tetrad[1],' ',tetrad[2],'\n')
    # print(filter.Mink_dot(tetrad[0],tetrad[1]), filter.Mink_dot(tetrad[0],tetrad[2]), filter.Mink_dot(tetrad[1],tetrad[2]))
    # print(type(tetrad))

    # smart_guess = micro_model.get_interpol_prim(['vx','vy'],[0.5,0.5,0.5])

    # CPU_start_time = time.process_time()
    # res = filter.Favre_residual(smart_guess,[0.5,0.5,0.5], 10)
    # print('Residual: ',res,f'\nElapsed CPU time is {time.process_time() - CPU_start_time} with {10**filter.spatial_dim} points per face\n')

    # CPU_start_time = time.process_time()
    # res, error = filter.Favre_residual_ib(smart_guess,[0.5,0.5,0.5])[:]
    # print('Residual: ',res,"\nError estimate: ",error,f'\nElapsed CPU time is {time.process_time() - CPU_start_time} with the inbuilt method')

    CPU_start_time = time.process_time()
    coord_range = [[9.995,10.005],[-0.2,-0.3],[0.5,0.7]]
    num_points = [1,1,1]
    
    min_res, failed_coord = Filter.find_observers(num_points, coord_range, 10)
    for i in range(len(min_res[0])):
        for j in range(len(min_res)):
            print(min_res[j][i])
        print('\n')

    num_minim = 1
    for x in num_points: 
        num_minim *= x
    print(f'Elapsed CPU time for finding {num_minim} observer(s) is {time.process_time() - CPU_start_time}.')
    print('Failed coordinates:', failed_coord)
    
    # Filter = Box_2D(0.1)
    MesoModel = NonIdealHydro(micro_model, Filter)
    MesoModel.calculate_coefficients()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    