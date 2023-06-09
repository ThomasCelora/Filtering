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
from Visualization import *



if __name__ == '__main__':
    

    CPU_start_time = time.process_time()

    FileReader = METHOD_HDF5('./Data/Testing/')
    # FileReader = METHOD_HDF5('../../Filtering/Data/KH/Ideal/t_1998_2002/')
    # micro_model = IdealMHD_2D()
    micro_model = IdealHydro_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()

    visualizer = Plotter_2D()
    visualizer.plot_vars(micro_model, ['v1','n'], t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2],\
                          interp_dims=(20,40), method='interpolate', component_indices=[(),()])
    
    # visualizer.plot_micro_var_2D_interpol('v1', t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2], dimensions=(20,40))
    # visualizer.plot_micro_var_2D_rawdata('v1', t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2])

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
    coord_range = [[10.000, 10.005],[-0.1,0.1], [-0.2,0.2]]
    num_points = [2,5,10]
    spacing = 10

    meso_model = NonIdealHydro2D(micro_model, Filter)
    meso_model.find_observers(num_points, coord_range, spacing)
    meso_model.setup_variables()
    meso_model.filter_micro_variables()
    meso_model.calculate_dissipative_coefficients()

    visualizer.plot_vars(meso_model, ['U','pi'], t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2],\
                      interp_dims=(20,40), method='interpolate', component_indices=[(1),(0,1)])
            
    visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
                                         t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2],\
                      interp_dims=(20,40), method='interpolate', component_index=(1,2))
        
        
    # visualizer.set_meso_model(meso_model)
    # visualizer.plot_meso_var_2D_rawdata('U', t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2])
    # visualizer.plot_meso_var_2D_interpol('U', t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2], dimensions=(20,40))

    # visualizer.compare_micro_meso_var_interpol(micro_var_str='v1', meso_var_str='U',\
    #            t=10.000, x_range=[-0.1,0.1], y_range=[-0.2,0.2], dimensions=(20,40))
    
 
    # min_res, failed_coord = Filter.find_observers(num_points, coord_range, 10)
    # for i in range(len(min_res[0])):
    #     for j in range(len(min_res)):
    #         print(min_res[j][i])
    #     print('\n')

    # num_minim = 1
    # for x in num_points: 
    #     num_minim *= x
    # print(f'Elapsed CPU time for finding {num_minim} observer(s) is {time.process_time() - CPU_start_time}.')
    # print('Failed coordinates:', failed_coord)
    
    # Filter = Box_2D(0.1)
    #meso_model = NonIdealHydro2D(micro_model, Filter)
    #meso_model.calculate_coefficients()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    