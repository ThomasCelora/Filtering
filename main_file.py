# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:45:31 2023

@author: marcu
"""

from multiprocessing import Process, Pool
from FileReaders import *
from Filters_TC import *
from MicroModels import *
from MesoModels import *
from Visualization import *
from Analysis import *

if __name__ == '__main__':
    
    # Pickling Options
    LoadMicroModelFromPickleFile = False
    MicroModelPickleLoadFile = 'IdealHydro2D.pickle'

    DumpMicroModelToPickleFile = True
    MicroModelPickleDumpFile = 'IdealHydro2D.pickle'

    LoadMesoModelFromPickleFile = False
    MesoModelPickleLoadFile = 'NonIdealHydro2D.pickle'
    
    DumpMesoModelToPickleFile = True
    MesoModelPickleDumpFile = 'NonIdealHydro2D.pickle'
    
    t_slice_plotting = 10.0
    x_range_plotting = [-0.2,0.2]
    y_range_plotting = [-0.3,0.3]
    
    # Start timing    
    CPU_start_time = time.process_time()

    # Read in data from file
    HDF5_Directory = './Data/Testing/20x40/'
    FileReader = METHOD_HDF5(HDF5_Directory)
    # FileReader = METHOD_HDF5('../../Filtering/Data/KH/Ideal/t_998_1002/')

    # Create and setup micromodel
    if LoadMicroModelFromPickleFile:
        with open(MicroModelPickleLoadFile, 'rb') as filehandle:
            micro_model = pickle.load(filehandle) 
    else:
        micro_model = IdealHydro_2D()
        FileReader.read_in_data(micro_model) 
        micro_model.setup_structures()

    if DumpMicroModelToPickleFile:
        with open(MicroModelPickleDumpFile, 'wb') as filehandle:
            pickle.dump(micro_model, filehandle)  

    # Create visualizer for plotting micro data
    visualizer = Plotter_2D()
    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(),(),()])
    visualizer.plot_vars(micro_model, ['BC'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='raw_data', components_indices=[(1,)])

    visualizer.plot_vars(micro_model, ['v1','v2','n'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(),(),()])
    visualizer.plot_vars(micro_model, ['BC'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                          interp_dims=(20,40), method='interpolate', components_indices=[(1,)])

    # Create the observer-finder and filter
    ObsFinder = FindObs_drift_root(micro_model,box_len=0.001)
    Filter = spatial_box_filter(micro_model,filter_width=0.001)
    
    CPU_start_time = time.process_time()
    coord_range = [[10.000,10.005],x_range_plotting, y_range_plotting]
    num_points = [2,10,20]
    # spacing = 2

    # Create MesoModel and find special observers
    if LoadMesoModelFromPickleFile:
        with open(MesoModelPickleLoadFile, 'rb') as filehandle:
            meso_model = pickle.load(filehandle) 
    else:
        meso_model = NonIdealHydro2D(micro_model, ObsFinder, Filter)
        meso_model.find_observers(num_points, coord_range)
        meso_model.setup_variables()
        meso_model.filter_micro_variables()
        meso_model.calculate_dissipative_coefficients()
        
    # CPU_start_time = time.process_time()
    # meso_model.find_observers(num_points, coord_range, spacing)
    # meso_model.find_observers(num_points, coord_range)

    # print(f'\nElapsed CPU time for observer-finding is {time.process_time() - CPU_start_time}\
    #       with {np.product(num_points)} and {filter.n_filter_points**filter.spatial_dim} points per face\n')

    # Having found observers, setup MesoModel
    meso_model.setup_variables()
    meso_model.filter_micro_variables()
    meso_model.calculate_dissipative_coefficients()

    if DumpMesoModelToPickleFile:
        with open(MesoModelPickleDumpFile, 'wb') as filehandle:
            pickle.dump(meso_model, filehandle) 

    # Plot MesoModel variables
    visualizer.plot_vars(meso_model, ['U'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,)])
            
    # visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
    #                                       t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
    #                   interp_dims=(20,40), method='raw_data', component_indices=(1,2))

    visualizer.plot_vars(meso_model, ['U'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', components_indices=[(1,)])
            
    visualizer.plot_var_model_comparison([micro_model, meso_model], 'SET', \
                                          t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='interpolate', component_indices=(1,2))
        
    # Analyse coefficients of the MesoModel
    analyzer = CoefficientAnalysis(visualizer)
    # Zeta
    visualizer.plot_vars(meso_model, ['Zeta'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[()])
        
    analyzer.DistributionPlot(meso_model, 'Zeta', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', component_indices=())

    analyzer.JointPlot(meso_model, 'Zeta', 'U', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', y_component_indices=(), x_component_indices=(0,))

    # Kappa    
    visualizer.plot_vars(meso_model, ['Kappa'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(2,)])
        
    analyzer.DistributionPlot(meso_model, 'Kappa', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                  interp_dims=(20,40), method='raw_data', component_indices=(2,))

    analyzer.JointPlot(meso_model, 'Kappa', 'T~', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', y_component_indices=(2,), x_component_indices=())

    # Eta
    visualizer.plot_vars(meso_model, ['Eta'], t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', components_indices=[(1,2)])
        
    analyzer.DistributionPlot(meso_model, 'Eta', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', component_indices=(1,2))

    analyzer.JointPlot(meso_model, 'Eta', 'U', t=10.000, x_range=x_range_plotting, y_range=y_range_plotting,\
                      interp_dims=(20,40), method='raw_data', y_component_indices=(1,2), x_component_indices=(0,))
        
    print(f'Total elapsed CPU time for finding is {time.process_time() - CPU_start_time}.')
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    