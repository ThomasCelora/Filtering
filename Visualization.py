# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 03:14:43 2023

@author: marcu
"""


import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn 
from system.BaseFunctionality import *

from MicroModels import * 
from MesoModels import *
from Filters import * 

class Plotter_2D(object):
    
    def __init__(self, screen_size):
        """
        Parameters: 
        -----------
        screen_size = list of 2 floats 
            width and height of the screen: used to rescale plots' size.
        """
        self.subplots_dims = {1 : (1,1),
                         2 : (1,2),
                         3 : (1,3),
                         4 : (2,2),
                         5 : (2,3),
                         6 : (2,3)}
        
        self.screen_size = np.array(screen_size)
    
    def get_var_data(self, model, var_str, t, x_range, y_range, interp_dims=None, method= 'raw_data', component_indices=()):
        """
        Retrieves the required data from model to plot a variable defined by
        var_str over coordinates t, x_range, y_range, either from the model's
        raw data or by interpolating between the model's raw data over the coords.

        Parameters
        ----------
        model : Micro or Meso Model
        var_str : str
            Must match a variable of the model.
        t : float
            time coordinate (defines the foliation).
        x_range : list of 2 floats: x_start and x_end
            defines range of x coordinates within foliation.
        y_range : list of 2 floats: y_start and y_end
            defines range of y coordinates within foliation.
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        method : str
            currently either raw_data or interpolate.
        component_indices : tuple
            the indices of the component to pick out if the variable is a vector/tensor.

        Returns
        -------
        data_to_plot : numpy array of floats
            the (2D) data to be plotted by plt.imshow().
        extent: list of floats 
            

        Notes:
        ------
        Logic: if method is raw_data, then no interp_dims are needed. 
        Better to have 'raw_data' and no interp_dims = None as default 
        """
        if var_str in model.get_all_var_strs():

            # Block to check component_indices passed are compatible with shape of var to be plotted. 
            st1 = len(component_indices)
            st2 = len(model.get_var_gridpoint(var_str, 0, 0, 0).shape)
            compatible = st1==st2 
            if not compatible: 
                if st2 == 0 : 
                    print('WARNING: {} is a scalar but you passed some indices, ignoring this and moving on.'.format(var_str))
                    component_indices = ()
                elif st2 != 0: 
                    print('WARNING: {} is a tensor but you passed more/fewer indices than required. Retrieving the "first component"!'.format(var_str))
                    component_indices = tuple([0 for _ in range(st2)])  
            
            extent = [*x_range, *y_range]

            if method == 'interpolate':
                compatible = interp_dims != None and len(interp_dims) ==2
                if not compatible: 
                    print('Error: when using (linearly spaced) interpolated data, you must \
                          specify # points in each direction! Exiting.')
                    return None
                
                nx, ny = interp_dims[:]
                xs, ys = np.linspace(x_range[0], x_range[1], nx), np.linspace(y_range[0], y_range[1], ny)
                data_to_plot = np.zeros((nx, ny))
                
                for i in range(nx):
                    for j in range(ny):
                        point = [t, xs[i], ys[j]]   
                        data_to_plot[i, j] = model.get_interpol_var(var_str, point)[component_indices]
                           
            elif method == 'raw_data':
                start_indices = Base.find_nearest_cell([t, x_range[0], y_range[0]], model.get_gridpoints())
                end_indices = Base.find_nearest_cell([t, x_range[1], y_range[1]], model.get_gridpoints())

                h = start_indices[0]
                i_s, i_f = start_indices[1], end_indices[1]
                j_s, j_f = start_indices[2], end_indices[2]                     
                        
                data_shape = (i_f+ 1- i_s, j_f+ 1- j_s)
                data_to_plot = np.zeros(data_shape)

                for i in range(i_f+ 1 - i_s): 
                    for j in range(j_f+ 1 - j_s): 
                        data_to_plot[i,j] = model.get_var_gridpoint(var_str, h, i, j)[component_indices]

            else:
                print('Data method is not a valid choice! Must be interpolate or raw_data.')
                return None
            
            # return data_to_plot, points
            return data_to_plot, extent
        
        else:
             print(f'{var_str} is not a plottable variable of the model!') 
             return None

    def plot_vars(self, model, var_strs, t, x_range, y_range, interp_dims=None, method='raw_data', components_indices=[()]):
        """
        Plot variable(s) from model, defined by var_strs, over coordinates 
        t, x_range, y_range. Either from the model's raw data or by interpolating 
        between the model's raw data over the coords.

        Parameters
        ----------
        model : Micro or Meso Model
        var_strs : list of str
            Must match entries in the models' 'vars' dictionary.
        t : float
            time coordinate (defines the foliation).
        x_range : list of 2 floats: x_start and x_end
            defines range of x coordinates within foliation.
        y_range : list of 2 floats: y_start and y_end
            defines range of y coordinates within foliation.
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        method : str
            currently either raw_data or interpolate.
        components_indices : list of tuple(s)
            the indices of the components to pick out if the variables are vectors/tensors.
            Can be omitted if all variables are scalars, otherwise must be a list
            of tuples matching the length of var_strs that corresponds with each
            variable in the list.

        Output
        -------
        Plots the (2D) data using imshow. Note that the plotting data's coordinates
        may not perfectly match the input coordinates if method=raw_data as
        nearest-cell data is used where the input coordinates do not coincide
        with the model's raw data coordinates.

        """
        n_plots = len(var_strs)
        n_rows, n_cols = self.subplots_dims[n_plots]

        # Block to determine adaptively the figsize. 
        figsizes = {1 : (1/3.,1/3.),
                2 : (2/3.,1/3.),
                3 : (1,1/3.),
                4 : (2/3.,2/3.),
                5 : (1,2/3.),
                6 : (1,2/3.)}
        for item in figsizes: 
            figsizes[item] = tuple(figsizes[item] * self.screen_size)
        figsize = figsizes[n_plots]
        
        
        fig, axes = plt.subplots(n_rows,n_cols, figsize=figsize) 
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for var_str, component_indices, ax in zip(var_strs, components_indices, axes):  
            # data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            data_to_plot, extent = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            im = ax.imshow(data_to_plot, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(var_str)
            ax.set_xlabel(r'$y$') # WHY?
            ax.set_ylabel(r'$x$')

        fig.suptitle('Snapshot from model {} at time {}'.format(model.get_model_name(), t), fontsize = 12)
        fig.tight_layout()
        plt.show()
        
    def plot_var_model_comparison(self, models, var_str, t, x_range, y_range, interp_dims=None, method='raw_data', component_indices=(), diff_plot=True):
        """
        Plot a variable from a number of models. If 2 models are given, a third
        plot of the difference will be automatically plotted, too. If 'raw_data'
        is chosen as the method, will check to see if the data points in the model
        lie at the same coordinates, which they must.

        Parameters
        ----------
        models : Micro or Meso Models
        var_sts : str
            Must match entries in the models' 'vars' dictionary.
        t : float
            time coordinate (defines the foliation).
        x_range : list of 2 floats: x_start and x_end
            defines range of x coordinates within foliation.
        y_range : list of 2 floats: y_start and y_end
            defines range of y coordinates within foliation.
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        method : str
            currently either raw_data or interpolate.
        component_indices : tuple
            the indices of the component to pick out if the variable is a vector/tensor.

        Output
        -------
        Plots the (2D) data using imshow. Note that the plotting data's coordinates
        may not perfectly match the input coordinates if method=raw_data as
        nearest-cell data is used where the input coordinates do not coincide
        with the model's raw data coordinates.

        """
        n_cols = len(models)
        if not n_cols == 2:
            diff_plot = False # Only plot difference of 2 models...
        n_rows = 1
        if diff_plot:
            n_cols+=1

        # Block to determine adaptively the figsize. 
        figsize = np.array([1,2/3.]) * self.screen_size

        fig, axes = plt.subplots(n_rows,n_cols,sharex='row',sharey='col',figsize=figsize)
        
        for model, ax in zip(models, axes.flatten()):
            # data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            data_to_plot, extent = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            im = ax.imshow(data_to_plot, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(model.get_model_name())
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$x$')

        if diff_plot:
            ax = axes.flatten()[-1]
            # data_to_plot1, points1 = self.get_var_data(models[0], var_str, t, x_range, y_range, interp_dims, method, component_indices)
            # data_to_plot2, points2 = self.get_var_data(models[1], var_str, t, x_range, y_range, interp_dims, method, component_indices)
            data_to_plot1, extent = self.get_var_data(models[0], var_str, t, x_range, y_range, interp_dims, method, component_indices)
            data_to_plot2 = self.get_var_data(models[1], var_str, t, x_range, y_range, interp_dims, method, component_indices)[0]
            # if len(points1) != len(points2):
            #     diff_plot = False
            #     pass
            # for t_points1, t_points2 in zip(points1, points2):
            #     print(t_points1, t_points2)
            #     if len(t_points1) != t_points2.shape:
            #         diff_plot = False
            #         continue
            #     if not np.allclose(t_points1, t_points2):
            #         diff_plot = False
            # if diff_plot:
            try:                
                # extent = [points1[2][0],points1[2][-1],points1[1][0],points1[1][-1]]
                im = ax.imshow(data_to_plot1 - data_to_plot2, extent=extent)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.set_title('Models difference')
                ax.set_xlabel(r'$y$')
                ax.set_ylabel(r'$x$')
            except(ValueError):
                print(f"Cannot plot the difference between {var_str} in the two "
                      "models. Likely due to the data coordinates not coinciding.")
        fig.suptitle('Contrasting {} against different models'.format(var_str))
        fig.tight_layout()
        plt.show()



if __name__ == '__main__':

    FileReader = METHOD_HDF5('./Data/test_res100/')
    micro_model = IdealMHD_2D()
    FileReader.read_in_data(micro_model)
    micro_model.setup_structures()

    visualizer = Plotter_2D([11.97, 8.36])

    # TESTING GET_VAR_DATA
    ######################  
    # var = 'BC'
    # components = (0,2)
    # data1= visualizer.get_var_data(micro_model, var, 1.502, [0.3, 0.4], [0.3,0.4], component_indices=components)[0]
    # data, extent= visualizer.get_var_data(micro_model, var, 1.502, [0.3, 0.4], [0.3,0.4], component_indices=components, method='interpolate', interp_dims=(20,20))
    # print(extent)

    # TESTING PLOT_VARS
    ###################
    vars = ['BC', 'vx', 'vy', 'Bz', 'p', 'W']
    components = [(0,), (), (), (), (), ()]
    model = micro_model
    # visualizer.plot_vars(model, vars, 1.502, [0.01, 0.98], [0.01, 0.98], components_indices=components)
    visualizer.plot_vars(model, vars, 1.502, [0.01, 0.98], [0.01, 0.98], method = 'interpolate', interp_dims=(100,100), components_indices=components)


    # TESTING PLOT_VAR_MODEL_COMPARISON
    ###################################
    # find_obs = FindObs_drift_root(micro_model, 0.001)
    # filter = spatial_box_filter(micro_model, 0.003)
    # meso_model = resMHD2D(micro_model, find_obs, filter)
    # ranges = [0.3, 0.4]
    # meso_model.setup_meso_grid([[1.501, 1.503],ranges, ranges], coarse_factor=2)
    # meso_model.find_observers()
    # meso_model.filter_micro_variables()

    # var = 'BC'
    # component = (0,)
    # models = [micro_model, meso_model]
    # smaller_ranges = [ranges[0]+0.01, ranges[1]- 0.01] # Needed to avoid interpolation errors at boundaries
    # # visualizer.plot_var_model_comparison(models, var, 1.502, smaller_ranges, smaller_ranges, \
    # #                                      method='interpolate', interp_dims=(30,30), component_indices=component)
    # visualizer.plot_var_model_comparison(models, var, 1.502, smaller_ranges, smaller_ranges, component_indices=component)