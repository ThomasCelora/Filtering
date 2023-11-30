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
    
    def __init__(self, screen_size = [11.97, 8.36]):
        """
        Parameters: 
        -----------
        screen_size = list of 2 floats 
            width and height of the screen: used to rescale plots' size.
        """
        self.plot_vars_subplots_dims = {1 : (1,1),
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
            
            # extent = [*x_range, *y_range]

            if method == 'interpolate':
                compatible = interp_dims != None and len(interp_dims) ==2
                if not compatible: 
                    print('Error: when using (linearly spaced) interpolated data, you must' +\
                          'specify # points in each direction! Exiting.')
                    return None
                
                nx, ny = interp_dims[:]
                xs, ys = np.linspace(x_range[0], x_range[1], nx), np.linspace(y_range[0], y_range[1], ny)
                data_to_plot = np.zeros((nx, ny))
                
                points = [t, xs, ys]
                extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]

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
                        
                gridpoints = model.get_gridpoints()
                points = [gridpoints[0][h], gridpoints[1][i_s:i_f+1], gridpoints[2][j_s:j_f+1]]
                extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]

                data_shape = (i_f - i_s + 1, j_f - j_s + 1) 
                data_to_plot = np.zeros(data_shape)

                for i in range(i_f - i_s + 1): 
                    for j in range(j_f - j_s + 1): 
                        data_to_plot[i,j] = model.get_var_gridpoint(var_str, h, i + i_s, j + j_s)[component_indices]

            else:
                print('Data method is not a valid choice! Must be interpolate or raw_data.')
                return None
            # return data_to_plot, points
            return data_to_plot, extent
        
        else:
             print(f'{var_str} is not a plottable variable of the model!') 
             return None

    def plot_vars(self, model, var_strs, t, x_range, y_range, interp_dims=None, method='raw_data', components_indices=None): #[()]):
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
            currently either raw_data or  .
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
        n_rows, n_cols = self.plot_vars_subplots_dims[n_plots]

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

        if not components_indices: 
            print('No list of components indices passed: setting this to an empty list.')
            components_indices = [ () for _ in range(len(var_strs))]
            
        for var_str, component_indices, ax in zip(var_strs, components_indices, axes):  
            # data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            data_to_plot, extent = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            im = ax.imshow(data_to_plot, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            title = var_str
            if component_indices != ():
                title = title + ", {}-component".format(component_indices)
            ax.set_title(title)
            ax.set_xlabel(r'$y$') # WHY?
            ax.set_ylabel(r'$x$')

        fig.suptitle('Snapshot from model {} at time {}'.format(model.get_model_name(), t), fontsize = 12)
        fig.tight_layout()
        # plt.show()
        return fig
        
    def plot_vars_models_comparison(self, models, var_strs, t, x_range, y_range, interp_dims=None, method='raw_data', \
                                    components_indices=None, diff_plot=False, rel_diff=False):
        """
        Plot variables from a number of models. If 2 models are given, a third
        plot of the difference can be added too. 
        If 'raw_data' is chosen as the method, will check to see if the data points in the model
        lie at the same coordinates, which they must. 

        Parameters
        ----------
        models : list of Micro or Meso Models
        var_strs : list of lists of strings
            each sublist must match entries in the models' 'vars' dictionary.
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
        component_indices : list of list of tuples
            each tuple identifies the indices of the component to pick out if the variable 
            is a vector/tensor.
        diff_plot: bool 
            Whether to add a column to show difference between models
        rel_diff: bool
            Whether to plot the absolute or relative difference between models

        Output
        -------
        Plots the (2D) data using imshow. Note that the plotting data's coordinates
        may not perfectly match the input coordinates if method=raw_data as
        nearest-cell data is used where the input coordinates do not coincide
        with the model's raw data coordinates.

        """
        if len(var_strs) != len(models):
            print("I need a list of vars to plot per model. Check!")
            return None
        num_vars_1st_model = len(var_strs[0])
        for i in range(1, len(var_strs)):  
            if len(var_strs[i]) != num_vars_1st_model:
                print("The number of variables per model must be the same. Exiting.")
                return None
        
        n_cols = len(models)
        if diff_plot:
            if len(models)!=2:
                print("Can plot the difference between TWO models, not more.")
            else:
                n_cols+=1
        if rel_diff:
            if len(models)!=2:
                print("Can plot the difference between TWO models, not more.")
            else:
                n_cols+=1

        n_rows = len(var_strs[0])
        if len(var_strs[0])>3:
            print("This function is meant to compare up to 3 vars. Plotting the first three.")
            n_rows = 3
        
        if not components_indices: 
            print('No list of components indices passed: setting this to an empty list.')
            empty_tuple_components = [ () for _ in range(len(var_strs[0]))]
            components_indices = []
            for i in range(len(models)):
                components_indices.append(empty_tuple_components)

        # Block to determine adaptively the figsize. 
        # figsize = np.array([1,2/3.]) * self.screen_size
        figsize = self.screen_size
        # gridspec_dict = {'wspace' : 0.3, 'hspace': 0.3}
        # fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, gridspec_kw=gridspec_dict, figsize=figsize)
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=figsize)
      
        for j in range(len(models)):
            for i in range(n_rows):                    
                data_to_plot, extent = self.get_var_data(models[j], var_strs[j][i], t, x_range, y_range, interp_dims, method, components_indices[j][i])
                im=axes[i,j].imshow(data_to_plot, extent=extent)
                divider = make_axes_locatable(axes[i,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                title = models[j].get_model_name() + "\n"+var_strs[j][i]
                if components_indices[j][i] != ():
                    title += " {}-component".format(components_indices[j][i])
                axes[i,j].set_title(title)
                axes[i,j].set_xlabel(r'$y$')
                axes[i,j].set_ylabel(r'$x$')


        if diff_plot and len(models)==2:
            try:
                for i in range(len(var_strs[0])):
                    data1, extent1 = self.get_var_data(models[0], var_strs[0][i], t, x_range, y_range, interp_dims, method, components_indices[0][i])
                    data2, extent2 = self.get_var_data(models[1], var_strs[1][i], t, x_range, y_range, interp_dims, method, components_indices[1][i])
                    if extent1 != extent2:
                        print("Cannot plot the difference between the vars: data not aligned.")
                        continue
                    data_to_plot = data1 - data2
                    im = axes[i,2].imshow(data_to_plot, extent=extent1)
                    divider = make_axes_locatable(axes[i,2])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    axes[i,2].set_title('Models difference')
                    axes[i,2].set_xlabel(r'$y$')
                    axes[i,2].set_ylabel(r'$x$')
            except (ValueError):
                print(f"Cannot plot the difference between {var_strs} in the two "+\
                      "models. Likely due to the data coordinates not coinciding.")


        if rel_diff and len(models)==2:
            try:
                for i in range(len(var_strs[0])):
                    data1, extent1 = self.get_var_data(models[0], var_strs[0][i], t, x_range, y_range, interp_dims, method, components_indices[0][i])
                    data2, extent2 = self.get_var_data(models[1], var_strs[1][i], t, x_range, y_range, interp_dims, method, components_indices[1][i])
                    if extent1 != extent2:
                        print("Cannot plot the difference between the vars: data not aligned.")
                        continue
                    ar_mean = (np.abs(data1) + np.abs(data2))/2
                    data_to_plot = np.abs(data1 -data2)/ar_mean 
                    if diff_plot:
                        column = 3
                    else:
                        column = 2
                    im = axes[i,column].imshow(data_to_plot, extent=extent1)
                    divider = make_axes_locatable(axes[i,column])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    axes[i,column].set_title('Relative difference')
                    axes[i,column].set_xlabel(r'$y$')
                    axes[i,column].set_ylabel(r'$x$')
            except (ValueError):
                print(f"Cannot plot the difference between {var_strs} in the two "+\
                      "models. Likely due to the data coordinates not coinciding.")
                

        models_names = [model.get_model_name() for model in models]
        suptitle = "Comparing "
        for i in range(len(models_names)):
            suptitle += models_names[i] + ", "
        suptitle += "models."
        fig.suptitle(suptitle)
        fig.tight_layout()
        # plt.subplot_tool()
        return fig



if __name__ == '__main__':
    
    FileReader = METHOD_HDF5('../Data/test_res100/')
    micro_model = IdealMHD_2D()
    FileReader.read_in_data(micro_model)
    micro_model.setup_structures()


    visualizer = Plotter_2D([11.97, 8.36])
    print('Finished initializing')

    # TESTING GET_VAR_DATA
    ######################  
    # var = 'BC'
    # components = (0,2)
    # data1= visualizer.get_var_data(micro_model, var, 1.502, [0.3, 0.4], [0.3,0.4], component_indices=components)[0]
    # data, extent= visualizer.get_var_data(micro_model, var, 1.502, [0.3, 0.4], [0.3,0.4], component_indices=components, method='interpolate', interp_dims=(20,20))
    # print(extent)

    # TESTING PLOT_VARS
    ###################
    # vars = ['BC', 'vx', 'vy', 'Bz', 'p', 'W']
    # components = [(0,), (), (), (), (), ()]
    # model = micro_model
    # visualizer.plot_vars(model, vars, 1.502, [0.01, 0.98], [0.01, 0.98], components_indices=components)
    # visualizer.plot_vars(model, vars, 1.502, [0.01, 0.98], [0.01, 0.98], method = 'interpolate', interp_dims=(100,100), components_indices=components)


    # TESTING PLOT_VAR_MODEL_COMPARISON
    ###################################
    find_obs = FindObs_drift_root(micro_model, 0.001)
    filter = spatial_box_filter(micro_model, 0.003)
    meso_model = resMHD2D(micro_model, find_obs, filter)
    ranges = [0.2, 0.25]
    meso_model.setup_meso_grid([[1.501, 1.503],ranges, ranges], coarse_factor=1)
    meso_model.find_observers()
    meso_model.filter_micro_variables()

    print("Finished filtering")

    vars = [['BC', 'BC', 'BC', 'BC'],['BC', 'BC' ,'BC' ,'BC']] 
    components = [[(0,), (0,), (0,), (0,)],[(0,), (0,), (0,), (0,)]]
    models = [micro_model, meso_model]
    # smaller_ranges = [ranges[0]+0.01, ranges[1]- 0.01] # Needed to avoid interpolation errors at boundaries
    # visualizer.plot_var_model_comparison(models, var, 1.502, smaller_ranges, smaller_ranges, \
    #                                      method='interpolate', interp_dims=(30,30), component_indices=component)
    visualizer.plot_vars_models_comparison(models, vars, 1.502, ranges, ranges, components_indices=components, diff_plot=True, rel_diff = False)
    plt.show()
