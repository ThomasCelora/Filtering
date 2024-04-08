# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 03:14:43 2023

@author: marcu
"""


import matplotlib.pyplot as plt
from matplotlib import colors 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator
import numpy as np
import h5py
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

        # Change to latex font
        plt.rc("font",family="serif")
        plt.rc("mathtext",fontset="cm")
    
    def get_var_data(self, model, var_str, t, x_range, y_range, component_indices=(), method= 'raw_data', interp_dims=None):
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
        component_indices : tuple
            the indices of the component to pick out if the variable is a vector/tensor.
        method : str
            currently either raw_data or interpolate.
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        
        Returns
        -------
        data_to_plot : numpy array of floats
            the 2D data to be plotted by plt.imshow()
        extent: list of floats 

            

        Notes:
        ------
        Logic: if method is raw_data, then no interp_dims are needed. 
        Better to have 'raw_data' and interp_dims = None as default 

        data_to_plot is transposed and extent is built to be used with: 
        origin=lower, extent=L,R,B,T by imshow

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
                          'specify # points in each spatial direction! Exiting.')
                    return None
                
                nx, ny = interp_dims[:]
                xs, ys = np.linspace(x_range[0], x_range[1], nx), np.linspace(y_range[0], y_range[1], ny)
                data_to_plot = np.zeros((nx, ny))
                
                points = [t, xs, ys]
                # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
                extent = [points[1][0],points[1][-1],points[2][0],points[2][-1]]

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
                # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
                extent = [points[1][0],points[1][-1],points[2][0],points[2][-1]]

                data_shape = (i_f - i_s + 1, j_f - j_s + 1) 
                data_to_plot = np.zeros(data_shape)

                for i in range(i_f - i_s + 1): 
                    for j in range(j_f - j_s + 1): 
                        data_to_plot[i,j] = model.get_var_gridpoint(var_str, h, i + i_s, j + j_s)[component_indices]

            else:
                print('Data method is not a valid choice! Must be interpolate or raw_data.')
                return None
            # return data_to_plot, points
            data_to_plot = np.transpose(data_to_plot)
            return data_to_plot, extent
        
        else:
             print(f'{var_str} is not a plottable variable of the model!') 
             return None

    def plot_vars(self, model, var_strs, t, x_range, y_range, components_indices=None, method='raw_data', interp_dims=None, 
                  norms=None, cmaps=None):
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
        components_indices : list of tuple(s)
            the indices of the components to pick out if the variables are vectors/tensors.
            Can be omitted if all variables are scalars, otherwise must be a list
            of tuples matching the length of var_strs that corresponds with each
            variable in the list.
        method : str
            currently either raw_data or interpolate
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        norms = list of strs
            each entry of the list is passed as option to imshow as norm=str
            when plotting the corresponding var
            
            valid choices include all the standard norms like log or symlog, 
            and 'mysymlog' which is implemented in BaseFunctionality

        cmaps = list of strs
            each entry of the list is passed to imshow as cmap=cmaps[i]
            when plotting the corresponding var

            valid choices are all the std ones
        
        Output
        -------
        Plots the (2D) data using imshow. Note that the plotting data's coordinates
        may not perfectly match the input coordinates if method=raw_data as
        nearest-cell data is used where the input coordinates do not coincide
        with the model's raw data coordinates.

        """
        n_plots = len(var_strs)

        if norms == None:
            norms = [None for _ in range(len(var_strs))]
        elif len(var_strs)!= len(norms):
            print('The norms provided are not the same number as the variables: setting these to auto')
            norms = [None for _ in range(len(var_strs))]

        if cmaps == None:
            cmaps = [None for _ in range(len(var_strs))]
        elif len(var_strs)!= len(cmaps):
            print('The norms provided are not the same number as the variables: setting these to auto')
            cmaps = [None for _ in range(len(var_strs))]

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
            
        for i, (var_str, component_indices, ax) in enumerate(zip(var_strs, components_indices, axes)):  
            # data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            # extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            data_to_plot, extent = self.get_var_data(model, var_str, t, x_range, y_range, component_indices, method, interp_dims)

            if norms[i] == 'mysymlog': 
                ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(data_to_plot)
                data_to_plot = MySymLogPlotting.symlog_var(data_to_plot)
                mynorm = MyThreeNodesNorm(nodes)
                im = ax.imshow(data_to_plot, extent=extent, origin='lower', norm=mynorm, cmap=cmaps[i])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                cbar.set_ticks(ticks)
                cbar.ax.set_yticklabels(labels)

            elif norms[i] != 'mysymlog':
                im = ax.imshow(data_to_plot, extent=extent, origin='lower', norm=norms[i], cmap=cmaps[i])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
            title = var_str
            if hasattr(model, 'labels_var_dict'):
                if title in model.labels_var_dict.keys():
                    title = model.labels_var_dict[title]
            if component_indices != ():
                title = title + ", {}-component".format(component_indices)
            ax.set_title(title)
            ax.set_xlabel(r'$x$') 
            ax.set_ylabel(r'$y$')

        time_for_filename = str(round(t,2))
        fig.suptitle('Snapshot from model {} at time {}'.format(model.get_model_name(), time_for_filename), fontsize = 12)
        fig.tight_layout()
        # plt.show()
        return fig
        
    def plot_vars_models_comparison(self, models, var_strs, t, x_range, y_range, components_indices=None, method='raw_data',\
                                    interp_dims=None, diff_plot=False, rel_diff=False, norms=None, cmaps=None):
        """
        Plot variables from a number of models. If 2 models are given, a third
        plot of the difference (relative or absolute) can be added too. 
        The method refers to the difference plot(s): for models with different grid spacings, data must 
        be extracted via interpolation, so setting method = 'interpolate'. 
        In any other case, should set method = 'raw_data'.

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
        component_indices : list of list of tuples
            each tuple identifies the indices of the component to pick out if the variable 
            is a vector/tensor.
        method : str
            currently either raw_data or interpolate.
        interp_dims : tuple of integers
            defines the number of points to interpolate at in x and y directions.
        diff_plot: bool 
            Whether to add a column to show difference between models
        rel_diff: bool
            Whether to plot the absolute or relative difference between models
        norms/maps = list of list of strs
            these have to be compatible with the final number of rows and columns 
            in the plot. First index in the list runs over the columns (models and their difference),
            second index runs over the rows (vars).

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
                
        inv_fig_shape = (n_cols, n_rows)
        if not norms or np.array(norms).shape != inv_fig_shape:
            print('Norms provided are not compatible with figure, setting these to auto')
            norms = [None for _ in range(n_rows)]
            norms = [norms for _ in range(n_cols)]
    
        if not cmaps or np.array(cmaps).shape != inv_fig_shape:
            print('Colormaps not compatible with figure, setting these to auto')
            cmaps = [None for _ in range(n_rows)]
            cmaps = [cmaps for _ in range(n_cols)]


        figsize = self.screen_size
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=figsize)
      
        for j in range(len(models)):
            for i in range(n_rows):                

                data_to_plot, extent = self.get_var_data(models[j], var_strs[j][i], t, x_range, y_range, components_indices[j][i], 'raw_data', None)

                if norms[j][i] == 'mysymlog': 
                    ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(data_to_plot)
                    data_to_plot = MySymLogPlotting.symlog_var(data_to_plot)
                    mynorm = MyThreeNodesNorm(nodes)
                    im = axes[i,j].imshow(data_to_plot, extent=extent, origin='lower', norm=mynorm, cmap=cmaps[j][i])
                    divider = make_axes_locatable(axes[i,j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.set_ticks(ticks)
                    cbar.ax.set_yticklabels(labels)

                elif norms[j][i] != 'mysymlog':
                    im = axes[i,j].imshow(data_to_plot, extent=extent, origin='lower', norm=norms[j][i], cmap=cmaps[j][i])
                    divider = make_axes_locatable(axes[i,j])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')

                # im=axes[i,j].imshow(data_to_plot, extent=extent, norm=norms[j][i], cmap=cmaps[j][i])
                # divider = make_axes_locatable(axes[i,j])
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # fig.colorbar(im, cax=cax, orientation='vertical')
                # title = models[j].get_model_name() + "\n"+var_strs[j][i]

                # title = models[j].get_model_name() + '\n'
                title = ''
                if hasattr(models[j], 'labels_var_dict'):
                    if var_strs[j][i] in models[j].labels_var_dict.keys():
                        title += models[j].labels_var_dict[var_strs[j][i]]
                    else: 
                        title += var_strs[j][i]
                else:
                    title += var_strs[j][i]

                if components_indices[j][i] != ():
                    title += " {}-component".format(components_indices[j][i])
                axes[i,j].set_title(title)
                axes[i,j].set_xlabel(r'$x$')
                axes[i,j].set_ylabel(r'$y$')



        if diff_plot and len(models)==2:
            try:
                for i in range(len(var_strs[0])):
                    data1, extent1 = self.get_var_data(models[0], var_strs[0][i], t, x_range, y_range, components_indices[0][i], method, interp_dims)
                    data2, extent2 = self.get_var_data(models[1], var_strs[1][i], t, x_range, y_range, components_indices[1][i], method, interp_dims)
                    if extent1 != extent2:
                        print("Cannot plot the difference between the vars: data not aligned.")
                        continue
                    data_to_plot = data1 - data2

                    if norms[2][i] == 'mysymlog': 
                        ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(data_to_plot)
                        data_to_plot = MySymLogPlotting.symlog_var(data_to_plot)
                        mynorm = MyThreeNodesNorm(nodes)
                        im = axes[i,2].imshow(data_to_plot, extent=extent, origin='lower', norm=mynorm, cmap=cmaps[2][i])
                        divider = make_axes_locatable(axes[i,2])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                        cbar.set_ticks(ticks)
                        cbar.ax.set_yticklabels(labels)

                    elif norms[2][i] != 'mysymlog':
                        im = axes[i,2].imshow(data_to_plot, extent=extent, origin='lower', norm=norms[2][i], cmap=cmaps[2][i])
                        divider = make_axes_locatable(axes[i,2])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im, cax=cax, orientation='vertical')


                    # im = axes[i,2].imshow(data_to_plot, extent=extent1, norm=norms[2][i], cmap=cmaps[2][i])
                    # divider = make_axes_locatable(axes[i,2])
                    # cax = divider.append_axes('right', size='5%', pad=0.05)
                    # fig.colorbar(im, cax=cax, orientation='vertical')


                    axes[i,2].set_title('Models difference')
                    axes[i,2].set_xlabel(r'$y$')
                    axes[i,2].set_ylabel(r'$x$')
            except ValueError as v:
                print(f"Cannot plot the difference between {var_strs} in the two "+\
                      f"models. Caught a value error: {v}")


        if rel_diff and len(models)==2:
            try:
                for i in range(len(var_strs[0])):
                    data1, extent1 = self.get_var_data(models[0], var_strs[0][i], t, x_range, y_range, components_indices[0][i], method, interp_dims)
                    data2, extent2 = self.get_var_data(models[1], var_strs[1][i], t, x_range, y_range, components_indices[1][i], method, interp_dims)
                    if extent1 != extent2:
                        print("Cannot plot the difference between the vars: data not aligned.")
                        continue
                    ar_mean = (np.abs(data1) + np.abs(data2))/2
                    data_to_plot = np.abs(data1 -data2)/ar_mean 
                    if diff_plot:
                        column = 3
                    else:
                        column = 2


                    if norms[column][i] == 'mysymlog': 
                        ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(data_to_plot)
                        data_to_plot = MySymLogPlotting.symlog_var(data_to_plot)
                        mynorm = MyThreeNodesNorm(nodes)
                        im = axes[i,column].imshow(data_to_plot, extent=extent, origin='lower', norm=mynorm, cmap=cmaps[column][i])
                        divider = make_axes_locatable(axes[i,column])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                        cbar.set_ticks(ticks)
                        cbar.ax.set_yticklabels(labels)

                    elif norms[column][i] != 'mysymlog':
                        im = axes[i,column].imshow(data_to_plot, extent=extent, origin='lower', norm=norms[column][i], cmap=cmaps[column][i])
                        divider = make_axes_locatable(axes[i,column])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im, cax=cax, orientation='vertical')

                    # im = axes[i,column].imshow(data_to_plot, extent=extent1, norm=norms[column][i], cmap=cmaps[column][i])
                    # divider = make_axes_locatable(axes[i,column])
                    # cax = divider.append_axes('right', size='5%', pad=0.05)
                    # fig.colorbar(im, cax=cax, orientation='vertical')

                    axes[i,column].set_title('Relative difference')
                    axes[i,column].set_xlabel(r'$y$')
                    axes[i,column].set_ylabel(r'$x$')
            except ValueError as v:
                print(f"Cannot plot the difference between {var_strs} in the two "+\
                      f"models. Caught a value error: {v}")
                

        models_names = [model.get_model_name() for model in models]
        suptitle = "Comparing "
        for i in range(len(models_names)):
            suptitle += models_names[i] + ", "
        suptitle += "models."
        # fig.suptitle(suptitle)
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

    vars = [['BC'], ['BC']]
    models = [micro_model, meso_model]
    components = [[(0,)],[(0,)]]
    norms = [['log'], ['log'], ['symlog']]
    cmaps=None
    # cmaps = [['seismic'], ['seismic'], ['seismic']]
    # smaller_ranges = [ranges[0]+0.01, ranges[1]- 0.01] # Needed to avoid interpolation errors at boundaries
    # visualizer.plot_var_model_comparison(models, var, 1.502, smaller_ranges, smaller_ranges, \
    #                                      method='interpolate', interp_dims=(30,30), component_indices=component)
    visualizer.plot_vars_models_comparison(models, vars, 1.502, ranges, ranges, components_indices=components, diff_plot=True, rel_diff = False, 
                                           norms=norms, cmaps=cmaps)
    plt.show()
