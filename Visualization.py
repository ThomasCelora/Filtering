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

class Plotter_2D(object):
    
    def __init__(self):
        """
        Blank as yet.
        """
        self.subplots_dims = {1 : (1,1),
                         2 : (1,2),
                         3 : (1,3),
                         4 : (2,2),
                         5 : (2,3),
                         6 : (2,3)}

        
    def get_var_data(self, model, var_str, t, x_range, y_range, interp_dims, method, component_indices):
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
        points : numpy array of floats
            the coordinates of the data-points.

        """
        if var_str in model.get_all_var_strs():
        
            if method == 'interpolate':
    
                nx, ny = interp_dims[:]
                xs, ys = np.linspace(x_range[0],x_range[1],nx), np.linspace(y_range[0],y_range[1],ny)
                points = [t,xs,ys]
                data_to_plot = np.zeros((nx,ny))
                
                for i in range(nx):
                    for j in range(ny):
                        point = [t,xs[i],ys[j]]
                        # print(component_indices)
                        # print(interpn(model.domain_vars['points'],\
                        #                     model.vars[var_str], point,\
                        #                     method = model.interp_method)[0])#[0][component_indices])
                        
                        data_to_plot[i,j] = interpn(model.domain_vars['points'],\
                                            model.vars[var_str], point,\
                                            method = model.interp_method)[0][component_indices]         
            elif method == 'raw_data':
    
                start_indices = Base.find_nearest_cell([t, x_range[0], y_range[0]], model.domain_vars['points'])
                end_indices = Base.find_nearest_cell([t, x_range[1], y_range[1]], model.domain_vars['points'])
                
                h = start_indices[0]
                i_s, i_f = start_indices[1], end_indices[1]
                j_s, j_f = start_indices[2], end_indices[2]
                
                points = [model.domain_vars['points'][0][h],\
                          model.domain_vars['points'][1][i_s:i_f+1],\
                          model.domain_vars['points'][2][j_s:j_f+1]]
                
                data_to_plot = model.vars[var_str][h:h+1, i_s:i_f+1, j_s:j_f+1][0]#[:, :, component_indices]
                # print(data_to_plot)
                if component_indices:
                    for component_index in component_indices:
                        data_to_plot = data_to_plot[:, :, component_index]


                
            else:
                print('Data method is not a valid choice! Must be interpolate or raw_data.')
        
        else:
             print(f'{var_str} is not a plottable variable of the model!') 

        return data_to_plot, points
    
    
    def plot_vars(self, model, var_strs, t, x_range, y_range, interp_dims=(), \
                  method='interpolate', components_indices=None):
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
            Can be omitted is all variables are scalars, otherwise must be a list
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
        fig, axes = plt.subplots(n_rows,n_cols, figsize=(16,16)) # figsize should be adaptive..
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for var_str, component_indices, ax in zip(var_strs, components_indices, axes):  
            data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            im = ax.imshow(data_to_plot, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(model.get_model_name())
            # fig.suptitle(model.get_model_name(), fontsize=16)
            ax.set_title(var_str)
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$x$')

        fig.tight_layout()
        plt.show()
        
 
    def plot_var_model_comparison(self, models, var_str, t, x_range, y_range, \
                            interp_dims=(), method='interpolate', component_indices=(), diff_plot=True):
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
        fig, axes = plt.subplots(n_rows,n_cols,sharex='row',sharey='col',figsize=(16,16))
        
        for model, ax in zip(models, axes.flatten()):
            data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            im = ax.imshow(data_to_plot, extent=extent)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_title(model.get_model_name())
            ax.set_xlabel(r'$y$')
            ax.set_ylabel(r'$x$')

        if diff_plot:
            ax = axes.flatten()[-1]
            data_to_plot1, points1 = self.get_var_data(models[0], var_str, t, x_range, y_range, interp_dims, method, component_indices)
            data_to_plot2, points2 = self.get_var_data(models[1], var_str, t, x_range, y_range, interp_dims, method, component_indices)
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
                extent = [points1[2][0],points1[2][-1],points1[1][0],points1[1][-1]]
                im = ax.imshow(data_to_plot1 - data_to_plot2, extent=extent)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                ax.set_title(model.get_model_name())
                ax.set_title('Model Difference')
                ax.set_xlabel(r'$y$')
                ax.set_ylabel(r'$x$')
            except(ValueError):
                print(f"Cannot plot the difference between {var_str} in the two "
                      "models. Likely due to the data coordinates not coinciding.")
        fig.tight_layout()
        plt.show()










