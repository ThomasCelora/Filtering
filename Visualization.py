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
        Parameters: 
        -----------
        micro_model: instance of class containing the microdata
        micro_model: instance of class containing the filtered data

        """
   
    def get_var_data(self, model, var_str, t, x_range, y_range, interp_dims, method, component_indices):

        if var_str in model.get_all_var_strs():
        
            if method == 'interpolate':
    
                nx, ny = interp_dims[:]
                xs, ys = np.linspace(x_range[0],x_range[1],nx), np.linspace(y_range[0],y_range[1],ny)
                points = [t,xs,ys]
                data_to_plot = np.zeros((nx,ny))
                
                for i in range(nx):
                    for j in range(ny):
                        point = [t,xs[i],ys[j]]
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
                
                data_to_plot = model.vars[var_str][h:h+1, i_s:i_f+1, j_s:j_f+1][0][component_indices]

                
            else:
                print('Data method is not a valid choice! Must be interpolate or raw_data.')
        
        else:
             print(f'{var_str} is not a plottable variable of the model!') 

        return data_to_plot, points
    
    
    def plot_vars(self, model, var_strs, t, x_range, y_range, interp_dims=(), method='interpolate', component_indices=[()]):

        fig, axes = plt.subplots(1,len(var_strs))

        for var_str, component_index, ax in zip(var_strs, component_indices, axes.flatten()):  
            data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_index)
            extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            ax.imshow(data_to_plot, extent=extent)
            ax.set_title(var_str)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')

        plt.show()       
        
 
    def plot_var_model_comparison(self, models, var_str, t, x_range, y_range, \
                            interp_dims=(), method='interpolate', component_index=()):

        fig, axes = plt.subplots(1,len(models))
        
        for model, ax in zip(models, axes.flatten()):
            data_to_plot, points = self.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_index)
            extent = [points[2][0],points[2][-1],points[1][0],points[1][-1]]
            ax.imshow(data_to_plot, extent=extent)
            ax.set_title(model.get_model_name())
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')

        plt.show()















