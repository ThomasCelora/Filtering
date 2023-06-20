# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
import seaborn as sns

class CoefficientAnalysis(object):
    
    def __init__(self, visualizer):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        self.visualizer = visualizer # need to re-structure this... or do I
        
    def JointPlot(self, model, var_str, t, x_range, y_range, interp_dims, method, component_indices):

        data_to_plot, points = \
            self.visualizer.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
        fig = plt.figure(figsize=(16,16))
        sns.joint_plot(data_to_plot)
        plt.title(var_str)
        fig.tight_layout()
        plt.show()
        
    def DistributionPlot(self, model, var_str, t, x_range, y_range, interp_dims, method, component_indices):

        data_to_plot, points = \
            self.visualizer.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)

        fig = plt.figure(figsize=(16,16))
        sns.displot(data_to_plot)
        plt.title(var_str)
        fig.tight_layout()
        plt.show()



















