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
import statsmodels.api as sm
    


class CoefficientsAnalysis(object): # 2D function? 
    """
    """

    def __init__(self, visualizer, spatial_dims):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        self.visualizer = visualizer # need to re-structure this... or do I
        self.spatial_dims = spatial_dims

    def JointPlot(self, model, y_var_str, x_var_str, t, x_range, y_range,\
                  interp_dims, method, y_component_indices, x_component_indices):
        y_data_to_plot, points = \
            self.visualizer.get_var_data(model, y_var_str, t, x_range, y_range, interp_dims, method, y_component_indices)
        x_data_to_plot, points = \
            self.visualizer.get_var_data(model, x_var_str, t, x_range, y_range, interp_dims, method, x_component_indices)
        fig = plt.figure(figsize=(16,16))
        sns.jointplot(x=x_data_to_plot.flatten(), y=y_data_to_plot.flatten(), kind="hex", color="#4CB391")
        plt.title(y_var_str+'('+x_var_str+')')
        fig.tight_layout()
        plt.show()
        
    def DistributionPlot(self, model, var_str, t, x_range, y_range, interp_dims, method, component_indices):

        data_to_plot, points = \
            self.visualizer.get_var_data(model, var_str, t, x_range, y_range, interp_dims, method, component_indices)
            
        # print(data_to_plot)

        fig = plt.figure(figsize=(16,16))
        sns.displot(data_to_plot)
        plt.title(var_str)
        fig.tight_layout()
        plt.show()

    def scalar_regression(self, y, X):
        """
        Routine to perform ordinary/weighted/multivariate regression on some gridded data. 

        Parameters: 
        -----------
        y: ndarray of shape (n,m)
            the "measurements" of the dependent quantity
        
        X: list of ndarrays of shape (n,m)
            the data for the regressors 

        weights: array of shape (n,m) 
            the weights of the data (variance of the measurement)
        """
        dep_shape = np.shape(y)
        n_reg = len(X)
    
        for i in range(n_reg):
            if np.shape(X[i]) != dep_shape:
                print('Dependent data is not aligned with regressor, removing regressor data.')
                X.remove(X[i])
            else: 
                X[i] = X[i].flatten()

        y = y.flatten()
        model = sm.OLS(y, X)
        result = model.fit()
        return result.params
    
    def scalar_weighted_regression():
        """
        Based on sm.WLS 
        Weights are computed externally by the MesoModel class
        """
        pass

    def tensor_components_regression(self, y, X, weights = None, components = None):
        """
        JUST JOTTING DOWN IDEAS FOR NOW
        SHOULD BE A WRAPPER OF scalar_weighted_regression()
        """
        if not components:
            components = []
            skipping_idx = tuple([0 for _ in range(self.spatial_dims)])
            for i in np.ndindex(y[skipping_idx].shape):
                components.append(i)

        # component_regress = np.zeros((len(components), len(X)))
        components_params = []
        for component in components:
            dep = y[component]
            reg = X[component]
            if weights:
                ws = weights[component] 
            # component_regress.append(self.simple_regression(dep, reg, ws))
            components_params.append( self.simple_regression(dep, reg, ws))
        components_params.reshape(len(X), len(y))

        avg_params = np.mean(components_params)
        return avg_params

    def tensor_components_weighted_regression():
        """
        Should be a wrapper of scalar_weighted_regression()
        Weights are computed externally by the MesoModel class
        """        
        pass

    def PCA_check_regressor_correlation():
        """
        Idea: pass a large list of quantities that are correlated with a residual/closure coefficient.
        Check if there is a smaller subset: pass percentage of variance as parameter, like 70%, 
        and the # of principal components to be retained will be computed from the eigenvalues of the 
        correlation matrix, that is the loadings. 
        Then return the PCA weights: this will be later used as regressors for the residuals. 
        """
        pass

    def PCA_find_regressors():
        """
        Idea pass both the residuals and a large list of quantities, identify which among these are
        correlated with the residual. Do this by checking the scores of the residual (the first var 
        of the dataset, say) on the various principal components. 
        Then return the weights. This will then be used in regression (possibyly after checking they're
        not correlated).
        """
        pass








