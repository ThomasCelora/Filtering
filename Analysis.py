# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import pickle
import seaborn as sns
import warnings
import statsmodels.api as sm

from system.BaseFunctionality import *
from MicroModels import * 
from FileReaders import * 
from Filters import *
from Visualization import *
from MesoModels import * 

    


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
        self.visualizer=visualizer # need to re-structure this... or do I
        self.spatial_dims=spatial_dims

    def trim_data(self, data, ranges, model_points):
        """
        Takes input gridded data 'data' (grid points given by input 'model_points')
        and trim this to lie within ranges. 

        Parameters: 
        -----------
        data: ndarray 
            gridded data, shape must be compatible with model_points 

        ranges: list of lists of 2 floats
            the min and max in each direction

        model_points: list of list of floats
        
        Returns:
        --------
        data trimmed to within ranges
        """
        if not (len(ranges)==len(model_points) and len(ranges) == len(data.shape)):
            print('Check: i) data incompatible with ranges or ii) ranges incompatible with model_points. Skipping!')
            return data
        else:
            mins = [i[0] for i in ranges]
            maxs = [i[1] for i in ranges]
            start_indices = Base.find_nearest_cell(mins, model_points)
            end_indices = Base.find_nearest_cell(maxs, model_points)

            IdxsToRemove = []
            num_points = [len(model_points[i]) for i in range(len(model_points))]
            for i in range(len(ranges)):
                IdxsToRemove.append([ j for j in range(num_points[i]) if j < start_indices[i] or j > end_indices[i]])

            for i in range(len(ranges)):
                data = np.delete(data, IdxsToRemove[i], axis=i)

            return data

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

    def scalar_regression(self, y, X, ranges = None, model_points = None):
        """
        Routine to perform ordinary (multivariate) regression on some gridded data. 

        Parameters: 
        -----------
        y: ndarray of gridded scalar data
            the "measurements" of the dependent quantity
        
        X: list of ndarrays of gridded data (treated as indep scalars)
            the "data" for the regressors 

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        Returns:
        --------
        list of fitted parameters
        list of std errors associated with the fitted parameters, 
        (i.e. parameter * std error of the corresponding regressor.)

        Notes:
        ------
        Gridded data and ranges are chosen at the model level, and passed here
        as parameters, so that this external method will not change/access any 
        model quantity. 

        Works in any dimensions!
        """
        # CHECKING ALIGNMENT OF PASSED DATA
        dep_shape = np.shape(y)
        n_reg = len(X)
        for i in range(n_reg):
            if np.shape(X[i]) != dep_shape:
                print(f'The {i}-th regressor data is not aligned with dependent data, removing {i}-th regressor data.')
                X.remove(X[i])

        # TRIMMING THE DATA TO WITHIN RANGES 
        if ranges != None and model_points != None:
            print('Trimming dataset for regression')
            y = self.trim_data(y, ranges, model_points)
            for x in X: 
                x = self.trim_data(x, ranges, model_points)

        # FLATTENING + FITTING
        y = y.flatten()
        for x in X: 
            x = x.flatten()
        n_reg = len(X)
        n_data = len(y)
        X = np.reshape(X, (n_data, n_reg))
        model = sm.OLS(y, X)
        result = model.fit()
        return result.params, result.bse
    
    def scalar_weighted_regression(self, y, X, W, ranges = None, model_points = None):
        """
        Routine to perform ordinary (multivariate) regression on some gridded data. 

        Parameters: 
        -----------
        y: ndarray of gridded scalar data
            the "measurements" of the dependent quantity
        
        X: list of ndarrays of gridded data (treated as indep scalars)
            the "data" for the regressors 

        W: ndarray of gridded weights (one per gridpoint)

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        Returns:
        --------
        list of fitted parameters
        list of std errors associated with the fitted parameters, 
        (i.e. parameter * std error of the corresponding regressor.)

        Notes:
        ------
        For the future, is it worth coding this as a decorator? 
        """
        # CHECKING ALIGNMENT OF PASSED DATA
        dep_shape = np.shape(y)
        n_reg = len(X)

        for i in range(n_reg):
            if np.shape(X[i]) != dep_shape:
                print(f'The {i}-th regressor data is not aligned with dependent data, removing {i}-th regressor data.')
                X.remove(X[i])
            elif np.shape(y) != np.shape(W):
                print(f'The weights passed are not not aligned with data, setting these to 1')
                W = [1 for i in range(len(W))]
        

        # TRIMMING THE DATA TO WITHIN RANGES 
        if ranges != None and model_points != None:
            print('Trimming dataset for regression')
            y = self.trim_data(y, ranges, model_points)
            for x in X: 
                x = self.trim_data(x, ranges, model_points)

        # FLATTENING + FITTING
        y = y.flatten()
        for x in X: 
            x = x.flatten()
        n_reg = len(X)
        n_data = len(y)
        X = np.reshape(X, (n_data, n_reg))
        model = sm.WLS(y, X, W)
        result = model.fit()
        return result.params, result.bse

    def visualize_correlation(self, x, y, xlabel=None, ylabel=None, ranges=None, model_points=None):
        """
        Method that returns an instance of JointGrid, with plotted the scatter plot and univariate distributions.
        Possibility to cut data to lie within ranges.

        Parameters:
        -----------
        x,y: nd.arrays 
            must be of the same shape

        xlabel, ylabel: strs
            labels for the quantites plotted

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        
        Returns:
        --------
        Instance of JointGrid. Figure can then be accessed at the model level
        for further adjustments. 
        """
        if x.shape != y.shape: 
            print('Cannot check correlation: data is misaligned!')
            return None

        if ranges != None and model_points != None:
            print('Trimming dataset for correlation plot')
            x = self.trim_data(x, ranges, model_points)
            y = self.trim_data(y, ranges, model_points)
            
        x=x.flatten()
        y=y.flatten()
         
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.JointGrid()
            sns.scatterplot(x=x, y=y, ax=g.ax_joint)
            sns.kdeplot(x=x, ax=g.ax_marg_x)
            sns.histplot(y=y, ax=g.ax_marg_y, kde=True)
        
            g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
            g.fig.tight_layout()
        return g

    def visualize_correlations(self, data, labels, ranges=None, model_points=None):
        """
        Method that returns an instance of PairGrid of correlation plots for a list of vars.
        Plotted is: 
            daigonal: univariate histogram (with kde superimposed)
            upper triangle: scatteplots
            lower triangle: bivariate kde(s)

        Possibility to cut data to lie within ranges.

        Parameters:
        -----------
        data: list of nd.arrays 
            must be of the same shape

        labels: list of strs
            labels for the quantites plotted

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        
        Returns:
        --------
        Instance of PairGrid. Figure can then be accessed at the model level
        for further adjustments.
        """
        ref_shape=data[0].shape
        Idx_to_delete = []
        for i in range(1,len(data)):
            if data[i].shape != ref_shape:
                print(f'Cannot use {labels[i]} data: misaligned with first array. Removing it.') 
                Idx_to_delete.append(i)

        if len(Idx_to_delete) >= 1:
            data=list(np.delete(data, Idx_to_delete, axis=0))

        if ranges != None and model_points != None:
            print('Trimming dataset for correlation plot')
            for i in range(len(data)):
                data[i]=self.trim_data(data[i], ranges, model_points)

        for i in range(len(data)):
            data[i]=data[i].flatten()

        data=np.column_stack(data)
        data_df = pd.DataFrame(data, columns=labels)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.PairGrid(data_df)
            g.map_upper(sns.scatterplot)
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.histplot, kde=True)
            g.fig.tight_layout()
        return g

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
        




