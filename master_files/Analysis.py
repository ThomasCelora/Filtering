# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

# USE !SCIKIT LEARN INSTEAD? IT'S THE PACKAGE FOR MACHINE LEARNING SO! 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import pickle
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA 
# import statsmodels.api as sm

from system.BaseFunctionality import *
from MicroModels import * 
from FileReaders import * 
from Filters import *
from Visualization import *
from MesoModels import * 

    

class CoefficientsAnalysis(object): 
    """
    Class containing a number of methods for performing statistical analysis on gridded data. 
    Methods include: regression, visualizing correlations, PCAs
    """
    def __init__(self): #, visualizer, spatial_dims):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        # self.visualizer=visualizer # need to re-structure this... or do I 
        # Do you really need the visualizer here? I don't think so! 
        # self.spatial_dims=spatial_dims

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

    def scalar_regression(self, y, X, ranges = None, model_points = None, weights=None, add_intercept=False):
        """
        Routine to perform ordinary or weighted (multivariate) regression on some gridded data. 

        Parameters: 
        -----------
        y: ndarray of gridded scalar data
            the "measurements" of the dependent quantity
        
        X: list of ndarrays of gridded data (treated as indep scalars)
            the "data" for the regressors 

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        weights: ndarray of gridded weights

        add_intercept: bool
            whether to extemd the dataset to account for a constant offset in the
            regression model.

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

        Intercept must be added manually: for the future, add this via true/false block.
        """
        # CHECKING ALIGNMENT OF PASSED DATA
        dep_shape = np.shape(y)
        n_reg = len(X)
        for i in range(n_reg):
            if np.shape(X[i]) != dep_shape:
                print(f'The {i}-th regressor data is not aligned with dependent data, removing {i}-th regressor data.')
                X.remove(X[i])
        
        if len(X)==0: 
            print('None of the passed regressors is compatible with dep data. Exiting.')
            return None
        
        if weights:
            if np.shape(y) != np.shape(weights):
                print(f'The weights passed are not not aligned with data, setting these to 1')
                weights=np.ones(y.shape)
        
        # TRIMMING THE DATA TO WITHIN RANGES 
        if ranges != None and model_points != None:
            print('Trimming dataset for regression')
            Y = self.trim_data(y, ranges, model_points)
            if weights:
                Weights = self.trim_data(weights, ranges, model_points)
            XX = []
            for x in X: 
                XX.append(self.trim_data(x, ranges, model_points))

        # FLATTENING + FITTING
        Y = Y.flatten()
        if weights:
            Weights = Weights.flatten()

        for i in range(len(X)): 
            XX[i]=XX[i].flatten()

        if add_intercept:
            const = np.ones(Y.shape)
            XX.insert(0,const)
        n_reg = len(XX)
        n_data = len(Y)
        XX= np.reshape(XX, (n_data, n_reg))
        
        # VERSION USING STATSMODELS
        # if weights:
        #     model=sm.WLS(y, Xfl, W)
        #     result= model.fit()
        #     return result.params, result.bse
        # else:
        #     model=sm.OLS(y, Xfl)
        #     result=model.fit()
        #     return result.params, result.bse

        # VERSION USING SKLEARN:
        model=LinearRegression(fit_intercept=False)
        if weights:
            model.fit(XX,Y,sample_weight=weights)
        else:
            model.fit(XX,Y)
        Y_hat = model.predict(XX)
        res = Y - Y_hat 
        res_sum_of_sq = np.dot(res,res)
        sigma_res_sq = res_sum_of_sq / (n_data-n_reg)
        var_beta = np.linalg.inv(np.einsum('ji,jl->il',XX,XX)) * sigma_res_sq
        bse = [var_beta[i,i]**0.5 for i in range(n_reg)]

        result = [(model.coef_[i], bse[i]) for i in range(n_reg)]
        return result

    def tensor_components_regression(self, y, X, spatial_dims, ranges=None, model_points=None, components=None, weights=None, add_intercept=False):
        """
        Wrapper of scalar_regression: if no components is passed, perform regression on each tensor component
        independently, otherwise only on a subset of these. All the component-wise results are returned.

        Parameters:
        -----------
        y: ndarray of gridded tensorial data
            the "measurements" of the dependent quantity
        
        X: list of ndarrays of gridded data 
            the "data" for the regressors 

        spatial_dims: int    
        
        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        components: list of tuples
            the tensor components to be considered for regression

        add_intercept: bool
            whether to extemd the dataset to account for a constant offset in the
            regression model.

        Returns:
        --------
        list of fitted parameters
        list of std errors associated with the fitted parameters, 
        (i.e. parameter * std error of the corresponding regressor.)

        Notes:
        ------
        Weights are taken as the same for each component here.
        For future: do you need to change this? 
        """
        # CHECKING THE RANK OF DATA AND REGRESSORS ARE COMPATIBLE
        l=[0 for i in range(spatial_dims+1)]
        dep_shape = y[tuple(l)].shape
        for i, x in enumerate(X):
            if x[tuple(l)].shape != dep_shape:
                print('The {}-th regressor shape {} is not compatible with dependent shape {}. Removing it!'.format(i, x[0,0,0].shape, dep_shape))
                X.remove(x)
        if len(X)==0: 
            print('None of the passed regressors is compatible with dep data. Exiting.')
            return None
        
        # PREPARING COMPONENTS TO LOOP OVER: user-provided or all?     
        if components:
            for c in components:
                if len(c) != len(dep_shape):
                    print(f'The components indices passed {c} are not compatible with data. Ignoring this and moving on!')
                    components.remove(c)
        else:
            components = []
            for i in np.ndindex(dep_shape):
                components.append(i)

        # RESHAPING: now grid indices come last, needed for flattening!
        tot_shape=y.shape
        reshaping=[tot_shape[i] for i in range(spatial_dims+1)] 
        reshaping=tuple(list(dep_shape)+reshaping)
        y = y.reshape(reshaping)
        for i in range(len(X)):
            X[i]= X[i].reshape(reshaping) 

        # SCALAR REGRESSION ON EACH COMPONENT
        results = []
        for c in components:
            yc = y[c]
            Xc = [ x[c] for x in X]
            results.append(self.scalar_regression(yc, Xc, ranges, model_points, weights, add_intercept))
        return results

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
            X = self.trim_data(x, ranges, model_points)
            Y = self.trim_data(y, ranges, model_points)
            print('Finished trimming data')
        else: 
            X, Y = x, y
        X=X.flatten()
        Y=Y.flatten()
        print('Data flattened')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.JointGrid()
            sns.scatterplot(x=X, y=Y, ax=g.ax_joint)
            sns.kdeplot(x=X, ax=g.ax_marg_x)
            sns.histplot(y=Y, ax=g.ax_marg_y, kde=True)
            g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
            g.fig.tight_layout()
        print('Figure produced inside Coeff Analysis')
        return g

    def visualize_many_correlations(self, data, labels, ranges=None, model_points=None):
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
            print('Finished trimming data')
        
        Data = []
        for i in range(len(data)):
            Data.append(data[i].flatten())

        print('Data_flattened')

        Data=np.column_stack(Data)
        Data_df = pd.DataFrame(Data, columns=labels)
        print('Flattened data converted to a dataframe')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.PairGrid(Data_df)
            g.map_upper(sns.scatterplot)
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.histplot, kde=True)
            g.fig.tight_layout()

        print('Finished producing figure inside CoefficientsAnalysis')
        return g

    def PCA_find_regressors_subset(self, data, ranges=None, model_points=None, var_wanted = 1.):
        """
        Idea: pass a large list of quantities that are correlated with a residual/closure coeff.
        Check if there is a smaller subset of principal components to be retained that are sufficient
        to explain the enough of the observed variance in the dataset. 

        AIM: linear dimensionality reductions for regressors

        Parameters: 
        -----------
        data: list of gridded data

        ranges: list of lists of 2 floats
            mins and max in each direction

        model_points: list of lists containing the gridpoints in each direction

        var_wanted: float
            should be a number between 0. and 1. : the percetage of variance to be retained.

        Returns:
        --------
        comp_decomp: array of shape (n_var, n_comp)
            Matrix whose columns contains the decomposition of the principal components 
            written in the basis of the untransformed (original) variables. 

        g: result of self.visualize_many_correlations --> show the PCs are indeed uncorrelated.

        Notes:
        ------
        
        """
        # CHECKING AND PREPROCESSING THE DATA
        ref_shape = data[0].shape
        n_vars = len(data)
        for i in range(1, n_vars):
            if data[i].shape != ref_shape:
                print(f'The {i}-th  feature passed is not aligned with the first, removing {i}-th feature.')
                data.remove(data[i])

        if len(data)==0: 
            print('No two vars are compatible. Exiting.')
            return None

        if ranges != None and model_points != None:
            print('Trimming dataset for PCA analysis.')
            for i in range(len(data)):
                data[i] = self.trim_data(data[i], ranges, model_points)

        for i in range(n_vars):
            data[i] = data[i].flatten()

        #STANDARDIZING DATA TO ZERO MEAN AND UNIT VARIANCE
        Data = []
        for i in range(len(data)):
            x = data[i]
            mean = np.mean(x)
            var = np.var(x)
            y = np.array([x[j]-mean for j in range(len(x))])
            Data.append(y/var)
            # Data.append(y)
        Data = np.column_stack(tuple([Data[i] for i in range(n_vars)]))

        # HOW MANY PRINCIPAL COMPONENTS HAVE TO BE RETAINED? 
        pca_model = PCA().fit(Data)
        n_comp=0
        exp_var_sum = np.cumsum(pca_model.explained_variance_ratio_)
        var_captured = exp_var_sum[n_comp]
        while var_captured < var_wanted:
            n_comp +=1
            var_captured= exp_var_sum[n_comp]
        n_comp = n_comp+1

        # WORKING OUT THE COMPONENTS DECOMPOSITION
        var2comp = pca_model.components_ # shape: n_comp * n_var
        comp_decomp = []
        for i in range(n_comp):
            # should be the inverse but var2comp is unitary, so transpose 
            comp_decomp.append(np.einsum('ij->ji', var2comp)[:,i])

        # CREATING THE FIGURE WITH THE COMPONENTS PROFILE TO CHECK THEY'RE UNCORRELATED
        # this block is temporary and will be removed later.
        pc_profiles = np.einsum('ij,kj->ik', Data, var2comp)
        labels = [f'{i} comp' for i in range(n_comp)]
        pc_for_plotting = [pc_profiles[:,i] for i in range(n_comp)]
        g = self.visualize_many_correlations(pc_for_plotting, labels)
        
        return comp_decomp, g

    def PCA_find_regressors():
        """
        Idea pass both the residuals and a large list of quantities, identify which among these are
        correlated with the residual. Do this by checking the scores of the residual (the first var 
        of the dataset, say) on the various principal components. 
        Then return the weights. This will then be used in regression (possibyly after checking they're
        not correlated).
        """
        pass
        
    # Not sure about these two methods. 
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


if __name__ == '__main__':
    pass
