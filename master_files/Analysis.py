# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

# USE !SCIKIT LEARN INSTEAD? IT'S THE PACKAGE FOR MACHINE LEARNING SO! 

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
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
    Methods include: regression, visualizing correlations, PCA routines
    """
    def __init__(self): #, visualizer, spatial_dims):
        """
        Nothing as yet...

        Returns
        -------
        Also nothing...
        
        """
        # Change to latex font
        plt.rc("font",family="serif")
        plt.rc("mathtext",fontset="cm")

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
                newdata = np.delete(data, IdxsToRemove[i], axis=i)

            return newdata
        
    def get_pos_or_neg_mask(self, pos_or_neg, array):
        """
        Function that return the mask that would be applied to an array in order to select 
        positive or negative values

        Parameters:
        -----------
        pos_or_neg: can be int (0 or 1) or bool True or False

        array: np.array

        Notes:
        ------
        To be combined with others before applying all together 
        """
        if pos_or_neg:
            mask = ma.masked_where(array<0, array, copy=True).mask
        else:
            mask = ma.masked_where(array>0, array, copy=True).mask

        return mask
        
    def preprocess_data(self, list_of_arrays, preprocess_data):#, ranges=None, model_points=None):
        """
        Takes input list of arrays with dictionary on how to preprocess_data and info about
        trimming of arrays to selected range within grid

        Parameters:
        ----------
        list_of_arrays: 
            list with the combined data to be processed

        preprocess_data: dictionary
            each value of the dictionary must be a list long as the arrays passed (checked for)
            Expected keys are: 

            'pos_or_neg' is 1 (0) if you want to select positive (negative) values

            'log_or_not' is 1 if you want to take the logarithm of the data 


        # ranges: list of lists of 2 floats --> this is passed to trim_data 
        #    the min and max in each direction

        # model_points: list of list of floats --> this is passed to trim_data 
        #    the gridpoints of the array
        """

        num_arrays = len(list_of_arrays)
        condition = False
        for key in preprocess_data:
            if len(preprocess_data[key]) != num_arrays:
                condition=True
        if condition: 
            print('Preprocess_data dictionary is not compatible with list_of_arrays')

        # # Trimming data to lie within range: you probably don't want this to habben here actually
        # if ranges != None and model_points != None:
        #     new_list = []
        #     for i in range(len(list_of_arrays)):
        #         new_list.append(self.trim_data(list_of_arrays[i], ranges=ranges, model_points=model_points))
        
        # Combining the masks of the different arrays
        masks = []
        for i in range(len(list_of_arrays)):
            pos_or_neg = preprocess_data['pos_or_neg'][i]
            temp = self.get_pos_or_neg_mask(pos_or_neg, list_of_arrays[i]) 
            masks.append(temp)
        
        tot_mask = masks[0]
        for i in range(1,len(masks)):
            tot_mask = np.logical_or(tot_mask, masks[i])
        
        # Masking the combined dataset with combined mask and taking log
        processed_list = []
        for i in range(len(list_of_arrays)):
            temp = ma.masked_array(list_of_arrays[i], tot_mask)
            processed_list.append(temp.compressed())
            #when array is compressed, this is automatically flattened!
        
        for i in range(len(processed_list)):
            if preprocess_data['log_or_not'][i]:
                processed_list[i] = np.log10(np.abs(processed_list[i]))

        return processed_list

    def extract_randomly(self, array, num_extractions):
        """
        Extract randomly from data. 

        Parameters:
        -----------
        data: np.array (flattened or not)
        num_extractions: number of values to be extracted 
        """
        new_array = array.copy()
        if new_array.shape != (1,):
            print('Array is not flat, flattening it.')
            new_array = new_array.flatten()

        max_idx = len(new_array) - 1
        extracted_vals = []
        for i in range(num_extractions):
            temp = new_array[random.randint(0,max_idx)]
            extracted_vals.append(temp)
        return np.array(extracted_vals)

    def scalar_regression(self, y, X, ranges=None, model_points=None, weights=None, add_intercept=False):
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
            whether to extend the dataset to account for a constant offset in the
            regression model.

        Returns:
        --------
        list of (fitted parameters, std errors)
        (std errors = parameter * std error of the corresponding regressor.)

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

    
        # TRIMMING THE DATA TO WITHIN RANGES + ADDING COLUMN FOR INTERCEPT (if necessary)
        if ranges != None and model_points != None:
            print('Trimming dataset for regression')
            Y = self.trim_data(y, ranges, model_points)
            XX = []
            for x in X: 
                XX.append(self.trim_data(x, ranges, model_points))
            if weights:
                Weights = self.trim_data(weights, ranges, model_points)            
        else: 
            XX , Y = X , y
            if weights:
                Weights = weights

        # Trim data first, then pre-process data
        # if preprocess_data then do it. 

        # FLATTENING + FITTING
        Y = Y.flatten()
        if weights:
            Weights = Weights.flatten()

        if add_intercept:
            const = np.ones(Y.shape)
            # print(const)
            XX.insert(0, const)
            n_reg = n_reg + 1 

        for i in range(len(XX)): 
            XX[i]=XX[i].flatten()
        XX = np.einsum('ij->ji', XX)
        
        # # VERSION USING STATSMODELS
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
            print('Fitting with weights')
            model.fit(XX, Y, sample_weight=Weights)
        else:
            print('Fitting without weights')
            model.fit(XX,Y)
        regress_coeff = list(model.coef_)
        
        # Computing std errors on the regressed coefficients
        n_data = len(Y)
        Y_hat = model.predict(XX)
        res = Y - Y_hat 
        res_sum_of_sq = np.dot(res,res)
        sigma_res_sq = res_sum_of_sq / (n_data-n_reg)
        var_beta = np.linalg.inv(np.einsum('ji,jl->il',XX,XX)) * sigma_res_sq
        bse = [var_beta[i,i]**0.5 for i in range(n_reg)]

        return regress_coeff, bse

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

    def visualize_correlation(self, x, y, xlabel=None, ylabel=None, ranges=None, model_points=None, \
                              hue_array=None, style_array=None, legend_dict=None, palette=None, markers=None):
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
            if hue_array is not None: 
                hue_array = self.trim_data(hue_array, ranges, model_points)
            if style_array is not None: 
                style_array = self.trim_data(style_array, ranges, model_points)
            print('Finished trimming data')
        else: 
            X, Y = x, y
        
        X=X.flatten()
        Y=Y.flatten()
        if hue_array is not None: 
            hue_array = hue_array.flatten()
        if style_array is not None: 
            style_array = style_array.flatten()
        print('Data flattened')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.JointGrid()
            scatter = sns.scatterplot(x=X, y=Y, ax=g.ax_joint, s=4, c='red', hue=hue_array, style=style_array, 
                                      palette=palette, markers=markers)
            sns.histplot(x=X, ax=g.ax_marg_x, kde=True, color= 'red')
            sns.histplot(y=Y, ax=g.ax_marg_y, kde=True, color= 'red')
            g.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
            g.fig.tight_layout()

            if legend_dict is not None: 
                handles, labels = scatter.get_legend_handles_labels()
                new_labels = [legend_dict[label] for label in labels]
                scatter.legend(handles, new_labels)

        # print('Figure produced inside Coeff Analysis')
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

        Data=np.column_stack(Data)
        Data_df = pd.DataFrame(Data, columns=labels)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            g=sns.PairGrid(Data_df)
            g.map_upper(sns.scatterplot, s=4, c='red')
            g.map_lower(sns.kdeplot, color='red')
            g.map_diag(sns.histplot, kde=True, color='red')
            g.fig.tight_layout()

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
        Data =[data[0]]
        for i in range(1, n_vars):
            if data[i].shape != ref_shape:
                print(f'The {i}-th  feature passed is not aligned with the first, removing {i}-th feature.')
            else: 
                Data.append(data[i])

        if len(Data)==0: 
            print('No two vars are compatible. Exiting.')
            return None

        if ranges != None and model_points != None:
            print('Trimming dataset for PCA analysis.')
            for i in range(len(Data)):
                Data[i] = self.trim_data(Data[i], ranges, model_points)

        for i in range(len(Data)):
            Data[i] = Data[i].flatten()

        #STANDARDIZING DATA TO ZERO MEAN AND UNIT VARIANCE
        for i in range(len(Data)):
            x = Data[i]
            mean = np.mean(x)
            var = np.var(x)
            y = np.array([x[j]-mean for j in range(len(x))])
            Data[i] = y/var
            # Data[i] = y
        Data = np.column_stack(tuple([Data[i] for i in range(len(Data))]))

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

    def PCA_find_regressors(self, dependent_var, explanatory_vars, ranges=None, model_points=None, pcs_num=1):
        """
        Idea: pass both the residuals/coefficient and a list of quantities. Identify which among these are
        correlated with the residual/coefficient, by checking the scores of this on the 
        principal components. Then return the linear combination of the (highest) components with respect 
        to the explanatory vars. 
        """
        # CHECKING AND PREPROCESSING THE DATA
        dep_shape = dependent_var.shape
        n_expl_vars = len(explanatory_vars)
        Expl_vars = []
        for i in range(0, n_expl_vars):
            if explanatory_vars[i].shape != dep_shape:
                print(f'The {i}-th  feature passed is not aligned with the first, removing {i}-th feature.')
            else: 
                Expl_vars.append(explanatory_vars[i])

        if len(Expl_vars)==0: 
            print('No two vars are compatible. Exiting.')
            return None

        if ranges != None and model_points != None:
            print('Trimming dataset for PCA analysis.')
            Dep_var = self.trim_data(dependent_var, ranges, model_points)
            for i in range(len(explanatory_vars)):
                Expl_vars[i] = self.trim_data(Expl_vars[i], ranges, model_points)

        Dep_var = Dep_var.flatten()
        for i in range(len(Expl_vars)):
            Expl_vars[i] = Expl_vars[i].flatten()

        #STANDARDIZING DATA TO ZERO MEAN AND UNIT VARIANCE
        Data = []
        for i in range(len(Expl_vars)):
            x = Expl_vars[i]
            mean = np.mean(x)
            var = np.var(x)
            y = np.array([x[j]-mean for j in range(len(x))])
            Data.append(y/var)
            # Data.append(y)
        mean = np.mean(Dep_var)
        var = np.var(Dep_var)
        y = (Dep_var - mean)
        y=y/var
        Data = np.column_stack(tuple([y] + [Data[i] for i in range(len(Expl_vars))]))

        #IDENTIFYING THE COMPONENTS WITH HIGHEST SCOREs ON THE DEPENDENT VAR
        pca_model = PCA().fit(Data)
        var2comp = pca_model.components_
        comp2var = np.einsum('ij->ji', var2comp)

        print('components_: \n{}\n'.format(var2comp))
        scores_of_dep_var = var2comp[:,0]
        print(f'Scores of dependent var on the principal components: \n{scores_of_dep_var}\n')
        # sorted_scores_indices = np.argsort(scores_of_dep_var)
        sorted_scores_indices = np.argsort(np.abs(scores_of_dep_var))
        pos_of_highest_pcs = sorted_scores_indices[-pcs_num:] #identifying the index of the ones with highest score
        # this is ordering the scores, but you should care about their magnitude, no?

        highest_pcs_decomp = []
        corresponding_scores = []  
        tot_explained_var = 0 
        for i in range(pcs_num-1, -1, -1):
            highest_pcs_decomp.append(comp2var[:,pos_of_highest_pcs[i]])
            corresponding_scores.append(scores_of_dep_var[pos_of_highest_pcs[i]])
            tot_explained_var += pca_model.explained_variance_ratio_[pos_of_highest_pcs[i]]

        print(f'Total explained variance: {tot_explained_var}\n')

            
        return highest_pcs_decomp, corresponding_scores

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

    # #TESTING REGRESSION
    # x0 = np.arange(100).reshape((10,10))
    # x1 = np.sin(np.arange(100).reshape((10,10)))
    # X = [x0, x1]
    # y = 1 + 2 * x0 + 3* x1 + random.randint(20,30)

    # statistical_tool = CoefficientsAnalysis()
    # result, errors = statistical_tool.scalar_regression(y,X, add_intercept=True) 
    # print('Coefficients: {}'.format(result))
    # print('Errors: {}\n'.format(errors))

    # #TESTING PRE-PROCESS DATA
    x = np.zeros((100,100))
    for idx in np.ndindex(x.shape):
        signum = [-1,1][random.randint(0,1)]
        x[idx] = signum * random.randint(0,100)

    y = np.zeros((100,100))
    for idx in np.ndindex(x.shape):
        signum = [-1,1][random.randint(0,1)]
        if signum==1:  
            exp = random.randint(-5,5)
        elif signum ==-1: 
            exp = random.randint(-3,3)
        y[idx] = signum * (10 ** exp)

    print('Max and min of x: {}, {}\n'.format(np.max(x), np.min(x)))
    print('Max and min of y: {}, {}\n'.format(np.max(y), np.min(y)))



    preprocess_data = {'pos_or_neg' : [1,0], 
                       'log_or_not' : [0,1]}
    
    statistical_tool = CoefficientsAnalysis()
    data = [x,y]


    x, y = statistical_tool.preprocess_data(data, preprocess_data)

    print('Processing data....\n')
    print('Max and min of x: {}, {}\n'.format(np.max(x), np.min(x)))
    print('Max and min of y: {}, {}\n'.format(np.max(y), np.min(y)))
    

    print('Extracting random vals from x...\n')
    extracted = statistical_tool.extract_randomly(x, 10)
    print(extracted)
