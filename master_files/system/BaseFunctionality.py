# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:21:02 2022

@author: mjh1n20
"""

# from multiprocessing import Process, Pool
import numpy as np
# from timeit import default_timer as timer
import cProfile, pstats, io
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

class Base(object):
    @staticmethod
    def Mink_dot(vec1, vec2):
        """
        Parameters:
        -----------
        vec1, vec2 : list of floats (or np.arrays)

        Return:
        -------
        mink-dot (cartesian) in 1+n dim
        """
        if len(vec1) != len(vec2):
            print("The two vectors passed to Mink_dot are not of same dimension!")

        dot = -vec1[0]*vec2[0]
        for i in range(1,len(vec1)):
            dot += vec1[i] * vec2[i]
        return dot
  
    @staticmethod
    def get_rel_vel(spatial_vels):
        """
        Build unit vectors starting from spatial components
        Needed as this will enter the minimization procedure

        Parameters:
        ----------
        spatial_vels: list of floats

        Returns:
        --------
        list of floats: the d+1 vector, normalized wrt Mink metric
        """
        W = 1 / np.sqrt(1-np.sum(spatial_vels**2))
        return W * np.insert(spatial_vels,0,1.0)

    @staticmethod
    def project_tensor(vector1_wrt, vector2_wrt, to_project):
        """
        """
        return np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
    
    
    @staticmethod
    def orthogonal_projector(u, metric):
        """
        Returns: 
        --------
        Orthogonal projector wrt vector u

        Notes:
        ------
        The vector u must be time-like
        """
        return metric + np.outer(u,u)    
 

    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    Should now work for any dimensional data.
    """
    @staticmethod
    def find_nearest(array, value):
        """
        Returns closest value to input 'value' in 'array'

        Parameters: 
        -----------
        array: np.array of shape (n,)

        value: float

        Returns:
        --------
        float 

        Note:
        -----
        To be used together with find_nearest_cell.  
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
    @staticmethod    
    def find_nearest_cell(point, points):
        """
        Use find nearest to find closest value in a list of input 'points' to 
        input 'point'. 

        Parameters:
        -----------
        point: list of d+1 float

        points: list of lists of d+1 floats 

        Returns:
        --------
        List of d+1 indices corresponding to closest value to point in points
        """
        if len(points) != len(point):
            print("find_nearest_cell: The length of the coordinate vector\
                   does not match the length of the coordinates.")
        positions = []
        for dim in range(len(point)):
            positions.append(Base.find_nearest(points[dim], point[dim]))
        return [np.where(points[i] == positions[i])[0][0] for i in range(len(positions))]
    
    def profile(self, fnc):
        """A decorator that uses cProfile to profile a function"""
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval
        return inner

class MySymLogPlotting(object):

    @staticmethod
    def symlog_num(num):
        """
        Return the symlog of a number

        Parameters:
        -----------
        num: float

        Returns:
        --------
        The symlog of the input num (separate copy)
        """
        if np.abs(num) +1. == 1.0:
            result = num
        else:
            result = np.sign(num) * np.log10(np.abs(num)+1.)
        return result

    @staticmethod
    def inverse_symlog_num(num):
        if num > 0:
            return 1 + 10 ** num
        elif num < 0:
            return 1- 10 ** (- num)
        else:
            return 0

    @staticmethod
    def symlog_var(var):
        """
        Return the symlog of an array.

        Parameters:
        -----------
        var: np.array of any shape

        Returns:
        --------
        The symlog of the input var (separate copy)
        """
        count_zeros=0
        temp = np.empty_like(var)
        for index in np.ndindex(var.shape):
            value = var[index]
            if value == 0: 
                count_zeros +=1
            else: 
                temp[index] = MySymLogPlotting.symlog_num(value)
        if count_zeros >= 1:
            print('Careful: there are {} zeros in the data'.format(count_zeros))
        return temp

    @staticmethod
    def get_mysymlog_var_ticks(var):
        """
        Method to automatize the computation of the ticks and nodes for a variable.
        Nodes are then to be used within the class MyThreeNodesNorm. 
        Ticks and labels are for the colorbar of the plot of input 'var'. 

        Parameters: 
        -----------
        var: np.array
            This HAS TO take both positive and negative values 

        Returns:
        --------
        ticks: list
            list of tick points to be used by the colorbar

        ticks_labels: list
            list of corresponding labels for the colorbar

        nodes: list of len=5
            the extrame and the three central nodes to be used by MyThreeNodesNorm

        Notes:
        ------
        The ticks/nodes and labels are computed like this: start from negative values, identify min 
        and max values of the negative part of input 'var' to identify relevant ticks and nodes. 
        Then add a zero (tick and node) and proceed to the positive values.
        """
        ticks = []
        ticks_labels = []
        nodes = []
    

        pos_var = np.ma.masked_where(var <0., var, copy=True).compressed()
        pos_var_small = np.ma.masked_where(pos_var >=1., pos_var, copy=True).compressed()
        pos_var_large = np.ma.masked_where(pos_var <1., pos_var, copy=True).compressed()

        neg_var = np.ma.masked_where(var >0., var, copy=True).compressed()
        neg_var_small = np.ma.masked_where(neg_var <=-1., neg_var, copy=True).compressed()
        neg_var_large = np.ma.masked_where(neg_var >-1., neg_var, copy=True).compressed()

        # Working out nodes, ticks and ticks_labels for the negative range
        if len(neg_var_large) >0: 
            # print('There are negative large values', flush=True)
            vmin = np.amin(neg_var_large)
            new_nodes = [MySymLogPlotting.symlog_num(vmin)]
            new_ticks = new_nodes
            new_ticks_labels = [r'$-10^{%d}$'%(int(np.log10(-vmin)))]
            
            nodes += new_nodes
            ticks += new_ticks
            ticks_labels += new_ticks_labels
            
            
            if len(neg_var_small) == 0: 
                # print('Actually: only negative large values', flush=True)
                vmax = np.amax(neg_var_large)
                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$-10^{%d}$'%(int(np.log10(-vmax)))]

                nodes += new_nodes
                ticks += new_ticks
                ticks_labels += new_ticks_labels

            else:
                # print('And also negative small values', flush=True)
                vmin = np.amin(neg_var_small)
                vmax = np.amax(neg_var_small)

                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                # new_ticks = [symlog_num(vmin), symlog_num(vmax)]
                new_ticks = [MySymLogPlotting.symlog_num(vmax)]
                # new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmin,-vmax])]
                new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmax])]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes
                
        else: # len(neg_var_large)==0:
            # print('Only negative small values', flush=True)
            vmin = np.amin(neg_var_small)
            vmax = np.amax(neg_var_small)

            # print(vmin, vmax, "\n")
            new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
            # print(new_nodes)
            new_ticks = new_nodes
            new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmin,-vmax])]

            ticks += new_ticks
            ticks_labels += new_ticks_labels
            nodes += new_nodes


        nodes += [0.]
        ticks += [0.]
        ticks_labels += ['0']

        # Working out the remaining nodes, ticks and ticks_labels for the positive range

        if len(pos_var_large)==0:
            # print('Only positive small values', flush=True)
            vmin = np.amin(pos_var_small)
            vmax = np.amax(pos_var_small)

            # print(vmin, vmax)
            new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
            # print(new_nodes)
            new_ticks = new_nodes
            new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]
    
            ticks += new_ticks
            ticks_labels += new_ticks_labels
            nodes += new_nodes

        else: # len(pos_var_large) > 0: 
            # print('There are positive large values', flush=True)

            if len(pos_var_small) >0:
                # print('And also positive small values', flush=True)
                vmin = np.amin(pos_var_small)
                vmax = np.amax(pos_var_small)

                # new_nodes = [symlog_num(vmax)]
                new_nodes = [MySymLogPlotting.symlog_num(vmin)]
                # new_ticks = [symlog_num(vmin), symlog_num(vmax)]
                new_ticks = [MySymLogPlotting.symlog_num(vmin)]
                # new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]
                new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin])]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes

                vmax = np.amax(pos_var_large)
                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$10^{%d}$'%(int(np.log10(vmax)))]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes
                

            else: # len(pos_var_small) ==0:
                vmin = np.amin(pos_var_large)
                vmax = np.amax(pos_var_large)

                new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]

                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes

        return ticks, ticks_labels, nodes

class MyThreeNodesNorm(mpl.colors.Normalize):
    """
    Sub-classing colors.Normalize: the norm has three inner nodes plus the extrema. 
    Within each segment (delimited by a node or extrema) you have linear interpolation. 
    
    Should be used when plotting quantities that are both positive and negative, and you
    want to highlight 1) where a critical value (middle_node) is 2) the closest values some 
    variable takes to its left and right
    """
    def __init__(self, nodes, clip=False):
        """
        Parameters: 

        nodes: array of five numbers in strictly ascending order (the nodes)
        """
        if len(nodes)!=5: 
            raise ValueError('The class MyThreeNodesNorm requires 5 nodes: the extrema and the three central')

        for i in range(len(nodes)-1):
            if nodes[i+1] <=nodes[i]:
                raise ValueError('nodes must be in monotonically ascending order!')
                
        super().__init__(nodes[0], nodes[4], clip)
        self.first_node = nodes[1]
        self.central_node = nodes[2]
        self.third_node = nodes[3]

    def __call__(self, value, clip=None):
        x = [self.vmin, self.first_node, self.central_node, self.third_node, self.vmax]
        y = [0, 0.4, 0.5, 0.6, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y = [self.vmin, self.first_node, self.central_node, self.third_node, self.vmax]
        x = [0, 0.4, 0.5, 0.6, 1.]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
