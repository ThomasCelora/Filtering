# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:21:02 2022

@author: mjh1n20
"""

from multiprocessing import Process, Pool
import os
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
import h5py
from scipy.interpolate import interpn
from scipy.optimize import root, minimize
#from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, quad
import cProfile, pstats, io
import math

class Base(object):

    @staticmethod
    def Mink_dot(vec1,vec2):
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
        return np.insert(spatial_vels,0,W)    

    @staticmethod
    def project_tensor(vector1_wrt, vector2_wrt, to_project):
        return np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
    
    @staticmethod
    def orthogonal_projector(u, metric):
        return metric + np.outer(u,u)    


    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to in put 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    Should now work for any dimensional data.
    """
    @staticmethod
    def find_nearest(array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]

    @staticmethod   
    def find_nearest_cell(point, points):
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
