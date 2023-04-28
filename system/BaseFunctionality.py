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

    def Mink_dot(self,vec1,vec2):
        """
        Inner-product in (n+1)-dimensions
        """
        dot = -vec1[0]*vec2[0] # time component
        for i in range(1,len(vec1)):
            dot += vec1[i]*vec2[i] # spatial components
        return dot
    
    def get_rel_vel(self, spatial_vels):
        """
        Construct (n+1)-velocity (meso) from spatial Cartesian (x,y,...) components
        """
        W = 1 / np.sqrt(1-np.sum(spatial_vels**2))
        return spatial_vels.insert(spatial_vels,0,W)    

    def get_U_mu_MagTheta(self, Vmag_Vtheta):
        """
        Construct (2+1)-velocity (meso) from spatial Polar (r, theta) components
        """
        Vmag, Vtheta = Vmag_Vtheta[0], Vmag_Vtheta[1]
        return self.get_U_mu([Vmag*np.cos(Vtheta),Vmag*np.sin(Vtheta)])
    
    def construct_tetrad(self, U):
        """
        Construct 2 tetrad vectors that are perpendicular to (2+1)-velocity U,
        and each other. These are used to define the box for filtering.
        """
        e_x = np.array([0.0,1.0,0.0]) # 1 + 2D
        E_x = e_x + np.multiply(self.Mink_dot(U,e_x),U)
        E_x = E_x / np.sqrt(self.Mink_dot(E_x,E_x)) # normalization
        e_y = np.array([0.0,0.0,1.0])
        E_y = e_y + np.multiply(self.Mink_dot(U,e_y),U) - np.multiply(self.Mink_dot(E_x,e_y),E_x)
        E_y = E_y / np.sqrt(self.Mink_dot(E_y,E_y))
        return E_x, E_y
        
    def find_boundary_pts(self, E_x,E_y,P,L):
        """
        Find the (four) corners of the box that is the filtering region.

        Parameters
        ----------
        E_x : list of floats
            One tetrad vector.
        E_y : list of floats
            Second tetrad vector.
        P : list of floats
            Coordinate of the centre of the box (t,x,y).
        L : float
            Filtering lengthscale (length of one side of the box).

        Returns
        -------
        corners : list of list of floats
            list of the coordinates of the box's corners.

        """
        c1 = P + (L/2)*(E_x + E_y)
        c2 = P + (L/2)*(E_x - E_y)
        c3 = P + (L/2)*(-E_x - E_y)
        c4 = P + (L/2)*(-E_x + E_y)
        corners = [c1,c2,c3,c4]
        return corners
    
    def surface_flux(self, x,E_x,E_y,P,direc_vec):
        point = P + x*(E_x + E_y)
        u, n = self.interpolate_u_n_point(point)
        n_mu = np.multiply(u,n)
        return self.Mink_dot(n_mu,direc_vec)
    
    def project_tensor(self, vector1_wrt, vector2_wrt, to_project):
        return np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
    
    def orthogonal_projector(self, u):
        return self.metric + np.outer(u,u)    


    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to in put 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    """
    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
    def find_nearest_cell(self, point):
        t_pos = self.find_nearest(self.ts,point[0])
        x_pos = self.find_nearest(self.xs,point[1])
        y_pos = self.find_nearest(self.ys,point[2])
        return [np.where(self.ts==t_pos)[0][0], np.where(self.xs==x_pos)[0][0], np.where(self.ys==y_pos)[0][0]]
    
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
