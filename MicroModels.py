# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:13:07 2023

@author: Marcus
"""

from FileReaders import *
import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import root, minimize

class IdealHydro(object):

    def __init__(self):
        
        self.nt = 0
        self.nx = 0
        self.ny = 0
        self.Nx = 0
        self.Ny = 0
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.xs = 0
        self.ys = 0
        self.dt = 0
        self.dx = 0
        self.dy = 0
        self.ts = []
        self.points = (self.ts,self.xs,self.ys)
        
        self.domain_vars = {'nt': self.nt,
                            'nx': self.nx,
                            'ny': self.ny,
                            'Nx': self.Nx,
                            'Ny': self.Ny,
                            'xmin': self.xmin,
                            'xmax': self.xmax,
                            'ymin': self.ymin,
                            'ymax': self.ymax,
                            'x': self.xs,
                            'y': self.ys,
                            'dt': self.dt,
                            'dx': self.dx,
                            'dy': self.dy}
 
        # Define fluid variables for both the fine and coarse data
        self.vxs = np.zeros((self.nt, self.nx, self.ny))
        self.vys = np.zeros((self.nt, self.nx, self.ny))
        self.uts = np.zeros((self.nt, self.nx, self.ny))
        self.uxs = np.zeros((self.nt, self.nx, self.ny))
        self.uys = np.zeros((self.nt, self.nx, self.ny))
        self.ns = np.zeros((self.nt, self.nx, self.ny))
        self.rhos = np.zeros((self.nt, self.nx, self.ny))
        self.ps = np.zeros((self.nt, self.nx, self.ny))
        self.Ws = np.zeros((self.nt, self.nx, self.ny))
        self.Ts = np.zeros((self.nt, self.nx, self.ny))
        self.hs = np.zeros((self.nt, self.nx, self.ny))
        self.Id_SETs = np.zeros((self.nt, self.nx, self.ny, 3, 3))
        
        self.prim_vars = {'v1': self.vxs,
                          'v2': self.vys,
                          'p': self.ps,
                          'rho': self.rhos,
                          'n': self.ns,
                          'u_t': self.uts,
                          'u_x': self.uxs,
                          'u_y': self.uys,
                          'Id_SET': self.Id_SETs}

        self.aux_vars = {'W': self.Ws,
                         'T': self.Ts,
                         'h': self.hs}

        self.prim_vars_strs = ['v1','v2','p','rho','n']
        self.aux_vars_strs= ['W','T','h']

        # def read_data(file_reader):
        #     file_reader = METHOD(self, './Data/Testing/')
        
        self.interpolator

    def find_observer(self, coordinate, residual):
        """
        Main function.
        Finds the meso-observers, U, that the fluid has no drift with respect to.
    
        Parameters
        ----------
        t_range, x_range, y_range : lists of 2 floats
            Define the coordinate ranges of the points to find observers at.
        L : Float
            Filtering lengthscale.
        n_ts, n_xs, n_ys : integers
            Number of points to find observers at in (t,x,y) dimensions.
        initial_guess (DEPRECATED): list of floats.
            An initial guess for U. Much simpler and still robust to just use 
            the micro velocity, u, at a given point at the initial guess.
    
        Returns
        -------
        list of coordinates, Us, and minimization errors.
    
        """
        u, n = self.interpolate_u_n_point(coordinate)
        initial_guess_vx_vy = [u[1]/u[0], u[2]/u[0]]
        try:
            sol = minimize(residual,x0=guess_vx_vy,args=(coordinate,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
            # Large error in root-find
            if (sol.fun > 1e-5):
                print("Warning! Residual is large: ",sol.fun)
        except:
            print("Failed for ", coordinate)
        finally:
            pass
        return sol
    
    def interpolate_u_n_point(self, point):
        """
        Same as interpolate_u_n_coords but takes a list [t,x,y] as a 'point'.
        Yes, one of these two functions is completely redundant...
        """
        n_interpd = interpn(self.points,self.ns,point)
        vx_interpd = interpn(self.points,self.vxs,point)
        vy_interpd = interpn(self.points,self.vys,point)
        W_interpd = 1/np.sqrt(1 - (vx_interpd**2 + vy_interpd**2))
        u_interpd = W_interpd, vx_interpd, vy_interpd
        return [u_interpd[0][0], u_interpd[1][0], u_interpd[2][0]], n_interpd[0]

MicroModel = IdealHydro()
FileReader = METHOD(MicroModel, './Data/Testing/Rotor_2D/')
















