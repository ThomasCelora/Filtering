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
        # self.vxs = np.zeros((self.nt, self.nx, self.ny))
        # self.vys = np.zeros((self.nt, self.nx, self.ny))
        # self.uts = np.zeros((self.nt, self.nx, self.ny))
        # self.uxs = np.zeros((self.nt, self.nx, self.ny))
        # self.uys = np.zeros((self.nt, self.nx, self.ny))
        # self.ns = np.zeros((self.nt, self.nx, self.ny))
        # self.rhos = np.zeros((self.nt, self.nx, self.ny))
        # self.ps = np.zeros((self.nt, self.nx, self.ny))
        # self.Ws = np.zeros((self.nt, self.nx, self.ny))
        # self.Ts = np.zeros((self.nt, self.nx, self.ny))
        # self.hs = np.zeros((self.nt, self.nx, self.ny))
        # self.Id_SETs = np.zeros((self.nt, self.nx, self.ny, 3, 3))
        
        self.vxs = []
        self.vys = []
        self.ps = []
        self.ns = []
        self.rhos = []

        self.prim_vars = {'v1': self.vxs,
                          'v2': self.vys,
                          'p': self.ps,
                          'rho': self.rhos,
                          'n': self.ns}

        self.uts = []
        self.uxs = []
        self.uys = []
        self.Id_SETs = []
        
        self.vars =       {'u_t': self.uts,
                          'u_x': self.uxs,
                          'u_y': self.uys,
                          'Id_SET': self.Id_SETs}

        self.Ws = []
        self.Ts = []
        self.hs = []
        
        self.aux_vars = {'W': self.Ws,
                         'T': self.Ts,
                         'h': self.hs}

        self.prim_vars_strs = ['v1','v2','p','rho','n']
        self.var_strs = ['u_t','u_x','u_y']
        self.aux_vars_strs= ['W','T','h']

        # Define Minkowski metric
        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

    def setup(self):
        self.uts = self.Ws[:]
        self.uxs = np.zeros((self.nt,self.nx,self.ny))
        self.uys = np.zeros((self.nt,self.nx,self.ny))
        self.Id_SETs = np.zeros((self.nt,self.nx,self.ny,3,3))
        for h in range(self.nt):
            for i in range(self.nx):
                for j in range(self.ny):
                    self.uts[h][i,j] = self.Ws[h][i,j]
                    self.uxs[h][i,j] = self.Ws[h][i,j]*self.vxs[h][i,j]
                    self.uys[h][i,j] = self.Ws[h][i,j]*self.vys[h][i,j]
                    u_vec = np.array([self.uts[h][i,j],self.uxs[h][i,j],self.uys[h][i,j]])
                    self.Id_SETs[h][i,j] = self.rhos[h][i,j]*np.outer(u_vec,u_vec)\
                        + self.ps[h][i,j]*(self.metric + np.outer(u_vec,u_vec))

    def find_observer(self, point, residual):
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
        u, n = self.interpolate_u_n_point(point)
        initial_guess_vx_vy = [u[1]/u[0], u[2]/u[0]]
        try:
            sol = minimize(residual,x0=guess_vx_vy,args=(point,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
            # Large error in root-find
            if (sol.fun > 1e-5):
                print("Warning! Residual is large: ",sol.fun)
        except:
            print("Failed for ", coordinate)
        finally:
            pass
        return sol
    

    def interpolate_var(self, point, var_str):
        return interpn(self.points,self.vars[var_str],point)[0]


















