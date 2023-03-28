# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:13:07 2023

@author: Marcus
"""

from FileReaders import *
import numpy as np

class IdealHydro(object):

    def __init__(self):

        self.nx = 0
        self.ny = 0
        self.xs = 0
        self.ys = 0
        self.dx = 0
        self.dy = 0
        self.ts = []
        self.points = (self.ts,self.xs,self.ys)


        
        self.domain_vars = {'nx': self.nx,
                            'ny': self.ny,
                            'x': self.xs,
                            'y': self.ys,
                            'dx': self.dx,
                            'dy': self.dy}
 
        # Define fluid variables for both the fine and coarse data
        self.vxs = np.zeros((num_files, self.nx, self.ny))
        self.vys = np.zeros((num_files, self.nx, self.ny))
        self.uts = np.zeros((num_files, self.nx, self.ny))
        self.uxs = np.zeros((num_files, self.nx, self.ny))
        self.uys = np.zeros((num_files, self.nx, self.ny))
        self.ns = np.zeros((num_files, self.nx, self.ny))
        self.rhos = np.zeros((num_files, self.nx, self.ny))
        self.ps = np.zeros((num_files, self.nx, self.ny))
        self.Ws = np.zeros((num_files, self.nx, self.ny))
        self.Ts = np.zeros((num_files, self.nx, self.ny))
        self.hs = np.zeros((num_files, self.nx, self.ny))
        self.Id_SETs = np.zeros((num_files, self.nx, self.ny, 3, 3))
        
        self.prim_vars = {'v1': self.vxs,
                          'v2': self.vys,
                          'n': self.ns,
                          'rho': self.rhos,
                          'p': self.ps,
                          'W': self.Ws,
                          'u_t': self.uts,
                          'u_x': self.uxs,
                          'u_y': self.uys,
                          'T': self.Ts,
                          'Id_SET': self.Id_SETs}

        self.aux_vars = {'T': self.Ts,
                         'W': self.Ws,
                         'h': self.hs}

        self.prim_vars_strs = ['v1','v2','p','rho','n']
        self.aux_vars_strs= ['W','T','h']

        # def read_data(file_reader):
        #     file_reader = METHOD(self, './Data/Testing/')

        def find_observer(self, coordinate, residual):
        def find_observers(self, t_range,x_range,y_range,L,n_ts,n_xs,n_ys,initial_guess):
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
        t_coords = np.linspace(t_range[0],t_range[-1],n_ts)
        x_coords = np.linspace(x_range[0],x_range[-1],n_xs)
        y_coords = np.linspace(y_range[0],y_range[-1],n_ys)
        funs = []
        vectors = []
        coord_list = []
        for t in t_coords:
            for x in x_coords:
                for y in y_coords:
                    u, n = self.interpolate_u_n_coords(t,x,y)
                    # guess_vx_vy = initial_guess
                    guess_vx_vy = [u[1]/u[0], u[2]/u[0]]
                    coords = [t,x,y]
                    #sol = minimize(self.residual_ib,x0=guess_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
                    #vectors.append(self.get_U_mu(sol.x))
                    #funs.append(sol.fun)
                    #coord_list.append(coords)
                    #guess_vx_vy = [sol.x[0],sol.x[1]]
                    try:
                        sol = minimize(self.residual_ib,x0=guess_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
                        vectors.append(self.get_U_mu(sol.x))
                        funs.append(sol.fun)
                        coord_list.append(coords)
                        #guess_vx_vy = [sol.x[0],sol.x[1]]
                        if (sol.fun > 1e-5):
                            print("Warning! Residual is large: ",sol.fun)
                    except:
                        print("Failed for ",coords)
                    finally:
                        pass
        # f_to_write.write(str(coord_list)+str(vectors)+str(funs))
        # DON'T THINK THIS WAS DOING ANYTHING??
        # with open('KH_observers.pickle', 'wb') as handle:
        #     pickle.dump(np.array([coord_list, vectors, funs]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [coord_list, vectors, funs]            


IHD = IdealHydro()

















