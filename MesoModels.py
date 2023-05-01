# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

import numpy as np
from scipy.integrate import quad
from multiprocessing import Process, Pool

class NonIdealHydro2D(object):

    def __init__(self, MicroModel, Filter):
        self.Filter = Filter
        self.MicroModel = MicroModel
        #self.Observers = Observers
        self.spatial_dims = 2

        # Run some compatability test...
        compatible = True
        if(compatible):
            print("Meso and Micro models are compatible!")
        else:
            print("Meso and Micro models are incompatible!")
            
        #Dictionary for 'local' variables - ones we won't need derivatives of
        self.local_var_strs = ("n","SET")
        self.local_vars = dict.fromkeys(self.local_var_strs)
        for str in self.local_vars:
            self.local_vars[str] = []

        #Dictionary for 'non-local' variables - ones we need to take derivatives of
        self.nonlocal_var_strs = ("T_tilde","U")
        self.nonlocal_vars = dict.fromkeys(self.nonlocal_var_strs)
        for str in self.nonlocal_var_strs:
            self.nonlocal_vars[str] = []

    def find_observers(self, num_points, ranges, spacing):
        self.U_coords, self.Us, self.U_errors = self.Filter.find_observers(num_points, ranges, spacing)[0]
            
    def p_from_EoS(self, rho, n):
        """
        Calculate pressure from EoS
        """
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p
    
       
    def filter_variables(self):
        # Filter necessary variables from the micromodel
        filter_args = [(coord, U) for coord, U in zip(self.U_coords, self.Us)]
        print(filter_args)
        for local_var_str in self.local_var_strs:
            with Pool(2) as p:
                self.local_vars[local_var_str] = p.starmap(self.Filter.filter_var, filter_args)
 
        for coord, U in zip(self.U_coords, self.Us):
            self.Ns = self.Filter.filter_var(coord, U, 'n')
            self.SETs = self.Filter.filter_var(coord, U, 'SET')
    
    def calculate_nonlocal_variables(self, coords):
        # Size of box for spatial filtering 
        # the numerical coefficient ~ determines #cells along side of filtering box
        self.L = 4*np.sqrt(self.dx*self.dy) 
        self.dT = 0.01 # steps to take for differential calculations
        self.dX = 0.01
        self.dY = 0.01
        # Stencils for finite-differencing
        self.cen_SO_stencil = [1/12, -2/3, 0, 2/3, -1/12]
        self.cen_FO_stencil = [-1/2, 0, 1/2]
        self.fw_FO_stencil = [-1, 1]
        self.bw_FO_stencil = [-1, 1]
        
        # Calculate time derivatives for central slices
        for t_slice in range(self.n_obs_t):
            # Central-difference                                
            for i in range(1,self.n_obs_x-1):
                for j in range(1,self.n_obs_y-1): # fix these to use self.n_x_pts etc.
                    self.dtUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[t_slice][i][j] / self.dt_obs
                    self.dtUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[t_slice][i][j] / self.dt_obs
                    self.dtUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[t_slice][i][j] / self.dt_obs
                    # pick out central slice with first [1]    
                    self.dxUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[1][i-1+t_slice][j] / self.dx_obs
                    self.dxUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[1][i-1+t_slice][j] / self.dx_obs
                    self.dxUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[1][i-1+t_slice][j] / self.dx_obs

                    self.dyUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[1][i][j-1+t_slice] / self.dy_obs
                    self.dyUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[1][i][j-1+t_slice] / self.dy_obs
                    self.dyUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[1][i][j-1+t_slice] / self.dy_obs                    
        
        
    def calculate_dissipative_residuals(self, coord, U):
        # Filter the scalar fields
        N = self.Filter.filter_scalar(coord, U, 'n')

        # Construct filtered Id SET         
        filtered_Id_SET = self.Filter.filter_scalar(coord, U, 'Id_SET')
        
        # Do required projections of SET
        h_mu_nu = self.orthogonal_projector(U)
        rho_res = self.project_tensor(U,U,filtered_Id_SET)
        q_res = np.einsum('ij,i,jk',filtered_Id_SET,U,h_mu_nu)            
        tau_res = np.einsum('ij,ik,jl',filtered_Id_SET,h_mu_nu,h_mu_nu) # tau = p + Pi+ pi
        
        # Calculate Pi and pi residuals
        tau_trace = np.trace(tau_res)#
        p_tilde = self.p_from_EoS(rho_res, N)
        T_tilde = p_tilde/N
        self.T_tildes[h, i, j] = T_tilde
        Pi_res = tau_trace - p_tilde
        pi_res = tau_res - np.dot((p_tilde + Pi_res),h_mu_nu)
        
        return Pi_res, q_res, pi_res

    def calculate_nonlocal_variables(self, coord, indices):
        """
        Calculate the non-ideal, dissipation terms (without coefficeints).

        Parameters
        ----------
        indices : list of ints
             Indices of data-point.
        coord : list of floats
             Coordinates in (t,x,y).

        Returns
        -------
        Theta : float
            Divergence of the observer velocity.
        omega : vector of floats
            Transverse momentum.
        sigma : tensor of floats
            Symmetric, trace-free bla bla.

        """
        h, i, j = indices
        # T = self.values_from_hdf5(coord, 'T') # Fix this - should be from EoS(N,p_tilde)
        T = self.T_tildes[h, i, j]
        dtT = self.calc_t_deriv('T',coord)[0]
        dxT = self.calc_x_deriv('T',coord)[0]
        dyT = self.calc_y_deriv('T',coord)[0]
        # print(T.shape,dxT.shape)
        Ut = self.Uts[h,i,j]
        Ux = self.Uxs[h,i,j]
        Uy = self.Uys[h,i,j]
        dtUt = self.dtUts[h,i,j]
        dtUx = self.dtUxs[h,i,j]
        dtUy = self.dtUys[h,i,j]
        dxUt = self.dxUts[h,i,j]
        dxUx = self.dxUxs[h,i,j]
        dxUy = self.dxUys[h,i,j]
        dyUt = self.dyUts[h,i,j]
        dyUx = self.dyUxs[h,i,j]
        dyUy = self.dyUys[h,i,j]
   
        Theta = dtUt + dxUx + dyUy
        a = np.array([Ut*dtUt + Ux*dxUt + Uy*dyUt, Ut*dtUx + Ux*dxUx + Uy*dyUx, Ut*dtUy + Ux*dxUy + Uy*dyUy])#,ux*dxuz+uy*dyuz+uz*dzuz])

        omega = np.array([dtT, dxT, dyT]) + np.multiply(T,a) # FIX
        sigma = np.array([[2*dtUt - (2/3)*Theta, dtUx + dxUt, dtUy + dyUt],\
                                                  [dxUt + dtUx, 2*dxUx - (2/3)*Theta, dxUy + dyUx],
                                                  [dyUt + dtUy, dyUx + dxUy, 2*dyUy - (2/3)*Theta]])

        return Theta, omega, sigma

        
    def calculate_coefficients(self, point):
        Pi_res, q_res, pi_res = self.calculate_dissipative_residuals(point)
        Theta, omega, sigma = self.calculate_nonlocal_variables(point)
        return -Pi_res/Theta, -q_res/omega, -pi_res/sigma
        
        
class meso_model_example(object):

    def __init__(self, micro_model, constraint, filter):
        pass
        
        
        
        
        
        
        
        
        
        
        
        