# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

import numpy as np
from scipy.integrate import quad
from multiprocessing import Process, Pool
from system.BaseFunctionality import *
import pickle

class NonIdealHydro2D(object):

    def __init__(self, MicroModel, Filter, interp_method = "linear"):
        self.Filter = Filter
        self.MicroModel = MicroModel
        #self.Observers = Observers
        self.spatial_dims = 2
        self.interp_method = interp_method
        
        self.domain_var_strs = ('Nt','Nx','Ny','dT','dX','dY','points')
        self.domain_vars = dict.fromkeys(self.domain_var_strs)
        
        #Dictionary for 'local' variables, 
        #obtained from filtering the appropriate MicroModel variables
        self.filter_var_strs = ("n","SET")
        self.filter_vars = dict.fromkeys(self.filter_var_strs)

        #Dictionary for MesoModel variables
        self.meso_var_strs = ("U","U_coords","U_errors","T~")
        self.meso_vars = dict.fromkeys(self.meso_var_strs)

        #Strings for 'non-local' variables - ones we need to take derivatives of
        self.nonlocal_var_strs = ("U","T~")

        #Dictionary for derivative variables - calculated by finite differencing
        self.deriv_var_strs = ("dtU","dxU","dyU","dtT~","dxT~","dyT~")
        self.deriv_vars = dict.fromkeys(self.deriv_var_strs)

        #Dictionary for dissipative residuals - from the NonId-SET that we take
        #projections of
        self.diss_residual_strs = ("Pi","q","pi")
        self.diss_residuals = dict.fromkeys(self.diss_residual_strs)

        self.diss_var_strs = ("Theta","Omega","Sigma")
        self.diss_vars = dict.fromkeys(self.diss_residual_strs)

        self.diss_coeff_strs = ("Zeta", "Kappa", "Eta")
        self.diss_coeffs = dict.fromkeys(self.diss_coeff_strs)
        
        self.coefficient_strs = ("Gamma")
        self.coefficients = dict.fromkeys(self.diss_coeff_strs)
        self.coefficients['Gamma'] = 4.0/3.0

        #Dictionary for all vars - useful for e.g. plotting
        self.all_var_strs = self.filter_var_strs + self.meso_var_strs + self.deriv_var_strs\
                        + self.diss_residual_strs + self.diss_var_strs + self.diss_coeff_strs
        
        # Stencils for finite-differencing
        self.cen_SO_stencil = [1/12, -2/3, 0, 2/3, -1/12]
        self.cen_FO_stencil = [-1/2, 0, 1/2]
        self.fw_FO_stencil = [-1, 1]
        self.bw_FO_stencil = [-1, 1]

        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        # Run some compatability test...
        compatible = True
        if self.spatial_dims != self.MicroModel.spatial_dims:
            compatible = False
        for filter_var_str in self.filter_var_strs:
            if not (filter_var_str in MicroModel.vars.keys()):
                compatible = False

        if compatible:
            print("Meso and Micro models are compatible!")
        else:
            print("Meso and Micro models are incompatible!")        

    def get_all_var_strs(self):
        return self.all_var_strs
    
    def get_model_name(self):
        return 'NonIdealHydro2D'
    
    def find_observers(self, num_points, ranges, spacing):
        self.meso_vars['U_coords'], self.meso_vars['U'], self.meso_vars['U_errors'] = \
            self.Filter.find_observers(num_points, ranges, spacing)[0]
        self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny'] = num_points[:]
        self.domain_vars['dT'] = (ranges[0][-1] - ranges[0][0]) / self.domain_vars['Nt']
        self.domain_vars['dX'] = (ranges[1][-1] - ranges[1][0]) / self.domain_vars['Nx']
        self.domain_vars['dY'] = (ranges[2][-1] - ranges[2][0]) / self.domain_vars['Ny']
        self.domain_vars['points'] = [np.linspace(ranges[0][0], ranges[0][-1], num_points[0]),\
                                      np.linspace(ranges[1][0], ranges[1][-1], num_points[1]),\
                                      np.linspace(ranges[2][0], ranges[2][-1], num_points[2])]
    
    def setup_variables(self):
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        n_dims = self.spatial_dims+1
        self.meso_vars['U_coords'] = np.array(self.meso_vars['U_coords']).reshape([Nt, Nx, Ny, n_dims])
        self.meso_vars['U'] = np.array(self.meso_vars['U']).reshape([Nt, Nx, Ny, n_dims])
        self.meso_vars['U_errors'] = np.array(self.meso_vars['U_errors']).reshape([Nt, Nx, Ny])
        self.meso_vars['T~'] = np.zeros((Nt, Nx, Ny))

        self.filter_vars['n'] = np.zeros((Nt, Nx, Ny))
        self.filter_vars['SET'] = np.zeros((Nt, Nx, Ny,n_dims,n_dims))
        
        for nonlocal_var_str in self.nonlocal_var_strs:
            self.deriv_vars['dt'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])
            self.deriv_vars['dx'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])
            self.deriv_vars['dy'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])

        self.diss_residuals['Pi'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_vars['Theta'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_residuals['q'] = np.zeros((Nt, Nx, Ny,n_dims)) 
        self.diss_vars['Omega'] = np.zeros((Nt, Nx, Ny,n_dims)) 
        self.diss_residuals['pi'] = np.zeros((Nt, Nx, Ny,n_dims,n_dims)) 
        self.diss_vars['Sigma'] = np.zeros((Nt, Nx, Ny,n_dims,n_dims)) 

        # Single value for each coefficient (per data point) for now...
        self.diss_coeffs['Zeta'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_coeffs['Kappa'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_coeffs['Eta'] = np.zeros((Nt, Nx, Ny))
        
        self.vars = self.filter_vars
        self.vars.update(self.meso_vars)
        self.vars.update(self.deriv_vars)
        self.vars.update(self.diss_residuals)
        self.vars.update(self.diss_vars)
        self.vars.update(self.diss_coeffs)        
        
    def p_from_EoS(self, rho, n):
        """
        Calculate pressure from EoS using rho (energy density) and n (number density)
        """
        p = (self.coefficients['Gamma']-1)*(rho-n)
        return p
       
    def filter_micro_variables(self):
        """
        'Spatially' average required variables from the micromodel w.r.t.
        the observers that have been found.
        """
        for h in range(self.domain_vars['Nt']):
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                        self.filter_vars['n'][h,i,j] =\
                            self.Filter.filter_prim_var(self.meso_vars['U_coords'][h,i,j], self.meso_vars['U'][h,i,j], 'n')
                        self.filter_vars['SET'][h,i,j] =\
                            self.Filter.filter_struc(self.meso_vars['U_coords'][h,i,j], self.meso_vars['U'][h,i,j], 'SET')
                    
        # Should be able to convert here to only 1 filter function... (not prim/struct)
        # for filter_var_str in self.filter_var_strs:
        #     filter_args = [(coord, U, filter_var_str) for coord, U in zip(self.U_coords, self.Us)]
        #     with Pool(2) as p:
        #         self.micro_vars[micro_var_str] = p.starmap(self.Filter.filter_var, filter_args)
        #         self.micro_vars[micro_var_str] = p.starmap(self.Filter.filter_prim_var, filter_args)
    
    def calculate_derivatives(self):
        """
        Calculate the required derivatives of MesoModel variables for 
        constructing the non-ideal terms.
        """
        for nonlocal_var_str in self.nonlocal_var_strs:
            self.calculate_time_derivatives(nonlocal_var_str)
            self.calculate_x_derivatives(nonlocal_var_str)
            self.calculate_y_derivatives(nonlocal_var_str)
    
    def calculate_time_derivatives(self, nonlocal_var_str):
        deriv_var_str = 'dt'+nonlocal_var_str
        for h in range(self.domain_vars['Nt']):
            if h == 0:
                stencil = self.fw_FO_stencil
                samples = [0,1]
            elif h == (self.domain_vars['Nt']-1):
                stencil = self.fw_FO_stencil
                samples = [-1,0]
            else:
                stencil = self.cen_FO_stencil
                samples = [-1,0,1]
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                    for s in range(len(samples)):
                        self.deriv_vars[deriv_var_str][h,i,j] \
                        += (stencil[s]*self.meso_var_strs[nonlocal_var_str][h+samples[s],i,j]) / self.domain_vars['dT']
                        
    def calculate_x_derivatives(self, nonlocal_var_str):
        deriv_var_str = 'dx'+nonlocal_var_str
        for h in range(self.domain_vars['Nt']):
          for i in range(self.domain_vars['Nx']):
              if i == 0:
                  stencil = self.fw_FO_stencil
                  samples = [0,1]
              elif i == (self.domain_vars['Nx']-1):
                  stencil = self.fw_FO_stencil
                  samples = [-1,0]
              else:
                  stencil = self.cen_FO_stencil
                  samples = [-1,0,1]
              for j in range(self.domain_vars['Ny']):
                  for s in range(len(samples)):
                      self.deriv_vars[deriv_var_str][h,i,j] \
                      += (stencil[s]*self.meso_var_strs[nonlocal_var_str][h,i+samples[s],j]) / self.domain_vars['dX']
                      
    def calculate_y_derivatives(self, nonlocal_var_str):
        deriv_var_str = 'dy'+nonlocal_var_str
        for h in range(self.domain_vars['Nt']):
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                    if j == 0:
                        stencil = self.fw_FO_stencil
                        samples = [0,1]
                    elif j == (self.domain_vars['Ny']-1):
                        stencil = self.fw_FO_stencil
                        samples = [-1,0]
                    else:
                        stencil = self.cen_FO_stencil
                        samples = [-1,0,1]   
                    for s in range(len(samples)):
                        self.deriv_vars[deriv_var_str][h,i,j] \
                        += (stencil[s]*self.meso_var_strs[nonlocal_var_str][h,i,j+samples[s]]) / self.domain_vars['dY']                  

    def calculate_dissipative_residuals(self, h, i, j):
        """
        Returns the inferred (residual) values of the 
        dissipative terms from the filtered MicroModel SET.        

        Parameters
        ----------
        indices : TYPE
            DESCRIPTION.

        Returns
        -------
        Pi_res : scalar float
            Bulk viscosity
        q_res : (d+1) vector of floats
            Heat flux.
        pi_res : (d+1)x(d+1) tensor of floats
            Shear viscosity.

        """
        # Move this
        # h, i, j = indices
        # Filter the scalar fields
        N = self.filter_vars['n'][h,i,j]
        Id_SET = self.filter_vars['SET'][h,i,j]
        U = self.meso_vars['U'][h,i,j]
        
        # Do required projections of SET
        h_mu_nu = Base.orthogonal_projector(U, self.metric)
        rho_res = Base.project_tensor(U,U,Id_SET)

        # Set Meso temperature from filtered quantities using EoS
        p_tilde = self.p_from_EoS(rho_res, N)
        T_tilde = p_tilde/N
        self.meso_vars['T~'][h,i,j] = T_tilde
        
        # Calculate dissipative residual with tensor manipulation
        q_res = np.einsum('ij,i,jk',Id_SET,U,h_mu_nu)            
        tau_res = np.einsum('ij,ik,jl',Id_SET,h_mu_nu,h_mu_nu)
        tau_trace = np.trace(tau_res)
        Pi_res = tau_trace - p_tilde
        pi_res = tau_res - np.dot((p_tilde + Pi_res),h_mu_nu)

        self.diss_residuals['Pi'][h,i,j] = Pi_res
        self.diss_residuals['q'][h,i,j] = q_res
        self.diss_residuals['pi'][h,i,j] = pi_res

    def calculate_dissipative_variables(self, h, i, j):
        """
        Calculates the non-ideal, dissipation terms (without coefficeints)
        for the non-ideal MesoModel SET.

        Parameters
        ----------
        indices : list of ints
             Indices of data-point.
        coord : list of floats
             Coordinates in (t,x,y).

        Returns
        -------
        Decomposition of the observer velocity:
            
        Theta : float (scalar)
            Divergence (isotropic).
        omega : vector of floats
            Transverse momentum.
        sigma : (d+1)x(d+1) tensor of floats
            Symmetric, trace-free.

        """
        # h, i, j = indices

        T = self.meso_vars['T~'][h, i, j]
        dtT = self.deriv_vars['dtT~'][h, i, j]
        dxT = self.deriv_vars['dxT~'][h, i, j]
        dyT = self.deriv_vars['dyT~'][h, i, j]

        # U = self.meso_vars['U'][h,i,j][:]
        Ut, Ux, Uy = self.meso_vars['U'][h,i,j][:]

        # dtU = self.deriv_vars['dtU'][h,i,j]
        dtUt, dtUx, dtUy = self.deriv_vars['dtU'][h,i,j][:]
        dxUt, dxUx, dxUy = self.deriv_vars['dxU'][h,i,j][:]
        dyUt, dyUx, dyUy = self.deriv_vars['dyU'][h,i,j][:]
        
        # Need to do this with Einsum...
        Theta = dtUt + dxUx + dyUy
        a = np.array([Ut*dtUt + Ux*dxUt + Uy*dyUt, Ut*dtUx + Ux*dxUx + Uy*dyUx, Ut*dtUy + Ux*dxUy + Uy*dyUy])

        Omega = np.array([dtT, dxT, dyT]) + np.multiply(T,a)
        Sigma = np.array([[2*dtUt - (2/3)*Theta, dtUx + dxUt, dtUy + dyUt],\
                                                  [dxUt + dtUx, 2*dxUx - (2/3)*Theta, dxUy + dyUx],
                                                  [dyUt + dtUy, dyUx + dxUy, 2*dyUy - (2/3)*Theta]])
        self.diss_vars['Theta'][h,i,j] = Theta
        self.diss_vars['Omega'][h,i,j] = Omega
        self.diss_vars['Sigma'][h,i,j] = Sigma
        
    def calculate_dissipative_coefficients(self):
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        # parallel_args = [(h, i, j) for h in range(Nt) for i in range(Nx) for j in range(Ny)]
        # # print(parallel_args)
        # with Pool(2) as p:
        #     p.starmap(self.calculate_dissipative_residuals, parallel_args)
        #     p.starmap(self.calculate_dissipative_variables, parallel_args)

        # self.diss_coeffs['Zeta'] = -self.diss_residuals['Pi'] / self.diss_vars['Theta']
        for h in range(self.domain_vars['Nt']):
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                      self.calculate_dissipative_residuals(h,i,j)
                      self.calculate_dissipative_variables(h,i,j)
                      self.diss_coeffs['Zeta'][h,i,j] = -self.diss_residuals['Pi'][h,i,j] / self.diss_vars['Theta'][h,i,j]
                    
        for diss_coeff_str in self.diss_coeff_strs:
            coeffs_handle = open(diss_coeff_str+'.pickle', 'wb')
            pickle.dump(self.diss_coeffs[diss_coeff_str], coeffs_handle, protocol=pickle.HIGHEST_PROTOCOL)       
        
