# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

import numpy as np
from scipy.interpolate import interpn 
import pickle
from multiprocessing import Process, Pool
from multimethod import multimethod

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scst    

from system.BaseFunctionality import *
from MicroModels import * 
from FileReaders import * 
from Filters import *
from Visualization import *

class NonIdealHydro2D(object):

    def __init__(self, MicroModel, ObsFinder, Filter, interp_method = "linear"):
        self.MicroModel = MicroModel
        self.ObsFinder = ObsFinder
        self.Filter = Filter
        self.spatial_dims = 2
        self.n_dims = self.spatial_dims + 1
        self.interp_method = interp_method
        
        self.domain_var_strs = ('Nt','Nx','Ny','dT','dX','dY','points')
        self.domain_vars = dict.fromkeys(self.domain_var_strs)
        
        #Dictionary for 'local' variables, 
        #obtained from filtering the appropriate MicroModel variables
        self.filter_var_strs = ("BC","SET")
        self.filter_vars = dict.fromkeys(self.filter_var_strs)

        #Dictionary for MesoModel variables
        self.meso_var_strs = ("U","U_coords","U_errors","T~","N")
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
    
    def find_observers(self, num_points, ranges):#, spacing):
        # self.meso_vars['U_coords'], self.meso_vars['U'], self.meso_vars['U_errors'] = \
        #     self.ObsFinder.find_observers(num_points, ranges, spacing)[0]
        self.meso_vars['U_coords'], self.meso_vars['U'], self.meso_vars['U_errors'] = \
            self.ObsFinder.find_observers_ranges(num_points, ranges)[0]
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
        self.meso_vars['U_errors'] = np.array(self.meso_vars['U_errors']).reshape([Nt, Nx, Ny, 2])
        self.meso_vars['T~'] = np.zeros((Nt, Nx, Ny))
        self.meso_vars['N'] = np.zeros((Nt, Nx, Ny))

        self.filter_vars['BC'] = np.zeros((Nt, Nx, Ny, n_dims))
        self.filter_vars['SET'] = np.zeros((Nt, Nx, Ny, n_dims,n_dims))
        
        for nonlocal_var_str in self.nonlocal_var_strs:
            self.deriv_vars['dt'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])
            self.deriv_vars['dx'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])
            self.deriv_vars['dy'+nonlocal_var_str] = np.zeros_like(self.meso_vars[nonlocal_var_str])

        self.diss_residuals['Pi'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_vars['Theta'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_residuals['q'] = np.zeros((Nt, Nx, Ny, n_dims)) 
        self.diss_vars['Omega'] = np.zeros((Nt, Nx, Ny, n_dims)) 
        self.diss_residuals['pi'] = np.zeros((Nt, Nx, Ny, n_dims, n_dims)) 
        self.diss_vars['Sigma'] = np.zeros((Nt, Nx, Ny, n_dims, n_dims)) 

        # Single value for each coefficient (per data point) for now...
        self.diss_coeffs['Zeta'] = np.zeros((Nt, Nx, Ny)) 
        self.diss_coeffs['Kappa'] = np.zeros((Nt, Nx, Ny, n_dims))
        self.diss_coeffs['Eta'] = np.zeros((Nt, Nx, Ny, n_dims, n_dims))
        
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
                        self.filter_vars['BC'][h,i,j] = self.Filter.filter_var_point('BC', self.meso_vars['U_coords'][h,i,j],
                                                     self.meso_vars['U'][h,i,j])
                        self.filter_vars['SET'][h,i,j] = self.Filter.filter_var_point('SET', self.meso_vars['U_coords'][h,i,j],
                                                     self.meso_vars['U'][h,i,j])

                        # self.filter_vars['n'][h,i,j] =\
                        #     self.Filter.filter_prim_var(self.meso_vars['U_coords'][h,i,j], self.meso_vars['U'][h,i,j], 'n')
                        # self.filter_vars['SET'][h,i,j] =\
                        #     self.Filter.filter_struc(self.meso_vars['U_coords'][h,i,j], self.meso_vars['U'][h,i,j], 'SET')
                    
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
        stencil = self.cen_FO_stencil
        samples = [-1,0,1]
        for h in range(self.domain_vars['Nt']):
            if h == 0:
                stencil = self.fw_FO_stencil
                samples = [0,1]
            if h == (self.domain_vars['Nt']-1):
                stencil = self.fw_FO_stencil
                samples = [-1,0]
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                    for s in range(len(samples)):
                        self.deriv_vars[deriv_var_str][h,i,j] \
                        += (stencil[s]*self.meso_vars[nonlocal_var_str][h+samples[s],i,j]) / self.domain_vars['dT']
                        
    def calculate_x_derivatives(self, nonlocal_var_str):
        deriv_var_str = 'dx'+nonlocal_var_str
        stencil = self.cen_FO_stencil
        samples = [-1,0,1]
        for h in range(self.domain_vars['Nt']):
          for i in range(self.domain_vars['Nx']):
              if i == 0:
                  stencil = self.fw_FO_stencil
                  samples = [0,1]
              if i == (self.domain_vars['Nx']-1):
                  stencil = self.fw_FO_stencil
                  samples = [-1,0]
              for j in range(self.domain_vars['Ny']):
                  for s in range(len(samples)):
                      # print((stencil[s]*self.meso_vars[nonlocal_var_str][h,i+samples[s],j]) / self.domain_vars['dX'])
                      self.deriv_vars[deriv_var_str][h,i,j] \
                      += (stencil[s]*self.meso_vars[nonlocal_var_str][h,i+samples[s],j]) / self.domain_vars['dX']
                      
    def calculate_y_derivatives(self, nonlocal_var_str):
        deriv_var_str = 'dy'+nonlocal_var_str
        stencil = self.cen_FO_stencil
        samples = [-1,0,1]
        for h in range(self.domain_vars['Nt']):
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):
                    if j == 0:
                        stencil = self.fw_FO_stencil
                        samples = [0,1]
                    if j == (self.domain_vars['Ny']-1):
                        stencil = self.fw_FO_stencil
                        samples = [-1,0]
                    for s in range(len(samples)):
                        self.deriv_vars[deriv_var_str][h,i,j] \
                        += (stencil[s]*self.meso_vars[nonlocal_var_str][h,i,j+samples[s]]) / self.domain_vars['dY']                  

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
        BC = self.filter_vars['BC'][h,i,j]
        N = np.sqrt(-Base.Mink_dot(BC, BC))
        self.meso_vars['N'][h,i,j] = N
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
        
        # print(Pi_res, q_res, pi_res)

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

        # print(Ut, Ux, Uy)

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
        
        # Derivative calculations need to be done after all residuals
        self.calculate_derivatives()
        for h in range(self.domain_vars['Nt']):
            for i in range(self.domain_vars['Nx']):
                for j in range(self.domain_vars['Ny']):                     
                      self.calculate_dissipative_variables(h,i,j)
                      self.diss_coeffs['Zeta'][h,i,j] = -self.diss_residuals['Pi'][h,i,j] / self.diss_vars['Theta'][h,i,j]
                      for k in range(self.n_dims):
                          self.diss_coeffs['Kappa'][h,i,j,k] = -self.diss_residuals['q'][h,i,j,k] / self.diss_vars['Omega'][h,i,j,k]
                          for l in range(self.n_dims):
                              self.diss_coeffs['Eta'][h,i,j,k,l] = -self.diss_residuals['pi'][h,i,j,k,l] / self.diss_vars['Sigma'][h,i,j,k,l]
                    
        for diss_coeff_str in self.diss_coeff_strs:
            coeffs_handle = open(diss_coeff_str+'.pickle', 'wb')
            pickle.dump(self.diss_coeffs[diss_coeff_str], coeffs_handle, protocol=pickle.HIGHEST_PROTOCOL)       
        
class resMHD2D(object):
    """
    Idea: alternative meso_models (e.g. with different closures scheme) will 
    only have to change the method model_residuals and list of non_local_vars 
    (possibly but less likely decompose_structure_gridpoint)
    """
    def __init__(self, micro_model, find_obs, filter, interp_method = 'linear'):
        """
        ADD DESCRIPTION OF THIS
        """
        self.micro_model = micro_model
        self.find_obs = find_obs
        self.filter = filter
        self.spatial_dims = 2
        self.interp_method = interp_method

        self.domain_int_strs = ('Nt','Nx','Ny')
        self.domain_float_strs = ("Tmin","Tmax","Xmin","Xmax","Ymin","Ymax","Dt","Dx","Dy")
        self.domain_array_strs = ("T","X","Y","Points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs+self.domain_array_strs)
        for var in self.domain_vars: 
            self.domain_vars[var] = []

        # This is the Levi-Civita symbol, not tensor, so be careful when using it 
        self.Levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
                      for k in range(3)]for j in range(3) ] for i in range(3) ])
        
        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        self.filter_vars_strs = ['U', 'U_errors', 'U_success']
        self.filter_vars = dict.fromkeys(self.filter_vars_strs)
        for var in self.filter_vars:
            var = []

        self.meso_structures_strs  = ['SETfl', 'SETem', 'BC', 'Fab']
        self.meso_structures = dict.fromkeys(self.meso_structures_strs) 
        for var in self.meso_structures:
                self.meso_structures[var] = []

        self.meso_scalars_strs = ['eps_tilde', 'n_tilde', 'p', 'eos_res', 'Pi_res', 'sigma_tilde', 'b_tilde']
        self.meso_vectors_strs = ['u_tilde', 'q_res', 'e_tilde', 'j', 'J_tilde']
        self.meso_r2tensors_strs = ['pi_res']
        self.meso_vars_strs = self.meso_scalars_strs + self.meso_vectors_strs + self.meso_r2tensors_strs
        self.meso_vars = dict.fromkeys(self.meso_vars_strs)

        # Dictionary with stencils and coefficients for finite differencing
        self.differencing = dict.fromkeys((1, 2))
        self.differencing[1] = dict.fromkeys(['fw', 'bw', 'cen']) 
        self.differencing[1]['fw'] = {'coefficients' : [-1, 1] , 'stencil' : [0, 1]}
        self.differencing[1]['bw'] = {'coefficients' : [1, -1] , 'stencil' : [0, -1]}
        self.differencing[1]['cen'] = {'coefficients' : [-1/2., 0, 1/2.] , 'stencil' : [-1, 0, 1]}
        self.differencing[2] = dict.fromkeys(['fw', 'bw', 'cen']) 
        self.differencing[2]['fw'] = {'coefficients' : [-3/2., 2., -1/2.] , 'stencil' : [0, 1, 2]}
        self.differencing[2]['bw'] = {'coefficients' : [3/2., -2., 1/2.] , 'stencil' : [0, -1, -2]}
        self.differencing[2]['cen'] = {'coefficients' : [1/12., -2/3., 0., 2/3., -1/12.] , 'stencil' : [-2, -1, 0, 1, 2]}


        # dictionary with non-local quantities (keys must match one of meso_vars or structure)
        self.nonlocal_vars_strs = ['u_tilde', 'Fab'] 
        Dstrs = ['D_' + i for i in self.nonlocal_vars_strs]
        self.deriv_vars = dict.fromkeys(Dstrs)

        # Additional, closure-scheme specific vars must be added appropriately using model_residuals()

        # self.all_var_strs = self.meso_vars_strs  + Dstrs + self.meso_structures_strs
        # Run some compatibility test... 
        compatible = True
        error = ''
        if self.spatial_dims != micro_model.get_spatial_dims(): 
            compatible = False
            error += '\nError: different dimensions.'
        for struct in self.meso_structures_strs:
            if struct not in self.micro_model.get_structures_strs():
                compatible = False
                error += f'\nError: {struct} not in micro_model!'

        if not compatible:
            print("Meso and Micro models are incompatible:"+error) 

    def get_all_var_strs(self): 
        return list(self.meso_vars.keys())  + list(self.deriv_vars.keys()) + list(self.meso_structures.keys())

    def get_gridpoints(self): 
        """
        Pretty self-explanatory
        """
        return self.domain_vars['Points']

    def get_interpol_var(self, var, point):
        """
        Returns the interpolated variables at the point.

        Parameters
        ----------
        var: str corresponding to meso_structure, meso_vars or deriv_vars
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Interpolated values/arrays corresponding to variable. 
        Empty list if none of the variables is not meso_structures/meso_vars or deriv_vars of the model.

        Notes
        -----
        Interpolation gives errors when applied to boundary 
        """
        if var in self.meso_structures:
            return interpn(self.domain_vars['Points'], self.meso_structures[var], point, method = self.interp_method)[0]
        elif var in self.meso_vars: 
            return interpn(self.domain_vars['Points'], self.meso_vars[var], point, method = self.interp_method)[0]
        elif var in self.deriv_vars: 
            return interpn(self.domain_vars['Points'], self.deriv_vars[var], point, method = self.interp_method)[0]
        else: 
            print('{} does not belong to either meso_structures/meso_vars or deriv_vars. Check!'.format(var))
    
    @multimethod
    def get_var_gridpoint(self, var: str, h: object, i: object, j: object):
        """
        Returns variable corresponding to input 'var' at gridpoint 
        identified by h,i,j

        Parameters:
        -----------
        var: string corresponding to structure, meso_vars or deriv_vars

        h,i,j: int
            integers corresponding to position on the grid. 

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest 
        grid-point to input 'point'. 

        Notes:
        ------
        This method is useful e.g. for plotting the raw data. 
        """
        if var in self.meso_structures:
            return self.meso_structures[var][h,i,j]  
        elif var in self.meso_vars:
            return self.meso_vars[var][h,i,j]  
        elif var in self.deriv_vars:
            return self.deriv_vars[var][h,i,j]
        else: 
            print("{} is not a variable of resMHD_2D!".format(var))
            return None

    @multimethod
    def get_var_gridpoint(self, var: str, point: object):
        """
        Returns variable corresponding to input 'var' at gridpoint 
        closest to input 'point'.

        Parameters:
        -----------
        vars: string corresponding to structure, meso_vars or deriv_vars

        point: list of 2+1 floats

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest 
        grid-point to input 'point'. 

        Notes:
        ------
        This method should be used in case using interpolated values becomes 
        too expensive. 
        """
        indices = Base.find_nearest_cell(point, self.domain_vars['Points'])
        if var in self.meso_structures:
            return self.meso_structures[var][tuple(indices)]  
        elif var in self.meso_vars:
            return self.meso_vars[var][tuple(indices)]   
        elif var in self.deriv_vars:
            return self.deriv_vars[var][tuple(indices)]
        else: 
            print("{} is not a variable of resMHD_2D!".format(var))
            return None

    def get_model_name(self):
        return 'resMHD2D'
    
    def set_find_obs_method(self, find_obs):
        self.find_obs = find_obs

    def setup_meso_grid(self, patch_bdrs, coarse_factor = 1): 
        """
        Builds the meso_model grid using the micro_model grid points contained in 
        the input patch (defined via 'patch_bdrs'). The method allows for coarse graining 
        the grid in the spatial directions ONLY. 
        Then store the info about the meso grid and set up arrays of definite rank and size 
        for the quantities needed later. 

        Parameters: 
        -----------
        patch_bdrs: list of lists of two floats, 
            [[tmin, tmax],[xmin,xmax],[ymin,ymax]]

        coarse_factor: integer   

        Notes: 
        ------
        If the patch_bdrs are larger than micro_grid, the method will not set-up the meso_grid, 
        and an error message is printed. This is extra safety measure!
        """

        # Is the patch within the micro_model domain? 
        conditions = patch_bdrs[0][0] < self.micro_model.domain_vars['tmin'] or \
                    patch_bdrs[0][1] > self.micro_model.domain_vars['tmax'] or \
                    patch_bdrs[1][0] < self.micro_model.domain_vars['xmin'] or \
                    patch_bdrs[1][1] > self.micro_model.domain_vars['xmax'] or \
                    patch_bdrs[2][0] < self.micro_model.domain_vars['ymin'] or \
                    patch_bdrs[2][1] > self.micro_model.domain_vars['ymax']
        
        if conditions: 
            print('Error: the input region for filtering is larger than micro_model domain!')
            return None 

        #Find the nearest cell to input patch bdrs
        patch_min = [patch_bdrs[0][0], patch_bdrs[1][0], patch_bdrs[2][0]]
        patch_max = [patch_bdrs[0][1], patch_bdrs[1][1], patch_bdrs[2][1]]
        idx_mins = Base.find_nearest_cell(patch_min, self.micro_model.domain_vars['points'])
        idx_maxs = Base.find_nearest_cell(patch_max, self.micro_model.domain_vars['points'])

        # Set meso_grid spacings
        self.domain_vars['Dt'] = self.micro_model.domain_vars['dt']
        self.domain_vars['Dx'] = self.micro_model.domain_vars['dx'] * coarse_factor
        self.domain_vars['Dy'] = self.micro_model.domain_vars['dy'] * coarse_factor

        # Building the meso_grid
        h, i, j = idx_mins[0], idx_mins[1], idx_mins[2]
        while h <= idx_maxs[0]:
            t = self.micro_model.domain_vars['t'][h]
            self.domain_vars['T'].append(t)
            h += 1
        while i <= idx_maxs[1]:
            x = self.micro_model.domain_vars['x'][i]
            self.domain_vars['X'].append(x)
            i += coarse_factor
        while j <= idx_maxs[2]:
            y = self.micro_model.domain_vars['y'][j]
            self.domain_vars['Y'].append(y)
            j += coarse_factor
                
        # Saving the info about the meso_grid
        self.domain_vars['Points'] = [self.domain_vars['T'], self.domain_vars['X'], self.domain_vars['Y']]
        self.domain_vars['Tmin'] = np.amin(self.domain_vars['T'])
        self.domain_vars['Xmin'] = np.amin(self.domain_vars['X'])
        self.domain_vars['Ymin'] = np.amin(self.domain_vars['Y'])
        self.domain_vars['Tmax'] = np.amax(self.domain_vars['T'])
        self.domain_vars['Xmax'] = np.amax(self.domain_vars['X'])
        self.domain_vars['Ymax'] = np.amax(self.domain_vars['Y'])
        self.domain_vars['Nt'] = len(self.domain_vars['T'])
        self.domain_vars['Nx'] = len(self.domain_vars['X'])
        self.domain_vars['Ny'] = len(self.domain_vars['Y'])

        # Setup arrays for structures
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        self.meso_structures['BC'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
        self.meso_structures['SETfl'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))
        self.meso_structures['SETem'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))
        self.meso_structures['Fab'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))

        # Setup arrays for meso_vars 
        for str in self.meso_scalars_strs:
            self.meso_vars[str] = np.zeros((Nt, Nx, Ny))
        for str in self.meso_vectors_strs:
            self.meso_vars[str] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
        for str in self.meso_r2tensors_strs: 
            self.meso_vars[str] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))

        # Setup arrays for filter_vars
        self.filter_vars['U'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
        self.filter_vars['U_errors'] = np.zeros((Nt, Nx, Ny))
        self.filter_vars['U_success'] = dict()


        # Setup arrays for derivatives of the model. 
        # THOMAS'S WAY (SAVE THEM AS TENSORS)
        self.deriv_vars['D_u_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))
        self.deriv_vars['D_Fab'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1, self.spatial_dims+1))

        # MARCUS'S WAY 
        # for str in self.non_local_vars_strs: 
        #     if str in self.meso_vars:
        #         self.deriv_vars.update({'dt'+str:np.zeros_like(self.meso_vars[str])})
        #         self.deriv_vars.update({'dx'+str:np.zeros_like(self.meso_vars[str])})
        #         self.deriv_vars.update({'dy'+str:np.zeros_like(self.meso_vars[str])})
        #     if str in self.meso_structures:
        #         self.deriv_vars.update({'dt'+str:np.zeros_like(self.meso_structures[str])})
        #         self.deriv_vars.update({'dx'+str:np.zeros_like(self.meso_structures[str])})
        #         self.deriv_vars.update({'dy'+str:np.zeros_like(self.meso_structures[str])})
        
    def find_observers(self): 
        """
        Method to compute filtering observers at grid points built with setup_meso_grid. 
        The observers found (and relative errors) are saved in the dictionary self.filter_vars.
        Set up the entry self.filter_vars['U_success'] as a dictionary with (tuples of) indices
        on the meso_grid as keys, and bool as values (true if the observer has been found, false otherwise)

        Notes:
        ------
        Requires setup_meso_grid() and setup_variables() to be called first. 
        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']): 
                    point = [t,x,y]
                    sol = self.find_obs.find_observer(point)
                    if sol[0]:
                        self.filter_vars['U'][h,i,j] = sol[1]
                        self.filter_vars['U_errors'][h,i,j] = sol[2]
                        self.filter_vars['U_success'].update({(h,i,j) : True})

                    if not sol[0]: 
                        self.filter_vars['U_success'].update({(h,i,j) : False})
                        print('Careful: obs could not be found at: ', self.domain_vars['Points'][h][i][j])

    def filter_micro_variables(self):
        """
        Filter all meso_model structures AND micro pressure at the grid_points built with setup_meso_grid(). 
        This method relies on filter_var_point implemented separately for the filter, e.g. spatial_box_filter

        Parameters: 
        -----------

        Notes:
        ------
        Requires find_observers() and setup_meso_grid() to be called first.
        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']):
                    point = [t, x, y]
                    if self.filter_vars['U_success'][h,i,j]:
                        obs = self.filter_vars['U'][h,i,j]
                        for struct in self.meso_structures:
                            self.meso_structures[struct][h,i,j] = self.filter.filter_var_point(struct, point, obs)
                        self.meso_vars['p'][h,i,j] = self.filter.filter_var_point('p', point, obs)
                    else: 
                        print('Could not filter at {}: observer not found.'.format(point))
        
    def decompose_structures_gridpoint(self, h, i, j): 
        """
        Decompose the fluid part of SET as well as Fab at grid point (h,i,j)

        Parameters:
        -----------
        h, i, j: integers
            the indices on the grid where the decomposition is performed. 

        Returns: 
        --------
        None

        Notes:
        ------
        Decomposition of BC, fluid SET and Faraday tensor. The EM part of SET is decomposed later 
        via non local operations (same story for the charge current).

        """
        # Computing the Favre density and velocity
        n_t = np.sqrt(-Base.Mink_dot(self.meso_structures['BC'][h,i,j], self.meso_structures['BC'][h,i,j]))
        u_t = np.multiply(1 / n_t, self.meso_structures['BC'][h,i,j])
        T_ab = self.meso_structures['SETfl'][h,i,j,:,:] # Remember this is rank (2,0)
    
        # Computing the decomposition at each point
        eps_t = np.einsum('i,j,ik,jl,kl', u_t, u_t, self.metric, self.metric, T_ab)
        h_ab = np.einsum('ij,jk->ik', self.metric + np.einsum('i,j->ij', u_t, u_t), self.metric) # This is a rank (1,1) tensor, i.e. a real projector.
        q_a = np.einsum('ij,jk,kl,l->i',h_ab, T_ab, self.metric, u_t)
        s_ab = np.einsum('ij,kl,jl->ik',h_ab, h_ab, T_ab)
        s = np.einsum('ii',s_ab)

        # Storing the decomposition with appropriate names. 
        self.meso_vars['n_tilde'][h,i,j] = n_t
        self.meso_vars['u_tilde'][h,i,j,:] = u_t
        self.meso_vars['eps_tilde'][h,i,j] = eps_t
        self.meso_vars['q_res'][h,i,j,:] = q_a
        self.meso_vars['pi_res'][h,i,j,:,:] = s_ab - np.multiply(s / self.spatial_dims , self.metric +np.outer(u_t,u_t))
        p_filt = self.meso_vars['p'][h,i,j]
        self.meso_vars['eos_res'][h,i,j] = 0 
        # p_t =  p_from_EOS method(eps_t, n_t) 
        # self.meso_vars['eos_res'][h,i,j] = p_filt - p_t
        self.meso_vars['Pi_res'][h,i,j] = s - p_filt -self.meso_vars['eos_res'][h,i,j]
        

        # Repeat for the Faraday tensor (which is rank (0,2))
        Fab = self.meso_structures['Fab'][h,i,j]
        self.meso_vars['e_tilde'][h,i,j,:] = np.einsum('ij,j->i',Fab, u_t)
        self.meso_vars['b_tilde'][h,i,j] = np.einsum('ijk,i,jl,km,lm', self.Levi3D, u_t, self.metric, self.metric, Fab)

        # Corresponding 3+1 d formulae: only change in B
        # np.einsum('ijkl,i,km,ln,mn->i', self.Levi4D, u_t, self.metric, self.metric, Fab)

    def decompose_structures(self):
        """
        Decompose structures at all points on the meso_grid where observers could be found. 
        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']): 
                    point = [t,y,x]
                    if self.filter_vars['U_success'][h,i,j]:
                        self.decompose_structures_gridpoint(h,i,j)
                    else: 
                        print('Structures not decomposed at {}: observer could not be found.'.format(point))

    def calculate_derivative_gridpoint(self, nonlocal_var_str, h, i, j, direction, order = 1): 
        """
        Calculate partial derivative in the input 'direction' of the variable corresponding to 'nonlocal_var_str' 
        at the position on the grid identified by indices h,i,j. The order of the differencing scheme 
        can also be specified, default to 1.

        Parameters: 
        -----------
        nonlocal_var_str: string
            quantity to be taken derivate of, must be in self.nonlocal_vars_strs()

        h, i ,j: integers
            indices for point on the meso_grid

        direction: integer < self.spatial_dim + 1

        order: integer, defatult to 1
            order of the differencing scheme            

        Returns: 
        --------
        Finite-differenced quantity at (h,i,j) 
                
        Notes:
        ------
        The method returns the value instead of storing it, so that these can be 
        rearranged as preferred later, that is in calculate_derivatives() 
        """
        
        if direction > self.spatial_dims: 
            print('Directions are numbered from 0 up to {}'.format(self.spatial_dims))
            return None
        
        if order > len(self.differencing): 
            print('Maximum order implemented is {}: continuing with it.'.format(len(self.differencing)))
            order = len(self.differencing)

        # Forward, backward or centered differencing? 
        k = [h,i,j][direction]
        N = len(self.domain_vars['Points'][direction])

        if k in [l for l in range(order)]:
            coefficients = self.differencing[order]['fw']['coefficients']
            stencil = self.differencing[order]['fw']['stencil']
        elif k in [N-1-l for l in range(order)]:
            coefficients = self.differencing[order]['bw']['coefficients']
            stencil = self.differencing[order]['bw']['stencil']
        else:
            coefficients = self.differencing[order]['cen']['coefficients']
            stencil = self.differencing[order]['cen']['stencil']

        temp = 0 
        for s, sample in enumerate(stencil): 
            idxs = [h,i,j]
            idxs[direction] += sample
            temp += np.multiply( coefficients[s] / self.domain_vars['Dx'], self.get_var_gridpoint(nonlocal_var_str, *idxs))
        return temp
    
    def calculate_derivatives(self):
        """
        Compute all the derivatives of the quantities corresponding to nonlocal_vars_strs, for all
        gridpoints on the meso-grid. 

        Notes: 
        ------
        The derived quantities are stored as 'tensors' as follows: 
            1st three indices refer to the position on the grid 

            4th index refers to the directionality of the derivative 

            last indices (1 or 2) correspond to the components of the quantity to be derived 

        The index corresponding to the derivative is covariant, i.e. down. 
        
        Example:

            Fab [h,i,j,a,b] : h,i,j grid; a,b, spacetime components

            D_Fab[h,i,j,c,a,b]: h,i,j grid; c direction of derivative; a,b as for Fab

        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']): 
                    point = [t,y,x]
                    if self.filter_vars['U_success'][h,i,j]:
                        for dir in range(self.spatial_dims+1):
                            for str in self.nonlocal_vars_strs: 
                                dstr = 'D_' + str
                                self.deriv_vars[dstr][h,i,j,dir] = self.calculate_derivative_gridpoint(str, h, i, j, dir)
                    else: 
                        print('Derivatives not calculated at {}: observer could not be found.'.format(point))

    # closure_ingredients_gridpoint() ??
    def model_residuals_gridpoint(self, h, i, j):
        """
        Decompose quantities obatined via non_local operations (i.e. derivatives) as needed 
        for the closure scheme. This is done at gridpoint (h,i,j) and then relevant values are 
        returned 

        Parameters:
        -----------
        h, i, j: integers
            the indices of the gridpoint

        Returns: 
        --------
        list of strings: names of the quantities computed 

        list of nd.arrays with quantities computed

        Notes: 
        ------
        The ingredients for the closure scheme are model specific: it makes sense that the relevant 
        dictionary is set up not on construction. 

        UNDER CONSTRUCTION: THE EM PART NEEDS MORE THINKING 
        """
        # COMPUTING THE MESO-CURRENT AND DECOMPOSING IT WRT FAVRE_OBS    
        u_t = self.meso_vars['u_tilde'][h,i,j]
        DFab = self.deriv_vars['D_Fab'][h,i,j]

        current = np.einsum('ij, kl, kjl->i', self.metric, self.metric, DFab)
        sigma_t = np.einsum('i,ij,j->',current, self.metric, u_t)
        J_t = current - np.multiply(sigma_t, u_t)

        # MODELLING THE DISSIPATIVE TERMS IN THE FLUID SET 
        nabla_u = self.deriv_vars['D_u_tilde'][h,i,j] # This is a rank (1,1) tensor
        acc_t = np.einsum('i,ij->j',u_t, nabla_u)
        should_be_zero = np.einsum('ij,i,kj',self.metric, u_t, nabla_u)

        h_ab = np.einsum('ij,jk->ik', self.metric + np.einsum('i,j->ij', u_t, u_t), self.metric)
        Daub = nabla_u + np.einsum('ij,j,l', self.metric, u_t, acc_t) + np.einsum('ij,j,l', self.metric, should_be_zero, u_t)
        exp_t = np.einsum('ii->', Daub)
        shear_t = np.multiply(1/2., Daub + np.einsum('ij->ji', Daub)) - np.multiply( 1/self.spatial_dims * exp_t, h_ab) 

        # THINK WHAT TO DO WITH THE HEAT FLUX --> GRADIENTS IN T OR N AND EPS
        # THINK ABOUT THE EM PART, DO YOU NEED MORE?  
        # COME BACK TO THIS LATER?
        # self.meso_vars['j'][h,i,j,:] = current ?? 

        closure_vars_strs = ['shear_tilde', 'exp_tilde', 'acc_tilde', 'sigma_tilde', 'J_tilde']
        closure_vars = [shear_t, exp_t, acc_t, sigma_t, J_t]

        # I THINK THERE'S A MISTAKE WITH THE SHEAR CALCULATION HERE, CHECK!!!
        # PRINT IT TO CHECK IT IS REALLY SYMMETRIC AS IT SHOULD!
        return closure_vars_strs, closure_vars
    #closure_ingredients ??
    def model_residuals(self): 
        """
        Wrapper of the corresponding gridpoint method. 
        Set up the dictionary for the closure variables not yet defined.
        Loop over the grid and store the decomposed variables. 
        """        
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        for h in range(Nt): 
            for i in range(Nx): 
                for j in range(Ny): 
                    if self.filter_vars['U_success'][h,i,j]: 
                        keys, values = self.model_residuals_gridpoint(h,i,j)
                        # try-except block to extend the meso_vars dictionary 
                        for idx, key in enumerate(keys): 
                            try: 
                                self.meso_vars[key]
                            except KeyError:
                                print('The key {} does not belong to meso_vars yet, adding it!'.format(key))
                                shape = values[idx].shape
                                self.meso_vars.update({key: np.zeros(([Nt, Nx, Ny] + list(shape)))})
                            finally: 
                                    self.meso_vars[key][h,i,j] = values[idx]

    def shear_regression_test(self):
        """
        WORK IN PROGRESS: A MINIMAL REGRESSION ANALYSIS ON THE COMPONENTS OF THE SHEAR TENSOR
        NEED TO FIRST WORK ON WRAPPER OF MODEL_RESIDUALS THAT STORES ALL THE INFO AT ALL POINTS 
        Notes:
        ------
        Automatic looping over all components: how to make use of the fact that the shear is symmetric? 
        Better to use a wrapper? 
        """
        # LINEAR REGRESSION: LOOP OVER COMPONENTS AND AVERAGE

        shape = self.meso_vars['shear_tilde'][0,0,0].shape
        statistics = np.zeros((tuple(list(shape)+ [3])))
        for component_idx in np.ndindex(shape):
            xs = self.meso_vars['shear_tilde'][0,:,:,component_idx]
            ys = self.meso_vars['pi_res'][0,:,:,component_idx]
            xs = xs.flatten()
            ys = ys.flatten()
            sns_plot = sns.regplot(x = xs, y = ys)
            # print(scst.pearsonr(xs,ys))
            statistics[component_idx] = np.array(scst.linregress(xs,ys)[:3])

            fig = sns_plot.get_figure()
            fig.tight_layout()
            # plt.show()

        m_statistics = np.zeros((3,))
        for i in range(len(m_statistics)):
            # for idx in np.ndindex(shape): 
            idx = np.ndindex(shape)
            m_statistics[i] = np.mean(statistics[:,:,i])


        # PREPARING THE DATA FOR CONTRASTING RESIDUAL AND MODEL
        eta = m_statistics[0]
        # pi_model = np.zeros(self.meso_vars['pi_res'][0,:,:,:,:].shape)

        # Nx, Ny = self.domain_vars['Nx'], self.domain_vars['Ny']
        # for i in range(Nx): 
        #         for j in range(Ny):
        #             if self.filter_vars['U_success'][0,i,j]:
        #                 shear = self.model_residuals_gridpoint(0,i,j)[1][0]
        #                 pi_model[i,j,:,:] = np.multiply(eta, shear) # Calling the same function twice! Save data! 
        # pi_res = self.meso_vars['pi_res'][0,:,:,:,:]

        # THIS IS A TRICK: YOU WANT TO COMPARE THE RESIDUAL WITH MODEL = SHEAR x ETA 
        # self.meso_vars['shear_tilde'][:,:,:] = np.multiply(eta, self.meso_vars['shear_tilde'][:,:,:] )
        for idx in np.ndindex((3,3)): 
            self.meso_vars['shear_tilde'][:,:,:,idx]  = np.multiply(eta, self.meso_vars['shear_tilde'][:,:,:,idx])
        

        # # SETTING UP AND PLOTTING VIA MARCUS'S ROUTINES
        visualizer = Plotter_2D()
        t = self.domain_vars['T'][0]
        x_range, y_range = [self.domain_vars['Xmin'], self.domain_vars['Xmax']] ,\
                                     [self.domain_vars['Ymin'], self.domain_vars['Ymax']]
        # print(t, x_range, y_range)

        # print(visualizer.get_var_data(self, 'shear_tilde', t, x_range, y_range, component_indices=(0,0)))
        visualizer.plot_vars(self, ['pi_res', 'shear_tilde'], t, x_range, y_range, components_indices = [(0,0),(0,0)])

    def EL_style_closure_gridpoint():
        """
        Compute bulk, shear via scalarization + PA trick, thermal conductivity later. 
        At a point. 
        Returns: one coefficient at a point.
        """
        pass

    def EL_style_closure(): 
        """
        Wrapper of EL_style_closure_gridpoint()    
        """
        pass

    def EL_style_closure_regression():
        """
        Takes in a list of correlation quantities (strings? The regressors needed are computed 
        previously via closure ingredients)
        Pass the dataset to a CoefficientAnalysis instance. 
        Plot the correlations with the regressors and perform the regression using methods from a Coefficient 
        Analysis instance. 
        """
        pass


if __name__ == '__main__':


    FileReader = METHOD_HDF5('./Data/test_res100/')
    micro_model = IdealMHD_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()
    find_obs = FindObs_drift_root(micro_model, 0.001)
    filter = spatial_box_filter(micro_model, 0.003)


    CPU_start_time = time.process_time()
    meso_model = resMHD2D(micro_model, find_obs, filter)

    meso_model.setup_meso_grid([[1.501, 1.504],[0.3, 0.4],[0.4, 0.5]],1)
    print(meso_model.domain_vars['Points'])
    meso_model.find_observers()
    meso_model.filter_micro_variables()
    
    # meso_model.decompose_structures_gridpoint(1,1,1)
    meso_model.decompose_structures()

    
    # print(meso_model.calculate_derivative_gridpoint('u_tilde', 0,5,5,1, order=2))
    meso_model.calculate_derivatives()

    # meso_model.model_residuals_gridpoint(1,1,0)
    meso_model.model_residuals()
    meso_model.shear_regression_test()

    # print('Total time is {}'.format(time.process_time() - CPU_start_time))    

