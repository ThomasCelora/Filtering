# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

import numpy as np
from scipy.interpolate import interpn 
import multiprocessing as mp
from multimethod import multimethod

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scst    
from scipy.linalg import det


from system.BaseFunctionality import *
from MicroModels import * 
from FileReaders import * 
from Filters import *
from Visualization import *
from Analysis import *

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

        self.meso_scalars_strs = ['eps_tilde', 'n_tilde', 'p_tilde', 'p_filt', 'eos_res', 'Pi_res', 'sigma_tilde', 'b_tilde', 'T_tilde']
        self.meso_vectors_strs = ['u_tilde', 'q_res', 'e_tilde', 'j', 'J_tilde']
        self.meso_r2tensors_strs = ['pi_res']
        self.meso_vars_strs = self.meso_scalars_strs + self.meso_vectors_strs + self.meso_r2tensors_strs
        self.meso_vars = dict.fromkeys(self.meso_vars_strs)

        self.coefficient_strs = ["Gamma"]
        self.coefficients = dict.fromkeys(self.coefficient_strs)
        self.coefficients['Gamma'] = 4.0/3.0

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
        self.nonlocal_vars_strs = ['u_tilde', 'Fab', 'T_tilde'] 
        # self.nonlocal_vars_strs = ['u_tilde', 'Fab'] 
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
        elif var in self.filter_vars: 
            return interpn(self.domain_vars['Points'], self.filter_vars[var], point, method = self.interp_method)[0]
        else: 
            print('{} does not belong to either meso_structures/meso_vars/deriv_vars/filter_vars. Check! (Method: get_interpol_var(var, point))'.format(var))
    
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
        elif var in self.filter_vars:
            return self.filter_vars[var][h,i,j]
        else: 
            print('{} does not belong to either meso_structures/meso_vars/deriv_vars/filter_vars. Check! (Method: get_var_gridpoint(h,i,j))'.format(var))
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
        elif var in self.filter_vars:
            return self.filter_vars[var][tuple(indices)]
        else: 
            print('{} does not belong to either meso_structures/meso_vars/deriv_vars/filter_vars. Check! (Method: get_var_gridpoint(point))'.format(var))
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
        self.deriv_vars['D_T_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))

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
                        self.meso_vars['p_filt'][h,i,j] = self.filter.filter_var_point('p', point, obs)
                    else: 
                        print('Could not filter at {}: observer not found.'.format(point))

    def p_from_EOS(self, eps, n):
        """
        Compute pressure from Gamma-law EoS 
        """
        return (self.coefficients['Gamma']-1)*eps*n

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
        p_t = self.p_from_EOS(eps_t, n_t) 
    

        # Storing the decomposition with appropriate names. 
        self.meso_vars['n_tilde'][h,i,j] = n_t
        self.meso_vars['u_tilde'][h,i,j,:] = u_t
        self.meso_vars['eps_tilde'][h,i,j] = eps_t
        self.meso_vars['q_res'][h,i,j,:] = q_a
        self.meso_vars['pi_res'][h,i,j,:,:] = s_ab - np.multiply(s / self.spatial_dims , self.metric +np.outer(u_t,u_t))
        self.meso_vars['p_tilde'][h,i,j] =  p_t
        self.meso_vars['Pi_res'][h,i,j] = s - p_t
        self.meso_vars['eos_res'][h,i,j] = self.meso_vars['p_filt'][h,i,j] - p_t
        self.meso_vars['T_tilde'][h,i,j] = p_t / n_t

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

    def calculate_derivatives_gridpoint(self, nonlocal_var_str, h, i, j, direction, order = 1): 
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
                                self.deriv_vars[dstr][h,i,j,dir] = self.calculate_derivatives_gridpoint(str, h, i, j, dir)
                    else: 
                        print('Derivatives not calculated at {}: observer could not be found.'.format(point))

    def closure_ingredients_gridpoint(self, h, i, j):
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

        Rename: closure_ingredients_gridpoint() ??
        Coefficients can be computed here directly, but more general models will require more work.
        Do not split this, at least for now
        """
        # COMPUTING THE MESO-CURRENT AND DECOMPOSING IT WRT FAVRE_OBS    
        u_t = self.meso_vars['u_tilde'][h,i,j]
        u_t_cov = np.einsum('ij,j->i', self.metric, u_t)
        Nabla_Fab = self.deriv_vars['D_Fab'][h,i,j]

        # CLOSURE INGREDIENTS: FLUID BIT - WORKING WITH (2,0)
        nabla_u = self.deriv_vars['D_u_tilde'][h,i,j] # This is a rank (1,1) tensor
        nabla_u = np.einsum('ij,jk->ik', self.metric, nabla_u) #this is a rank (2,0) tensor
        acc_t = np.einsum('i,ij', u_t_cov, nabla_u) # vector

        # The following two quantities should be identically zero, but they won't be due to numerical errors.
        # Computing them and removing to project velocity gradients
        unit_norm_violation = np.einsum('ij,j', nabla_u, u_t_cov) #vector
        acc_orthogonality_violation = np.einsum('i,i->', u_t_cov, acc_t)
        Daub = nabla_u + np.einsum('i,j->ij', u_t, acc_t) + np.einsum('i,j->ij', unit_norm_violation, u_t) +\
            np.multiply(acc_orthogonality_violation, np.einsum('i,j->ij', u_t, u_t)) #Should be a (2,0) tensor
        h_ab = self.metric + np.einsum('i,j->ij', u_t, u_t)
        exp_t = np.einsum('ii->',Daub)
        shear_t = np.multiply(1/2., Daub + np.einsum('ij->ji', Daub)) - np.multiply( 1/self.spatial_dims * exp_t, h_ab) 
        vort_t = np.multiply(1/2., Daub - np.einsum('ij->ji', Daub))

    
        # CLOSURE INGREDIENTS: HEAT FLUX 
        nabla_T = self.deriv_vars['D_T_tilde'][h,i,j]
        projector = np.zeros((3,3))
        np.fill_diagonal(projector, 1) 
        projector += np.einsum('i,j->ij',u_t_cov, u_t)
        DaT = np.einsum('ij,j->i', projector, nabla_T)
        Theta_tilde = DaT + np.multiply(self.meso_vars['T_tilde'][h,i,j], np.einsum('ij,j->i', self.metric, acc_t))

        # CLOSURE INGREDIENTS: EM PART - DO YOU NEED MORE? 
        # current = np.einsum('ij, kl, kjl->i', self.metric, self.metric, DFab)
        # sigma_t = np.einsum('i,ij,j->',current, self.metric, u_t)
        # J_t = current - np.multiply(sigma_t, u_t)
        # # self.meso_vars['j'][h,i,j,:] = current ?? 

        closure_vars_strs = ['shear_tilde', 'exp_tilde', 'acc_tilde', 'Theta_tilde']
        closure_vars = [shear_t, exp_t, acc_t, Theta_tilde]
        return closure_vars_strs, closure_vars
    
    def closure_ingredients(self): 
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
                        keys, values = self.closure_ingredients_gridpoint(h,i,j)
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

    def EL_style_closure_gridpoint(self, h, i, j):
        """
        Compute bulk, shear via scalarization + PA trick, thermal conductivity later. 
        At a point. 
        Returns: coefficient(s) at a point.
        """
        coefficients_names=[]
        coefficients=[]
        
        # CALCULATING BULK VISCOUS COEFF
        zeta = self.meso_vars['Pi_res'][h,i,j] / self.meso_vars['exp_tilde'][h,i,j] 
        coefficients_names.append('zeta')
        coefficients.append(zeta)

        # CALCULATING SHEAR VISCOUS COEFF
        pi_res_sq = np.einsum('ij,kl,ik,jl->', self.meso_vars['pi_res'][h,i,j], self.meso_vars['pi_res'][h,i,j], self.metric, self.metric)
        shear_sq = np.einsum('ij,kl,ik,jl->', self.meso_vars['shear_tilde'][h,i,j], self.meso_vars['shear_tilde'][h,i,j], self.metric, self.metric)
        eta = pi_res_sq/shear_sq
        # Compute sign of eta by looking at the PA of pi_res with higher eigenvalue. 
        # Change shear to the eigenbasis of pi_res (using that the matrix is unitary as pi_res is sym)
        pi_eig, pi_eigv = np.linalg.eigh(self.meso_vars['pi_res'][h,i,j])
        transformed_shear = np.einsum('ji,jk,kl', pi_eigv, self.meso_vars['shear_tilde'][h,i,j], pi_eigv)
        abs_pi_eig = np.abs(pi_eig)
        pos = list(abs_pi_eig).index(np.max(abs_pi_eig))
        eta = eta * np.sign(pi_eig[-1] / transformed_shear[pos,pos])
        coefficients_names.append('eta')
        coefficients.append(eta)

        # CALCULATING THE HEAT CONDUCTIVITIY
        q_res = self.meso_vars['q_res'][h,i,j]
        Theta_t = self.meso_vars['Theta_tilde'][h,i,j]
        q_res_sq = np.einsum('i,ij,j', q_res, self.metric, q_res)
        Theta_t_sq = np.einsum('i,ij,j', Theta_t, self.metric, Theta_t)
        cos_angle = np.einsum('i,ij,j', q_res, self.metric, Theta_t) / (q_res_sq * Theta_t_sq)
        sign = np.sign(cos_angle)
        kappa = sign * np.sqrt(q_res_sq / Theta_t_sq)
        coefficients_names.append('kappa')
        coefficients.append(kappa)

        return coefficients_names, coefficients

    def EL_style_closure(self): 
        """
        Wrapper of EL_style_closure_gridpoint()    
        """
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        for h in range(Nt): 
            for i in range(Nx): 
                for j in range(Ny): 
                    if self.filter_vars['U_success'][h,i,j]: 
                        keys, values = self.EL_style_closure_gridpoint(h,i,j)
                        # try-except block to extend the meso_vars dictionary with the dissipative coefficients
                        for idx, key in enumerate(keys): 
                            try: 
                                self.meso_vars[key]
                            except KeyError:
                                print('The key {} does not belong to meso_vars yet, adding it!'.format(key))
                                shape = values[idx].shape
                                self.meso_vars.update({key: np.zeros(([Nt, Nx, Ny] + list(shape)))})
                            finally: 
                                    self.meso_vars[key][h,i,j] = values[idx]
                    
    def EL_style_closure_regression(self, CoefficientAnalysis):
        """
        Takes in a list of correlation quantities (strings? The regressors needed are computed 
        previously via closure ingredients)
        Pass the dataset to a CoefficientAnalysis instance. 
        Plot the correlations with the regressors and perform the regression using methods from a Coefficient 
        Analysis instance. 
        """
        # FICTITIOUS RANGES AND POINTS TO TEST TRIM_DATA ROUTINE
        ranges = [[self.domain_vars['Tmin'], self.domain_vars['Tmax']],\
                [self.domain_vars['Xmin'], self.domain_vars['Xmax']],\
                [self.domain_vars['Ymin'], self.domain_vars['Ymax']]]
        points = self.domain_vars['Points']

        # TESTING SCALAR REGRESSION ROUTINE
        # y = self.meso_vars['zeta']
        # X = [self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # stats_result = CoefficientAnalysis.scalar_regression(y, X, ranges, points)
        # print(*stats_result)

        # TESTING TENSOR REGRESSION ROUTINE
        # y=self.meso_vars['q_res']
        # X=[self.meso_vars['e_tilde'], self.meso_vars['u_tilde']]
        # # stats_result = CoefficientAnalysis.tensor_components_regression(y, X, 2,ranges, points, components=[(0,),(2,)])
        # stats_result = CoefficientAnalysis.tensor_components_regression(y, X, 2,ranges, points, add_intercept=False)
        # for i in range(len(stats_result)):
        #     print('{}-th component: \n {} \n\n'.format(i, stats_result[i]))

        # TESTING CORRELATION VISUALIZATION ROUTINES
        # x = self.meso_vars['n_tilde']
        # y = self.meso_vars['eps_tilde']
        # labels = ['n_tilde','eps_tilde']
        # g1=CoefficientAnalysis.visualize_correlation(x, y, xlabel=labels[0], ylabel=labels[1])
        # fig=g1.figure
        # fig.savefig('Single_correlation_plot.pdf', format='pdf')

        # data=[self.meso_vars['eta'], self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['eta', 'b_tilde', 'eps_tilde', 'n_tilde']
        # g2=CoefficientAnalysis.visualize_many_correlations(data, labels)

        # data=[self.meso_vars['zeta'], self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['zeta', 'b_tilde', 'eps_tilde', 'n_tilde']
        # gbis= CoefficientAnalysis.visualize_many_correlations(data, labels)
        # # plt.savefig('Many_correlations_plot.pdf', format='pdf')
        # plt.show()
        
        # TESTING THE CHECK REGRESSORS ROUTINE:
        # data=[self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['b_tilde', 'eps_tilde', 'n_tilde']
        # g1=CoefficientAnalysis.visualize_many_correlations(data, labels)
        # comps, g2 = CoefficientAnalysis.PCA_find_regressors_subset(data, ranges = ranges, model_points = points)
        # g2.fig.suptitle("Correlation between PC (standardized data)")
        # g2.fig.subplots_adjust(top=0.94) 
        # print(comps)
        # plt.show()

class resHD2D(object):
    """
    CUT AND PAST OF resMHD_2D --> removing the magnetic bits
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

        self.meso_structures_strs  = ['SET', 'BC']
        self.meso_structures = dict.fromkeys(self.meso_structures_strs) 
        for var in self.meso_structures:
                self.meso_structures[var] = []

        self.meso_scalars_strs = ['eps_tilde', 'n_tilde', 'p_tilde', 'p_filt', 'eos_res', 'Pi_res', 'T_tilde']
        self.meso_vectors_strs = ['u_tilde', 'q_res']
        self.meso_r2tensors_strs = ['pi_res']
        self.meso_vars_strs = self.meso_scalars_strs + self.meso_vectors_strs + self.meso_r2tensors_strs
        self.meso_vars = dict.fromkeys(self.meso_vars_strs)

        self.coefficient_strs = ["Gamma"]
        self.coefficients = dict.fromkeys(self.coefficient_strs)
        self.coefficients['Gamma'] = 4.0/3.0

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
        self.nonlocal_vars_strs = ['u_tilde', 'T_tilde', 'n_tilde', 'eps_tilde'] 
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

        self.labels_var_dict = {'SET' : r'$<T^{ab}>$', 
                                'BC' : r'$<n^a>$',
                                'U' : r'$U^a$',
                                'Gamma' : r'$\Gamma$',
                                'eos_res' : r'$M$',
                                'Pi_res' : r'$\tilde{\Pi}$',
                                'q_res' : r'$\tilde{q}^a$',
                                'pi_res' : r'$\tilde{\pi}^{ab}$',
                                'eps_tilde' : r'$\tilde{\varepsilon}$',
                                'n_tilde' : r'$\tilde{n}$',
                                'p_tilde' : r'$\tilde{p}$',
                                'p_filt' : r'$<p>$',
                                'T_tilde' : r'$\tilde{T}$',
                                'u_tilde' : r'$\tilde{u}^a$',
                                'D_u_tilde' : r'$\nabla_{a}\tilde{u}^b$',
                                'D_T_tilde' : r'$\nabla_{a}\tilde{T}$',
                                'D_n_tilde' : r'$\nabla_{a}\tilde{n}$',
                                'D_eps_tilde' : r'$\nabla_{a}\tilde{\varepsilon}$',
                                'n_tilde_dot' : r'$\dot{\tilde{n}}$',
                                'T_tilde_dot' : r'$\dot{\tilde{T}}$',
                                'sD_T_tilde': r'$D_{a}\tilde{T}$',
                                'sD_n_tilde': r'$D_{a}\tilde{n}$',
                                'shear_tilde' : r'$\tilde{\sigma}^{ab}$',
                                'acc_tilde' : r'$\tilde{a}^a$',
                                'exp_tilde' : r'$\tilde{\theta}$',
                                'Theta_tilde' : r'$\tilde{\Theta}^a$',
                                'eta' : r'$\eta$',
                                'zeta' : r'$\zeta$',
                                'kappa' : r'$\kappa$',
                                'Pi_res_sq' : r'$\tilde{\Pi}^2$',
                                'pi_res_sq' : r'$\tilde{\pi}_{ab}\tilde{\pi}^{ab}$',
                                'shear_sq' : r'$\tilde{\sigma}_{ab}\tilde{\sigma}^{ab}$',
                                'Theta_sq': r'$\tilde{\Theta}_a\tilde{\Theta}^a$',
                                'q_res_sq': r'$\tilde{q}_a \tilde{q}^a$',
                                'det_shear': r'$det(\sigma)$',
                                'vort_sq' : r'$\omega_{ab}\omega^{ab}$',
                                'acc_mag': r'$|a|$', 
                                'sD_n_tilde_sq' : r'$D_{a}\tilde{n}D^{a}\tilde{n}$',
                                'dot_Dn_Theta' : r'$D_{a}\tilde{n}\Theta^{a}$',
                                'Q1' : r'$\tilde{\sigma}_{ab}\tilde{\sigma}^{ab} - \tilde{\omega}_{ab}\tilde{\omega}^{ab}$',
                                'Q2' : r'$\tilde{\sigma}_{ab}\tilde{\sigma}^{ab}/\tilde{\omega}_{ab}\tilde{\omega}^{ab}$',
                                'weights' : r'$w$'}

    def update_labels_dict(self, entry_dict):
        """
        pretty self-explanatory
        """
        self.labels_var_dict.update(entry_dict)

    def set_find_obs(self, find_obs):
        self.find_obs = find_obs

    def set_filter(self, filter):
        self.filter = filter

    def get_all_var_strs(self): 
        return list(self.meso_vars.keys())  + list(self.deriv_vars.keys()) + list(self.meso_structures.keys()) + \
            list(self.filter_vars.keys())

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
        elif var in self.filter_vars:
            return interpn(self.domain_vars['Points'], self.filter_vars[var], point, method=self.interp_method)[0]
        else: 
            print('Cannot interpolate value of {} using data fromfilter_vars meso_structures/meso_varsderiv_vars/filter_vars. Check!'.format(var))
    
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
        elif var in self.filter_vars:
            return self.filter_vars[var][h,i,j]
        else: 
            print(f'Cannot get value of {var} at h,i,j from data in meso_vars/meso_structures/deriv_vars/filter_vars')
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
        elif var in self.filter_vars:
            return self.filter_vars[var][tuple(indices)]
        else: 
            print(f'Cannot get value of {var} at h,i,j from data in meso_vars/meso_structures/deriv_vars/filter_vars')
            return None

    def get_model_name(self):
        return 'resHD2D'
    
    def set_find_obs_method(self, find_obs):
        self.find_obs = find_obs

    def setup_meso_grid(self, patch_bdrs, coarse_factor = 1, coarse_time = False): 
        """
        Builds the meso_model grid using the micro_model grid points contained in 
        the input patch (defined via 'patch_bdrs'). The method allows for coarse graining 
        the grid (both in space dirs only or also in time)
        Then store the info about the meso grid and set up arrays of definite rank and size 
        for the quantities needed later. 

        Parameters: 
        -----------
        patch_bdrs: list of lists of two floats, 
            [[tmin, tmax],[xmin,xmax],[ymin,ymax]]

        coarse_factor: integer   

        coarse_time: boolean
            If true, coarsening is also applied to the time direction.
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
        if coarse_time:
            self.domain_vars['Dt'] = self.micro_model.domain_vars['dt'] * coarse_factor
        else:
             self.domain_vars['Dt'] = self.micro_model.domain_vars['dt']
        self.domain_vars['Dx'] = self.micro_model.domain_vars['dx'] * coarse_factor
        self.domain_vars['Dy'] = self.micro_model.domain_vars['dy'] * coarse_factor

        # Building the meso_grid
        h, i, j = idx_mins[0], idx_mins[1], idx_mins[2]
        while h <= idx_maxs[0]:
            t = self.micro_model.domain_vars['t'][h]
            self.domain_vars['T'].append(t)
            if coarse_time:
                h += coarse_factor
            else:
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
        self.meso_structures['SET'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))

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

        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    self.filter_vars['U_success'].update({(h,i,j): False})

        # Setup arrays for derivatives of the model. 
        self.deriv_vars['D_u_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1, self.spatial_dims+1))
        self.deriv_vars['D_T_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
        self.deriv_vars['D_eps_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
        self.deriv_vars['D_n_tilde'] = np.zeros((Nt, Nx, Ny, self.spatial_dims+1))
 
    def find_observers(self): 
        """
        Method to compute filtering observers at grid points built with setup_meso_grid. 
        The observers found (and relative errors) are saved in the dictionary self.filter_vars.
        Set up the entry self.filter_vars['U_success'] as a dictionary with (tuples of) indices
        on the meso_grid as keys, and bool as values (true if the observer has been found, false otherwise)

        Notes:
        ------
        Requires setup_meso_grid() to be called first. 
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
                        # self.filter_vars['U_success'].update({(h,i,j) : False})
                        # No need to update the dictionary as this has been initialized to False everywhere. 
                        print('Careful: obs could not be found at: ', self.domain_vars['Points'][h][i][j])

    def find_observers_parallel(self, n_cpus):
        """
        Method to find observers at all points on meso-grid, parallelized version. 
        The observers found (and relative errors) are saved in the dictionary self.filter_vars.
        Set up the entry self.filter_vars['U_success'] as a dictionary with (tuples of) indices
        on the meso_grid as keys, and bool as values (true if the observer has been found, false otherwise)

        Parameters:
        -----------

        n_cpus: int
            number of processes to run in parallel 

        Notes:
        ------
        This method relies on the routine find_obs.find_observers_parallel().
        So meso_class must be constructed passing parallelized class for finding observers. 
        """
        ts = self.domain_vars['T']
        xs = self.domain_vars['X']
        ys = self.domain_vars['Y']

        t_idxs = np.arange(len(ts))
        x_idxs = np.arange(len(xs))
        y_idxs = np.arange(len(ys))

        points = []
        for elem in product(ts,xs,ys):
            points.append(list(elem))

        indices_meso_grid = []
        for elem in product(t_idxs, x_idxs, y_idxs):
            indices_meso_grid.append(elem)

        successes, failures = self.find_obs.find_observers_parallel(points, n_cpus)

        for i in range(len(successes[0])):
            point_indxs_meso_grid = indices_meso_grid[successes[0][i]]
            self.filter_vars['U'][point_indxs_meso_grid] = successes[1][i]
            self.filter_vars['U_errors'][point_indxs_meso_grid] = successes[2][i]
            self.filter_vars['U_success'].update({(point_indxs_meso_grid): True})

        if len(failures)!=0:
            print('Observers could not be found at the following points:\n')
            for i in range(len(failures)):
                failed_idxs_meso_grid = indices_meso_grid[failures[i]]
                print('{}\n'.format(failed_idxs_meso_grid))

    def filter_micro_variables(self):
        """
        Filter all meso_model structures AND micro pressure within the input ranges. 
        If no range is provided in one direction, routine will try and filter at all points
        on the meso-grid. Note this would require the grid to be set up wisely so to avoid
        problems at the boundaries. 
        
        This method relies on filter_var_point implemented separately for the filter, e.g. spatial_box_filter

        Notes:
        ------
        Requires setup_meso_grid() to be called first.
        Also find_observers() should be called first, although not doing so won't crash it. 
        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']):
                    point = [t,x,y]
                    if self.filter_vars['U_success'][h,i,j]:
                        obs = self.filter_vars['U'][h,i,j]
                        for struct in self.meso_structures:
                            self.meso_structures[struct][h,i,j] = self.filter.filter_var_point(struct, point, obs)
                        self.meso_vars['p_filt'][h,i,j] = self.filter.filter_var_point('p', point, obs)
                    else: 
                        print('Could not filter at {}: observer not found.'.format(point))

    def filter_micro_vars_parallel(self, n_cpus):
        """
        Filter all meso_model structures AND micro pressure. 
        Routine will try and filter at all points on the meso-grid. 
        Note this would require the grid to be set up wisely so to avoid
        problems at the boundaries. 
        
        This method relies on filter_vars_parallel implemented separately for the 
        filter class, e.g. as in box_filter_parallel
        
        Parameters:
        -----------

        n_cpus: int
            number of processes for parallelization

        Notes:
        ------
        Requires setup_meso_grid() to be called first.
        Also find_observers() should be called first, although not doing so won't crash it. 
        """
        ts = self.domain_vars['T']
        xs = self.domain_vars['X']
        ys = self.domain_vars['Y']

        t_idxs = np.arange(len(ts))
        x_idxs = np.arange(len(xs))
        y_idxs = np.arange(len(ys))

        points = []
        for elem in product(ts,xs,ys):
            points.append(list(elem))

        indices_meso_grid = []
        for elem in product(t_idxs, x_idxs, y_idxs):
            indices_meso_grid.append(elem)

        observers = []
        for elem in product(t_idxs, x_idxs, y_idxs):
            if self.filter_vars['U_success'][elem]:
                observers.append(self.filter_vars['U'][elem])
            else:
                print('Observers are not computed on (parts of) the grid!')
                return None

        vars = ['BC', 'SET', 'p']
        points_observers = []
        for i in range(len(points)):
            # args_for_filtering_parallel.append([points[i], observers[i], vars])
            points_observers.append([points[i], observers[i]])
            
        # positions, filtered_vars = self.filter.filter_vars_parallel(args_for_filtering_parallel, n_cpus)
        filtered_vars = dict.fromkeys(vars)
        for var in vars:
            positions, filtered_vars[var] = self.filter.filter_var_parallel(points_observers, var, n_cpus)

        for i in range(len(positions)):
            point_indxs_meso_grid = indices_meso_grid[positions[i]]
            self.meso_structures['BC'][point_indxs_meso_grid] = filtered_vars['BC'][i]
            self.meso_structures['SET'][point_indxs_meso_grid] = filtered_vars['SET'][i]
            self.meso_vars['p_filt'][point_indxs_meso_grid] = filtered_vars['p'][i]
            # self.meso_structures['BC'][point_indxs_meso_grid] = filtered_vars[i][0]
            # self.meso_structures['SET'][point_indxs_meso_grid] = filtered_vars[i][1]
            # self.meso_vars['p_filt'][point_indxs_meso_grid] = filtered_vars[i][2]
        
    def p_from_EOS(self, eps, n):
        """
        Compute pressure from Gamma-law EoS 
        """
        return (self.coefficients['Gamma']-1)*(eps-n)

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
        u_t = np.multiply(1. / n_t, self.meso_structures['BC'][h,i,j])
        T_ab = self.meso_structures['SET'][h,i,j,:,:] # Remember this is rank (2,0)
    
        # Computing the decomposition at each point
        eps_t = np.einsum('i,j,ik,jl,kl', u_t, u_t, self.metric, self.metric, T_ab)
        h_ab = np.einsum('ij,jk->ik', self.metric + np.einsum('i,j->ij', u_t, u_t), self.metric) # This is a rank (1,1) tensor, i.e. a real projector.
        q_a = np.einsum('ij,jk,kl,l->i',h_ab, T_ab, self.metric, u_t) # There might be missing a minus sign here?
        s_ab = np.einsum('ij,kl,jl->ik',h_ab, h_ab, T_ab)
        s = np.einsum('ii',s_ab)
        p_t = self.p_from_EOS(eps_t, n_t) 
    

        # Storing the decomposition with appropriate names. 
        self.meso_vars['n_tilde'][h,i,j] = n_t
        self.meso_vars['u_tilde'][h,i,j,:] = u_t
        self.meso_vars['eps_tilde'][h,i,j] = eps_t
        self.meso_vars['q_res'][h,i,j,:] = q_a
        self.meso_vars['pi_res'][h,i,j,:,:] = s_ab - np.multiply(s / self.spatial_dims , self.metric +np.outer(u_t,u_t))
        self.meso_vars['p_tilde'][h,i,j] =  p_t
        self.meso_vars['Pi_res'][h,i,j] = s - p_t
        self.meso_vars['eos_res'][h,i,j] = self.meso_vars['p_filt'][h,i,j] - p_t
        self.meso_vars['T_tilde'][h,i,j] = p_t / n_t

    def decompose_structures(self):
        """
        Decompose structures at all points on the meso_grid where observers could be found. 

        Parameters:
        -----------
        t_range: list of two floats
            ranges in the t-direction, used for selecting the sublist of points 

        x_range, y_range: similar to above. 
        """
        for h, t in enumerate(self.domain_vars['T']):
            for i, x in enumerate(self.domain_vars['X']):
                for j, y in enumerate(self.domain_vars['Y']):
                    point = [t,y,x]
                    if self.filter_vars['U_success'][h,i,j]:
                        self.decompose_structures_gridpoint(h,i,j)
                    else: 
                        print('Structures not decomposed at {}: observer could not be found.'.format(point))

    @staticmethod
    def p_Gamma_law(eps, n, Gamma):
        """
        staticmethod used by decompose_structures_task, that is the 
        parallel version decompose_structures_gridpoint 

        Parameters: 
        -----------
        eps: float  
            the energy density of the fluid at a point

        n: float
            the baryon number density of the fluid at a point 

        Gamma: float
            Gamma factor of the Gamma law 
        """
        return (Gamma-1)* (eps-n)

    @staticmethod
    def decompose_structures_task(BC, SET , p_filt, h, i, j):
        """
        Task to be executed in parallel: decomposing the structures at a point 

        Parameters: 
        -----------

        BC: np.array (3,)
            the baryon current, rank: (1,0)

        SET: np.array (3,3)
            the Stress-Energy tensor, rank (2,0)

        p_filt: float
            the filtered pressure, scalar

        h,i,j: integers
            indices of the corresponding gridpoint on meso-grid
        
        Returns: 
        --------
        (list, list):
            first list contains the decomposition of structures at point
            second list contains indices of point on mesogrid
        """
        # As this is staticmethod, no access to self.metric
        metric = np.zeros((3,3))
        metric[0,0] = -1.
        metric[1,1] = metric[2,2] = 1.
        spatial_dims = 2. 

        # Computing the Favre density and velocity
        n_t = np.sqrt(-Base.Mink_dot(BC, BC))
        u_t = np.multiply(1./ n_t, BC)

        # remember SET is a rank (2,0) tensor
        # Computing the SET decomposition at each point
        eps_t = np.einsum('i,j,ik,jl,kl', u_t, u_t, metric, metric, SET)
        h_ab = np.einsum('ij,jk->ik', metric + np.einsum('i,j->ij', u_t, u_t), metric) # This is a rank (1,1) tensor, i.e. a real projector.
        q_a = - np.einsum('ij,jk,kl,l->i', h_ab, SET, metric, u_t)
        s_ab = np.einsum('ij,kl,jl->ik', h_ab, h_ab, SET)
        s = np.einsum('ij,ji->', s_ab, metric)
        # s = np.multiply(1/spatial_dims, s)
        s_ab_tracefree = s_ab - np.multiply(s/3, metric + np.einsum('i,j->ij', u_t, u_t)) 

        p_t = resHD2D.p_Gamma_law(eps_t, n_t, 4.0/3.0) 
        # rechange this: this was just to test the impact of EOS residuals onto the modelling of zeta
        # Pi_res = s - p_t
        Pi_res = s - p_filt
        EOS_res = p_filt - p_t
        T_t = p_t/n_t
        
        return [n_t, u_t, eps_t, q_a, s_ab_tracefree, p_t, Pi_res, EOS_res, T_t], [h,i,j] #Uncomment if EoS residual is modelled
        # return [n_t, u_t, eps_t, q_a, s_ab_tracefree, p_t, Pi_res, T_t], [h,i,j]

    def decompose_structures_parallel(self, n_cpus):
        """
        Routine to decompose structures on the entire grid in 
        parallel. 

        Parameters:
        -----------
        n_cpus: int
            numbers of processes for parallelization
        """
        # Preparing arguments for pool 
        args_for_pool=[]
        for h in range(len(self.domain_vars['T'])):
            for i in range(len(self.domain_vars['X'])):
                for j in range(len(self.domain_vars['Y'])):
                    BC = self.meso_structures['BC'][h,i,j]
                    SET = self.meso_structures['SET'][h,i,j]
                    p_filt = self.meso_vars['p_filt'][h,i,j]
                    args_for_pool.append((BC, SET, p_filt, h,i,j))

        with mp.Pool(processes=n_cpus) as pool:
            print('Decomposing structures in parallel with {} processes'.format(pool._processes), flush=True)
            for result in pool.starmap(resHD2D.decompose_structures_task, args_for_pool):
                h,i,j = result[1]
                self.meso_vars['n_tilde'][h,i,j] = result[0][0]
                self.meso_vars['u_tilde'][h,i,j,:] = result[0][1]
                self.meso_vars['eps_tilde'][h,i,j] = result[0][2]
                self.meso_vars['q_res'][h,i,j,:] = result[0][3]
                self.meso_vars['pi_res'][h,i,j,:,:] = result[0][4]
                self.meso_vars['p_tilde'][h,i,j] =  result[0][5]
                self.meso_vars['Pi_res'][h,i,j] = result[0][6]
                self.meso_vars['eos_res'][h,i,j] = result[0][7] #If EoS is not modelled you should adjust this!
                self.meso_vars['T_tilde'][h,i,j] = result[0][8] 

    def calculate_derivatives_gridpoint(self, nonlocal_var_str, h, i, j, direction, order = 1): 
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
                                self.deriv_vars[dstr][h,i,j,dir] = self.calculate_derivatives_gridpoint(str, h, i, j, dir)
                    else: 
                        print('Derivatives not calculated at {}: observer could not be found.'.format(point))

    def closure_ingredients_gridpoint(self, h, i, j):
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

        Rename: closure_ingredients_gridpoint() ??
        Coefficients can be computed here directly, but more general models will require more work.
        Do not split this, at least for now
        """
        #  FAVRE_OBS    
        u_t = self.meso_vars['u_tilde'][h,i,j]
        u_t_cov = np.einsum('ij,j->i', self.metric, u_t)

        # CLOSURE INGREDIENTS: FAVRE OBS DERIVATIVE DECOMPOSITION - WORKING WITH (2,0)
        nabla_u = self.deriv_vars['D_u_tilde'][h,i,j] # This is a rank (1,1) tensor
        nabla_u = np.einsum('ij,jk->ik', self.metric, nabla_u) #this is a rank (2,0) tensor
        acc_t = np.einsum('i,ij', u_t_cov, nabla_u) # vector

        # The following two quantities should be identically zero, but they won't be due to numerical errors.
        # Computing them and removing to project velocity gradients
        unit_norm_violation = np.einsum('ij,j', nabla_u, u_t_cov) #vector
        acc_orthogonality_violation = np.einsum('i,i->', u_t_cov, acc_t)
        Daub = nabla_u + np.einsum('i,j->ij', u_t, acc_t) + np.einsum('i,j->ij', unit_norm_violation, u_t) +\
            np.multiply(acc_orthogonality_violation, np.einsum('i,j->ij', u_t, u_t)) #Should be a (2,0) tensor
        h_ab = self.metric + np.einsum('i,j->ij', u_t, u_t)
        exp_t = np.einsum('ii->',Daub)
        shear_t = np.multiply(1/2., Daub + np.einsum('ij->ji', Daub)) - np.multiply( 1/self.spatial_dims * exp_t, h_ab) 
        vort_t = np.multiply(1/2., Daub - np.einsum('ij->ji', Daub))

    
        # CLOSURE INGREDIENTS: HEAT FLUX 
        nabla_T = self.deriv_vars['D_T_tilde'][h,i,j]
        projector = np.zeros((3,3))
        np.fill_diagonal(projector, 1) 
        projector += np.einsum('i,j->ij',u_t_cov, u_t)
        DaT = np.einsum('ij,j->i', projector, nabla_T)
        Theta_tilde = DaT + np.multiply(self.meso_vars['T_tilde'][h,i,j], np.einsum('ij,j->i', self.metric, acc_t))

        closure_vars_strs = ['shear_tilde', 'exp_tilde', 'acc_tilde', 'Theta_tilde']
        closure_vars = [shear_t, exp_t, acc_t, Theta_tilde]
        return closure_vars_strs, closure_vars
    
    def closure_ingredients(self): 
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
                        keys, values = self.closure_ingredients_gridpoint(h,i,j)
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
    
    @staticmethod
    def closure_ingredients_task(u_t, nabla_u, T_t, nabla_T, nabla_n, h, i, j):
        """
        Task to decompose Favre obs + Temperature derivatives
        These will be used in EL_style_closure to extract the 
        turbulent effective dissipative coefficients

        Parameters:
        -----------
        u_t: np.array (3,)
            the Favre-observer, rank: (1,0)

        nabla_u: np.array(3,3)
            derivatives of Favre observer, rank: (1,1)
        
        T_t: float
            The temperature from filtered EoS 
        
        nabla_T: np.array (3,0)
            Temperature derivatives, rank: (0,1)

        h,i,j: integers
            the indices of point on meso-grid

        Returns: 
        --------
        (list, list, list):
            first list contains the strings of the returned quantities 
            (needed as the meso-vars dictionary is extended appropriately)

            second list contains the corresponding np.arrays
            
            third list contains the indices on meso-grid
        """
        # building blocks: this task is static so no access to self
        spatial_dims = 2.
        metric = np.zeros((3,3))
        metric[0,0] = -1
        metric[1,1] = metric[2,2] = 1

        # CLOSURE INGREDIENTS: FAVRE OBS DERIVATIVE DECOMPOSITION - WORKING WITH (2,0)
        # The decomposition below is exact if certain algebraic constraints are satisfied, which they won't be
        # due to numerical errors. 
        # However, it appears the only quantity to be corrected is the acceleration
        u_t_cov = np.einsum('ij,j->i', metric, u_t)
        nabla_u = np.einsum('ij,jk->ik', metric, nabla_u) #this is a rank (2,0) tensor
        acc_t = np.einsum('i,ij', u_t_cov, nabla_u) # vector

        acc_orthogonality_violation = np.einsum('i,i->', u_t_cov, acc_t)
        acc_t = acc_t + acc_orthogonality_violation * u_t 

        projector = np.zeros((3,3))
        np.fill_diagonal(projector, 1) 
        projector += np.einsum('i,j->ij',u_t, u_t_cov)
        h_ab = metric + np.einsum('i,j->ij', u_t, u_t) #Rank (2,0) tensor

        Daub = np.einsum('ij,kl,jl->ik', projector, projector, nabla_u)
        exp_t = np.einsum('ij,ji->', Daub, metric) 
        shear_t = np.multiply(1/2., Daub + np.einsum('ij->ji', Daub)) - np.multiply( exp_t/ spatial_dims, h_ab) 
        vort_t = np.multiply(1/2., Daub - np.einsum('ij->ji', Daub))

        # #The following should be used if the violation of the algebraic constraints become too large
        # Daub = np.einsum('ij,kl,jl->ik', projector, projector, nabla_u)
        # orthogonality_violation_1 = np.einsum('i,ij->j', u_t_cov, Daub)
        # orthogonality_violation_2 = np.einsum('ij,j->i', Daub, u_t_cov)
        # orthogonality_violation_3 = np.einsum('i,j,ij->', u_t_cov, u_t_cov, Daub)
        # Daub = Daub + np.einsum('i,j->ij', u_t, orthogonality_violation_1) + np.einsum('i,j->ij', orthogonality_violation_2, u_t) + \
        #     np.multiply(orthogonality_violation_3, np.einsum('i,j->ij', u_t, u_t) )
        # exp_t = np.einsum('ij,ji->', Daub, metric) 
        # shear_t = np.multiply(1/2., Daub + np.einsum('ij->ji', Daub)) - np.multiply( exp_t/ spatial_dims, h_ab) 
        # vort_t = np.multiply(1/2., Daub - np.einsum('ij->ji', Daub))

        
        # CLOSURE INGREDIENTS:  DERIVATIVES OF N_TILDE AND T_TILDE  
        projector = np.zeros((3,3))
        np.fill_diagonal(projector, 1)
        projector += np.einsum('i,j->ij',u_t_cov, u_t) #This is different from the projector above: look at indices!

        sD_n_tilde = np.einsum('ij,j->i', projector, nabla_n)
        sD_T_tilde = np.einsum('ij,j->i', projector, nabla_T)

        n_tilde_dot = np.einsum('i,i->', u_t, nabla_n)
        T_tilde_dot = np.einsum('i,i->', u_t, nabla_T)

        Theta_t = sD_T_tilde + np.multiply(T_t, np.einsum('ij,j->i', metric, acc_t))

        closure_vars_strs = ['shear_tilde', 'exp_tilde', 'acc_tilde', 'vort_tilde','Theta_tilde', 'n_tilde_dot', \
                             'T_tilde_dot', 'sD_n_tilde', 'sD_T_tilde' ]
        closure_vars = [shear_t, exp_t, acc_t, vort_t, Theta_t, n_tilde_dot, T_tilde_dot, sD_n_tilde, sD_T_tilde]
        return closure_vars_strs, closure_vars, [h,i,j]

    def closure_ingredients_parallel(self, n_cpus):
        """
        Routine to compute and store the closure ingredients at all gridpoints in parallel

        Parameters:
        -----------
        n_cpus: int
            number of processes for parallelization

        Notes: 
        ------
        worth thinking about merging this with decompose structures?     
        """
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        args_for_pool = []
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    u_t = self.meso_vars['u_tilde'][h,i,j]
                    nabla_u = self.deriv_vars['D_u_tilde'][h,i,j]
                    T_t = self.meso_vars['T_tilde'][h,i,j]
                    nabla_T = self.deriv_vars['D_T_tilde'][h,i,j]
                    nabla_n = self.deriv_vars['D_n_tilde'][h,i,j]
                    args_for_pool.append((u_t, nabla_u, T_t, nabla_T, nabla_n, h, i, j))

        with mp.Pool(processes=n_cpus) as pool:
            print('Computing closure ingredients with {} processes'.format(pool._processes), flush=True)
            for result in pool.starmap(resHD2D.closure_ingredients_task, args_for_pool):
                keys, values, grid_idxs = result
                # if self.filter_vars['U_success'][tuple(grid_idxs)]: #this check is in practice always True
                for idx, key in enumerate(keys):
                    try:
                        self.meso_vars[key]
                    except KeyError:
                        print('The key {} does not belong to meso_vars yet, adding it!'.format(key), flush=True)
                        shape = values[idx].shape
                        self.meso_vars.update({key : np.zeros(([Nt,Nx,Ny]+ list(shape)))})
                    finally: 
                        self.meso_vars[key][tuple(grid_idxs)] = values[idx]

    def EL_style_closure_gridpoint(self, h, i, j):
        """
        Compute bulk, shear via scalarization + PA trick, thermal conductivity later. 
        At a point. 
        Returns: coefficient(s) at a point.
        """
        coefficients_names=[]
        coefficients=[]
        
        # CALCULATING BULK VISCOUS COEFF
        zeta = self.meso_vars['Pi_res'][h,i,j] / self.meso_vars['exp_tilde'][h,i,j] 
        coefficients_names.append('zeta')
        coefficients.append(zeta)

        # CALCULATING SHEAR VISCOUS COEFF
        pi_res_sq = np.einsum('ij,kl,ik,jl->', self.meso_vars['pi_res'][h,i,j], self.meso_vars['pi_res'][h,i,j], self.metric, self.metric)
        shear_sq = np.einsum('ij,kl,ik,jl->', self.meso_vars['shear_tilde'][h,i,j], self.meso_vars['shear_tilde'][h,i,j], self.metric, self.metric)
        eta = pi_res_sq/shear_sq
        # Compute sign of eta by looking at the PA of pi_res with higher eigenvalue. 
        # Change shear to the eigenbasis of pi_res (using that the matrix is unitary as pi_res is sym)
        pi_eig, pi_eigv = np.linalg.eigh(self.meso_vars['pi_res'][h,i,j])
        transformed_shear = np.einsum('ji,jk,kl', pi_eigv, self.meso_vars['shear_tilde'][h,i,j], pi_eigv)
        abs_pi_eig = np.abs(pi_eig)
        pos = list(abs_pi_eig).index(np.max(abs_pi_eig))
        eta = eta * np.sign(pi_eig[pos] / transformed_shear[pos,pos])
        coefficients_names.append('eta')
        coefficients.append(eta)

        # CALCULATING THE HEAT CONDUCTIVITIY
        q_res = self.meso_vars['q_res'][h,i,j]
        Theta_t = self.meso_vars['Theta_tilde'][h,i,j]
        q_res_sq = np.einsum('i,ij,j', q_res, self.metric, q_res)
        Theta_t_sq = np.einsum('i,ij,j', Theta_t, self.metric, Theta_t)
        cos_angle = np.einsum('i,ij,j', q_res, self.metric, Theta_t) / (q_res_sq * Theta_t_sq)
        sign = np.sign(cos_angle)
        kappa = sign * np.sqrt(q_res_sq / Theta_t_sq)
        coefficients_names.append('kappa')
        coefficients.append(kappa)

        return coefficients_names, coefficients

    def EL_style_closure(self): 
        """
        Wrapper of EL_style_closure_gridpoint()    
        """
        Nt, Nx, Ny = self.domain_vars['Nt'], self.domain_vars['Nx'], self.domain_vars['Ny']
        for h in range(Nt): 
            for i in range(Nx): 
                for j in range(Ny): 
                    if self.filter_vars['U_success'][h,i,j]: 
                        keys, values = self.EL_style_closure_gridpoint(h,i,j)
                        # try-except block to extend the meso_vars dictionary with the dissipative coefficients
                        for idx, key in enumerate(keys): 
                            try: 
                                self.meso_vars[key]
                            except KeyError:
                                print('The key {} does not belong to meso_vars yet, adding it!'.format(key))
                                shape = values[idx].shape
                                self.meso_vars.update({key: np.zeros(([Nt, Nx, Ny] + list(shape)))})
                            finally: 
                                    self.meso_vars[key][h,i,j] = values[idx]

    @staticmethod
    def EL_style_closure_task(Pi_res, exp, pi_res, shear, q_res, Theta, h, i, j):
        """
        Task for computing the effective dissipative coefficients at all gridpoint 
        in parallel

        Parameters:
        -----------
        Pi_res: float
            Pressure residual using meso EoS

        exp: float
            the Favre expansion rate 

        pi_res: np.array (3,3)
            the residual anisotropic stresses (should be trace-free)

        shear: np.array (3,3)
            the shear matrix computed from Favre obs derivatives
        
        q_res: np.array (3,)
            the residual heat-flux
        
        Theta: np.array (3,)
            the temperature spatial gradients with acceleration term

        h,i,j: integers
            grid indixes corresponding to point on grid

        Return: 
        -------
        (coeff names, coeff, [h,i,j])
            
        """
        coefficients_names=[]
        coefficients=[]

        metric = np.zeros((3,3))
        metric[0,0] = -1
        metric[1,1] = metric[2,2] = +1

        # CALCULATING BULK VISCOUS COEFF
        zeta = - Pi_res / exp 
        coefficients_names.append('zeta')
        coefficients.append(zeta)

        # CALCULATING SHEAR VISCOUS COEFF
        pi_res_sq = np.einsum('ij,kl,ik,jl->', pi_res, pi_res, metric, metric)
        shear_sq = np.einsum('ij,kl,ik,jl->', shear, shear, metric, metric)
        eta = np.sqrt(pi_res_sq/shear_sq)
        # Compute sign of eta by looking at the PA of pi_res with higher eigenvalue. 
        # Change shear to the eigenbasis of pi_res (using that the matrix is unitary as pi_res is sym)
        pi_eig, pi_eigv = np.linalg.eigh(pi_res)
        transformed_shear = np.einsum('ji,jk,kl', pi_eigv, shear, pi_eigv)
        abs_pi_eig = np.abs(pi_eig)
        pos = list(abs_pi_eig).index(np.max(abs_pi_eig))
        eta = eta * np.sign( - pi_eig[pos] / transformed_shear[pos,pos])
        coefficients_names.append('eta')
        coefficients.append(eta)

        # CALCULATING THE HEAT CONDUCTIVITIY
        q_res_sq = np.einsum('i,ij,j', q_res, metric, q_res)
        Theta_sq = np.einsum('i,ij,j', Theta, metric, Theta)
        cos_angle = np.einsum('i,ij,j', q_res, metric, Theta) / (q_res_sq * Theta_sq)
        sign = - np.sign(cos_angle)
        kappa = sign * np.sqrt(q_res_sq / Theta_sq)
        coefficients_names.append('kappa')
        coefficients.append(kappa)
        

        return coefficients_names, coefficients, [h,i,j]

    def EL_style_closure_parallel(self, n_cpus):
        """
        Routine to launch EL_style_closure_task in parallel across multiple gridpoints

        Parameters:
        -----------
        n_cpus: int
            number of processes for parallelization

        Notes:
        ------
        worth thinking about merging this with decompose_structures_task and closure_ingredients_task? 
        could compute just u_t at all points, so to be able to take derivatives of it
        Then since you're accesing a single point, you could decompose, compute ingredients and extract coeff
        in one go. Might give you a speed up: the loop for preparning arguments for pool is done once
        and not twice/thrice, and the processes are opened/closed half the times. 
        """
        args_for_pool=[]

        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']


        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    Pi_res = self.meso_vars['Pi_res'][h,i,j]
                    pi_res = self.meso_vars['pi_res'][h,i,j]
                    q_res = self.meso_vars['q_res'][h,i,j]
                    exp_t = self.meso_vars['exp_tilde'][h,i,j]
                    shear_t = self.meso_vars['shear_tilde'][h,i,j]
                    Theta_t = self.meso_vars['Theta_tilde'][h,i,j]

                    args_for_pool.append((Pi_res, exp_t, pi_res, shear_t, q_res, Theta_t, h,i,j))


        with mp.Pool(processes=n_cpus) as pool: 
            print('Computing dissipative coefficients with {} processes'.format(pool._processes), flush=True)
            for result in pool.starmap(resHD2D.EL_style_closure_task, args_for_pool):
                keys, values, grid_idxs = result
                # if self.filter_vars['U_success'][tuple(grid_idxs)]: #this check is in practice always True
                for idx, key in enumerate(keys):
                    try:
                        self.meso_vars[key]
                    except KeyError:
                        print('The key {} does not belong to meso_vars yet, adding it!'.format(key), flush=True)
                        shape = values[idx].shape
                        self.meso_vars.update({key : np.zeros(([Nt,Nx,Ny]+ list(shape)))})
                    finally: 
                        self.meso_vars[key][tuple(grid_idxs)] = values[idx]

    @staticmethod 
    def EL_componentwise_task(pi_res, shear, h, i, j):
        """
        Task for computing the shear coefficients componentwise

        Parameters:
        -----------
        pi_res: nd.array
            the anisotropic stress residual at gridpoint h,i,j

        shear: nd.array
            the (Favre-observer) shear tensor at gridpoint h,i,j

        Returns:
        --------
        list of strs:
            the names by which you want to store data in the class instance
        
        nd.array:
            the componentwise values of the coefficient, for now eta only
        
        list:
            the gridpoint indices on the meso-grid

        Notes:
        ------
        """
        eta_componentwise = np.zeros(pi_res.shape)
        eta_componentwise =  - np.divide(pi_res, shear)
        coeff_name = 'eta_cw'

        return [coeff_name], [eta_componentwise], [h, i, j]
    
    def EL_componentwise_parallel(self, n_cpus, store=False):
        """
        Wrapper of EL_componentwise_task: launch the task in parallel with n_cpus

        Parameters:
        -----------
        n_cpus: int
            number of processes for parallelization

        store: bool 
            If true, the meso_vars dictionary is extended to include these 
            Otherwise, the dictionary is created and returned
        """
        args_for_pool=[]

        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        printshit = self.meso_vars['shear_tilde'].shape
        print(f'shear_tilde.shape: {printshit}')
        printshit = self.meso_vars['pi_res'].shape
        print(f'pi_res.shape: {printshit}')

        for h in range(Nt): 
            for i in range(Nx):
                for j in range(Ny):
                    shear_t = self.meso_vars['shear_tilde'][h,i,j]
                    pi_res = self.meso_vars['pi_res'][h,i,j]

                    args_for_pool.append((pi_res, shear_t, h, i, j))

        with mp.Pool(processes=n_cpus) as pool: 
            print('Computing vars for modelling coefficients with {} processes'.format(pool._processes), flush=True) 

            for result in pool.starmap(resHD2D.EL_componentwise_task, args_for_pool):
                keys, values, grid_idxs = result

                if store: 
                    for idx, key in enumerate(keys):
                        try:
                            self.meso_vars[key]
                        except KeyError:
                            print('The key {} does not belong to meso_vars yet, adding it!'.format(key), flush=True)
                            try: 
                                shape = values[idx].shape
                            except AttributeError: 
                                shape = []
                            self.meso_vars.update({key : np.zeros(([Nt,Nx,Ny]+ list(shape)))})
                        finally: 
                            self.meso_vars[key][tuple(grid_idxs)] = values[idx]
                
                else:
                    for i, key in enumerate(keys):
                        try:
                            results_dictionary[key]
                        except UnboundLocalError:
                            results_dictionary = dict.fromkeys([key])
                            shape = values[i].shape
                            results_dictionary[key] = np.zeros(([Nt, Nx, Ny] + list(shape)))
                        except KeyError:
                            shape = values[i].shape
                            results_dictionary.update( {key : np.zeros(([Nt, Nx, Ny] + list(shape)))} )

                        finally:
                            results_dictionary[key][tuple(grid_idxs)] = values[i]

        if not store:
            return results_dictionary

    @staticmethod
    def modelling_coefficients_task(shear, vort, acc, Theta, sD_n_tilde, Pi_res, pi_res, q_res, h, i, j):
        """
        Task (to be used in parallel) to compute various quantities that will be used later 
        for modelling the extracted closure coefficients. 
        """
        var_names = []
        vars = []

        metric = np.zeros((3,3))
        metric[0,0] = -1
        metric[1,1] = metric[2,2] = +1

        # Computing various invariants of the velocity gradients

        # This is required as the determinant is an invariant (does not change under coordinates changes) 
        # only when the tensor is written as a rank (1,1)
        shear_rank11 = np.einsum('ij,jk->ik', shear, metric)
        det_shear = det(shear_rank11)
        var_names.append('det_shear')
        vars.append(det_shear)

        shear_sq = np.einsum('ij,kl,ik,jl->', shear, shear, metric, metric)
        var_names.append('shear_sq')
        vars.append(shear_sq)

        vort_sq = np.einsum('ij,kl,ik,jl->', vort, vort, metric, metric)
        var_names.append('vort_sq')
        vars.append(vort_sq)

        acc_mag = np.sqrt(np.einsum('i,ij,j->', acc, metric, acc))
        var_names.append('acc_mag')
        vars.append(acc_mag)

        # Computing quantities related to the Q-criterion
        Q1 = shear_sq - vort_sq
        var_names.append('Q1')
        vars.append(Q1)

        Q2 = shear_sq/vort_sq
        var_names.append('Q2')
        vars.append(Q2)

        # Computing scalars out of thermo gradients
        Theta_sq = np.einsum('i,ij,j', Theta, metric, Theta)
        var_names.append('Theta_sq')
        vars.append(Theta_sq)

        sD_n_tilde_sq = np.einsum('i,ij,j->', sD_n_tilde, metric, sD_n_tilde)
        var_names.append('sD_n_tilde_sq')
        vars.append(sD_n_tilde_sq)

        dot_Dn_Theta = np.einsum('i,ij,j->', sD_n_tilde, metric, Theta )
        var_names.append('dot_Dn_Theta')
        vars.append(dot_Dn_Theta)

        # Computing squares of residuals and also of Theta_tilde
        Pi_res_sq = Pi_res * Pi_res
        var_names.append('Pi_res_sq')
        vars.append(Pi_res_sq)

        pi_res_sq = np.einsum('ij,kl,ik,jl->', pi_res, pi_res, metric, metric)
        var_names.append('pi_res_sq')
        vars.append(pi_res_sq)

        q_res_sq = np.einsum('i,ij,j', q_res, metric, q_res)
        var_names.append('q_res_sq')
        vars.append(q_res_sq)

        return var_names, vars, [h,i,j]
    
    def modelling_coefficients_parallel(self, n_cpus):
        """
        Routine to parallelize the task 'Computing_calibration_vars_task'
        """
        args_for_pool=[]

        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        for h in range(Nt): 
            for i in range(Nx):
                for j in range(Ny):
                    shear_t = self.meso_vars['shear_tilde'][h,i,j]
                    vort_t = self.meso_vars['vort_tilde'][h,i,j]
                    acc_t = self.meso_vars['acc_tilde'][h,i,j]
                    Theta_t = self.meso_vars['Theta_tilde'][h,i,j]  
                    sD_n_tilde = self.meso_vars['sD_n_tilde'][h,i,j]
                    Pi_res = self.meso_vars['Pi_res'][h,i,j]
                    pi_res = self.meso_vars['pi_res'][h,i,j]
                    q_res = self.meso_vars['q_res'][h,i,j]

                    args_for_pool.append((shear_t, vort_t, acc_t, Theta_t, sD_n_tilde, Pi_res, pi_res, q_res, h, i, j))

        with mp.Pool(processes=n_cpus) as pool: 
            print('Computing vars for modelling coefficients with {} processes'.format(pool._processes), flush=True)
            
            for result in pool.starmap(resHD2D.modelling_coefficients_task, args_for_pool):
                keys, values, grid_idxs = result

                for idx, key in enumerate(keys):
                    try:
                        self.meso_vars[key]
                    except KeyError:
                        print('The key {} does not belong to meso_vars yet, adding it!'.format(key), flush=True)
                        try: 
                            shape = values[idx].shape
                        except AttributeError: 
                            shape = []
                        self.meso_vars.update({key : np.zeros(([Nt,Nx,Ny]+ list(shape)))})
                    finally: 
                        self.meso_vars[key][tuple(grid_idxs)] = values[idx]

    def weights_Q1_skew(self):
        """
        Build weights for gridpoints: downplay points where shear is small, but take into 
        account both positive and negative values of Q1
        """
        # def symlog(array):
        #     return np.sign(array) * np.log10(np.abs(array)+1) 
        
        Q1 = self.meso_vars['Q1']
        symlog_Q1 = MySymLogPlotting.symlog_var(Q1)
        
        M_pos = np.amax(symlog_Q1)
        m_neg = np.amin(symlog_Q1)

        symlog_Q1_pos = np.ma.masked_where(symlog_Q1 < 0, symlog_Q1, copy=True).compressed()
        symlog_Q1_neg = np.ma.masked_where(symlog_Q1 > 0, symlog_Q1, copy=True).compressed()
        M_neg = np.amax(symlog_Q1_neg)
        m_pos = np.amin(symlog_Q1_pos)

        c_pos = (M_pos + m_pos)/2.
        c_neg = (M_neg + m_neg)/2.

        def get_weights(x, c_pos, c_neg):
            if x >= 0: 
                result = (np.tanh(x-c_pos)+1)/2
            else: 
                result = (-np.tanh(x-c_neg)+1)/2
            return (result+1.)/2.
        
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        weights = np.zeros((Nt, Nx, Ny))
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    weights[h,i,j] = get_weights(symlog_Q1[h,i,j], c_pos, c_neg)

        self.meso_vars.update({'weights' : weights})

    def weights_Q1_non_neg(self):
        """
        Build weights for gridpoints: downplay points where Q1 is negative
        """
        # def symlog(array):
        #     return np.sign(array) * np.log10(np.abs(array)+1) 
        
        Q1 = self.meso_vars['Q1']
        symlog_Q1 = MySymLogPlotting.symlog_var(Q1)
        symlog_Q1_pos = np.ma.masked_where(symlog_Q1 < 0, symlog_Q1, copy=True).compressed()
        symlog_Q1_neg = np.ma.masked_where(symlog_Q1 > 0, symlog_Q1, copy=True).compressed()

        M_pos = np.amax(symlog_Q1_pos)
        m_pos = np.amin(symlog_Q1_pos)
        M_neg = np.amax(symlog_Q1_neg)
        m_neg = np.amin(symlog_Q1_neg)

        c = (M_pos + m_neg)/2.
        a_pos = (M_pos + m_pos)/2.
        a_neg = np.abs(M_neg + m_neg)/2.
        a = (a_pos + a_neg)/2.

        def get_weights(x, c, a):
            result = np.tanh((x-c)/a)
            return (result+1.)/2.
        
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        weights = np.zeros((Nt, Nx, Ny))
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    weights[h,i,j] = get_weights(symlog_Q1[h,i,j], c, a)

        self.meso_vars.update({'weights' : weights})

    def weights_Q2(self): 
        """
        Build weights for gridpoints based on Q2
        """
        Q2 = np.log10(self.meso_vars['Q2'])

        M = np.amax(Q2)
        m = np.amin(Q2)
        scale = (M+m)/2

        def get_weights(x, scale):
            result = np.tanh(x/scale)
            result = (result +1)/2
            return result
        
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        weights = np.zeros((Nt, Nx, Ny))
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    weights[h,i,j] = get_weights(Q2[h,i,j], scale)

        self.meso_vars.update({'weights' : weights})

    def residual_weights(self, residual_str):
        """
        Build weights based on residual corresponding to input 'residual_str'. 
        Downplay points where this is small, that is those points where a closure is less required!
        """

        residual = np.log10(self.meso_vars[residual_str])

        M = np.amax(residual)
        m = np.amin(residual)
        scale = (M+m)/2

        def get_weights(x, scale):
            result = np.tanh(x/scale)
            result = (result +1)/2
            return result
        
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        weights = np.zeros((Nt, Nx, Ny))
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    weights[h,i,j] = get_weights(residual[h,i,j], scale)

        self.meso_vars.update({'weights' : weights})

    def denominator_weights(self, coeff_denominator_str):
        """
        Build weights based on quantity corresponding to 'coeff_denominator_str'. 
        Downplay points where this is small, that is points where the extracted coefficient are less 
        trustworthy
        """

        denominator = np.log10(self.meso_vars[coeff_denominator_str])

        M = np.amax(denominator)
        m = np.amin(denominator)
        scale = (M+m)/2

        def get_weights(x, scale):
            result = np.tanh(x/scale)
            result = (result +1)/2
            return result
        
        Nt = self.domain_vars['Nt']
        Nx = self.domain_vars['Nx']
        Ny = self.domain_vars['Ny']

        weights = np.zeros((Nt, Nx, Ny))
        for h in range(Nt):
            for i in range(Nx):
                for j in range(Ny):
                    weights[h,i,j] = get_weights(denominator[h,i,j], scale)

        self.meso_vars.update({'weights' : weights})

    def EL_style_closure_regression(self, CoefficientAnalysis):
        """
        Takes in a list of correlation quantities (strings? The regressors needed are computed 
        previously via closure ingredients)
        Pass the dataset to a CoefficientAnalysis instance. 
        Plot the correlations with the regressors and perform the regression using methods from a Coefficient 
        Analysis instance. 
        """
        # FICTITIOUS RANGES AND POINTS TO TEST TRIM_DATA ROUTINE
        ranges = [[self.domain_vars['Tmin'], self.domain_vars['Tmax']],\
                [self.domain_vars['Xmin'], self.domain_vars['Xmax']],\
                [self.domain_vars['Ymin'], self.domain_vars['Ymax']]]
        points = self.domain_vars['Points']

        # TESTING SCALAR REGRESSION ROUTINE
        # y = self.meso_vars['zeta']
        # X = [self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # stats_result = CoefficientAnalysis.scalar_regression(y, X, ranges, points)
        # print(*stats_result)

        # TESTING TENSOR REGRESSION ROUTINE
        # y=self.meso_vars['q_res']
        # X=[self.meso_vars['e_tilde'], self.meso_vars['u_tilde']]
        # # stats_result = CoefficientAnalysis.tensor_components_regression(y, X, 2,ranges, points, components=[(0,),(2,)])
        # stats_result = CoefficientAnalysis.tensor_components_regression(y, X, 2,ranges, points, add_intercept=False)
        # for i in range(len(stats_result)):
        #     print('{}-th component: \n {} \n\n'.format(i, stats_result[i]))

        # TESTING CORRELATION VISUALIZATION ROUTINES
        # x = self.meso_vars['n_tilde']
        # y = self.meso_vars['eps_tilde']
        # labels = ['n_tilde','eps_tilde']
        # g1=CoefficientAnalysis.visualize_correlation(x, y, xlabel=labels[0], ylabel=labels[1])
        # fig=g1.figure
        # fig.savefig('Single_correlation_plot.pdf', format='pdf')

        # data=[self.meso_vars['eta'], self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['eta', 'b_tilde', 'eps_tilde', 'n_tilde']
        # g2=CoefficientAnalysis.visualize_many_correlations(data, labels)

        # data=[self.meso_vars['zeta'], self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['zeta', 'b_tilde', 'eps_tilde', 'n_tilde']
        # gbis= CoefficientAnalysis.visualize_many_correlations(data, labels)
        # # plt.savefig('Many_correlations_plot.pdf', format='pdf')
        # plt.show()
        
        # TESTING THE CHECK REGRESSORS ROUTINE:
        # data=[self.meso_vars['b_tilde'], self.meso_vars['eps_tilde'], self.meso_vars['n_tilde']]
        # labels=['b_tilde', 'eps_tilde', 'n_tilde']
        # g1=CoefficientAnalysis.visualize_many_correlations(data, labels)
        # comps, g2 = CoefficientAnalysis.PCA_find_regressors_subset(data, ranges = ranges, model_points = points)
        # g2.fig.suptitle("Correlation between PC (standardized data)")
        # g2.fig.subplots_adjust(top=0.94) 
        # print(comps)
        # plt.show()

if __name__ == '__main__':



    ########################################################
    # TESTING SERIAL IMPLEMENTATION
    ######################################################## 
    # FileReader = METHOD_HDF5('/Users/thomas/Dropbox/Work/projects/Filtering/Data/test_res100/')
    # micro_model = IdealHD_2D()
    # FileReader.read_in_data(micro_model) 
    # micro_model.setup_structures()
    # find_obs = FindObs_drift_root(micro_model, 0.001)
    # filter = spatial_box_filter(micro_model, 0.003)


    # CPU_start_time = time.process_time()
    # meso_model = resHD2D(micro_model, find_obs, filter)

    # # print(micro_model.domain_vars['tmin'], micro_model.domain_vars['tmax'])

    # t_range = [1.502, 1.505]
    # x_range = [0.29, 0.36]
    # y_range = [0.39, 0.46]

    # meso_model.setup_meso_grid([t_range, x_range, y_range])
    # meso_model.find_observers()
    # meso_model.filter_micro_variables()

    
    # print('Filter stage ended')
    # # # meso_model.decompose_structures_gridpoint(1,1,1)
    # meso_model.decompose_structures()

    # print('Decomposition stage ended') 
    # # print(meso_model.calculate_derivative_gridpoint('u_tilde', 0,1,1,0))
    # meso_model.calculate_derivatives()

    # # meso_model.closure_ingredients_gridpoint(1,1,0)
    # # meso_model.closure_ingredients()
    # # print('Derivatives and Decomposition stage ended')

    # # meso_model.EL_style_closure_gridpoint(1,2,3)
    # # meso_model.EL_style_closure()
    # # regressor = CoefficientsAnalysis()
    # # meso_model.EL_style_closure_regression(regressor)

    # # print('Total time is {}'.format(time.process_time() - CPU_start_time))    


    # ########################################################
    # # TESTING PARALLEL IMPLEMENTATION: find obs + filter
    # ######################################################## 
    # FileReader = METHOD_HDF5('../Data/test_res100/')
    # micro_model = IdealHD_2D()
    # FileReader.read_in_data(micro_model) 
    # micro_model.setup_structures()

    # t_range = [1.502, 1.504]
    # x_range = [0.05, 0.95]
    # y_range = [0.05, 0.95]

    
    # # find obs - serial 
    # start_time = time.perf_counter()
    # find_obs = FindObs_drift_root(micro_model, 0.001)
    # filter = spatial_box_filter(micro_model, 0.003)
    # meso_model = resHD2D(micro_model, find_obs, filter)
    # meso_model.setup_meso_grid([t_range, x_range, y_range])

    # num_points = meso_model.domain_vars['Nt'] * meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny']
    # print(f'Testing parallelization with {num_points} points\n')
    
    # meso_model.find_observers()
    # serial_time = time.perf_counter() - start_time
    # print('Serial execution time: {}\n'.format(serial_time))

    
    # # fin obs - parallel
    # start_time = time.perf_counter()
    # find_obs = FindObs_root_parallel(micro_model, 0.001)
    # filter = spatial_box_filter(micro_model, 0.003)
    # meso_model = resHD2D(micro_model, find_obs, filter)
    # meso_model.setup_meso_grid([t_range, x_range, y_range])
    # meso_model.find_observers_parallel()
    # parallel_time = time.perf_counter() - start_time
    # print('Finished finding observers in parallel, execution time: {}\n'.format(parallel_time))
    # print('Speed-up factor: {}'.format(serial_time/parallel_time))


    # # now filtering serial
    # start_time = time.perf_counter()
    # meso_model.filter_micro_variables()
    # serial_time = time.perf_counter() - start_time
    # print('Finished filtering in serial, time taken {}\n'.format(serial_time))

    # # now filtering in parallel
    # parallel_filter = box_filter_parallel(micro_model, 0.003)
    # meso_model.set_filter(parallel_filter)
    # start_time = time.perf_counter()
    # meso_model.filter_micro_vars_parallel()
    # parallel_time = time.perf_counter() - start_time
    # print('Finished filtering in parallel, time taken {}\n'.format(parallel_time))
    # print('Speed-up factor: {}'.format(serial_time/parallel_time))


    ########################################################
    # TESTING PARALLEL IMPLEMENTATION: meso routines
    ######################################################## 
    directory = '../Data/test_res100/'
    FileReader = METHOD_HDF5(directory)
    micro_model = IdealHD_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()

    t_range = [1.502, 1.504]
    x_range = [0.05, 0.06]
    y_range = [0.05, 0.06]

    # set up meso model and grid
    find_obs = FindObs_root_parallel(micro_model, 0.001)
    filter = box_filter_parallel(micro_model, 0.003)
    meso_model = resHD2D(micro_model, find_obs, filter)
    meso_model.setup_meso_grid([t_range, x_range, y_range])

    num_points = meso_model.domain_vars['Nt'] * meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny']
    print(f'Testing parallelization with {num_points} points\n')

    # fin obs - parallel
    start_time = time.perf_counter()
    meso_model.find_observers_parallel(os.cpu_count())
    parallel_time = time.perf_counter() - start_time
    print('Finished finding observers in parallel, execution time: {}\n'.format(parallel_time))

    # now filtering in parallel
    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(os.cpu_count())
    parallel_time = time.perf_counter() - start_time
    print('Finished filtering in parallel, time taken {}\n'.format(parallel_time))

    # # now testing serial vs parallel decomposition
    # start_time = time.perf_counter()
    # meso_model.decompose_structures()
    # serial_time = time.perf_counter() - start_time
    # print('Finished decomposing in serial, time taken {}\n'.format(serial_time))

    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(os.cpu_count())
    parallel_time = time.perf_counter() - start_time
    print('Finished decomposing in parallel, time taken {}\n'.format(parallel_time))
    # print('Speed-up factor: {}'.format(serial_time/parallel_time))
    
    