# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

import numpy as np
from scipy.integrate import quad

class NonIdealHydro2D(object):

    def __init__(self, MicroModel, Filter):
        self.Filter = Filter
        self.MicroModel = MicroModel
        print("Compatible!")

    def p_from_EoS(self, rho, n):
        """
        Calculate pressure from EoS
        """
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p
    
    def residual(self, V0_V1,P,L):
        """
        Calculate the residual to be minimized. This is the integral of the
        number flux across the sides of the box.
        """
        U = self.get_U_mu(V0_V1)
        # U = get_U_mu_MagTheta(V0_V1)
        E_x, E_y = self.construct_tetrad(U)
        corners = self.find_boundary_pts(E_x,E_y,P,L)
        flux = 0
        for i in range(3,-1,-1):
            surface = np.linspace(corners[i],corners[i-1],10)
            for coords in surface:
                u, n = self.interpolate_u_n_point(coords)
                n_mu = np.multiply(u,n) # construct particle drift
                if i == 3:
                    flux += self.Mink_dot(n_mu,-E_x) # Project wrt orthonormal tetrad and sum (surface integral)
                elif i == 2:
                    flux += self.Mink_dot(n_mu,-E_y) 
                elif i == 1:
                    flux += self.Mink_dot(n_mu,E_x)
                elif i == 0:
                    flux += self.Mink_dot(n_mu,E_y)
        return abs(flux) # **2 for minimization rather than r-f'ing?

    def residual_ib(self, V0_V1,P,L):
        """
        An alternative residual that uses in-build scipy method quad. This is
        vastly slower than manual integral calculation in residual function.
        """
        U = self.get_U_mu(V0_V1)
        # U = get_U_mu_MagTheta(V0_V1)
        E_x, E_y = self.construct_tetrad(U)
        flux = 0
        for i in range(3,-1,-1):
            if i == 3:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(E_x,E_y,P,-E_x))[0]
            elif i == 2:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(E_x,-E_y,P,-E_y))[0]
            elif i == 1:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(-E_x,-E_y,P,E_x))[0]
            elif i == 0:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(-E_x,E_y,P,E_y))[0]
            return abs(flux)
        
    def calculate_local_quantities(self, coord):
        # Need U, T to be calculated first
        N = self.Filter.filter_scalar(coord, U, 'n')

                    
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        