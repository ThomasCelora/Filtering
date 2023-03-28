# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:53:53 2023

@author: Marcus
"""

class NonIdealHydro(object):

    def __init__(self, micro_model):

        



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