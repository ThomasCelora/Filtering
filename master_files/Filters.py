# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:36:01 2023

@author: Thomas
"""

import numpy as np
import time 
import os
import scipy.integrate as integrate 
import multiprocessing as mp
from scipy.optimize import minimize, root
from scipy.interpolate import interpn 
from itertools import product
from system.BaseFunctionality import *

from MicroModels import *
from FileReaders import *
from system.BaseFunctionality import *

class FindObs_flux_min(object): 
    """
    Class for computing the observer by minimizing the (micro) baryon current 
    surface flux over a box. 
    Work in any dimension, read on construction from the micro-model. 
    """
    def __init__(self, micro_model, box_len):
        """
        Parameters: 
        -----------
        micro_model: instance of class containing the microdata

        box_len: float, side of the box
        """
        self.micro_model = micro_model
        self.spatial_dims = micro_model.get_spatial_dims()
        self.L = box_len

        self.flux_methods = { 
            "gauss" : self.flux_residual_Gauss ,
            "linear" : self.flux_residual ,
            "inbuilt" : self.flux_residual_ib
            }

    def set_box_length(self, bl):
        self.L = bl
        
    def get_tetrad_from_U(self, U):
        """
        Build tetrad orthogonal to unit velocity with from complete velocity vector

        Parameters:
        -----------
        U: list of d+1 floats, with d the number of spatial dimensions 

        Return:
        -------
        list of arrays: U + d unit vectors that complete it to a orthonormal basis
        """
        if len(U) != 1+self.spatial_dims:
            print('The dimension of U passed is not compatible with \
                  micro_model dimensionality!')
            return None
        
        es =[]
        for _ in range(self.spatial_dims):
            es.append(np.zeros(self.spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1    
        tetrad = [U]
        for i, vec in enumerate(es): #enumerate returns a tuple: so acts by value not reference!
            vec = vec + np.multiply(Base.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(Base.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1 / np.sqrt(Base.Mink_dot(vec, vec)))
            tetrad += [es[i]]
        return tetrad        

    def get_tetrad_from_vels(self, spatial_vels):
        """
        Build tetrad orthogonal to unit velocity with spatial velocities spatial_vels

        Parameters:
        -----------
        spatial_vels: list of d floats, with d the number of spatial dimensions 

        Return:
        -------
        list of arrays: U + d unit vectors that complete it to a orthonormal basis
        """
        if len(spatial_vels) != self.spatial_dims:
            print('The number of spatial velocities passed is not compatible with \
                  micro_model dimensionality!')
            return None

        U = np.array(Base.get_rel_vel(spatial_vels))
        return self.get_tetrad_from_U(U)

    def flux_residual(self, spatial_vels, point, lin_spacing = 10):
        """
        Compute the drift of baryons through the box built from spatial_vels.
        First get the center of the 2*(d+1) faces of the (d+1)-box, then build coords 
        for points to sample the flux through each face.
        Next, approximate the flux integral as a sum

        Parameters:
        -----------
        spatial_vels: list of d (spatial dimension) floats, spatial coord of vel
        
        point: list of d+1 floats (t,x,y) for the box center

        lin_spacing: integer, lin_spacing**spatial_dim is the # of points used to 
            sample the flux through each face. 

        Returns:
        --------
        float: absolute flux 

        Notes:
        ------
        Much faster than method based on inbuilt dblquad.
        """
        tetrad = self.get_tetrad_from_vels(spatial_vels)
        flux = 0

        xs = []
        for i in range(self.spatial_dims):
            xs.append(np.linspace(-self.L /2 , self.L /2, lin_spacing))
        coords = []
        for element in product(*xs):
            coords.append(np.array(element))

        for vec in tetrad: 
            rem_vecs = [x for x in tetrad if not (x==vec).all()]
            for i in range(2):
                center = point + np.multiply( (-1)**i * self.L / 2, vec)

                surf_coords = []
                for coord in coords:
                    temp = center
                    for i in range(self.spatial_dims):
                        temp += np.multiply(coord[i], rem_vecs[i])
                    surf_coords.append(temp)

                for coord in surf_coords: 
                    flux += Base.Mink_dot(self.micro_model.get_interpol_var('BC', coord), vec)

        flux *= (self.L / lin_spacing) ** self.spatial_dims
        return abs(flux)
     
    def flux_residual_Gauss(self, spatial_vels, point, order = 3):
        """
        Alternative for computing manually the flux residual, using the Gauss
        Legendre method. 

         Parameters:
        -----------
        spatial_vels: list of d (spatial dimension) floats, spatial coord of vel
        
        point: list of d+1 floats (t,x,y) for the box center 

        order: integer, order of the Gauss-Legendre sampling method. 

        Returns:
        --------
        float: absolute flux 

        Notes:
        ------
        Much faster than method based on inbuilt dblquad. Should be more accurate 
        than the one based on linearly spaced points. 
        """
        ps1d = []
        ws1d = []
        if order == 3: 
            ps1d = [0, + np.sqrt(3/5), - np.sqrt(3/5)]
            ws1d = [8./9. , 5./9. , 5./9.]
        elif order == 4: 
            p1 = np.sqrt(3./7 - 2/7 * np.sqrt(6/5))
            p2 = np.sqrt(3./7 + 2/7 * np.sqrt(6/5))
            w1 = (18. + np.sqrt(30) ) / 36.
            w2 = (18. - np.sqrt(30) ) / 36.
            ps1d = [p1, -p1, p2, -p2]
            ws1d = [w1, w1, w2, w2]
        elif order == 5: 
            p1 = np.sqrt(5 - 2* np.sqrt(10/7.))/3
            p2 = np.sqrt(5 + 2* np.sqrt(10/7.))/3
            w1 = (322 + 13 * np.sqrt(70)) /900
            w2 = (322 - 13 * np.sqrt(70)) /900
            ps1d = [0, p1, -p1, p2, -p2]
            ws1d = [128/225, w1, w1, w2, w2]
        else: 
            print("The method is implemented for Gauss-Legendre quadrature of order 3, 4 and 5 only!")
            return []

        xs = []
        ws = []
        for _ in range(self.spatial_dims):
            xs.append(ps1d)
            ws.append(ws1d)
        
        coords = []
        for element in product(*xs):
            coords.append(np.multiply(self.L/2, np.array(element) ))    

        totws = []
        for element in product(*ws):
            temp = 1.
            for w in element:
                temp *= w 
            totws.append(temp)

        tetrad = self.get_tetrad_from_vels(spatial_vels)
        flux = 0.
        for vec in tetrad: 
            rem_vecs = [x for x in tetrad if not (x==vec).all()]
            for i in range(2):
                center = point + np.multiply( (-1)**i * self.L / 2, vec)

                surf_coords = []
                for coord in coords:
                    temp = center
                    for i in range(self.spatial_dims):
                        temp += np.multiply(coord[i], rem_vecs[i])
                    surf_coords.append(temp)

                for i, coord in enumerate(surf_coords): 
                    Na = self.micro_model.get_interpol_var('BC',coord)
                    flux += Base.Mink_dot(Na, vec) * totws[i]

        flux *= (self.L /2  ) ** self.spatial_dims 
        return abs(flux)

    def flux_residual_ib(self, spatial_vels, point, abserr = 1e-7):
        """
        Compute the flux residual using inbuilt method dblquad or tplquad
        Based on function point_flux (nested definition)

        Parameters:
        -----------
        Vs: list of d floats, spatial components of the velocity vec

        point: float, center of the box

        Returns:
        --------
        tuple: absolute flux and error estimate

        Notes:
        ------
        Vastly slower than method above. 
        As this is based on dblquad, this method works fine only if it's 2+1 dim

        Also, dblquad returns an estimate of the error. This info is presently discarded
        in order to have the method return a scalar value so that this can be minimized 
        by find observer. 

        """
        tetrad = self.get_tetrad_from_vels(spatial_vels)
        flux = 0 
        partial_flux = 0
        error =  0
        partial_error = 0

        if self.spatial_dims == 2: 
            def point_flux(x, y , center, Vx, Vy, normal):
                coords = center + np.multiply(x, Vx) + np.multiply(y, Vy)
                Na = self.micro_model.get_interpol_var('BC', coords)
                flux = Base.Mink_dot(Na, normal)
                return flux

            for vec in tetrad: 
                rem_vecs = [x for x in tetrad if not (x==vec).all()]
                for i in range(2):
                    center = point + np.multiply( (-1)**i * self.L / 2, vec)
                    partial_flux, partial_error = integrate.dblquad(point_flux, -self.L / 2, self.L / 2, -self.L / 2, self.L / 2, \
                                                                    args = (center, rem_vecs[0], rem_vecs[1], vec), epsabs = abserr)[:]
                    flux += partial_flux
                    error += partial_error
            return abs(flux)
        
        elif self.spatial_dims == 3: 
            def point_flux(x, y, z , center, Vx, Vy, Vz, normal):
                coords = center + np.multiply(x, Vx) + np.multiply(y, Vy) + np.multiply(z, Vz)
                Na = self.micro_model.get_interpol_var('BC', coords)
                flux = Base.Mink_dot(Na, normal)
                return flux

            for vec in tetrad: 
                rem_vecs = [x for x in tetrad if not (x==vec).all()]
                for i in range(2):
                    center = point + np.multiply( (-1)**i * self.L / 2, vec)
                    partial_flux, partial_error = integrate.tplquad(point_flux, -self.L / 2, self.L / 2, -self.L / 2, self.L / 2, -self.L / 2, self.L / 2,\
                                                                    args = (center, rem_vecs[0], rem_vecs[1], vec), epsabs = abserr)[:]
                    flux += partial_flux
                    error += partial_error
            return abs(flux)

    def find_observer(self, point, flux_str = "gauss", initial_guess = None):
        """
        Key function: minimize the flux residual and find the observer at point. 
        
        Parameters: 
        -----------
        point: list of spatial_dims+1 floats (t,x,y)

        flux_str: string, default to "gauss"
            string for the method used to compute the flux, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        initial_guess: DEPRECATED, list of d floats: the spatial velocities of the initial guess
            if the number of spatial vels passed does not match the micro_model dimensionality, the 
            pointwise velocity is used instead. 

        Returns:
        --------
        Successful minimization: Boolean True, observer, error
        Failed minimization: Boolean False, coordinates

        Notes:
        ------
        To change the number of points to be used with linear or Gauss-Legendre methods, 
        or the absolute relative of the inbuilt method, change the default values of 
        the optional arguments in the corresponding methods.
        """

        guess = []
        if initial_guess is not None and len(initial_guess) == self.spatial_dims:
            guess = initial_guess
        else: 
            U = np.multiply(1 / self.micro_model.get_interpol_var('n', point) , self.micro_model.get_interpol_var('BC', point) )
            for i in range(1, len(U)):
                guess.append(U[i] / U[0])
            guess = np.array(guess)

        try: 
            sol = minimize(self.flux_methods[flux_str], x0 = guess, args = (point), bounds=((-0.8,0.8),(-0.8,0.8)),tol=1e-6)
            if sol.success: 
                observer = Base.get_rel_vel(sol.x)
                if sol.fun > 1e-5:
                    print(f'Warning: residual is large at {point}', sol.fun)
                return sol.success, observer, sol.fun
            if not sol.success: 
                return sol.success, point 
        except  KeyError: 
            print(f"The method you want to use for computing the flux, {flux_str}, does not exist!")
            return None
    
    def find_observers_points(self, points, flux_str = "gauss"):
        """
        Key function: minimize the flux residual and find the observers for points.
        flux_str determines the method to compute the flux residual.

        Parameters:
        -----------
        points: list of spatial_dims+1 floats, ordered as (t,x,y) 
 
        flux_str: string, default to "gauss"
            string for the method used to compute the flux, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        Returns:
        --------
        list of:
            1) coordinates at which minimization is successful 
            2) corresponding observers 
            3) corresponding residual 

        list of coordinates at which the minimization failed 

        Notes:
        ------
        To change the number of points to be used with linear or Gauss-Legendre methods,
        change the default values of the optional arguments in the corresponding methods.
        """
        observers = []
        errors = []
        success_coords = []
        failed_coords = []

        for point in points:
            sol = self.find_observer(point, flux_str)
            if sol[0]: 
                observers.append(sol[1])
                errors.append(sol[2])
                success_coords.append(point)
                
            if not sol[0]: 
                failed_coords.append[point]

        return [success_coords, observers, errors] , failed_coords
    
    def find_observers_ranges(self, num_points, ranges, flux_str = "gauss" ): 
        """
        Key function: minimize the flux_residual and find the observers for points
        in the ranges. flux_str determines the method to compute the flux residual. 

        Parameters:
        -----------
        num_points: list of integers 
            number of points to find observers at in each direction

        ranges: list of lists of two floats [[t_min,t_max], ...]
            Define the coord ranges to find observers at 

        flux_str: string, default to "gauss"
            string for the method used to compute the flux, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        Returns:
        --------
        list of:
            1) coordinates at which minimization is successful 
            2) corresponding observers 
            3) corresponding residual 

        list of coordinates at which the minimization failed 

        Notes:
        ------
        To change the number of points to be used with linear or Gauss-Legendre methods,
        change the default values of the optional arguments in the corresponding methods.
        """

        list_of_coords = []
        for i in range(len(num_points)):
            list_of_coords.append( np.linspace( ranges[i][0], ranges[i][-1] , num_points[i]) )

        points = []
        for element in product(*list_of_coords):
            points.append(np.array(element))
        
        return self.find_observers_points(points, flux_str)


class FindObs_drift_root(object):
    """
    Class for computing the observer by root-finding on the net baryon current 
    drift over a box. 
    Work in any dimension, read on construction from the micro-model. 
    """
    def __init__(self, micro_model, box_len):
        """
        Parameters:
        -----------
        micro_model: instance of micro_model, the micro data to be filtered
        
        box_len: float, side of the box for computing drift
        """
        self.micro_model = micro_model
        self.L = box_len
        self.spatial_dims = micro_model.get_spatial_dims()

        self.drift_methods = { 
            "gauss" : self.drift_residual_gauss ,
            "linear" : self.drift_residual ,
            "inbuilt" : self.drift_residual_ib
            }

    def set_box_len(self, box_len):
        self.L = box_len
    
    def get_tetrad_from_vels(self, spatial_vels):
        """
        Build tetrad orthogonal to unit velocity with spatial velocities spatial_vels

        Parameters:
        -----------
        spatial_vels: list of d floats, with d the number of spatial dimensions 

        Return:
        -------
        list of arrays: U + d unit vectors that complete it to a orthonormal basis
        """
        if len(spatial_vels) != self.spatial_dims:
            print('The number of spatial velocities passed is not compatible with \
                  micro_model dimensionality!')
            return None
        U = Base.get_rel_vel(spatial_vels)
        es =[]
        for _ in range(self.spatial_dims):
            es.append(np.zeros(self.spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1    
        tetrad = [U]
        for i, vec in enumerate(es): #enumerate returns a tuple: so acts by value not reference!
            vec = vec + np.multiply(Base.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(Base.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1 / np.sqrt(Base.Mink_dot(vec, vec)))
            tetrad += [es[i]]
        return tetrad 

    def drift_residual(self, spatial_vels, point, lin_spacing = 10):
        """
        Compute the averaged baryon current over box, using linearly spaced sample points. 
        Then compute the drift with respect to each element in the triad completing U (built from 
        spatial vels) to a tetrad. 

        Parameters:
        -----------
        spatial_vels: list of d floats, the spatial vels of the observer
        
        point: list of d+1 floats, the coord of the center of the box

        lin_spacing: integer, number of points (in each direction) used for sampling the box 

        Returns:
        --------
        list of d floats, the drifts in each direction of the triad. 
        """
        tetrad = self.get_tetrad_from_vels(spatial_vels)
        
        xs = []
        for i in range(self.spatial_dims+1):
            xs.append(np.linspace(-self.L /2 , self.L /2, lin_spacing))
        adapt_coords = []
        for element in product(*xs):
            adapt_coords.append(np.array(element))

        coords = []
        for coord in adapt_coords: 
            temp = np.array(point)
            for i in range(self.spatial_dims+1):
                temp += np.multiply(coord[i], tetrad[i])
            coords.append(temp)

        integral = np.zeros(1+self.spatial_dims)
        for coord in coords: 
            DeltaV = (self.L / lin_spacing)**(1+self.spatial_dims)
            integral += np.multiply(DeltaV, self.micro_model.get_interpol_var('BC', coord))
        
        drifts = []
        for i in range(len(tetrad)-1):
            drifts.append(Base.Mink_dot(integral, tetrad[i+1]))

        return drifts

    def drift_residual_gauss(self, spatial_vels, point, order = 3):
        """
        Method to compute net drift residual through the box, using Gauss-Legendre method

        Parameters:
        -----------
        spatial_vels: list of d floats, the spatial velocities of the observer

        point: list of d+1 floats ordered as (t,x,y,...), the centre of the box

        order: integer, the order of the Gauss-Legendre approximation

        Returns:
        --------
        list of d floats, the average drift in each direction of the triad.

        """
        ps1d = []
        ws1d = []
        if order == 3: 
            ps1d = [0, + np.sqrt(3/5), - np.sqrt(3/5)]
            ws1d = [8./9. , 5./9. , 5./9.]
        elif order == 4: 
            p1 = np.sqrt(3./7 - 2/7 * np.sqrt(6/5))
            p2 = np.sqrt(3./7 + 2/7 * np.sqrt(6/5))
            w1 = (18. + np.sqrt(30) ) / 36.
            w2 = (18. - np.sqrt(30) ) / 36.
            ps1d = [p1, -p1, p2, -p2]
            ws1d = [w1, w1, w2, w2]
        elif order == 5: 
            p1 = np.sqrt(5 - 2* np.sqrt(10/7.))/3
            p2 = np.sqrt(5 + 2* np.sqrt(10/7.))/3
            w1 = (322 + 13 * np.sqrt(70)) /900
            w2 = (322 - 13 * np.sqrt(70)) /900
            ps1d = [0, p1, -p1, p2, -p2]
            ws1d = [128/225, w1, w1, w2, w2]
        else: 
            print("The method is implemented for Gauss-Legendre quadrature of order 3, 4 and 5 only!")
            return []

        xs = []
        ws = []
        for _ in range(self.spatial_dims+1):
            xs.append(ps1d)
            ws.append(ws1d)
        
        adapt_coords = []
        for element in product(*xs):
            adapt_coords.append(np.multiply(self.L/2, np.array(element) ))    

        totws = []
        for element in product(*ws):
            temp = 1.
            for w in element:
                temp *= w 
            totws.append(temp)

        tetrad = self.get_tetrad_from_vels(spatial_vels)
        coords = []
        for coord in adapt_coords:
            temp = np.array(point)
            for i in range(self.spatial_dims+1):
                temp += np.multiply(coord[i], tetrad[i])
            coords.append(temp)

        integral = np.zeros(1+self.spatial_dims)
        for i, coord in enumerate(coords): 
            integral += np.multiply(totws[i], self.micro_model.get_interpol_var('BC', coord))
        integral *= (self.L /2  ) ** (self.spatial_dims+1)

        drifts = []
        for i in range(len(tetrad)-1):
            drifts.append(Base.Mink_dot(integral, tetrad[i+1]))

        return drifts

    def drift_residual_ib(self, spatial_vels, point, abserr = 1e-7):
        """
        Method to compute net drift residual through the box, using Gauss-Legendre method

        Parameters:
        -----------
        spatial_vels: list of d floats, the spatial velocities of the observer

        point: list of d+1 floats ordered as (t,x,y,...), the centre of the box

        abserr = optional float, absolute error to be passed to tplquad

        Returns:
        --------
        list of d floats, the average drift in each direction of the triad.
        """
        if self.spatial_dims != 2: 
            print('In built methods for integrals can be used only in 2+1 dimensions (i.e. tplquad). The 1+1 \
                  dimensional case is not really interesting so not implemented.')
            return None
        else:
            tetrad = self.get_tetrad_from_vels(spatial_vels)
            integral = np.zeros(3)
            error = np.zeros(3)

            for i in range(3):
                def BC_point_value(t, x, y):
                    coord = point + np.multiply(t, tetrad[0]) + np.multiply(x, tetrad[1]) + np.multiply(y, tetrad[2])   
                    return self.micro_model.get_interpol_var('BC', coord)[i]

                integral[i], error[i] = integrate.tplquad(BC_point_value, -self.L / 2, self.L / 2, -self.L / 2, self.L / 2, \
                                                -self.L/2, self.L/2, args = (), epsabs = abserr)[:]

            drifts = []
            for i in range(len(tetrad)-1):
                drifts.append(Base.Mink_dot(integral, tetrad[i+1]))
    
            return drifts

    def find_observer(self, point, drift_str = "gauss", initial_guess= None):
        """
        Key method: use optimize.root to find the roots of the drift function. 

        Parameters: 
        -----------
        point: list of d+1 floats, ordered (t,x,y,...), the coordinates of the box centre

        drift_str: string, the method used to compute the net drift
            must be either 'gauss', 'linear', 'inbuilt'

        initial_guess: DEPRECATED, list of d floats: the spatial velocities of the initial guess
            if the number of spatial vels passed does not match the micro_model dimensionality, the 
            pointwise velocity is used instead. 

        Returns:
        --------
        Successful root-finding: Boolean True, observer, avg error
        Failed minimization: Boolean False, coordinates 

        Notes:
        ------
        Change the optional values of the corresponding drift methods to change 
        the number of points used to sample the d+1 box.
        """
        guess = []
        if initial_guess is not None and len(initial_guess) == self.spatial_dims:
            guess = initial_guess
        else: 
            U = np.multiply(1 / self.micro_model.get_interpol_var('n', point) , self.micro_model.get_interpol_var('BC', point) )
            for i in range(1, len(U)):
                guess.append(U[i] / U[0])
        guess = np.array(guess)

        try: 
            sol = root(self.drift_methods[drift_str], x0 = guess, args = (point))
            if sol.success: 
                observer = Base.get_rel_vel(sol.x)
                error = 0 
                for i in range(len(sol.fun)):
                    error += sol.fun[i]
                avg_error = error/ len(sol.fun)
                if avg_error > 1e-5: 
                    print(f'Warning: residual is large at {point}', avg_error)
                return sol.success, observer, avg_error
            if not sol.success: 
                return sol.success, point 
        except  KeyError: 
            print(f"The method you want to use for computing the flux, {drift_str}, does not exist!")
            return None  

    def find_observers_points(self, points, drift_str = "gauss"):
        """
        Key method: use optimize.root to find the roots of the drift function.

        Parameters:
        -----------
        points: list of spatial_dims+1 floats, ordered as (t,x,y) 
 
        drift_str: string, default to "gauss"
            string for the method used to compute the drifts, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        Returns:
        --------
        list of:
            1) coordinates at which root finding is successful 
            2) corresponding observers 
            3) corresponding residual 

        list of coordinates at which root finding failed 

        Notes:
        ------
        Change the optional values of the corresponding drift methods to change 
        the number of points used to sample the d+1 box. 
        """
        observers = []
        avg_errors = []
        success_coords = []
        failed_coords = []

        for point in points:
            sol = self.find_observer(point, drift_str)
            if sol[0]: 
                observers.append(sol[1])
                avg_errors.append(sol[2])
                success_coords.append(point)
                
            if not sol[0]: 
                failed_coords.append[point]

        return [success_coords, observers, avg_errors] , failed_coords

    def find_observers_ranges(self, num_points, ranges, drift_str = "gauss" ): 
        """
        Key function: minimize the flux_residual and find the observers for points
        in the ranges. drift_str determines the method to compute the flux residual. 

        Parameters:
        -----------
        num_points: list of integers 
            number of points to find observers at in each direction

        ranges: list of lists of two floats [[t_min,t_max], ...]
            Define the coord ranges to find observers at 

        drift_str: string, default to "gauss"
            string for the method used to compute the flux, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        Returns:
        --------
        list of:
            1) coordinates at which root finding is successful 
            2) corresponding observers 
            3) corresponding residual 

        list of coordinates at which the root finding failed 

        Notes:
        ------
        Change the optional values of the corresponding drift methods to change 
        the number of points used to sample the d+1 box.
        """

        list_of_coords = []
        for i in range(len(num_points)):
            list_of_coords.append( np.linspace( ranges[i][0], ranges[i][-1] , num_points[i]) )

        points = []
        for element in product(*list_of_coords):
            points.append(np.array(element))
        
        return self.find_observers_points(points, drift_str)


class FindObs_root_parallel(object):
    """
    Parallel version (streamlined) of FindObs_drift_root
    Currently: based on gauss-quadrature with order 3 
               interpolation of quantities from micro_model.
    """
    def __init__(self, micro_model, box_len):
        """
        Parameters:
        -----------
        micro_model: instance of micro_model, the micro data to be filtered
        
        box_len: float, side of the box for computing drift
        """
        self.micro_model = micro_model
        self.L = box_len

    def set_box_length(self, box_len):
        """
        Method to change the width of the filter. 

        Parameters:
        -----------
        filter_width: float

        Returns:
        --------
        None
        """
        self.L = box_len

    @staticmethod
    def get_tetrad_from_vels(spatial_vels):
        """
        Build tetrad orthogonal to unit velocity with spatial velocities spatial_vels

        Parameters:
        -----------
        spatial_vels: list of d floats 

        Return:
        -------
        list of arrays: U + d unit vectors that complete it to a orthonormal basis
        """
        spatial_dims = len(spatial_vels)
        U = Base.get_rel_vel(spatial_vels) 
        es =[]
        for _ in range(spatial_dims):
            es.append(np.zeros(spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1    
        tetrad = [U]
        for i, vec in enumerate(es): 
            vec = vec + np.multiply(Base.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(Base.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1 / np.sqrt(Base.Mink_dot(vec, vec)))
            tetrad += [es[i]]
        return tetrad 
    
    @staticmethod
    def initializer(L, grid, BC):
        """
        Initializer for processes in pool. 
        Build the adapted coordinates and weights: this is independent of specific point
        so can be done once for all workers within a process managed by pool.

        Parameters:
        -----------
        spatial dimensions: int 
        L: float

        Notes:
        ------
        As this is built as static method, spatial_dims and L cannot be read 
        from the specific instance of class
        """
        global adapt_coords
        global totws
        global micro_spatial_dims 
        global micro_grid
        global micro_BC
        global box_len

        micro_grid = grid
        micro_spatial_dims = len(grid)-1
        micro_BC = BC
        box_len = L
        # print(f'Baryon current passed to initializer: {micro_BC}')

        ps1d = [0, + np.sqrt(3/5), - np.sqrt(3/5)]
        ws1d = [8./9. , 5./9. , 5./9.]
        xs = []
        ws = []
        for _ in range(micro_spatial_dims+1):
            xs.append(ps1d)
            ws.append(ws1d)

        adapt_coords = []
        for element in product(*xs):
            adapt_coords.append(np.multiply(box_len/2, np.array(element)))  

        totws = []
        for element in product(*ws):
            temp = 1.
            for w in element:
                temp *= w 
            totws.append(temp)
        # print('Initialized process in the pool', flush=True)

    @staticmethod
    def find_observer_Gauss(point, pos_in_list_points, initial_guess=None):
        """
        CPU-bound task to be run in parralel. 
        The routine combines what has been split into many in the serial 
        version of this class. 

        Parameters:
        -----------
        point_pos: list
            point_pos[0] contains the coordinates of point where to find observer

            point_pos[1] contains the position in a larger list passed to pool.map()

        Returns:
        --------
        Successful root-finding: Boolean True, position of point in list passed to pool.map(),
                                observer, avg error

        Failed minimization: Boolean False, position of point in list passed to pool.map() 

        Notes:
        ------
        To be combined with method in MesoModel.find_obsevers_parallel()
        """
        # Declaring global vars set up by initializer
        global adapt_coords
        global totws
        global micro_spatial_dims 
        global micro_grid
        global micro_BC
        global box_len

        # Building the initial guess 
        # point = point_pos[0]
        # pos_in_list_points = point_pos[1]
        # CHECK IF YOU REALLY WANT THIS: USEFUL FOR ROOTVSMIN.PY
        if initial_guess is not None: 
            guess = initial_guess
        else:
            guess = []
            BC_point = interpn(micro_grid, micro_BC, point)[0]
            n_point = np.sqrt(-Base.Mink_dot(BC_point, BC_point))
            U_point = np.multiply( 1 / n_point , BC_point)
            # U_point = np.multiply( 1 / self.micro_model.get_interpol_var('n', point), self.micro_model.get_interpol_var('BC', point)
            for i in range(1, len(U_point)):
                guess.append(U_point[i] / U_point[0])
        

        # Routine to compute drift residual via Gauss-Legendre quadrature.
        def residual_gauss(spatial_vels, point): #, micro_model):
            # spatial_dims = micro_model.get_spatial_dims()
            tetrad = FindObs_root_parallel.get_tetrad_from_vels(spatial_vels)
            coords = []
            for coord in adapt_coords:
                temp = np.array(point)
                for i in range(micro_spatial_dims+1):
                    temp += np.multiply(coord[i], tetrad[i])
                coords.append(temp)

            integral = np.zeros(1 + micro_spatial_dims)
            for i, coord in enumerate(coords): 
                BC_coord = interpn(micro_grid, micro_BC, coord)[0]
                # integral += np.multiply(totws[i], micro_model.get_interpol_var('BC', coord))
                integral += np.multiply(totws[i], BC_coord)
            integral *= (box_len /2) ** (micro_spatial_dims+1)

            drifts = []
            for i in range(len(tetrad)-1):
                drifts.append(Base.Mink_dot(integral, tetrad[i+1]))
            return drifts
        
        # observer: root of the residual gauss routine
        sol = root(residual_gauss, x0 = guess, args = (point)) #, self.micro_model))
        if sol.success: 
            observer = Base.get_rel_vel(sol.x)
            avg_error = np.sum(sol.fun[1])
            avg_error /= len(sol.fun)
            if avg_error > 1e-5: 
                print(f'Warning: residual is large at {point}: ', avg_error, flush=True)
            return sol.success, pos_in_list_points, observer, avg_error
        if not sol.success: 
            return sol.success, pos_in_list_points

    def find_observers_parallel(self, points, n_cpus, initial_guesses=None):
        """
        Method to run find_observer_Gauss in parallel on a list of points

        Parameters:
        -----------
        points: list of d+1 float, d is the spatial dimension of micro_model

        n_cpus: int
            number of processes

        Returns:
        --------

        successes: list of lists
            successes[0]: positions of point in input list of points
            successes[1]: observers found at successful points
            successes[2]: average error 

        failures: list (typically empty)
            position of failed points in input list of points

        there is no default value for n_cpus so that this has to be passed 
        explicitely in the tests below, or decided at the MesoModel level. 
        """
        observers = []
        avg_errors = []
        success_pos = []
        failed_pos = []

        # spatial_dims = self.micro_model.get_spatial_dims()
        L = self.L
        BC = self.micro_model.vars['BC']
        grid = self.micro_model.domain_vars['points'] 
        args_for_pool = [ (points[i], i) for i in range(len(points))]
        if initial_guesses is not None: 
            if len(initial_guesses) == len(points):
                print('Using provided initial guesses')
                args_for_pool = [ (points[i], i, initial_guesses[i]) for i in range(len(points))]
        else: 
            args_for_pool = [ (points[i], i) for i in range(len(points))]

        init = FindObs_root_parallel.initializer
        # initargs = (spatial_dims, L)
        initargs = (L, grid, BC)

        with mp.Pool(initializer=init, initargs=initargs, processes=n_cpus) as pool:
            print('Finding observers in parallel with {} processes\n'.format(pool._processes), flush=True)
            for result in pool.starmap(self.find_observer_Gauss, args_for_pool):
                if (result[0] == True): 
                    success_pos.append(result[1])
                    observers.append(result[2])
                    avg_errors.append(result[3])
                elif (result[0] == False):
                    failed_pos.append(result[1])

        return [success_pos, observers, avg_errors] , failed_pos


class spatial_box_filter(object):
    """
    Class for box-filtering the variables of a micro_model. 
    Work in any dimensions, read on construction from the micro_model.
    The observers for covariant filtering are computed separately and must be
    passed when the fitlering methods are called. 
    """
    def __init__(self, micro_model, filter_width):
        """
        Parameters: 
        ----------
        filter_width: float, width of the filter window in each direction

        micro_model: instance of micro_model class, so to have access to its structures
        """
        self.micro_model = micro_model
        self.spatial_dims = micro_model.get_spatial_dims()
        self.filter_width = filter_width

    def set_filter_width(self, filter_width):
        """
        Method to change the width of the filter. 

        Parameters:
        -----------
        filter_width: float

        Returns:
        --------
        None
        """
        self.filter_width = filter_width

    def set_micro_model(self, micro_model): 
        """
        Method to change the micro_model to be filtered. 

        Parameters:
        -----------
        micro_model: instance of micro_model class

        Returns:
        --------
        None
        """
        self.micro_model = micro_model
        self.spatial_dims = micro_model.get_spatial_dims()

    def complete_U_tetrad(self, U): 
        """
        Build unit vectors orthogonal to observer U
        U has to be normalized to -1.

        Parameters:
        -----------
        U: np.array of shape (1+spatial_dims,)

        Returns:
        list of spatial_dims numpy arrays of shape (1+spatial_dims,) 
        which complete U to a orthonormal tetrad
        """
        es =[]
        for _ in range(self.spatial_dims):
            es.append(np.zeros(self.spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1.
        triad = []
        for i, vec in enumerate(es): #enumerate returns a tuple: so acts by value not reference!
            vec = vec + np.multiply(Base.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(Base.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1/np.sqrt(Base.Mink_dot(vec, vec)))
            triad += [es[i]]
        return triad

    def filter_var_point(self, var_str, point, observer, sample_method = "gauss", num_points = 3):
        """
        First complete the observer to a tetrad at the point. Then build coords for 
        sample points in the spatial directions adapted to observer. Then approximate the 
        filter integral as a Riemann sum.

        Parameters:
        -----------
        var_str: string corresponding to a variable of the micro_model

        point: list of floats (t,x,y) 

        observer: np.array of shape (1+spatial_dims,) .

        sample_method: string, either 'gauss' or 'linear'. Decide how to sample the d-volume for
            filtering.
            
        num_points: integer, number of sample points in each spatial direction. 

        Returns: 
        --------
        nd.array with shape of the variable, the box_filtered quantity. 

        Notes: 
        ------
        Current version uses interpolated values. Should this be too expensive, change
        get_interpol_var for get_var_gridpoint! 
        
        Gauss quadrature gives more accurate results for low values of num-points, hence 
        reduce number of interpolations.
        The alternative is to use the gridpoint method with linearly spaced sampling points.
        """
        xs = []
        coords = []  
        if sample_method == "linear":
            for i in range(self.spatial_dims):
                xs.append(np.linspace(-self.filter_width /2 , self.filter_width /2, num_points))
            for element in product(*xs):
                coords.append(np.array(element))

            vecs = self.complete_U_tetrad(observer)
            sample_points = []
            for coord in coords:
                temp = np.array(point)
                for i in range(self.spatial_dims):
                    temp += np.multiply(coord[i], vecs[i])
                sample_points.append(temp)
            
            filtered_var = np.zeros(self.micro_model.get_interpol_var(var_str, point).shape)
            for sample in sample_points:
                filtered_var += self.micro_model.get_interpol_var(var_str, sample)

            return np.multiply(filtered_var, 1 / (num_points**self.spatial_dims))

        elif sample_method == "gauss":
            ps1d = []
            ws1d = []
            ws = []
            totws = []
            if num_points == 3: 
                ps1d = [0, + np.sqrt(3/5), - np.sqrt(3/5)]
                ws1d = [8./9. , 5./9. , 5./9.]
            elif num_points == 4: 
                p1 = np.sqrt(3./7 - 2/7 * np.sqrt(6/5))
                p2 = np.sqrt(3./7 + 2/7 * np.sqrt(6/5))
                w1 = (18. + np.sqrt(30) ) / 36.
                w2 = (18. - np.sqrt(30) ) / 36.
                ps1d = [p1, -p1, p2, -p2]
                ws1d = [w1, w1, w2, w2]
            elif num_points == 5: 
                p1 = np.sqrt(5 - 2* np.sqrt(10/7.))/3
                p2 = np.sqrt(5 + 2* np.sqrt(10/7.))/3
                w1 = (322 + 13 * np.sqrt(70)) /900
                w2 = (322 - 13 * np.sqrt(70)) /900
                ps1d = [0, p1, -p1, p2, -p2]
                ws1d = [128/225, w1, w1, w2, w2]
            else: 
                print("The method is implemented for Gauss-Legendre quadrature of order 3, 4 and 5 only!")
                return None

            for _ in range(self.spatial_dims):
                xs.append(ps1d)
                ws.append(ws1d)
        
            for element in product(*xs):
                coords.append(np.multiply(self.filter_width/2, np.array(element) ))    
            
            for element in product(*ws):
                temp = 1.
                for w in element:
                    temp *= w 
                totws.append(temp)

            vecs = self.complete_U_tetrad(observer)
            sample_points = []
            for coord in coords:
                temp = np.array(point)
                for i in range(self.spatial_dims):
                    temp += np.multiply(coord[i], vecs[i])
                sample_points.append(temp)
            
            filtered_var = np.zeros(self.micro_model.get_interpol_var(var_str, point).shape)
            for i, sample in enumerate(sample_points):
                filtered_var += totws[i] * self.micro_model.get_interpol_var(var_str, sample)

            return np.multiply(filtered_var, 1 / (2**self.spatial_dims))

        else: 
            print("Sample methods to filter variable must be either 'gauss' or 'linear'! ")
            return None

    def filter_var_manypoints(self, var_str, points, observers, sample_method = "gauss", num_points = 3):
        """
        Method to filter a variable in the micro_model given a list of points and observers.

        Parameters:
        -----------
        var_str: string corresponding to a variable in the micro_model

        points: list of N lists of floats, ordered as (t,x,y)

        observers: list of N np.array of shape (1 + spatial_dims,)

        sample_method, num_points: string, int passed to filter method at a point to choose how to 
            sample the d-volume for box filtering. 

        Returns:
        --------
        List with the filtered var (np.float or nd.array depending on var_str) at all points

        Notes:
        ------
        Current version uses interpolated values. Should this be too expensive, change
        get_interpol_var for get_var_gridpoint!
        """
        if len(points) != len(observers):
            print("The number of points and observers do not match!")
            return []
        
        filtered_var = []
        for i, point in enumerate(points): 
            filtered_var.append(self.filter_var_point(var_str, point, observers[i], sample_method = sample_method, num_points = num_points ))
        
        return filtered_var

    def filter_var_point_inbuilt(self, var_str, point, observer):
        """
        Computed the filtered variable using the inbuilt scipy quad method and the interpolated 
        values of the quantity. 

        Parameters: 
        -----------
        var_str: string, must correspond to a string in the micro_model 
        
        point: list of 1+spatial_dims floats, center of the box-filter

        observer: nd.array of shape (1+spatial_dims,)

        Returns: 
        --------
        The filtered quantity at the point, or none if the dimensionality is more than 3+1.

        Notes:
        ------
        Much slower than corresponding filter_var_manypoints_ip method, this has been implemented
        to check the accuracy of the alternative methods. 
        """
        vecs = self.complete_U_tetrad(observer)
        integrand = np.zeros(self.micro_model.get_interpol_var(var_str,point).shape)
        error = np.zeros(self.micro_model.get_interpol_var(var_str,point).shape)

        if self.spatial_dims == 2: 
            def interpol_var_adapt_coord(x, y, *ind):
                p = np.array(point)
                p += np.multiply(x, vecs[0]) + np.multiply(y, vecs[1]) 
                return self.micro_model.get_interpol_var(var_str, p)[ind]

            for ind in np.ndindex(integrand.shape):
                integrand[ind], error[ind] = integrate.dblquad(interpol_var_adapt_coord, -self.filter_width/2 , self.filter_width/2, -self.filter_width/2, self.filter_width/2, \
                                                               args = (ind), epsabs = 1e-7 )[:]
    
        elif self.spatial_dims ==3: 
            def interpol_var_adapt_coord(self, x, y, z, *ind):
                p = np.array(point)
                p += np.multiply(x, vecs[0]) + np.multiply(y, vecs[1]) + np.multiply(z, vecs[2]) 
                return self.micro_model.get_interpol_var(var_str, p)[ind]
            
            for ind in np.ndindex(integrand.shape):
                integrand[ind], error[ind] = integrate.tplquad(interpol_var_adapt_coord, -self.filter_width/2, self.filter_width/2, -self.filter_width/2, self.filter_width/2, \
                                                               -self.filter_width/2, self.filter_width/2, args = (point), epsabs = 1e-7 )[:]
        else: 
            print("The inbuilt method is implemented in 2+1 and 3+1 dims only!")
            return None 
        
        return np.multiply( integrand, 1 / (self.filter_width**self.spatial_dims)), error


class box_filter_parallel(object):
    """
    Parallel version (streamlined) of spatial_box_filter
    Currently: based on gauss-quadrature with order 3 
               interpolation of quantities from micro_model.
    """
    def __init__(self, micro_model, filter_width):
        """
        Constructor

        Parameters:
        -----------
        micro_model: instance of a micro_model class
            micro data to be filtered

        filter_width: float
        """
        self.micro_model = micro_model
        self.spatial_dims = micro_model.get_spatial_dims()
        self.filter_width = filter_width

    def set_filter_width(self, filter_width):
        """
        Method to change the width of the filter. 

        Parameters:
        -----------
        filter_width: float

        Returns:
        --------
        None
        """
        self.filter_width = filter_width
        
    @staticmethod
    def complete_U_tetrad(U):
        """
        Given a time-like vector in d+1 dims, build and return 
        d vectors that complete U to an ON basis.

        Notes:
        ------
        This method is static: does not depend on instance vars of the class
        But it makes sense to have it belong to the class nonetheless.
        """
        spatial_dims = len(U)-1
        es =[]
        for _ in range(spatial_dims):
            es.append(np.zeros(spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1.
        triad = []
        for i, vec in enumerate(es): #enumerate returns a tuple: so acts by value not reference!
            vec = vec + np.multiply(Base.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(Base.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1/np.sqrt(Base.Mink_dot(vec, vec)))
            triad += [es[i]]
        return triad

    @staticmethod
    def initializer(spatial_dims, filter_width, grid, var):
        """
        Initializer for processes in pool. 
        Build the adapted coordinates and weights: this is independent of specific point
        so can be done once for all workers within a process managed by pool.

        Currently: gauss-legendre quadrature of order 3.
        """
        global abstract_coords
        global totws
        global micro_spatial_dims
        global micro_var
        global micro_grid


        micro_spatial_dims = spatial_dims
        micro_grid = grid
        micro_var = var

        ps1d = [0, + np.sqrt(3/5), - np.sqrt(3/5)]
        ws1d = [8./9. , 5./9. , 5./9.]

        xs = []
        ws = []
        for _ in range(spatial_dims):
            xs.append(ps1d)
            ws.append(ws1d)
        
        abstract_coords = []
        for element in product(*xs):
            abstract_coords.append(np.multiply(filter_width/2, np.array(element) ))    
        
        totws = []
        for element in product(*ws):
            temp = 1.
            for w in element:
                temp *= w 
            totws.append(temp)

    @staticmethod
    def filter_var_point_gauss(point, observer, pos_in_list_points):
        """
        CPU-bound task to be run in parralel. 
        The routine combines what has been split into many in the serial 
        version of this class. 

        Parameters:
        -----------
        packed_args: [[point, observer, [vars_to_be_filtered]], position]

            point: list of floats
                the point at which filtering

            observer: nd.array
                the observer wrt which filtering

            [vars_to_be_filtered]: list of strs
                strings corresponding to vars in micro_model to be filtered
            
            position: integer
                number of specific point etc in larger list passed to pool.map()
        
        Returns:
        --------
        position: same as input
            
        filtered_vars: list containing vars corresponding to pass list, filtered
            at point 'point' wrt observer 'observer'

        """
        # declaring global vars set up by initializer
        global abstract_coords
        global totws
        global micro_spatial_dims
        global micro_var
        global micro_grid


        # Unpacking arguments passed by pool.map
        # unpacked_args= packed_args[0]
        # pos = packed_args[1]
        # point = unpacked_args[0]
        # observer = unpacked_args[1]
        # vars_strs = unpacked_args[2]

        # From abstract to "real" coordinates
        vecs = box_filter_parallel.complete_U_tetrad(observer)
        sample_points = []
        for coord in abstract_coords:
            temp = np.array(point)
            for i in range(micro_spatial_dims):
                temp += np.multiply(coord[i], vecs[i])
            sample_points.append(temp)

        # Filtering the var
        # filtered_vars = []
        # for var in vars_strs:
        #     filtered_var  = np.zeros(self.micro_model.get_var_gridpoint(var,0,0,0).shape)
        filtered_var = np.zeros(micro_var[tuple([ 0 for _ in range(len(micro_grid))])].shape)
        for i, sample in enumerate(sample_points):
            filtered_var += totws[i] * interpn(micro_grid, micro_var, sample)[0]
        filtered_var = np.multiply(filtered_var, 1 / (2**micro_spatial_dims))
        # filtered_vars.append(filtered_var)
        
        # Different variables are returned as list (ordered as var_strs)
        # Safer to return a dictionary: check redux in performance though.
        return pos_in_list_points, filtered_var

    def filter_var_parallel(self, points_observers, var, n_cpus):
        """
        Method to run filter_vars_point_gauss in parallel given a list of points, 
        observers and vars. 

        Parameters: 
        -----------

        list_packed_args: [[point,observer, [vars_to_be_filtered]]]

            point: list of floats

            observer: nd.array 

            [vars_to_be_filtered]: list of strs
                each item must match a var in micromodel

        n_cpus: int
            number of processes

        Returns:
        --------
        position_in_list: list of integers
            position in original list of points at which filtering
            Required as pool.map not necessarily returns processes in order

        filtered_vars: list of lists containing filtered vars

        there is no default value for n_cpus so that this has to be passed 
        explicitely in the tests below, or decided at the MesoModel level. 
        """
        position_in_list = []
        filtered_var = []
        args_for_pool = [ tuple([*points_observers[i], i]) for i in range(len(points_observers))]
        # print(args_for_pool[0])

        init = box_filter_parallel.initializer
        micro_grid = self.micro_model.domain_vars['points']
        micro_var = self.micro_model.vars[var]
        initargs=(self.spatial_dims, self.filter_width, micro_grid, micro_var)

        with mp.Pool(initializer=init, initargs=initargs, processes=n_cpus) as pool:
            print('Filtering {} in parallel with {} processes\n'.format(var, pool._processes), flush=True)
            for result in pool.starmap(self.filter_var_point_gauss, args_for_pool):
                position_in_list.append(result[0])
                filtered_var.append(result[1])
        
        return position_in_list, filtered_var


if __name__ == '__main__':

    ########################################################
    # TESTING SERIAL IMPLEMENTATION
    ######################################################## 
    # CPU_start_time = time.process_time()

    # FileReader = METHOD_HDF5('../Data/test_res100/')
    # micro_model = IdealMHD_2D()
    # FileReader.read_in_data(micro_model) 
    # micro_model.setup_structures()

    # find_obs = FindObs_drift_root(micro_model, 0.001)
    # # find_obs = FindObs_flux_min(micro_model, 0.001)
    # filter = spatial_box_filter(micro_model, 0.003)

    # vars = ['BC','SETfl', 'Fab', 'SETem']
    # points = [[1.502,0.3,0.5],[1.503,0.4,0.2]]
    # observers = find_obs.find_observers_points(points)[0][1]

    # for i in range(len(points)):   
    #     for  var in vars:   
    #         CPU_start_time = time.process_time()
    #         filtvar1 = filter.filter_var_point_inbuilt(var, points[i], observers[i])
    #         inbuilt_time = time.process_time() - CPU_start_time

    #         CPU_start_time = time.process_time()
    #         filtvar2 = filter.filter_var_point(var, points[i], observers[i])
    #         gauss_time = time.process_time() - CPU_start_time
    #         print(f"CPU time speed-up to filter {var} with Gauss method at {points[i]} is {inbuilt_time/gauss_time}. ")
    #         print(f'Filtered quantity with Gauss is \n {filtvar2}')
    #         print(f"Difference with method based on inbuilt integration is: \n {filtvar1[0] - filtvar2}")  
    #         print('\n************************\n')

    ########################################################
    # TESTING PARALLEL IMPLEMENTATION
    ########################################################
    FileReader = METHOD_HDF5('../Data/test_res100/')
    micro_model = IdealHD_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()

    # setting up the points for testing - pass all the points within a range
    t_range = [1.502, 1.504]
    x_range = [0.05, 0.15]
    y_range = [0.05, 0.15]

    patch_min = [t_range[0], x_range[0], y_range[0]]
    patch_max = [t_range[1], x_range[1], y_range[1]]
    idx_mins = Base.find_nearest_cell(patch_min, micro_model.domain_vars['points'])
    idx_maxs = Base.find_nearest_cell(patch_max, micro_model.domain_vars['points'])

    ts = micro_model.domain_vars['t'][idx_mins[0]:idx_maxs[0]]
    xs = micro_model.domain_vars['x'][idx_mins[1]:idx_maxs[1]]
    ys = micro_model.domain_vars['y'][idx_mins[2]:idx_maxs[2]]
    
    points = []
    for elem in product(ts,xs,ys):
        points.append(list(elem))

    print('Number of points: {}'.format(len(points)))

    # Now find observers - serial version
    # start_time = time.perf_counter()
    # find_obs_serial = FindObs_drift_root(micro_model, 0.001)
    # observers = find_obs_serial.find_observers_points(points)
    # serial_time = time.perf_counter() - start_time
    # print('Serial time: {}\n'.format(serial_time))

    # Now find observers - parallel version
    n_cpus = os.cpu_count()
    start_time = time.perf_counter()
    find_obs_parallel = FindObs_root_parallel(micro_model, 0.001)
    point = points[0]
    result, failed = find_obs_parallel.find_observers_parallel(points, n_cpus)
    print('Number of points failed: {}'.format(len(failed)))
    parallel_time = time.perf_counter() - start_time
    print('Finished finding observers. Parallel time: {}\n'.format(parallel_time))
    # print('Speed-up factor: {}\n'.format(serial_time/parallel_time))


    observers = result[1]
    vars = ['BC', 'SET']
    # Filtering serial
    # start_time = time.perf_counter()
    # serial_filter = spatial_box_filter(micro_model, 0.003)
    # for var in vars: 
    #     serial_filter.filter_var_manypoints(var, points, observers)
    # serial_time = time.perf_counter() - start_time
    # print('Finished filtering in serial, time-taken: {}'.format(serial_time))

    # Preparing args for parallel filter routine
    args_list = []
    for elem in zip(points, observers):
        # args_list.append([*elem, vars])
        args_list.append(tuple(elem))

    start_time = time.perf_counter()
    parallel_filter = box_filter_parallel(micro_model, 0.003)
    for var in vars:
        parallel_filter.filter_var_parallel(args_list, var, n_cpus)
    parallel_time = time.perf_counter() - start_time
    print('Finished filtering in parallel, time-taken: {}'.format(parallel_time))
    # print('Speed-up factor: {}\n'.format(serial_time/parallel_time))

