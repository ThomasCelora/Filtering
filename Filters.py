# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:36:01 2023

@author: Thomas
"""

import numpy as np
import time 
import scipy.integrate as integrate 
from scipy.optimize import minimize
from itertools import product
from system.BaseFunctionality import *

from MicroModels import *
from FileReaders import *


class FindObs_drift_min(object): 
    """
    Class for computing the Favre observer given a micromodel. 
    Work in any dimension, read on construction from the micro-model. 
    """
    def __init__(self, micro_model, box_len):
        """
        Parameters: 
        -----------
        micro_model: instance of class containing the microdata

        Note:
        -----
        To-do: think about checking compatibility (dimension + baryon currrent)
        """
        self.micro_model = micro_model
        self.spatial_dims = micro_model.get_spatial_dims()
        self.L = box_len

        self.residuals_methods = { 
            "gauss" : self.drift_residual_Gauss ,
            "linear" : self.drift_residual ,
            "inbuilt" : self.drift_residual_ib
            }

    def set_box_length(self, bl):
        self.L = bl
        
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
        Compute the drift of baryons through the box built from vx_vy.
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
     
    def drift_residual_Gauss(self, spatial_vels, point, order = 3):
        """
        Alternative for computing manually the Favre residual, using the Gauss
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

        # for i in range(len(coords)):
        #     print(f'Coord: {coords[i]} , weight: {weights[i]}')
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

    def drift_residual_ib(self, spatial_vels, point, abserr = 1e-7):
        """
        Compute the residual using inbuilt method dblquad
        Based on function point_flux

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
                flux = self.Mink_dot(Na, normal)
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
                flux = self.Mink_dot(Na, normal)
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

    def find_observer(self, point, residual_str = "gauss"):
        """
        Key function: minimize the Favre_residual and find the Favre observer at point. 
        
        Parameters: 
        -----------
        point: list of spatial_dims+1 floats (t,x,y)

        residual_str: string, default to "gauss"
            string for the method used to compute the residual, must be chosen within
            ['gauss', 'linear', 'inbuilt']

        Returns:
        --------
        OptimizedResult of the minimization via scipy.optimize.minimize

        Notes:
        ------
        To change the number of points to be used with linear or Gauss-Legendre methods, 
        or the absolute relative of the inbuilt method, change the default values of 
        the optional arguments in the corresponding methods.
        """

        U = np.multiply(1 / self.micro_model.get_interpol_var('n', point) , self.micro_model.get_interpol_var('BC', point) )
        guess = []
        for i in range(1, len(U)):
            guess.append(U[i] / U[0])
        guess = np.array(guess)
        try: 
            sol = minimize(self.residuals_methods[residual_str], x0 = guess, args = (point), bounds=((-0.8,0.8),(-0.8,0.8)),tol=1e-6)
            return sol
        except  KeyError: 
            print(f"The method you want to use for computing the residual, {residual_str}, does not exist!")
            return None
    
    def find_observers_points(self, points, residual_str = "gauss"):
        """
        Key function: minimize the Favre_residual and find the Favre observers for points.
        Spacing is the param passed to Favre_residual to sample the box faces.

        Parameters:
        -----------
        points: list of spatial_dims+1 floats, ordered as (t,x,y) 
 
        residual_str: string, default to "gauss"
            string for the method used to compute the residual, must be chosen within
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
        funs = []
        success_coords = []
        failed_coords = []

        for point in points:
            sol = self.find_observer(point, residual_str)
            if sol.success:
                observers.append(self.get_U_from_vels(sol.x))
                funs.append(sol.fun)
                success_coords.append(point)

                if (sol.fun > 1e-5): 
                    print(f"Warning, residual is large at {point}: ", sol.fun)
            else: 
                print(f'Failed for coordinates: {point}, due to', sol.message)
                failed_coords.append(point)
        return [success_coords, observers, funs] , failed_coords
    
    def find_observers_ranges(self, num_points, ranges, residual_str = "gauss" ): 
        """
        Key function: minimize the Favre_residual and find the Favre observers for points
        in linearly spaced (with spacing num_points) in the ranges. 
        Spacing is the param passed to Favre_residual to sample the box faces.

        Parameters:
        -----------
        num_points: list of integers 
            number of points to find observers at in each direction

        ranges: list of lists of two floats [[t_min,t_max], ...]
            Define the coord ranges to find observers at 

        residual_str: string, default to "gauss"
            string for the method used to compute the residual, must be chosen within
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
        
        return self.find_observers_points(points, residual_str)


class Box_filter(object):
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

if __name__ == '__main__':
    CPU_start_time = time.process_time()

    FileReader = METHOD_HDF5('./Data/test_res100/')
    micro_model = IdealMHD_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()
    constraint = FindObs_drift_min(micro_model,0.001)
    filter = Box_filter(micro_model, 0.001)

    var = "SET"
    point = [1.502,0.3,0.5]
    observer = Base.get_rel_vel(constraint.find_observer(point).x)
    CPU_start_time = time.process_time()
    filtvar1 = filter.filter_var_point_inbuilt(var, point, observer)
    print(f"CPU time to filter SET with in-built method is {time.process_time()- CPU_start_time}. ")

    CPU_start_time = time.process_time()
    filtvar2 = filter.filter_var_point(var, point, observer)
    print(f"CPU time to filter SET with Gauss method is {time.process_time()- CPU_start_time}. ")

    print(f"Difference between the two is: \n {filtvar1[0] - filtvar2}")

    