# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:36:01 2023

@author: Marcus
"""

import numpy as np
import scipy.integrate as integrate 
from scipy.optimize import minimize
from itertools import product

from MicroModels import *
from FileReaders import *
from system.BaseFunctionality import *

class Favre_observers(object): 

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
        self.spatial_dims = micro_model.spatial_dims
        self.L = box_len

    def set_box_length(self, bl):
        self.L = bl

    def Mink_dot(self, vec1, vec2):
        """
        Parameters:
        -----------
        vec1, vec2 : list of floats (or np.arrays)

        Return:
        -------
        mink-dot (cartesian) in 1+n dim
        """
        if len(vec1) != len(vec2):
            print("The two vectors passed to Mink_dot are not of same dimension!")

        dot = -vec1[0]*vec2[0]
        for i in range(1,len(vec1)):
            dot += vec1[i] * vec2[i]
        return dot


    def get_U_from_vels(self, spatial_vels):
        """
        Build unit vectors starting from spatial components
        Needed as this will enter the minimization procedure

        Parameters:
        ----------
        spatial_vels: list of floats

        Returns:
        --------
        list of floats: the d+1 vector, normalized wrt Mink metric
        """

        temp = 0
        for i in range(len(spatial_vels)):
            temp += spatial_vels[i]**2
        W = 1 / np.sqrt(1 - temp)
        U = [W]
        for i in range(len(spatial_vels)):
            U.append(W * spatial_vels[i])
        return U
        
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
        es =[]
        for _ in range(self.spatial_dims):
            es.append(np.zeros(self.spatial_dims+1))
        for i in range(len(es)):
            es[i][i+1]  = 1    
        tetrad = [U]
        for i, vec in enumerate(es): #enumerate returns a tuple: so acts by value not reference!
            vec = vec + np.multiply(self.Mink_dot(vec, U), U)
            for j in range(i-1,-1,-1):
                vec = vec - np.multiply(self.Mink_dot(vec, es[j]), es[j])
            es[i] = np.multiply(vec, 1 / np.sqrt(self.Mink_dot(vec, vec)))
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

        U = np.array(self.get_U_from_vels(spatial_vels))
        return self.get_tetrad_from_U(U)


    def Favre_residual(self, spatial_vels, point, lin_spacing):
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
        Much faster than method based on inbuilt dblquad: 100 points (per face) gives decent 
        results, and is 250 times faster.
        """
        
        tetrad = self.get_tetrad_from_vels(spatial_vels)
        flux = 0

        for vec in tetrad: 
            rem_vecs = [x for x in tetrad if not (x==vec).all()]
            for i in range(2):
                center = point + np.multiply( (-1)**i * self.L / 2, vec)

                xs = []
                for i in range(self.spatial_dims):
                    xs.append(np.linspace(-self.L /2 , self.L /2, lin_spacing))
                coords = []
                for element in product(*xs):
                    coords.append(np.array(element))

                surf_coords = []
                for coord in coords:
                    temp = center
                    for i in range(self.spatial_dims):
                        temp += np.multiply(coord[i], rem_vecs[i])
                    surf_coords.append(temp)

                for coord in surf_coords: 
                    U = self.micro_model.get_interpol_var(['bar_vel'], coord)[0]
                    rho = self.micro_model.get_interpol_var(['rho'], coord)
                    Na = np.multiply(rho, U)
                    flux += self.Mink_dot(Na, vec)

        flux *= (self.L / lin_spacing) ** self.spatial_dims
        return abs(flux)

        
    def point_flux(self, x, y , point, Vx, Vy, normal):
        """
        Compute the baryon flux at a point given two coordinates that param the surface
        Identified by normal. 

        Parameters:
        -----------
        coords: d floats, adapted coordinates of the box face

        point: list of floats (t,x,y) 

        Vx, Vy: (2+1)-arrays, tangent vectors to the box face

        normal: (2+1)-array, normal to the box face

        Returns:
        --------
        Float: flux at the point
        
        Notes:
        ------
        As this is used in Favre_residual_ib - which uses dblquad - this method has been 
        developed for the 2+1 dimensional case only. 
        """
        coords = point + np.multiply(x, Vx) + np.multiply(y, Vy)
        U = self.micro_model.get_interpol_struct('bar_vel', coords)
        rho = self.micro_model.get_interpol_prim(['rho'], coords)
        Na = np.multiply(rho, U)
        flux = self.Mink_dot(Na, normal)
        return flux

    def Favre_residual_ib(self, vx_vy, point):
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

        """
        if self.spatial_dim != 2: 
            print('This method uses a 2-dim integrator, so works fine only for 2 spatial dimensions ')
            return []
        
        xy_range = [- self.L / 2, + self.L / 2]
        tetrad = self.get_tetrad_from_vels(vx_vy)
        flux = 0 
        partial_flux = 0
        error =  0
        partial_error = 0

        for vec in tetrad: 
            rem_vecs = [x for x in tetrad if not (x==vec).all()]
            for i in range(2):
                partial_flux, partial_error = integrate.dblquad(self.point_flux, xy_range[0], xy_range[1], xy_range[0], xy_range[1], \
                                          args = (point, rem_vecs[0],rem_vecs[1],vec), epsabs = 1e-6 )[:]
                flux += partial_flux
                error += partial_error
        return abs(flux) , error


    def find_observers(self, num_points, ranges, spacing): 
        """
        Main function: minimize the Favre_residual and find the Favre observers for points
        in linearly spaced (with spacing num_points) in the ranges. 
        Spacing is the param passed to Favre_residual to sample the box faces.

        Parameters:
        -----------
        num_points: list of floats (t,x,y,(z))
            number of points to find observers at 

        ranges: list of lists of two floats [[t_min,t_max], ...]
            Define the coord ranges to find observers at 

        spacing: integer
            param to be passed to Favre_residual

        Returns:
        --------
        list of:
            1) coordinates at which minimization is successful 
            2) corresponding observers 
            3) corresponding residual 

        list of coordinates at which the minimization failed 

        Notes:
        ------
        This uses the faster residual, not the one based on the inbuilt dblquad 
        """

        list_of_coords = []
        for i in range(len(num_points)):
            list_of_coords.append( np.linspace( ranges[i][0], ranges[i][-1] , num_points[i]) )

        coords = []
        for element in product(*list_of_coords):
            coords.append(np.array(element))
        
        observers = []
        funs = []
        success_coords = []
        failed_coords = []

        for coord in coords:
            U = self.micro_model.get_interpol_var(['bar_vel'], coord)[0]
            guess = []
            for i in range(1, len(U)):
                guess.append(U[i] / U[0])
            guess = np.array(guess)
            sol = minimize(self.Favre_residual, x0 = guess, args = (coord, spacing), bounds=((-0.8,0.8),(-0.8,0.8)),tol=1e-6)
            # This rearrangement shouldn't be necessary!?!?
            try: 
                if sol.success:
                    observers.append(self.get_U_from_vels(sol.x))
                    funs.append(sol.fun)
                    success_coords.append(coord)

                    if (sol.fun > 1e-5): 
                        print(f"Warning, residual is large at {coord}: ", sol.fun)
            except:
                print(f'Failed for coordinates: {coord}, due to', sol.message)
                failed_coords.append(coord)

        return [success_coords, observers, funs] , failed_coords

    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to in put 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    """
    def find_nearest(self, array, value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
    def find_nearest_cell(self, coord):
        t_pos = self.find_nearest(self.micro_model.domain_vars['t'],coord[0])
        x_pos = self.find_nearest(self.micro_model.domain_vars['x'],coord[1])
        y_pos = self.find_nearest(self.micro_model.domain_vars['y'],coord[2])
        return [np.where(self.micro_model.domain_vars['t']==t_pos)[0][0],\
                np.where(self.micro_model.domain_vars['x']==x_pos)[0][0],\
                np.where(self.micro_model.domain_vars['y']==y_pos)[0][0]]

    def filter_prim_var(self, centre_coord, U, var_str, shape='box'):
        """
        Filter a variable over the volume of a box with centre given by 'point'.
        Originally done by scipy integration over the volume, but again incredibly
        slow so now a manual sum over all the cells within the box and then a 
        division by total number of cells.
        """
        tetrad = self.get_tetrad_from_U(U)
        start_coord, end_coord = centre_coord, centre_coord
        for coord, vec in zip(centre_coord, tetrad):
            start_coord -= np.array(vec)*self.L/2
            end_coord += np.array(vec)*self.L/2
        integrand = 0
        counter = 0
        start_cell, end_cell = self.find_nearest_cell(start_coord), self.find_nearest_cell(end_coord)
        for i in range(start_cell[0],end_cell[0]+1):
            for j in range(start_cell[1],end_cell[1]+1):
                for k in range(start_cell[2],end_cell[2]+1):
                    integrand += self.micro_model.prim_vars[var_str][i,j,k]
                    counter += 1
        return integrand/counter

    def filter_struc(self, centre_coord, U, var_str, shape='box'):
        """
        Filter a variable over the volume of a box with centre given by 'point'.
        Originally done by scipy integration over the volume, but again incredibly
        slow so now a manual sum over all the cells within the box and then a 
        division by total number of cells.
        """
        # contruct tetrad...
        tetrad = self.get_tetrad_from_U(U)
        start_coord, end_coord = centre_coord, centre_coord
        for coord, vec in zip(centre_coord, tetrad):
            start_coord -= np.array(vec)*self.L/2
            end_coord += np.array(vec)*self.L/2
        integrand = 0
        counter = 0
        start_cell, end_cell = self.find_nearest_cell(start_coord), self.find_nearest_cell(end_coord)
        for i in range(start_cell[0],end_cell[0]+1):
            for j in range(start_cell[1],end_cell[1]+1):
                for k in range(start_cell[2],end_cell[2]+1):
                    integrand += self.micro_model.structures[var_str][i,j,k]
                    counter += 1
        return integrand/counter

    def get_interpol_var(self, t, x, y, var_str):
        return self.micro_model.get_interpol_var(var_str, [t,x,y])

    def filter_var_ib(self, point, spatial_vels, var_str, shape='box'):
        """
        Filter a variable over the volume of a box with centre given by 'point'.
        Originally done by scipy integration over the volume, but again incredibly
        slow so now a manual sum over all the cells within the box and then a 
        division by total number of cells.
        """
        # contruct tetrad...
        tetrad = self.get_tetrad_from_vels(spatial_vels)
        # t_range = 
        # corners = self.find_boundary_pts(E_x,E_y,point,self.L)
        # start, end = corners[0], corners[2]
        integrand = 0
        integrand, error = integrate.tplquad(self.get_interpol_var, t_range[0], t_range[1],\
                                            x_range[0], x_range[1], y_range[0], y_range[1],\
                                            args = var_str, epsabs = 1e-6 )[:]
            
        return vol_integrand/(self.L)**3, error

# if __name__ == '__main__':
#     CPU_start_time = time.process_time()

#     FileReader = METHOD_HDF5('./Data/Testing/')
#     # micro_model = IdealMHD_2D()
#     micro_model = IdealHydro_2D()
#     FileReader.read_in_data(micro_model) 
#     micro_model.setup_structures()

#     filter = Favre_observers(micro_model,box_len=0.001)
    
#     # tetrad = filter.get_tetrad_from_vxvy([0.5,0.73])
#     # print(type(tetrad[0]),' ',type(tetrad[1]),' ',type(tetrad[2]),'\n', tetrad[0],' ',tetrad[1],' ',tetrad[2],'\n')
#     # print(filter.Mink_dot(tetrad[0],tetrad[1]), filter.Mink_dot(tetrad[0],tetrad[2]), filter.Mink_dot(tetrad[1],tetrad[2]))
#     # print(type(tetrad))

#     # smart_guess = micro_model.get_interpol_prim(['vx','vy'],[0.5,0.5,0.5])

#     # CPU_start_time = time.process_time()
#     # res = filter.Favre_residual(smart_guess,[0.5,0.5,0.5], 10)
#     # print('Residual: ',res,f'\nElapsed CPU time is {time.process_time() - CPU_start_time} with {10**filter.spatial_dim} points per face\n')

#     # CPU_start_time = time.process_time()
#     # res, error = filter.Favre_residual_ib(smart_guess,[0.5,0.5,0.5])[:]
#     # print('Residual: ',res,"\nError estimate: ",error,f'\nElapsed CPU time is {time.process_time() - CPU_start_time} with the inbuilt method')

#     CPU_start_time = time.process_time()
#     coord_range = [[9.995,10.005],[-0.2,-0.3],[0.5,0.7]]
#     num_points = [1,1,1]
    
#     min_res, failed_coord = filter.find_observers(num_points, coord_range, 10)
#     for i in range(len(min_res[0])):
#         for j in range(len(min_res)):
#             print(min_res[j][i])
#         print('\n')

#     num_minim = 1
#     for x in num_points: 
#         num_minim *= x
#     print(f'Elapsed CPU time for finding {num_minim} observer(s) is {time.process_time() - CPU_start_time}.')
#     print('Failed coordinates:', failed_coord)





    
    