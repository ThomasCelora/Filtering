# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:02 2023

@author: Thomas
"""

from FileReaders import *
from scipy.interpolate import interpn 
from system.BaseFunctionality import *
import numpy as np
import math
import time

# These are the symbols, so be careful when using these to construct vectors!
# levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
#                       for k in range(3)]for j in range(3) ] for i in range(3) ])

# levi4D = np.array([[[[ np.sign(i - j) * np.sign(j - k) * np.sign(k - l) * np.sign(i - l) \
#                        for l in range(4)] for k in range(4) ] for j in range(4)] for i in range(4)])


class IdealMHD_2D(object):

    def __init__(self, interp_method = "linear"):
        """
        Sets up the variables and dictionaries, strings correspond to 
        those used in METHOD

        Parameters
        ----------
        interp_method: str
            optional method to be used by interpn        
        """
        self.spatial_dims = 2
        self.interp_method = interp_method

        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        # This is the Levi-Civita symbol, not tensor, so be careful when using it 
        self.Levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
                      for k in range(3)]for j in range(3) ] for i in range(3) ])

        #Dictionary for grid: info and points
        self.domain_int_strs = ('nt','nx','ny')
        self.domain_float_strs = ("tmin","tmax","xmin","xmax","ymin","ymax","dt","dx","dy")
        self.domain_array_strs = ("t","x","y","points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs+self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []   

        #Dictionary for primitive var
        self.prim_strs = ("vx","vy","n","p", "Bz")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W","h","bz","bsq", "e")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        self.structures_strs = ("BC", "SETfl", "SETem", "Fab")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

    def get_spatial_dims(self):
        return self.spatial_dims

    def get_model_name(self):
        return 'IdealMHD_2D'

    def get_domain_strs(self):
        return self.domain_int_strs + self.domain_float_strs + self.domain_array_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.structures_strs
    
    def get_all_var_strs(self):
        return self.get_prim_strs() + self.get_aux_strs() + self.get_structures_strs()
    
    def setup_structures(self):
        """
        Set up the structures (i.e baryon (mass) current BC, Stress-Energy and Faraday tensors 

        Notes:
        ------
        Structures are built as multi-dim np.arrays, with the first three indices referring 
        to the grid, while the last one or two refer to space-time components.

        The Faraday tensor is stored as a fully co-variant tensor (both indices down)
        The stress-energy tensor is stored as a fully contra-variant tensor (both indices up)
        """
        self.structures["BC"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3))
        self.structures["SETfl"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3,3))
        self.structures["SETem"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3,3))
        self.structures["Fab"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3,3))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    vel_vec = np.array([self.aux_vars['W'][h,i,j],self.aux_vars['W'][h,i,j] * self.prim_vars['vx'][h,i,j] ,\
                                    self.aux_vars['W'][h,i,j] * self.prim_vars['vy'][h,i,j]])
                    
                    fibr_b = self.aux_vars['bz'][h,i,j]

                    self.structures['BC'][h,i,j,:] = np.multiply(self.prim_vars['n'][h,i,j], vel_vec )
                    
                    self.structures['SETfl'][h,i,j,:,:] = (self.prim_vars["n"][h,i,j] * self.aux_vars['h'][h,i,j]) * np.outer(vel_vec, vel_vec) \
                                                        + self.prim_vars['p'][h,i,j] * self.metric
                    
                    self.structures['SETem'][h,i,j:,:] = (fibr_b**2) * np.outer(vel_vec, vel_vec) + (fibr_b/2) * self.metric

                    self.structures['Fab'][h,i,j,:,:] = np.tensordot(self.Levi3D, vel_vec, axes = ([2,0])) * fibr_b
                    
                    
                    ################################
                    # CORRESPONDING 3+1-d FORMULAE
                    ################################

                    # vel_vec = np.array([self.aux_vars['W'][h,i,j],self.aux_vars['W'][h,i,j] * self.prim_vars['vx'][h,i,j] ,\
                    #                 self.aux_vars['W'][h,i,j] * self.prim_vars['vy'][h,i,j], self.aux_vars['W'][h,i,j] * self.prim_vars['vz'][h,i,j]])

                    # fibr_b = [0, self.aux_vars['bx'][h,i,j], self.aux_vars['by'][h,i,j], self.aux_vars['bz'][h,i,j]] #3+1d version

                    # self.structures['Fab'][h,i,j,:,:] = np.tensordot(np.tensordot(self.Levi4D, vel_vec, axes = ([2,0])), fibr_b, axes= ([2,0]))

                    # self.structures['SETem'][h,i,j,:,:] = Base.Mink_dot(fibr_b, fibr_b) * np.outer(vel_vec, vel_vec) +\
                    #                                     (Base.Mink_dot(fibr_b, fibr_b)/2) * self.metric -\
                    #                                     np.multiply(1/2., np.outer(fibr_b, fibr_b) )
                

        # This might be useful/needed
        # self.all_vars = self.prim_vars 
        # self.all_vars.update(self.aux_vars)
        # self.all_vars.update(self.structures)

    def get_var_gridpoint(self, var, point):
        """
        Returns variable corresponding to input 'var' at gridpoint 
        closest to input 'point'.

        Parameters:
        -----------
        vars: string corresponding to primitive, auxiliary or structre variable

        point: list of 2+1 floats

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest grid-point to input 'point'. 

        Notes:
        ------
        This method should be used in case using interpolated values 
        becomes too expensive. 
        """
        indices = Base.find_nearest_cell(point, self.domain_vars['points'])
        if var in self.get_prim_strs():
            return self.prim_vars[var][tuple(indices)]
            
        elif var in self.get_aux_strs():
            return self.aux_vars[var][tuple(indices)]
            
        elif var in self.get_structures_strs():
            if var == "BC":
                tmp = np.zeros(self.structures[var][0,0,0,:].shape)
                for a in range(len(self.structures[var][0,0,0,:])):
                    tmp[a] = self.structures[var][tuple(indices+[a])]
                return tmp
            else:
                tmp = np.zeros(self.structures[var][0,0,0,:,:].shape)
                for a in range(len(tmp[:,0])):
                    for b in range(len(tmp[0,:])):
                        tmp[a,b] = self.structures[var][tuple(indices+[a,b])]
                return tmp
        else: 
            print(f"{var} is not a variable of IdealMHD_2D!")
            return None

    def get_interpol_var(self, var, point):
        """
        Returns the interpolated variables at the point.

        Parameters
        ----------
        vars : str corresponding to primitive, auxiliary or structre variable
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Interpolated values/arrays corresponding to variable. 
        Empty list if none of the variables is a primitive, auxiliary o structure of the micro_model

        Notes
        -----
        Interpolation gives errors when applied to boundary 
        """

        if var in self.get_prim_strs():
            return interpn(self.domain_vars['points'], self.prim_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_aux_strs():
            return interpn(self.domain_vars['points'], self.aux_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_structures_strs():
            return interpn(self.domain_vars['points'], self.structures[var], point, method = self.interp_method)[0]
        else:
            print(f'{var} is not a primitive, auxiliary variable or structure of the micro_model!!')


class IdealHydro_2D(object):

    def __init__(self, interp_method = "linear"):
        """
        Sets up the variables and dictionaries, strings correspond to 
        those used in METHOD

        Parameters
        ----------
        interp_method: str
            optional method to be used by interpn        
        """
        self.spatial_dims = 2
        self.interp_method = interp_method

        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        # This is the Levi-Civita symbol, not tensor, so be careful when using it 
        self.Levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
                      for k in range(3)]for j in range(3) ] for i in range(3) ])

        #Dictionary for grid: info and points
        self.domain_int_strs = ('nt','nx','ny')
        self.domain_float_strs = ("tmin","tmax","xmin","xmax","ymin","ymax","dt","dx","dy")
        self.domain_array_strs = ("t","x","y","points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs + self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []   

        #Dictionary for primitive var
        self.prim_strs = ("v1","v2","rho","p","n")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W","h","T")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        self.structures_strs = ("bar_vel","SET")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

        #Dictionary for all vars
        self.var_strs = self.prim_strs + self.aux_strs + self.structures_strs
        # self.vars = self.prim_vars
        # self.vars.update(self.aux_vars)
        # self.vars.update(self.structures)   
        
        self.all_var_strs = self.prim_strs + self.aux_strs + self.structures_strs

    def get_model_name(self):
        return 'IdealHydro_2D'

    def get_spatial_dims(self):
        return self.spatial_dims

    def get_domain_strs(self):
        return self.domain_info_strs + self.domain_points_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.get_structures_strs
    
    def get_all_var_strs(self):
        return self.all_var_strs

    def setup_structures(self):
        """
        Set up the structures (i.e baryon vel, SET and Faraday) 

        Structures are built as multi-dim np.arrays, with the first indices referring 
        corrensponding to their components, the last three to the grid coordinates.
        """
        self.structures["bar_vel"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3))
        self.structures["SET"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3,3))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    self.structures["bar_vel"][h,i,j,0] = self.aux_vars['W'][h,i,j] 
                    self.structures["bar_vel"][h,i,j,1] = self.aux_vars['W'][h,i,j] * self.prim_vars['v1'][h,i,j]
                    self.structures["bar_vel"][h,i,j,2] = self.aux_vars['W'][h,i,j] * self.prim_vars['v2'][h,i,j]
                    vel = np.array(self.structures["bar_vel"][h,i,j,:])

                    
                    self.structures["SET"][h,i,j,:,:] = (self.prim_vars["rho"][h,i,j] + self.prim_vars["p"][h,i,j]) * \
                                                    np.outer(vel,vel) + self.prim_vars["p"][h,i,j] * self.metric

        self.vars = self.prim_vars
        self.vars.update(self.aux_vars)
        self.vars.update(self.structures)

    def get_interpol_prim(self, var_names, point): 
        """
        Returns the interpolated variable at the point

        Parameters
        ----------
        vars : list of strings 
            strings have to be in to prim_vars keys
        point : list of floats
            ordered coordinates: t,x,y
        Return
        ------
        list of floats corresponding to string of vars

        Notes
        -----
        Interpolation raises a ValueError when out of grid boundaries.   
        """
        res = []
        
        for var_name in var_names:
            try:
                res.append( interpn(self.domain_vars["points"], self.prim_vars[var_name], point, method = self.interp_method)[0]) #, bounds_error = False)
            except KeyError:
                print(f"{var_name} does not belong to the primitive variables of the micro_model!")    
        return res
    
    def get_interpol_aux(self, var_names, point): 
        """
        Returns the interpolated variable at the point

        Parameters
        ----------
        vars : list of strings 
            strings have to be in to aux_vars keys
        point : list of floats
            ordered coordinates: t,x,y
        Return
        ------
        list of floats corresponding to string of vars

        Notes
        -----
        Interpolation gives errors when applied to boundary  
        """

        res = []
        for var_name in var_names:
            try:
                res.append( interpn(self.domain_vars["points"], self.aux_vars[var_name], point, method = self.interp_method)[0])
            except KeyError:
                print(f"{var_name} does not belong to the auxiliary variables of the micro_model!")
        return res

    def get_interpol_struct(self, var_name, point): 
        """
        Returns the interpolated structure at the point

        Parameters
        ----------
        var : str corresponding to one of the structures
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Array with the interpolated values of the var structure
            Empty list if var is not a structure in the micro_model

        Notes
        -----
        Interpolation gives errors when applied to boundary  
        """
        res = []
        if var_name == "bar_vel":
            res = np.zeros(len(self.structures[var_name][:,0,0,0]))
            for a in range(len(self.structures[var_name][:,0,0,0])):
                res[a] = interpn(self.domain_vars["points"], self.structures[var_name][a,:,:,:], point, method = self.interp_method)[0]
        elif var_name == "SET":   
            res = np.zeros((len(self.structures[var_name][:,0,0,0,0]),len(self.structures[var][0,:,0,0,0])))
            for a in range(len(self.structures[var_name][:,0,0,0,0])):
                for b in range(len(self.structures[var_name][0,:,0,0,0])):
                    res[a,b] = interpn(self.domain_vars["points"],self.structures[var_name][a,b,:,:,:], point, method = self.interp_method)[0]                   
        else:
            print(f"{var} does not belong to the structures in the micro_model")
        return res    

    def get_interpol_var(self, var_names, point):
        """
        Returns the interpolated structure at the point

        Parameters
        ----------
        var : str corresponding to one of the structures
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Array with the interpolated values of any var
            Empty list if var is not a structure in the micro_model

        Notes
        -----
        Interpolation gives errors when applied to boundary  
        """
        res = []
        for var_name in var_names:
            try:
                res.append( interpn(self.domain_vars["points"], self.vars[var_name], point, method = self.interp_method)[0])
            except KeyError:
                print(f"{var_name} does not belong to the variables of the micro_model!")
        return res


# TC
if __name__ == '__main__':

    CPU_start_time = time.process_time()

    FileReader = METHOD_HDF5('./Data/test_res100/')
    micro_model = IdealMHD_2D()
    FileReader.read_in_data(micro_model) 
    micro_model.setup_structures()

    point = [1.502,0.4,0.2]
    vars = ['SETfl', 'BC', 'Fab', 'SETem']
    for var in vars: 
        res = micro_model.get_interpol_var(var, point)
        res2 = micro_model.get_var_gridpoint(var, point)
        print(f'{var}: \n {res} \n {res2} \n ********** \n ')

# MH
# if __name__ == '__main__':

#     CPU_start_time = time.process_time()

#     FileReader = METHOD_HDF5('./Data/Testing/')
#     # micro_model = IdealMHD_2D()
#     micro_model = IdealHydro_2D()
#     FileReader.read_in_data(micro_model) 
#     micro_model.setup_structures()

#     res = micro_model.get_interpol_var(['v1','rho'],[10.0,0.3,0.2])
#     print(type(res),"\n", res)

#     res = micro_model.get_interpol_var(['T','W'],[10.0,0.3,0.2])
#     print(type(res),"\n", res)

#     # res = micro_model.get_interpol_var(['bar_vel','SET'],[10.0,0.3,0.2])
#     # print(type(res),"\n", res)
#     # res = micro_model.get_interpol_aux(["b0","bx"],[0.5, 0.3,0.2])
#     # print(type(res),"\n", res)

#     # res = micro_model.get_interpol_struct("SET",[2.5, 0.3, 0.2])
#     # print(res.shape, res[0],'\n')

#     # print( micro_model.get_domain_strs() )

#     CPU_end_time = time.process_time()
#     CPU_time = CPU_end_time - CPU_start_time
#     print(f'The CPU time is {CPU_time} seconds')