# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:02 2023

@author: Thomas
"""

from FileReaders import *
from scipy.interpolate import interpn 
import numpy as np

# from scipy.optimize import root, minimize

# These are the symbols, so be careful when using these to construct vectors!
# levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
#                       for k in range(3)]for j in range(3) ] for i in range(3) ])

# levi4D = np.array([[[[ np.sign(i - j) * np.sign(j - k) * np.sign(k - l) * np.sign(i - l) \
#                        for l in range(4)] for k in range(4) ] for j in range(4)] for i in range(4)])


class IdealMHD_2D(object):

    def __init__(self, interp_method = "linear"):
        """
        Constructor, no parameters required. 

        Sets up the variables and dictionaries, strings correspond to 
        those used in METHOD

        Parameters
        ----------
        method: str
            optional method to be used by interpn        
        """
        self.interp_method = interp_method

        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        # This is the Levi-Civita symbol, not tensor, so be careful when using it 
        self.Levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
                      for k in range(3)]for j in range(3) ] for i in range(3) ])

        #Dictionary for grid: info and points
        self.domain_info_strs = ("nt","nx","ny","tmin","tmax","xmin","xmax","ymin","ymax","dt","dx","dy")
        self.domain_points_strs = ("ts","xs","ys","points")
        self.domain_vars = dict.fromkeys(self.domain_points_strs,self.domain_points_strs)
        for str in self.domain_points_strs:
            self.domain_points_vars = []   

        #Dictionary for primitive var
        self.prim_strs = ("vx","vy","rho","p","Bx","By")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W","h","b0","bx","by","bsq")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        self.structures_strs = ("bar_vel","SET","Faraday")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

    def get_domain_strs(self):
        return self.domain_info_strs + self.domain_points_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.get_structures_strs

    def setup_structures(self):
        """
        Set up the structures (i.e baryon vel, SET and Faraday) 

        Structures are built as multi-dim np.arrays, with the first indices referring 
        corrensponding to their components, the last three to the grid coordinates.
        """
        self.structures["bar_vel"] = np.zeros((3,self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny']))
        self.structures["SET"] = np.zeros((3,3,self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny']))
        self.structures["Faraday"] = np.zeros((3,3,self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny']))

        # print(type(self.aux_vars['W']))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    self.structures["bar_vel"][0,h,i,j] = self.aux_vars['W'][h,i,j] 
                    self.structures["bar_vel"][1,h,i,j] = self.aux_vars['W'][h,i,j] * self.prim_vars['vx'][h,i,j]
                    self.structures["bar_vel"][2,h,i,j] = self.aux_vars['W'][h,i,j] * self.prim_vars['vy'][h,i,j]
                    vel = np.array(self.structures["bar_vel"][:,h,i,j])

                    fibr_b = np.array([self.aux_vars['b0'][h,i,j],self.aux_vars['bx'][h,i,j],self.aux_vars['by'][h,i,j]])

                    # self.SET[:,:,h,i,j] = (self.rhos[h,i,j] + self.ps[h,i,j] + self.b2s[h,i,j]) * np.outer( vel,vel) + \
                    #              (self.ps[h,i,j] + self.b2s[h,i,j]/2) * self.metric - np.outer(fibr_b, fibr_b)
                    
                    self.structures["SET"][:,:,h,i,j] = (self.prim_vars["rho"][h,i,j] + self.prim_vars["p"][h,i,j] + self.aux_vars["bsq"][h,i,j]) * \
                                                    np.outer( vel,vel) + (self.prim_vars["p"][h,i,j] + self.aux_vars["bsq"][h,i,j]/2) * self.metric \
                                                    - np.outer(fibr_b, fibr_b)
                    
                    fol_vel_vec = np.zeros(3)
                    fol_vel_vec[0]=+1
                    fol_b_vec = np.array([self.aux_vars["b0"][h,i,j],self.aux_vars["bx"][h,i,j],self.aux_vars["by"][h,i,j]])
                    fol_e_vec = np.tensordot( self.Levi3D, np.outer(vel,fol_b_vec), axes = ([1,2],[0,1]))

                    self.structures['Faraday'][:,:,h,i,j] = np.outer( fol_vel_vec,fol_e_vec) - np.outer(fol_e_vec,fol_vel_vec) -\
                                                np.tensordot(self.Levi3D,fol_b_vec,axes=([2],[0]))


    def get_interpol_prim(self, vars, point): 
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
        Interpolation gives errors when applied to boundary  
        """
        res = []
        for var_name in vars:
            try:
                res.append( interpn(self.domain_vars["points"], self.prim_vars[var_name],point, method = self.interp_method)[0])
            except KeyError:
                print(f"{var_name} does not belong to the primitive variables of the micromodel!")    
        return res
    
    def get_interpol_aux(self, vars, point): 
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
        for var_name in vars:
            try:
                res.append( interpn(self.domain_vars["points"], self.aux_vars[var_name],point, method = self.interp_method)[0])
            except KeyError:
                print(f"{var_name} does not belong to the auxiliary variables of the micromodel!")
        return res
    

    def get_interpol_struct(self, var, point): 
        """
        Returns the interpolated structure at the point

        Parameters
        ----------
        vars : str corresponding to one of the structures
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Array with the interpolated values of the var structure
            Empty list if var is not a structure in the micromodel

        Notes
        -----
        Interpolation gives errors when applied to boundary  
        """
        res = []
        if var == "bar_vel":
            res = np.zeros(len(self.structures[var][:,0,0,0]))
            for a in range(len(self.structures[var][:,0,0,0])):
                res[a] = interpn(self.domain_vars["points"], self.structures[var][a,:,:,:],point, method = self.interp_method)[0]
        elif var == "SET":   
            res = np.zeros((len(self.structures[var][:,0,0,0,0]),len(self.structures[var][0,:,0,0,0])))
            for a in range(len(self.structures[var][:,0,0,0,0])):
                for b in range(len(self.structures[var][0,:,0,0,0])):
                    res[a,b] = interpn(self.domain_vars["points"],self.structures[var][a,b,:,:,:],point, method = self.interp_method)[0]                   
        elif var == "Faraday":
            res = np.zeros((len(self.structures[var][:,0,0,0,0]),len(self.structures[var][0,:,0,0,0])))
            for a in range(len(self.structures[var][:,0,0,0,0])):
                for b in range(len(self.structures[var][0,:,0,0,0])):
                    res[a,b]= interpn(self.domain_vars["points"],self.structures[var][a,b,:,:,:],point, method = self.interp_method)[0]
        else:
            print(f"{var} does not belong to the structures in the Micromodel")
        return res


if __name__ == '__main__':
    FileReader = METHOD_HDF5('./Data/test_res100/')
    MicroModel = IdealMHD_2D()
    FileReader.read_in_data(MicroModel) 
    MicroModel.setup_structures()

    # res = MicroModel.get_interpol_prim(["vz","vx","Bx"],[0.5, 0.3,0.2])
    # print(type(res),"\n", res)
    # res = MicroModel.get_interpol_aux(["b0","bx"],[0.5, 0.3,0.2])
    # print(type(res),"\n", res)

    # res = MicroModel.get_interpol_struct(["SET"],[2.5, 0.3,0.2])
    # print(res.shape, res[0])

    # print( MicroModel.get_domain_strs() )

