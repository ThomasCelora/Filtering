# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:58:02 2023

@author: Marcus
"""

import h5py
import glob 

class METHOD_HDF5(object):

    def __init__(self, directory):
        hdf5_filenames = glob.glob(directory+str('data_1*.hdf5'))
        self.hdf5_files = []
        for filename in hdf5_filenames:
            self.hdf5_files.append(h5py.File(filename,'r'))
        self.num_files = len(self.hdf5_files)
        self.key_strs = list(self.hdf5_files[0].keys())
        self.var_strs = []
        for key_str in self.key_strs:
            self.var_strs.append(list(self.hdf5_files[0][key_str].keys()))
            
        
    def read_in_data(self, micro_model):
        domain_const_strs = ['x','y']
        # domain_const_strs = ['nt','nx','ny','Nx','Ny','x','y''dx','dy']
        domain_var_strs = []# ['t']
        prim_vars_strs = ['v1','v2','p','rho','n']
        aux_vars_strs = ['W','T','h']
        
        # for domain_vars_str in micro_model.domain_vars:
        for domain_vars_str in domain_const_strs:
            micro_model.domain_vars[domain_vars_str] = self.hdf5_files[0]['Domain/'+domain_vars_str]

        for counter in range(self.num_files):
            for prim_vars_str in micro_model.prim_vars_strs:
                micro_model.prim_vars[prim_vars_str].append(self.hdf5_files[counter]['Primitive/'+prim_vars_str][:])

            for aux_vars_str in micro_model.aux_vars_strs:
                micro_model.aux_vars[aux_vars_str].append(self.hdf5_files[counter]['Auxiliary/'+aux_vars_str][:])




if __name__ == '__main__':
    from MicroModels import *

    FileReader = METHOD_HDF5('./Data/Testing/')
    MicroModel = IdealHydro()
    FileReader.read_in_data(MicroModel)
    
    
    
    
    
    
    
    
    