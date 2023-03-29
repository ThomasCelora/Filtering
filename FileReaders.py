# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:58:02 2023

@author: Marcus
"""

import h5py
import glob 

class METHOD(object):

    # def __init__(self, micro_model, directory)
        
    def read_in_data(self, micro_model, directory):
        hdf5_filenames = glob.glob(directory+str('data*.hdf5'))
        print(hdf5_filenames)
        hdf5_files = []
        num_files = len(hdf5_filenames)
        for filename in hdf5_filenames:
            hdf5_files.append(h5py.File(filename,'r'))
        print(hdf5_files[0]['Domain/x'])
            
        domain_const_strs = ['nt','nx','ny','Nx','Ny','x','y''dx','dy']
        domain_var_strs = ['t']
        prim_vars_strs = ['v1','v2','p','rho','n']
        aux_vars_strs = ['W','T','h']
        
        for domain_const_str in domain_const_strs:
            micro_model.domain_vars[domain_const_str] = hdf5_files[0]['Domain/'+domain_const_str]

        for counter in range(num_files):

            for domain_var_str in domain_var_strs:
                micro_model.domain_vars[domain_var_str].append(hdf5_files[counter]['Domain/'+domain_var_str])

            for prim_vars_str in prim_vars_strs:
                micro_model.prim_vars[domain_const_str].append(hdf5_files[counter]['Primitive/'+prim_vars_str][:])

            for aux_vars_str in aux_vars_strs:
                micro_model.aux_vars[aux_vars_str].append(hdf5_files[counter]['Auxiliary/'+aux_vars_str][:])

