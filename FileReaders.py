# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:58:02 2023

@author: Marcus
"""

import h5py
import glob 

class METHOD(object):

    def __init__(self, micro_model, directory):
        hdf5_filenames = glob.glob(directory+str('*.hdf5'))
        print(hdf5_filenames)
        hdf5_files = []
        num_files = len(hdf5_filenames)
        for filename in hdf5_filenames:
            hdf5_files.append(h5py.File(filename,'r'))

        domain_const_strs = ['x','y','nx','ny','dx','dy']
        domain_var_strs = ['t']
        prim_vars_strs = ['v1','v2','p','rho','n']
        aux_vars_strs = ['W','T','h']
        
        micro_model.points = (ts,xs,ys)
        micro_model.dx = (xs[-1] - xs[0])/nx
        micro_model.dy = (ys[-1] - ys[0])/ny
        micro_model.vxs = np.zeros((num_files, nx, ny))
        micro_model.vys = np.zeros((num_files, nx, ny))
        micro_model.ns = np.zeros((num_files, nx, ny))

        for domain_const_str in domain_const_strs:
            micro_model.domain_vars[domain_const_str] = hdf5_files[0]['Domain/'+domain_const_str]

        for counter in range(num_files):

            for domain_var_str in domain_var_strs:
                micro_model.domain_vars[domain_var_str].append(hdf5_files[counter]['Domain/'+domain_var_str])

            for prim_vars_str in prim_vars_strs:
                micro_model.prim_vars[domain_const_str].append(hdf5_files[counter]['Primitive/'+prim_vars_str][:])

            for aux_vars_str in aux_vars_strs:
                micro_model.aux_vars[aux_vars_str].append(hdf5_files[counter]['Auxiliary/'+aux_vars_str][:])
