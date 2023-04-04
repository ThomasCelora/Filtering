# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:00 2023

@author: Thomas
"""

import h5py
import glob
import numpy as np

class METHOD_HDF5(object):

    def __init__(self, directory):
        """
        Set up the list of files (from hdf5) and strings for variables and domain

        Parameters
        ----------
        directory: string 
            the filenames in the directory have to be incremental (sorted is used)
        """

        hdf5_filenames = sorted( glob.glob(directory+str('*.hdf5')))
        self.hdf5_files = []
        for filename in hdf5_filenames:
            self.hdf5_files.append(h5py.File(filename,'r'))
        self.num_files = len(self.hdf5_files)

        self.key_strs = list(self.hdf5_files[0].keys())
        self.var_strs = []
        for key_str in self.key_strs:
            self.var_strs.append(list(self.hdf5_files[0][key_str].keys()))

            
    def read_in_data(self,micro_model):   
        """
        Store data from files into micro_model 

        Parameters
        ----------
        micro_model: class MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.
        """ 
        for prim_var_str in  micro_model.prim_vars:
            for counter in range(self.num_files):
                micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+prim_var_str][:] )
                # The [:] is for returning the arrays not the dataset
            micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
        

        for aux_var_str in  micro_model.aux_vars:
            for counter in range(self.num_files):
                micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+aux_var_str][:] )
            micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
 

        # The following is temporary, as no extended info about domain are output by METHOD
        # Can recover all info about spatial grid, not about times

        micro_model.domain_vars["ts"] = np.arange(6)
        micro_model.domain_vars["xs"] = np.array(self.hdf5_files[0]['Domain/'+"x"][:])
        micro_model.domain_vars["ys"] = np.array(self.hdf5_files[0]['Domain/'+"y"][:])

        micro_model.domain_vars["nt"] = self.num_files
        micro_model.domain_vars["nx"] = len(micro_model.domain_vars["xs"])
        micro_model.domain_vars["ny"] = len(micro_model.domain_vars["ys"])

        micro_model.domain_vars["xmax"] = micro_model.domain_vars["xs"][-1]
        micro_model.domain_vars["xmin"] = micro_model.domain_vars["xs"][0]
        micro_model.domain_vars["ymax"] = micro_model.domain_vars["ys"][-1]
        micro_model.domain_vars["ymin"] = micro_model.domain_vars["ys"][0]         

        micro_model.domain_vars["dx"] = (micro_model.domain_vars["xmax"] - micro_model.domain_vars["xmin"]) \
                                        / micro_model.domain_vars["nx"]
        micro_model.domain_vars["dy"] = (micro_model.domain_vars["ymax"] - micro_model.domain_vars["ymin"]) \
                                        / micro_model.domain_vars["ny"]

        micro_model.domain_vars["points"] = [micro_model.domain_vars["ts"],micro_model.domain_vars["xs"],micro_model.domain_vars["ys"]]
    
    
if __name__ == '__main__':

    from MicroModels import * 

    FileReader = METHOD_HDF5('./Data/test_res100/')
    MicroModel = IdealMHD_2D()
    FileReader.read_in_data(MicroModel)

    
    