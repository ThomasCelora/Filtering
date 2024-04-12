# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:00 2023

@author: Thomas
"""

import h5py
import glob
import numpy as np

class METHOD_HDF5(object):

    def __init__(self, directory, fewer_snaps=False, smaller_list=None):
        """
        Set up the list of files (from hdf5) and dictionary with dataset names
        in the hdf5 file. 

        Parameters
        ----------
        directory: string 
            the filenames in the directory have to be incremental (sorted is used)

        fewer_snaps: bool
            set to true if you want micromodel to store fewer snapshots than can be found
            in directory 
        
        smaller_list: list
            indices to be retained of list orderd via sorted(glob.glob(directory+str('*.hdf5')))
        """ 
        hdf5_filenames = sorted(glob.glob(directory+str('*.hdf5')))
        if fewer_snaps:
            if smaller_list:
                temp = [hdf5_filenames[i] for i in smaller_list]
                hdf5_filenames = temp

        self.hdf5_files = []
        for filename in hdf5_filenames:
            self.hdf5_files.append(h5py.File(filename,'r'))
        self.num_files = len(self.hdf5_files)

        self.hdf5_keys = dict.fromkeys(list(self.hdf5_files[0].keys())) 
        for key in self.hdf5_keys: 
            self.hdf5_keys[key] = list(self.hdf5_files[0][key].keys())

    def get_hdf5_keys(self):
        return self.hdf5_keys

    def read_in_data(self, micro_model):   
        """
        Store data from files into micro_model 

        Parameters
        ----------
        micro_model: class MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.
        """ 

        self.translating_prims = dict.fromkeys(micro_model.get_prim_strs())
        for prim_str in micro_model.get_prim_strs():
            if prim_str == "n":
                self.translating_prims[prim_str] = "rho"
            else: 
                self.translating_prims[prim_str] = prim_str 

        for prim_var_str in  micro_model.prim_vars:
            try: 
                method_str = self.translating_prims[prim_var_str]
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][:] )
                    # The [:] is for returning the arrays not the dataset
                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Primitive/')
        

        self.translating_aux = dict.fromkeys(micro_model.get_aux_strs())
        for aux_str in micro_model.get_aux_strs():
            self.translating_aux[aux_str] = aux_str

        for aux_var_str in  micro_model.aux_vars:
            try: 
                method_str = self.translating_aux[aux_var_str]
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y']]

    def read_in_data_HDF5_missing_xy(self, micro_model):   
        """
        Store data from files into micro_model 

        Parameters
        ----------
        micro_model: class MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.
        """ 

        self.translating_prims = dict.fromkeys(micro_model.get_prim_strs())
        for prim_str in micro_model.get_prim_strs():
            if prim_str == "n":
                self.translating_prims[prim_str] = "rho"
            else: 
                self.translating_prims[prim_str] = prim_str 

        for prim_var_str in  micro_model.prim_vars:
            try: 
                method_str = self.translating_prims[prim_var_str]
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][:] )
                    # The [:] is for returning the arrays not the dataset
                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Primitive/')
        

        self.translating_aux = dict.fromkeys(micro_model.get_aux_strs())
        for aux_str in micro_model.get_aux_strs():
            self.translating_aux[aux_str] = aux_str

        for aux_var_str in  micro_model.aux_vars:
            try: 
                method_str = self.translating_aux[aux_var_str]
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        # Similar is for x,y which are not stored properly in by METHOD parallelSaveDataHDF5
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points', 'x', 'y']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])

        micro_model.domain_vars['x'] = np.zeros(micro_model.domain_vars['nx'])
        for i in range(len(micro_model.domain_vars['x'])):
            # micro_model.domain_vars['x'][i] = (micro_model.domain_vars['xmax']- micro_model.domain_vars['xmin']) / (2 * micro_model.domain_vars['nx']) + \
            #                                     i * micro_model.domain_vars['dx']
            micro_model.domain_vars['x'][i] = micro_model.domain_vars['xmin'] + i * micro_model.domain_vars['dx']
            
        micro_model.domain_vars['y'] = np.zeros(micro_model.domain_vars['ny'])
        for i in range(len(micro_model.domain_vars['y'])):
            # micro_model.domain_vars['y'][i] = (micro_model.domain_vars['ymax']- micro_model.domain_vars['ymin']) / (2 * micro_model.domain_vars['ny']) + \
            #                                     i * micro_model.domain_vars['dy']
            micro_model.domain_vars['y'][i] = micro_model.domain_vars['ymin'] + i * micro_model.domain_vars['dy']

        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y']]


if __name__ == '__main__':

    from MicroModels import * 

    FileReader = METHOD_HDF5('./Data/test_res100/')
    MicroModel = IdealMHD_2D()
    FileReader.read_in_data(MicroModel)
    # FileReader.read_in_data_HDF5_missing_xy(MicroModel)
