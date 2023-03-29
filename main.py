# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 01:45:31 2023

@author: marcu
"""

from multiprocessing import Process, Pool
from FileReaders import *
from Filters import *
from MicroModels import *
from MesoModels import *



if __name__ == '__main__':
    
    
    MicroModel = IdealHydro()
    FileReader = METHOD()
    FileReader.read_in_data(MicroModel, './Data/Testing/')
    Filter = Box_2D(0.1)
    MesoModel = NonIdealHydro(MicroModel)