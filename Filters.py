# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:36:01 2023

@author: Marcus
"""

import numpy as np

class Box_2D(object):
    
    def __init__(self, side_length):
        self.L = side_length
        
    
    def filter_scalar(self, point, U, quant_str):
        """
        Filter a variable over the volume of a box with centre given by 'point'.
        Originally done by scipy integration over the volume, but again incredibly
        slow so now a manual sum over all the cells within the box and then a 
        division by total number of cells.
        """
        # contruct tetrad...
        E_x, E_y = self.construct_tetrad(U)
        # corners = self.find_boundary_pts(E_x,E_y,point,self.L)
        # start, end = corners[0], corners[2]
        t, x, y = point
        integrand = 0
        counter = 0
        start_cell, end_cell = self.find_nearest_cell([t-(self.L/2),x-(self.L/2),y-(self.L/2)]), \
            self.find_nearest_cell([t+(self.L/2),x+(self.L/2),y+(self.L/2)])
        for i in range(start_cell[0],end_cell[0]+1):
            for j in range(start_cell[1],end_cell[1]+1):
                for k in range(start_cell[2],end_cell[2]+1):
                    integrand += self.vars[quant_str][i][j,k]
                    counter += 1
        return integrand/counter




