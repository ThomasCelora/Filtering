# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:30:01 2022

@author: mjh1n20
"""

import os
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
import h5py
from scipy.interpolate import interpn
from scipy.optimize import root, minimize
#from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, quad, tplquad, nquad
import cProfile, pstats, io
from system.BaseFunctionality import Base
import math

class PostProcessing(object):
        
    def __init__(self):
        fs_f = [] # fine
        fs_c = [] # coarse
        num_files = 11
        for n in range(num_files):
          fs_f.append(h5py.File('./Data/KH/Ideal/dp_400x400x0_'+str(n)+'.hdf5','r'))
          # fs_c.append(h5py.File('./Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        fss = [fs_f]
        self.nx, self.ny = int(400), int(400)
        self.c_nx, self.c_ny = int(self.nx/2), int(self.ny/2) # coarse
        # self.c_nx, self.c_ny = 200, 200 # coarse
        
        self.ts = np.linspace(0,30,11) # Need to actually get these
        self.xs = np.linspace(-0.5,0.5,self.nx)
        self.ys =  np.linspace(-1.0,1.0,self.ny)
        self.points = (self.ts,self.xs,self.ys)
        # self.dt get this from files...
        self.dx = (self.xs[-1] - self.xs[0])/self.nx # actual grid-resolution
        self.dy = (self.ys[-1] - self.ys[0])/self.ny

        # Define fluid variables for both the fine and coarse data
        self.vxs = []
        self.vys = []
        self.uts = []
        self.uxs = []
        self.uys = []
        self.ns = []
        self.rhos = []
        self.ps = []
        self.Ws = []
        self.Ts = []
        self.vars = {'v1': self.vxs,
                          'v2': self.vys,
                          'n': self.ns,
                          'rho': self.rhos,
                          'p': self.ps,
                          'W': self.Ws,
                          'u_t': self.uts,
                          'u_x': self.uxs,
                          'u_y': self.uys,
                          'T': self.Ts}

        self.vxs_c = []
        self.vys_c = []
        self.uts_c = []
        self.uxs_c = []
        self.uys_c = []
        self.ns_c = []
        self.rhos_c = []
        self.ps_c = []
        self.Ws_c = []
        self.Ts_c = []        
        self.vars_c = {'v1': self.vxs_c,
                          'v2': self.vys_c,
                          'n': self.ns_c,
                          'rho': self.rhos_c,
                          'p': self.ps_c,
                          'W': self.Ws_c,
                          'u_t': self.uts_c,
                          'u_x': self.uxs_c,
                          'u_y': self.uys_c,
                          'T': self.Ts_c}  
        
        self.prim_vars_strs = ['v1','v2','n','rho','p']
        self.aux_vars_strs= ['W','T']
        # for fs, c_fs in fss:
        # Load the data
        # for f_f, f_c in zip(fs_f,fs_c):
        for f_f in fs_f:
            for p_v_s in self.prim_vars_strs:
                self.vars[p_v_s].append(f_f['Primitive/'+p_v_s][:])
                # self.vars_c[p_v_s].append(f_c['Primitive/'+p_v_s][:])
            for a_v_s in self.aux_vars_strs:
                self.vars[a_v_s].append(f_f['Auxiliary/'+a_v_s][:])
                # self.vars_c[a_v_s].append(f_c['Primitive/'+a_v_s][:])
            # Artificial coarse data
            vxs_fine = f_f['Primitive/v1'][:]
            vxs_c = np.zeros((self.c_nx,self.c_ny))
            for i in range(self.c_nx):
                for j in range(self.c_ny):
                    vxs_c[i][j] = vxs_fine[i*2][j*2] + vxs_fine[i*2+1][j*2] \
                                       + vxs_fine[i*2][j*2+1] + vxs_fine[i*2][j*2+1]

        self.uts = self.Ws
        self.uxs = np.multiply(self.uts,self.vxs) # broken I think
        self.uys = np.multiply(self.uts,self.vys)
        self.uts_c = self.Ws_c
        self.uxs_c = np.multiply(self.uts_c,self.vxs_c)
        self.uys_c = np.multiply(self.uts_c,self.vys_c)
    
  
        # EoS & dissipation parameters
        self.coefficients = {'gamma': 5/3,
                        'zeta': 1e-2,
                        'kappa': 1e-4,
                        'eta': 1e-2}
        
        # Size of box for spatial filtering
        self.L = 2*np.sqrt(self.dx*self.dy) # filtering size
        self.dT = 0.01 # steps to take for differential calculations
        self.dX = 0.01
        self.dY = 0.01
        self.cen_SO_stencil = [1/12, -2/3, 0, 2/3, -1/12]

        # Define Minkowski metric
        self.metric = np.zeros((4,4))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = self.metric[3,3] = +1
        
        # Load the coordinates and observers already calculated
        with open('KH_observers.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.coord_list, self.vectors, self.funs = pickle.load(f)
     
    # def calc_4vel(W,vx,vy):
    #     return [W,W]
        
    def calc_NonId_terms(self,u,p,rho,n):
        # u = np.dot(W,[1,vx,vy]) # check this works...
        dtut = self.calc_t_deriv('u_t',point)
        dxux = self.calc_x_deriv('u_x',point)
        dyuy = self.calc_y_deriv('u_y',point)
        dxuy = self.calc_x_deriv('u_y',point)
        dyux = self.calc_y_deriv('u_x',point)
        dxT = self.calc_x_deriv('T',point)
        dyT = self.calc_y_deriv('T',point)
        Theta = dtut + dxux + dyuy
        Pi = -self.coefficients['zeta']*Theta
        q = -self.coefficients['kappa']*(dxT + dyT) # FIX
        pi = -self.coefficients['eta']*(dxuy + dyux - (2/3)*Theta)
        return Pi, q, pi
    
    def p_from_EoS(self,rho, n):
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p, rho, n
    
    def calc_Id_SET(self,u,p,rho):
        Id_SET = rho*np.outer(u,u) + p*self.metric
        return Id_SET

    def calc_NonId_SET(self,u,p,rho,n,coefficients):
        Pi, q, pi = self.calc_NonId_terms(u,p,rho,n)
        u_mu_u_nu = np.outer(u,u)
        h_mu_nu = self.metric + u_mu_u_nu
        NonId_SET = rho*u_mu_u_nu + (p+Pi)*h_mu_nu + np.outer(q,u) + np.outer(u,q) + pi
        return NonId_SET
    
    def calc_t_deriv(self, quant_str, point):
        t, x, y = point
        values = [self.scalar_val(T,x,y,quant_str) for T in np.linspace(t-2*self.dT,t+2*self.dT,5)]
        dt_quant = np.dot(self.cen_SO_stencil, values) / self.dT
        return dt_quant
    
    def calc_x_deriv(self, quant_str):
        t, x, y = point
        values = [self.scalar_val(t,X,y,quant_str) for X in np.linspace(x-2*self.dX,x+2*self.dX,5)]
        dX_quant = np.dot(self.cen_SO_stencil, values) / self.dX
        return dX_quant

    def calc_y_deriv(self, quant_str):
        t, x, y = point
        values = [self.scalar_val(t,x,Y,quant_str) for Y in np.linspace(y-2*self.dX,y+2*self.dY,5)]
        dX_quant = np.dot(self.cen_SO_stencil, values) / self.dY
        return dX_quant
    
    def scalar_val(self, t, x, y, quant_str):
        return interpn(self.points,self.vars[quant_str],[t,x,y])
    
    def scalar_val_point(self, point, quant_str):
        return interpn(self.points,self.vars[quant_str],point)
    
    def filter_scalar(self, point, U, quant_str):
        # contruct tetrad...
        E_x, E_y = Base.construct_tetrad(Base,U)
        corners = Base.find_boundary_pts(Base,E_x,E_y,point,L)
        start, end = corners[0], corners[2]
        t, x, y = point
        # integrated_quant = nquad(self.scalar_val,t-(L/2),t+(L/2),x-(L/2),x+(L/2),y-(L/2),y+(L/2),args=quant_str)
        print(quant_str)
        integrated_quant = nquad(func=self.scalar_val,ranges=[[start[0],end[0]],[start[1],end[1]],[start[2],end[2]]],args=[quant_str])
        # integrated_quant = nquad(func=self.scalar_val_point,
        #     ranges=[[start[0], start[1],start[2]],[end[0],end[1],end[2]]],args=quant_str)
        integrand = 0
        counter = 0
        start_cell, end_cell = self.find_nearest_cell([t-(L/2),x-(L/2),y-(L/2)]), self.find_nearest_cell([t+(L/2),x+(L/2),y+(L/2)])
        print(start_cell, end_cell)
        for i in range(start_cell[0],end_cell[0]+1):
            for j in range(start_cell[1],end_cell[1]+1):
                for k in range(start_cell[2],end_cell[2]+1):
                    integrand += self.vars[quant_str][i][j,k]
                    counter += 1
        print(counter)
        print(integrated_quant[0],integrand)
        return integrand/counter
        # for t_s in np.linspace(t-(L/2),t+(L/2),40):
        #     for x_pos np.linspace(x-2*self.dX,x+2*self.dX,40):
        #         for y_pos in np.linspace(y-2*self.dX,y+2*self.dY,40):
        #             integrand += self.vars[quant_str][self.find_nearest_cell(t_pos,x_pos,y_pos)]
        return integrated_quant[0] / (self.L**3) # seems too simple!?

    def project_tensor(vector1_wrt, vector2_wrt, to_project):
        projection = np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
        return projection
    
    def orthogonal_projector(self, u):
        return self.metric + np.outer(u,u)
    
    def values_from_hdf5(self, point, quant_str):
        t_label, x_label, y_label = self.find_nearest_cell(point)
        return self.vars[quant_str][t_label][x_label, y_label] # fix
    
    def find_nearest(self, array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
    def find_nearest_cell(self, point):
        t_pos = self.find_nearest(self.ts,point[0])
        x_pos = self.find_nearest(self.xs,point[1])
        y_pos = self.find_nearest(self.ys,point[2])
        return [np.where(self.ts==t_pos)[0][0], np.where(self.xs==x_pos)[0][0], np.where(self.ys==y_pos)[0][0]]
    
if __name__ == '__main__':
    
    Processor = PostProcessing()
    
    # f_obs = open("observers.txt", "r")
    # observers = f_obs.read()
    # print(observers)
    # print(observers[2:-1:1])
    #print(observers.split("]],"))
    with open('test_obs.pickle', 'rb') as filehandle:
        # Read the data as a binary data stream
        test_obs = pickle.load(filehandle)
    #print(test_obs)
    
    points = []
    Us = [] # Filtered
    for i in range(len(test_obs)): #awful
        points.append(test_obs[i][0])
        Us.append(test_obs[i][1])

    #print(points,Us)

    scalar_strs = ['rho', 'n', 'p']
    vector_strs = ['W', 'u_x', 'u_y']
    L = 0.1

    for point, U in zip(points, Us):
        #for scalar_str in scalar_strs:
            
            # Ns.append(Processor.filter_scalar(point, U, scalar_str, L))
            # print(Ns)

        # Filter scalar fields
        N = Processor.filter_scalar(point, U, scalar_strs[0])
        Rho = Processor.filter_scalar(point, U, scalar_strs[1])
        P = Processor.filter_scalar(point, U, scalar_strs[2])
        T = P/N
        U_t = Processor.filter_scalar(point, U, vector_strs[0])
        U_x = Processor.filter_scalar(point, U, vector_strs[1])
        U_y = Processor.filter_scalar(point, U, vector_strs[2])
      
        # Obtain coarse values
        n, rho, p = Processor.values_from_hdf5(point, scalar_strs[0]),\
            Processor.values_from_hdf5(point, scalar_strs[1]), Processor.values_from_hdf5(point, scalar_strs[2])
        W, u_x, u_y = Processor.values_from_hdf5(point, vector_strs[0]),\
            Processor.values_from_hdf5(point, vector_strs[1]), Processor.values_from_hdf5(point, vector_strs[2])
        u = [W, u_x, u_y]

        # Calculate Non-Ideal terms
        Pi, q, pi = Processor.calc_NonId_terms(u,p,rho) # coarse dissipative pieces

        # Construct coarse Id & Non-Id SET         
        coarse_Id_SET = Processor.calc_Id_SET(u, p, rho)
        coarse_nId_SET = Processor.calc_NonId_SET(u, p, rho, Pi, q, pi)
        # Construct filtered Id & Non-Id SET
        filtered_Id_SET = Processor.calc_Id_SET(U, P, Rho)
        filtered_nId_SET = Processor.calc_NonId_SET(U, P, Rho, Pi, q, pi)           
        
        # Do required projections of SET
        h_mu_nu = Processor.orthogonal_projector(U)
        parallel_proj = Processor.project_tensor(U,U,coarse_Id_SET)
        orthog_proj = Processor.project_tensor(h_mu_nu,h_mu_nu,coarse_Id_SET)
        mixed_proj = Processor.project_tensor(h_mu_nu, U, coarse_Id_SET)
    
        # Obtain residuals
        rho_res = parallel_proj
        q_res = mixed_proj / (2*U)
        S_mu_mu = np.trace(orthog_proj) # = (rho + p + Pi) u^2 + 2 q^mu u_mu + 4(p + Pi) CHECK
        Pi_res = (S_mu_mu - 4*(P+Pi) - 2*q_res*U ) / U**2 - rho_res - P
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    