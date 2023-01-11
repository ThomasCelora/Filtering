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
from multiprocessing import Process, Pool

class PostProcessing(object):
        
    def __init__(self):
        fs_f = [] # fine
        fs_c = [] # coarse
        num_files = 11
        for n in range(num_files):
          # fs_f.append(h5py.File('./Data/KH/Ideal/dp_400x400x0_'+str(n)+'.hdf5','r'))
          # fs_1f.append(h5py.File('./Data/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
          fs_f.append(h5py.File('./Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        fss = [fs_f]
        self.nx, self.ny = int(200), int(200)
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
        self.vxs = np.zeros((num_files, self.nx, self.ny))
        self.vys = np.zeros((num_files, self.nx, self.ny))
        self.uts = np.zeros((num_files, self.nx, self.ny))
        self.uxs = np.zeros((num_files, self.nx, self.ny))
        self.uys = np.zeros((num_files, self.nx, self.ny))
        self.ns = np.zeros((num_files, self.nx, self.ny))
        self.rhos = np.zeros((num_files, self.nx, self.ny))
        self.ps = np.zeros((num_files, self.nx, self.ny))
        self.Ws = np.zeros((num_files, self.nx, self.ny))
        self.Ts = np.zeros((num_files, self.nx, self.ny))
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
        for f_f, counter in zip(fs_f, range(num_files)):
            for p_v_s in self.prim_vars_strs:
                self.vars[p_v_s][counter] = f_f['Primitive/'+p_v_s][:]
                # self.vars_c[p_v_s][counter] = f_c['Primitive/'+p_v_s][:]
            for a_v_s in self.aux_vars_strs:
                self.vars[a_v_s][counter] = f_f['Auxiliary/'+a_v_s][:]
                # self.vars_c[a_v_s][counter] = f_c['Primitive/'+a_v_s][:]
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
        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1
        
        # Strings for iterating over for filtering in calc_residual
        self.scalar_strs = ['rho', 'n', 'p']
        self.vector_strs = ['W', 'u_x', 'u_y']
        
        # Load the coordinates and observers already calculated
        with open('test_obs.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            inputs = pickle.load(f)
            # print(inputs)
            self.coords = []
            self.vectors = []#np.zeros((len(inputs),3))
            for counter in range(len(inputs)):
                # print(inputs[counter][0])
                self.coords.append(inputs[counter][0])
                self.vectors.append(inputs[counter][1])

                # self.coords[counter] = inputs[counter][0]
                # self.vectors[counter] = inputs[counter][1]
            # print(self.vectors)
            # print(self.coords)

    # def calc_4vel(W,vx,vy):
    #     return [W,W]
        
    def calc_NonId_terms(self,u,p,rho,n,point):
        # u = np.dot(W,[1,vx,vy]) # check this works...
        dtut = self.calc_t_deriv('u_t',point)
        dxux = self.calc_x_deriv('u_x',point)
        dyuy = self.calc_y_deriv('u_y',point)
        dxuy = self.calc_x_deriv('u_y',point)
        dyux = self.calc_y_deriv('u_x',point)
        dxT = self.calc_x_deriv('T',point)
        dyT = self.calc_y_deriv('T',point)
        Theta = dtut + dxux + dyuy
        # a = np.array([ux*dxux+uy*dyux+uz*dzux,ux*dxuy+uy*dyuy+uz*dzuy,ux*dxuz+uy*dyuz+uz*dzuz])
        Pi = -self.coefficients['zeta']*Theta
        print(dxT.shape)
        q = -self.coefficients['kappa']*np.array([0.0, dxT, dyT]) # FI
        pi = -self.coefficients['eta']*np.array([[0.0, 2*dxux - (2/3)*Theta, dxuy + dyux],\
                                                 [dyux + dxuy,0.0,2*dyuy - (2/3)*Theta],
                                                 [dyux + dxuy,2*dyuy - (2/3)*Theta,0.0]])
        return Pi, q, pi
    
    def p_from_EoS(self,rho, n):
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p, rho, n
    
    def calc_Id_SET(self,u,p,rho):
        Id_SET = rho*np.outer(u,u) + p*self.metric
        return Id_SET

    def calc_NonId_SET(self,u,p,rho,n,Pi, q, pi):
        #Pi, q, pi = self.calc_NonId_terms(u,p,rho,n)
        u_mu_u_nu = np.outer(u,u)
        h_mu_nu = self.metric + u_mu_u_nu
        NonId_SET = rho*u_mu_u_nu + (p+Pi)*h_mu_nu + np.outer(q,u) + np.outer(u,q) + pi
        return NonId_SET
    
    def calc_t_deriv(self, quant_str, point):
        t, x, y = point
        values = [self.scalar_val(T,x,y,quant_str) for T in np.linspace(t-2*self.dT,t+2*self.dT,5)]
        dt_quant = np.dot(self.cen_SO_stencil, values) / self.dT
        return dt_quant
    
    def calc_x_deriv(self, quant_str, point):
        t, x, y = point
        values = [self.scalar_val(t,X,y,quant_str) for X in np.linspace(x-2*self.dX,x+2*self.dX,5)]
        dX_quant = np.dot(self.cen_SO_stencil, values) / self.dX
        return dX_quant

    def calc_y_deriv(self, quant_str, point):
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
        corners = Base.find_boundary_pts(Base,E_x,E_y,point,self.L)
        start, end = corners[0], corners[2]
        t, x, y = point
        # integrated_quant = nquad(self.scalar_val,t-(L/2),t+(L/2),x-(L/2),x+(L/2),y-(L/2),y+(L/2),args=quant_str)
        # print(quant_str)
        # integrated_quant = nquad(func=self.scalar_val,ranges=[[start[0],end[0]],[start[1],end[1]],[start[2],end[2]]],args=[quant_str])
        # integrated_quant = nquad(func=self.scalar_val_point,
        #     ranges=[[start[0], start[1],start[2]],[end[0],end[1],end[2]]],args=quant_str)
        integrand = 0
        counter = 0
        start_cell, end_cell = self.find_nearest_cell([t-(self.L/2),x-(self.L/2),y-(self.L/2)]), \
            self.find_nearest_cell([t+(self.L/2),x+(self.L/2),y+(self.L/2)])
        # print(start_cell, end_cell)
        for i in range(start_cell[0],end_cell[0]+1):
            for j in range(start_cell[1],end_cell[1]+1):
                for k in range(start_cell[2],end_cell[2]+1):
                    integrand += self.vars[quant_str][i][j,k]
                    counter += 1
        # print(counter)
        # print(integrated_quant[0]/counter,integrand/self.L**3)
        return integrand/counter
        # for t_s in np.linspace(t-(L/2),t+(L/2),40):
        #     for x_pos np.linspace(x-2*self.dX,x+2*self.dX,40):
        #         for y_pos in np.linspace(y-2*self.dX,y+2*self.dY,40):
        #             integrand += self.vars[quant_str][self.find_nearest_cell(t_pos,x_pos,y_pos)]
        # return integrated_quant[0] / (self.L**3) # seems too simple!?

    def project_tensor(self,vector1_wrt, vector2_wrt, to_project):
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
    
    def calc_residual(self, point, U):
            # point, U = point_and_vel
            
            # Filter the scalar fields
            N = self.filter_scalar(point, U, self.scalar_strs[0])
            Rho = self.filter_scalar(point, U, self.scalar_strs[1])
            P = self.filter_scalar(point, U, self.scalar_strs[2])
            T = P/N
            U_t = self.filter_scalar(point, U, self.vector_strs[0])
            U_x = self.filter_scalar(point, U, self.vector_strs[1])
            U_y = self.filter_scalar(point, U, self.vector_strs[2])
          
            # Obtain coarse values
            n, rho, p = self.values_from_hdf5(point, self.scalar_strs[0]),\
                self.values_from_hdf5(point, self.scalar_strs[1]), self.values_from_hdf5(point, self.scalar_strs[2])
            W, u_x, u_y = self.values_from_hdf5(point, self.vector_strs[0]),\
                self.values_from_hdf5(point, self.vector_strs[1]), self.values_from_hdf5(point, self.vector_strs[2])
            u = [W, u_x, u_y]

            # Calculate Non-Ideal terms
            Pi, q, pi = self.calc_NonId_terms(u,p,rho,n,point) # coarse dissipative pieces

            # Construct coarse Id & Non-Id SET         
            coarse_Id_SET = self.calc_Id_SET(u, p, rho)
            coarse_nId_SET = self.calc_NonId_SET(u, p, rho, n, Pi, q, pi)
            # Construct filtered Id & Non-Id SET
            filtered_Id_SET = self.calc_Id_SET(U, P, Rho)
            filtered_nId_SET = self.calc_NonId_SET(U, P, Rho, N, Pi, q, pi)           
            
            # Do required projections of SET
            h_mu_nu = self.orthogonal_projector(U)
            rho_res = self.project_tensor(U,U,filtered_nId_SET)
            tau_res = np.inner(h_mu_nu,filtered_nId_SET) # tau = p + Pi+ pi
            q_res = np.einsum('ij,i,jk',filtered_nId_SET,U,h_mu_nu)
            
            # orthog_proj = self.project_tensor(h_mu_nu,h_mu_nu,coarse_Id_SET)
            # mixed_proj = self.project_tensor(h_mu_nu, U, coarse_Id_SET)
        
            # Obtain residuals
            # rho_res = parallel_proj
            # q_res = mixed_proj / 2*np.sqrt(np.inner(U,U)) # np.outer(U,U) #(2*U)
            # S_mu_mu = np.trace(orthog_proj) # = (rho + p + Pi) u^2 + 2 q^mu u_mu + 4(p + Pi) CHECK
            # Pi_res = (S_mu_mu - 4*(P+Pi) - 2*q_res*U ) / np.inner(U,U) - rho_res - P
            return rho_res, q_res, Pi_res
            print("Rho residual: ", rho_res)
            print("q residual: ", q_res)
            print("Pi residual: ", Pi_res)    
    
    
if __name__ == '__main__':
    
    Processor = PostProcessing()
    
    # f_obs = open("observers.txt", "r")
    # observers = f_obs.read()
    # print(observers)
    # print(observers[2:-1:1])
    #print(observers.split("]],"))
    # with open('test_obs.pickle', 'rb') as filehandle:
    #     # Read the data as a binary data stream
    #     test_obs = pickle.load(filehandle)
    # print(test_obs)
    
    # points = []
    # Us = [] # Filtered
    # for i in range(len(test_obs)): #awful
    #     points.append(test_obs[i][0])
    #     Us.append(test_obs[i][1])

    #print(points,Us)


    args = [(coord, vector) for coord, vector in zip(Processor.coords, Processor.vectors)]

    residuals_handle = open('Residuals.pickle', 'wb')
    start = timer()
    with Pool(2) as p:
        residuals = p.starmap(Processor.calc_residual, args)
        pickle.dump(residuals, residuals_handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(residuals)
        # pickle.dump(p.starmap(Processor.calc_residual, args), residuals_handle, protocol=pickle.HIGHEST_PROTOCOL)
    end = timer()
    
    
    

            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    