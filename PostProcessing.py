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
# from scipy.optimize import root, minimize
#from mpl_toolkits.mplot3d import Axes3D
# from scipy.integrate import solve_ivp, quad, tplquad, nquad
import cProfile, pstats, io
#from system.BaseFunctionality import Base
import math
from multiprocessing import Process, Pool

class PostProcessing(object):
        
    def __init__(self):
        fs_f = [] # fine
        fs_c = [] # coarse
        num_files = 5
        for n in range(num_files):
          # fs_f.append(h5py.File('./Data/KH/Ideal/dp_400x400x0_'+str(n)+'.hdf5','r'))
          # fs_1f.append(h5py.File('./Data/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
          #fs_f.append(h5py.File('./Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
          fs_f.append(h5py.File('../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_998_1002/dp_400x800x0_'+str(n)+'.hdf5','r'))
        fss = [fs_f]
        self.nx, self.ny = int(400), int(800)
        # self.c_nx, self.c_ny = int(self.nx/2), int(self.ny/2) # coarse
        # self.c_nx, self.c_ny = 200, 200 # coarse
        
        
        self.ts = np.linspace(9.98,10.02,num_files) # Need to actually get these
        self.xs = np.linspace(-0.5,0.5,self.nx)
        self.ys =  np.linspace(-1.0,1.0,self.ny)
        self.points = (self.ts,self.xs,self.ys)
        # self.dt get this from files...
        self.dx = (self.xs[-1] - self.xs[0])/self.nx # actual grid-resolution
        self.dy = (self.ys[-1] - self.ys[0])/self.ny
        # Numer of observer time slices
        self.n_obs_t = 3 # = num_files / num_files - 2 ?
        # Number of observers calculated in x and y directions
        self.n_obs_x = 41
        self.n_obs_y = 21
        self.dt_obs = 0.01
        self.dx_obs = 0.02
        self.dy_obs = 0.02
        self.n_t_slices = self.n_obs_t - 2 # number of time slices for which to calculate residuals
        self.n_x_pts = self.n_obs_x - 2 # to avoid edges
        self.n_y_pts = self.n_obs_y - 2
        
        # Load coords and corresponding observers
        self.coords = np.loadtxt('coords998.txt').reshape(self.n_obs_t,self.n_obs_x,self.n_obs_y,3)
        self.Us = np.loadtxt('obs998.txt').reshape(self.n_obs_t,self.n_obs_x,self.n_obs_y,3)

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
        self.Id_SETs = np.zeros((num_files, self.nx, self.ny, 3, 3))

        self.vars = {'v1': self.vxs,
                          'v2': self.vys,
                          'n': self.ns,
                          'rho': self.rhos,
                          'p': self.ps,
                          'W': self.Ws,
                          'u_t': self.uts,
                          'u_x': self.uxs,
                          'u_y': self.uys,
                          'T': self.Ts,
                          'Id_SET': self.Id_SETs}

        self.prim_vars_strs = ['v1','v2','n','rho','p']
        self.aux_vars_strs= ['W','T']

        # Strings for iterating over for filtering in calc_residual
        self.scalar_strs = ['rho', 'n', 'p']
        self.vector_strs = ['W', 'u_x', 'u_y']
        self.tensor_strs = ['Id_SET']
        
        # Single slice for observers for now
        self.Uts = self.Us[:,:,:,0] # 3 slices for now 
        self.Uxs = self.Us[:,:,:,1]
        self.Uys = self.Us[:,:,:,2]
        # print(self.Uxs.shape)
        self.dtUts = np.zeros((self.n_obs_x,self.n_obs_y)) # single slice for now
        self.dtUxs = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dtUys = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dxUts = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dxUxs = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dxUys = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dyUts = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dyUxs = np.zeros((self.n_obs_x,self.n_obs_y))
        self.dyUys = np.zeros((self.n_obs_x,self.n_obs_y))

        # Define Minkowski metric
        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

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
            # Construct Ideal SET
            for i in range(self.nx):
                for j in range(self.ny): # Fix with ux not  vx... also v1???
                    self.uts[counter][i,j] = self.Ws[counter][i,j]
                    self.uxs[counter][i,j] = self.Ws[counter][i,j]*self.vxs[counter][i,j]
                    self.uys[counter][i,j] = self.Ws[counter][i,j]*self.vys[counter][i,j]
                    u_vec = np.array([self.uts[counter][i,j],self.uxs[counter][i,j],self.uys[counter][i,j]])
                    self.Id_SETs[counter][i,j] = f_f['Primitive/rho'][i,j]*np.outer(u_vec,u_vec)\
                        + f_f['Primitive/p'][i,j]*(self.metric + np.outer(u_vec,u_vec))
            
        # Size of box for spatial filtering
        self.L = 2*np.sqrt(self.dx*self.dy) # filtering size
        self.dT = 0.01 # steps to take for differential calculations
        self.dX = 0.01
        self.dY = 0.01
        self.cen_SO_stencil = [1/12, -2/3, 0, 2/3, -1/12]
        self.cen_FO_stencil = [-1/2, 0, 1/2]
        self.fw_FO_stencil = [-1, 1]
        self.bw_FO_stencil = [-1, 1]
        
        # Calculate time derivatives for central slices
        for t_slice in range(self.n_obs_t):
            # Central-difference                                
            for i in range(1,self.n_obs_x-1):
                for j in range(1,self.n_obs_y-1): # fix these to use self.n_x_pts etc.
                    self.dtUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[t_slice][i][j] / self.dt_obs
                    self.dtUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[t_slice][i][j] / self.dt_obs
                    self.dtUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[t_slice][i][j] / self.dt_obs
                    # pick out central slice with first [1]    
                    self.dxUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[1][i-1+t_slice][j] / self.dx_obs
                    self.dxUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[1][i-1+t_slice][j] / self.dx_obs
                    self.dxUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[1][i-1+t_slice][j] / self.dx_obs

                    self.dyUts[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uts[1][i][j-1+t_slice] / self.dy_obs
                    self.dyUxs[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uxs[1][i][j-1+t_slice] / self.dy_obs
                    self.dyUys[i,j] += self.cen_FO_stencil[t_slice]*\
                        self.Uys[1][i][j-1+t_slice] / self.dy_obs
                        
                    # Need some BCs to e.g. copy to edges that are missed here...

        # EoS & dissipation parameters
        self.coefficients = {'gamma': 5/3,
                        'zeta': 1e-2,
                        'kappa': 1e-4,
                        'eta': 1e-2}
        self.calculated_coefficients = np.zeros((self.n_t_slices,self.n_x_pts,self.n_y_pts))
        
        
    # def calc_4vel(W,vx,vy):
    #     return [W,W]
        
    def calc_NonId_terms(self,obs_indices,point):
        # u = np.dot(W,[1,vx,vy]) # check this works...
        h, i, j = obs_indices
        print(obs_indices)
        T = self.values_from_hdf5(point, 'T') # Fix this - should be from EoS(N,p_tilde)
        dtT = self.calc_t_deriv('T',point)[0]
        dxT = self.calc_x_deriv('T',point)[0]
        dyT = self.calc_y_deriv('T',point)[0]
        print(T.shape,dxT.shape)
        Ut = self.Uts[h,i,j]
        Ux = self.Uxs[h,i,j]
        Uy = self.Uys[h,i,j]
        dtUt = self.dtUts[i,j]
        dtUx = self.dtUxs[i,j]
        dtUy = self.dtUys[i,j]
        dxUt = self.dxUts[i,j]
        dxUx = self.dxUxs[i,j]
        dxUy = self.dxUys[i,j]
        dyUt = self.dyUts[i,j]
        dyUx = self.dyUxs[i,j]
        dyUy = self.dyUys[i,j]
   
        Theta = dtUt + dxUx + dyUy
        a = np.array([Ut*dtUt + Ux*dxUt + Uy*dyUt, Ut*dtUx + Ux*dxUx + Uy*dyUx, Ut*dtUy + Ux*dxUy + Uy*dyUy])#,ux*dxuz+uy*dyuz+uz*dzuz])
        print(Theta,a)

        omega = np.array([dtT, dxT, dyT]) + np.multiply(T,a) # FIX
        print(np.array([dtT, dxT, dyT]))
        print(np.multiply(T,a))
        print(omega.shape)
        sigma = np.array([[2*dtUt - (2/3)*Theta, dtUx + dxUt, dtUy + dyUt],\
                                                  [dxUt + dtUx, 2*dxUx - (2/3)*Theta, dxUy + dyUx],
                                                  [dyUt + dtUy, dyUx + dxUy, 2*dyUy - (2/3)*Theta]])

        # return -self.coefficients['zeta']*Theta, -self.coefficients['kappa']*omega, -self.coefficients['eta']*sigma
        return Theta, omega, sigma

    
    def p_from_EoS(self,rho, n):
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p
    
    # def calc_Id_SET(self,u,p,rho):
    #     Id_SET = rho*np.outer(u,u) + p*(self.metric + np.outer(u,u))
    #     return Id_SET

    # def calc_NonId_SET(self,u,p,rho,n,Pi, q, pi):
    #     #Pi, q, pi = self.calc_NonId_terms(u,p,rho,n)
    #     u_mu_u_nu = np.outer(u,u)
    #     h_mu_nu = self.metric + u_mu_u_nu
    #     NonId_SET = rho*u_mu_u_nu + (p+Pi)*h_mu_nu + np.outer(q,u) + np.outer(u,q) + pi
    #     return NonId_SET
    
    def calc_t_deriv(self, quant_str, point):
        t, x, y = point
        # print(t,x,y)
        # values = [self.scalar_val(T,x,y,quant_str) for T in np.linspace(t-2*self.dT,t+2*self.dT,5)]
        values = [self.scalar_val(T,x,y,quant_str) for T in np.linspace(t-1*self.dT,t+1*self.dT,3)]
        # dt_quant = np.dot(self.cen_SO_stencil, values) / self.dT
        dt_quant = np.dot(self.cen_FO_stencil, values) / self.dT
        return dt_quant
    
    def calc_x_deriv(self, quant_str, point):
        t, x, y = point
        # values = [self.scalar_val(t,X,y,quant_str) for X in np.linspace(x-2*self.dX,x+2*self.dX,5)]
        values = [self.scalar_val(t,X,y,quant_str) for X in np.linspace(x-1*self.dX,x+1*self.dX,3)]
        # dX_quant = np.dot(self.cen_SO_stencil, values) / self.dX
        dX_quant = np.dot(self.cen_FO_stencil, values) / self.dX
        return dX_quant

    def calc_y_deriv(self, quant_str, point):
        t, x, y = point
        # values = [self.scalar_val(t,x,Y,quant_str) for Y in np.linspace(y-2*self.dX,y+2*self.dY,5)]
        values = [self.scalar_val(t,x,Y,quant_str) for Y in np.linspace(y-1*self.dY,y+1*self.dY,3)]
        # dY_quant = np.dot(self.cen_SO_stencil, values) / self.dY
        dY_quant = np.dot(self.cen_FO_stencil, values) / self.dY
        return dY_quant
    
    def scalar_val(self, t, x, y, quant_str):
        return interpn(self.points,self.vars[quant_str],[t,x,y])
    
    def scalar_val_point(self, point, quant_str):
        return interpn(self.points,self.vars[quant_str],point)

    def construct_tetrad(self, U):
        e_x = np.array([0.0,1.0,0.0]) # 1 + 2D
        E_x = e_x + np.multiply(self.Mink_dot(U,e_x),U)
        E_x = E_x / np.sqrt(self.Mink_dot(E_x,E_x)) # normalization
        e_y = np.array([0.0,0.0,1.0])
        E_y = e_y + np.multiply(self.Mink_dot(U,e_y),U) - np.multiply(self.Mink_dot(E_x,e_y),E_x)
        E_y = E_y / np.sqrt(self.Mink_dot(E_y,E_y))
        return E_x, E_y
    
    def Mink_dot(self,vec1,vec2):
        dot = -vec1[0]*vec2[0] # time component
        for i in range(1,len(vec1)):
            dot += vec1[i]*vec2[i] # spatial components
        return dot
    
    def find_boundary_pts(self, E_x,E_y,P,L):
        c1 = P + (L/2)*(E_x + E_y)
        c2 = P + (L/2)*(E_x - E_y)
        c3 = P + (L/2)*(-E_x - E_y)
        c4 = P + (L/2)*(-E_x + E_y)
        corners = [c1,c2,c3,c4]
        return corners
    
    def filter_scalar(self, point, U, quant_str):
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


    def project_tensor(self,vector1_wrt, vector2_wrt, to_project):
        projection = np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
        return projection
    
    def orthogonal_projector(self, u):
        return self.metric + np.outer(u,u)
    
    def values_from_hdf5(self, point, quant_str):
        t_label, x_label, y_label = self.find_nearest_cell(point)
        return self.vars[quant_str][t_label][x_label, y_label] # fix
    
    def find_nearest(self, array, value):
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
    
    def calc_coeffs(self, point, U, obs_indices):
            # point, U = point_and_vel
            h, i, j = obs_indices
            # Filter the scalar fields
            N = self.filter_scalar(point, U, self.scalar_strs[0])
            # Rho = self.filter_scalar(point, U, self.scalar_strs[1])
            # P = self.filter_scalar(point, U, self.scalar_strs[2])
            # T = P/N
            # U_t = self.filter_scalar(point, U, self.vector_strs[0])
            # U_x = self.filter_scalar(point, U, self.vector_strs[1])
            # U_y = self.filter_scalar(point, U, self.vector_strs[2])
          
            # Obtain coarse values
            # n, rho, p = self.values_from_hdf5(point, self.scalar_strs[0]),\
            #     self.values_from_hdf5(point, self.scalar_strs[1]), self.values_from_hdf5(point, self.scalar_strs[2])
            # W, u_x, u_y = self.values_from_hdf5(point, self.vector_strs[0]),\
            #     self.values_from_hdf5(point, self.vector_strs[1]), self.values_from_hdf5(point, self.vector_strs[2])
            # u = [W, u_x, u_y]
            # t = n/p

            # Construct filtered Id SET         
            # coarse_Id_SET = self.calc_Id_SET(u, p, rho)
            filtered_Id_SET = self.filter_scalar(point, U, self.tensor_strs[0])        

            # Do required projections of SET
            h_mu_nu = self.orthogonal_projector(U)
            rho_res = self.project_tensor(U,U,filtered_Id_SET)
            q_res = np.einsum('ij,i,jk',filtered_Id_SET,U,h_mu_nu)            
            tau_res = np.einsum('ij,ik,jl',filtered_Id_SET,h_mu_nu,h_mu_nu) # tau = p + Pi+ pi
            
            # Calculate Pi and pi residuals
            tau_trace = np.trace(tau_res)#
            # print('tau_trace ',tau_trace)
            p_tilde = self.p_from_EoS(rho_res, N)
            # print('N, rho_res ',N, rho_res)
            # print('p_tilde ', p_tilde)
            Pi_res = tau_trace - p_tilde
            pi_res = tau_res - np.dot((p_tilde + Pi_res),h_mu_nu)
            
            # print('rho','Pi','q','pi','residuals')
            print('rho_res ',rho_res)
            print('Pi_res ',Pi_res)
            print('q_res ',q_res)
            print('pi_res',pi_res)
            # Calculate Non-Ideal terms
            # need to calc. derivatives here!
            # T_tilde = p_tilde/N
            # Theta, omega, sigma = self.calc_NonId_terms(T_tildes, U_tildes) # coarse dissipative pieces (without coefficients)

            # Temp hack - see PDF from Ian
            Theta, omega, sigma = self.calc_NonId_terms(obs_indices,point)
            zeta = -Pi_res/Theta
            kappa = np.average(-q_res/omega)
            eta = np.average(-pi_res/(2*sigma))
            # print('Theta ', Theta)
            # print('omega ',omega)
            # print('sigma ',sigma)
            print('zeta ', zeta)
            print('kappa ',kappa)
            print('eta ',eta)
            return zeta, kappa, eta
    
           
    
    
if __name__ == '__main__':
    
    Processor = PostProcessing()

    with open('Processor.pickle', 'wb') as filehandle:
        pickle.dump(Processor, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('Processor.pickle', 'rb') as filehandle:
    #     Processor = pickle.load(filehandle)
   
    # args = [(coord, vector) for coord, vector in zip(Processor.coords, Processor.Us)]
    # for h in range(Processor.n_obs_t):
    #     for i in range(Processor.n_obs_x):
    #         for j in range(Processor.n_obs_y):
    for h in range(Processor.n_t_slices):
        for i in range(Processor.n_x_pts):
            for j in range(Processor.n_y_pts):
                Processor.calculated_coefficients[h,i,j] = Processor.calc_coeffs(Processor.coords[h,i,j],Processor.Us[h,i,j],[h,i,j])

    np.savetxt('cald_coeff.txt', Processor.calculated_coefficients)

    # residuals_handle = open('Residuals.pickle', 'wb')
    # start = timer()
    # with Pool(2) as p:
    #     residuals = p.starmap(Processor.calc_residual, args)
    #     pickle.dump(residuals, residuals_handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     # print(residuals)
    #     # pickle.dump(p.starmap(Processor.calc_residual, args), residuals_handle, protocol=pickle.HIGHEST_PROTOCOL)
    # end = timer()
    
    
    

            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
