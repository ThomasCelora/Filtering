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


class PostProcessing(object):
        
    def __init__(self):
        fs1 = []
        num_files = 11
        for n in range(num_files):
          fs1.append(h5py.File('./Data/KH/Ideal/dp_400x400x0_'+str(n)+'.hdf5','r'))
        fss = [fs1]
        nx, ny = int(400), int(400)
        c_nx, c_ny = int(nx/2), int(ny/2) # coarse
        
        ts = np.linspace(0,30,11) # Need to actually get these
        xs = np.linspace(-0.5,0.5,nx)
        ys =  np.linspace(-1.0,1.0,ny)
        self.points = (ts,xs,ys)
        # self.dt get this...
        self.dx = (xs[-1] - xs[0])/nx # actual grid-resolution
        self.dy = (ys[-1] - ys[0])/ny
        self.vxs = []
        self.vys = []
        self.ns = []
        self.rhos = []
        self.ps = []
        self.Ws = []
        self.Ts = []
        for fs in fss[0]:
            self.vxs.append(fs['Primitive/v1'][:])
            self.vys.append(fs['Primitive/v2'][:])
            self.ns.append(fs['Primitive/n'][:])
            self.rhos.append(fs['Primitive/rho'][:])
            self.ps.append(fs['Primitive/p'][:])
            self.Ws.append(fs['Auxiliary/W'][:])
            self.Ts.append(fs['Auxiliary/T'][:])
            vxs_fine = fs['Primitive/v1'][:]
            vxs_coarse = np.zeros((c_nx,c_ny))
            for i in range(c_nx):
                for j in range(c_ny):
                    vxs_coarse[i][j] = vxs_fine[i*2][j*2] + vxs_fine[i*2+1][j*2] \
                                       + vxs_fine[i*2][j*2+1] + vxs_fine[i*2][j*2+1]

        self.ut = self.Ws
        self.ux = np.dot(self.ut,self.vxs)
        self.uy = np.dot(self.ut,self.vys)
    
        self.vars = {'v_x': self.vxs,
                          'v_y': self.vys,
                          'n': self.ns,
                          'rho': self.rhos,
                          'p': self.ps,
                          'W': self.Ws,
                          'u_t': self.ut,
                          'u_x': self.ux,
                          'u_y': self.uy,
                          'T': self.Ts}
        
        
        # EoS & dissipation parameters
        self.coefficients = {'gamma': 5/3,
                        'zeta': 1e-2,
                        'kappa': 1e-4,
                        'eta': 1e-2}
        
        self.L = 0.1
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
     
    def calc_4vel(W,vx,vy):
        return [W,W]
        
    def calc_NonId_terms(self,u,p,rho,n):
        # u = np.dot(W,[1,vx,vy]) # check this works...
        T = p/n
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
    
    def filter_scalar(self, point, U, quant_str, L):
        # contruct tetrad...
        E_x, E_y = Base.construct_tetrad(Base,U)
        corners = Base.find_boundary_pts(Base,E_x,E_y,point,L)
        start, end = corners[0], corners[2]
        t, x, y = point
        # integrated_quant = nquad(self.scalar_val,t-(L/2),t+(L/2),x-(L/2),x+(L/2),y-(L/2),y+(L/2),args=quant_str)
        print(start,end)
        print(start[0],end[2])
        print(quant_str)
        # integrated_quant = nquad(func=self.scalar_val,ranges=[[start[0],end[0]],[start[1],end[1]],[start[2],end[2]]],args=quant_str)
        integrated_quant = nquad(func=self.scalar_val_point,
            ranges=[[start[0], start[1],start[2]],[end[0],end[1],end[2]]],args=quant_str)
        return integrated_quant[0] / (L**3) # seems too simple!?

    def project_tensor(vector1_wrt, vector2_wrt, to_project):
        projection = np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
        return projection
    
    def orthogonal_projector(self, u):
        return self.metric + np.outer(u,u)
    
    def values_from_hdf5(self, point, quant_str):
        return self.macros[quant_str][point[0],point[1],point[2]] # fix
    
if __name__ == '__main__':

    import pickle
    
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

    scalar_strs = ['n', 'rho', 'p']
    vector_strs = ['W', 'u_x', 'u_y']
    micros = []
    L = 0.1
    Ns = []

    for point, U in zip(points, Us):
        #for scalar_str in scalar_strs:
            
            # Ns.append(Processor.filter_scalar(point, U, scalar_str, L))
            # print(Ns)

        # Filter scalar fields
        N, Rho, P = Processor.filter_scalar(point, U, scalar_strs[0], L),\
            Processor.filter_scalar(point, U, scalar_strs[1], L), Processor.filter_scalar(point, U, scalar_strs[2], L)
        T = P/N

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
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    