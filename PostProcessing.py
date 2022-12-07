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
from scipy.integrate import solve_ivp, quad, tplquad
import cProfile, pstats, io


# if __name__ == '__main__':

class PostProcessing(object):
        
    def __init__(self):
        fs1 = []
        num_files = 11
        for n in range(num_files):
          fs1.append(h5py.File('./Data/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
        fss = [fs1]
        nx = ny = 800
        
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
        for fs in fss[0]:
            self.vxs.append(fs['Primitive/v1'][:])
            self.vys.append(fs['Primitive/v2'][:])
            self.ns.append(fs['Primitive/n'][:])
            self.rhos.append(fs['Primitive/rho'][:])
            self.ps.append(fs['Primitive/p'][:])
        
        # EoS & dissipation parameters
        self.coefficients = {'gamma': 5/3,
                        'zeta': 1e-2,
                        'kappa': 1e-4,
                        'eta': 1e-2}
        
        self.dT = 0.01 # steps to take for differential calculations
        self.dX = 0.01
        
        # Define Minkowski metric
        self.metric = np.zeros((4,4))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = self.metric[3,3] = +1
        
        # Load the coordinates and observers already calculated
        with open('KH_observers.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            self.coord_list, self.vectors, self.funs = pickle.load(f)
        
    # def calc_NonId_terms(u,p,T):
    #     Pi = -coefficients['zeta']*
    #     q = coefficients['kappa']*
    #     pi = -coefficients['eta']*()
    #     return Pi, q, pi
    
    def p_from_EoS(rho, n):
        p = (self.coefficients['gamma']-1)*(rho-n)
        return p, rho, n
    
    def calc_Id_SET(u,p,rho):
        Id_Set = rho*np.outer(u,u) + p*metric
        return Id_SET

    def calc_NonId_SET(u,p,rho,coefficients):
        u_mu_u_nu = np.outer(u,u)
        h_mu_nu = self.metric + u_mu_u_nu
        NonId_SET = rho*u_mu_u_nu + (p+Pi)*h_mu_nu + np.outer(q,u) + np.outer(u,q) + pi
        return NonId_SET
    
    def calc_t_deriv(quant, point):
        t, x, y = point
        # values = [interpolate_quant(,,[ts,x,y]) for t in np.linspace(t-2*self.dT,t+2*self.dT,5)]
        stencil = [1/12, -2/3, 0, 2/3, -1/12]
        dt_quant = np.dot(stencil, values) / self.dT
        return dt_quant
    
    def calc_X_deriv(quant):
        dX_quant = np.dot(stencil, values) / self.dX
        return dX_quant
    
    def surface_flux(self, x,E_x,E_y,P,direc_vec):
        point = P + x*(E_x + E_y)
        u, n = self.interpolate_u_n_point(point)
        n_mu = np.multiply(u,n)
        return self.Mink_dot(n_mu,direc_vec)
    
    def residual_ib(self, V0_V1,P,L):
        U = self.get_U_mu(V0_V1)
        # U = get_U_mu_MagTheta(V0_V1)
        E_x, E_y = self.construct_tetrad(U)
        flux = 0
        for i in range(3,-1,-1):
            if i == 3:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(E_x,E_y,P,-E_x))[0]
            elif i == 2:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(E_x,-E_y,P,-E_y))[0]
            elif i == 1:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(-E_x,-E_y,P,E_x))[0]
            elif i == 0:
                flux += quad(func=self.surface_flux,a=-L/2,b=L/2,args=(-E_x,E_y,P,E_y))[0]
            return abs(flux)
    
    def scalar_val(self, t, x, y, quant_str):
        return interpn(self.points,self.vars[quant_str],[t,x,y])
    
    def filter_scalar(self, point, U, quant_str, L):
        # contruct tetrad...
        E_x, E_y = Base.construct_tetrad(U)
        corners = Base.find_boundary_pts(E_x,E_y,P,L)
        t, x, y = point
        integrated_quant = tplquad(self.scalar_val,t-(L/2),t+(L/2),x-(L/2),x+(L/2),y-(L/2),y+(L/2),args=quant_str)
        return integrated_quant / (L**3) # seems too simple!?

    def project_tensor(self, vector1_wrt, vector2_wrt, to_project):
        projection = np.inner(vector1_wrt,np.inner(vector2_wrt,to_project))
        return projection
    
    def orthogonal_projector(self, u):
        return self.metric + np.outer(u,u)
    
    def values_from_hdf5(self, point, quant_str):
        return self.macros[quant_str][point[0],point[1],point[2]] # fix
    
if __name__ == '__main__':

    import pickle
    
    Processor = PostProcessing()
    
    f_obs = open("observers.txt", "r")
    observers = f_obs.read()
    # print(observers)
    # print(observers[2:-1:1])
    #print(observers.split("]],"))
    pickle_file = open("KH_observers.pickle",'r')
    obs_pickle = pickle.load(pickle_file)
    print(obs_pickle)
    
    points = 
    Us = # Filtered

    scalar_strs = ['n', 'T', 'rho', 'p']
    micros = []

    for point, U in zip(points, Us):
        for scalar_str in scalar_strs:
            filtered N, Rho, P = Processor.filter_scalar(point, U, quant_str, L)
            coarse n, rho, p = Processor.ns[point, scalar_str], Processor.Ts[point, scalar_str], ...       
            coarse W, vx, vy = Processor.Ws[point, scalar_str], Processor.vxs[point, scalar_str], ...   
            coarse_Id_SET = calc_Id_SET(u, p, rho)
            # filtered_Id_SET = calc_Id_SET(U, P, Rho)
            
            rho_res = Processor.project_tensor(U,U,coarse_Id_SET)
            
            Pi, q, pi = Processor.calc_NonId_terms(u,p,rho) # coarse dissipative pieces
            
            # coarse_nId_SET = Processor.calc_NonId_SET(u, p, rho, Pi, q, pi)
            filtered_nId_SET = Processor.calc_NonId_SET(U, P, Rho, Pi, q, pi)
            
            h_mu_nu = orthogonal_projector(self, U)
            parallel_proj = Processor.project_tensor(U,U,coarse_Id_SET)
            orthog_proj = Processor.project_tensor(h_mu_nu,h_mu_nu,coarse_Id_SET)
            mixed_proj = Processor.project_tensor(h_mu_nu, U, coarse_Id_SET)
            q_res = mixed_proj / (2*U)
            S_mu_mu = np.trace(momentum_proj) # = (rho + p + Pi) u^2 + 2 q^mu u_mu + 4(p + Pi) CHECK
            Pi_res = (S_mu_mu - 4(P+Pi) - 2*q_res*U ) / U**2 - rho_res - P
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    