# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:21:02 2022

@author: mjh1n20
"""

from multiprocessing import Process, Pool
import os
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer
import h5py
from scipy.interpolate import interpn
from scipy.optimize import root, minimize
#from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, quad
import cProfile, pstats, io


class Base(object):
    
    def __init__(self):
        self.fs1 = []
        fs2 = []
        fs3 = []
        fs4 = []
        num_files = 5
        for n in range(num_files):
          # fs1.append(h5py.File('./Data/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
           # self.fs1.append(h5py.File('../Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
          # self.fs1.append(h5py.File('../../../../scratch/mjh1n20/Filtering_Data/KH/dp_400x800x0_'+str(n)+'.hdf5','r'))
            self.fs1.append(h5py.File('../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_998_1002/dp_400x800x0_'+str(n)+'.hdf5','r'))
          # self.fs1.append(h5py.File('../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_1998_2002/dp_400x800x0_'+str(n)+'.hdf5','r'))
          # self.fs1.append(h5py.File('../../../../scratch/mjh1n20/Filtering_Data/KH/Ideal/t_2998_3002/dp_400x800x0_'+str(n)+'.hdf5','r'))
          # fs1.append(h5py.File('./Data/KH/Shear/dp_400x800x0_'+str(n)+'.hdf5','r'))
        #   fs2.append(h5py.File('../Git/Plotting/BDNK/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
        #   fs2.append(h5py.File('../Git/Plotting/ISCE/KH/Ideal/dp_400x400x0_'+str(n)+'.hdf5','r'))
        #   fs3.append(h5py.File('../Git/Plotting/ISCE/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs4.append(h5py.File('../Git/Plotting/BDNK/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs3.append(h5py.File('../Git/Plotting/BDNK/KH/eta0_5em4/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs4.append(h5py.File('../Git/Plotting/ISCE/KH/Shear/tau5em2eta5em3/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs2.append(h5py.File('../Git/Plotting/ISCE/KH/Shear/dp_800x800x0_'+str(n)+'.hdf5','r'))
        #   fs2.append(h5py.File('../Git/Plotting/ISCE/KH/Ideal/tau5em3eta5em4/dp_400x400x0_'+str(n)+'.hdf5','r'))
        #   fs2.append(h5py.File('../Git/Plotting/ISCE/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs3.append(h5py.File('../Git/Plotting/IS/KH/Ideal/dp_800x800x0_'+str(n)+'.hdf5','r'))
        #   fs4.append(h5py.File('../Git/Plotting/IS/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
        #   fs3.append(h5py.File('ISCE/Rotor/lowres/data_serial_'+str(n)+'.hdf5','r'))
        #   fs4.append(h5py.File('ISCE/Rotor/long_highres/data_serial_'+str(n)+'.hdf5','r'))
        # fss = [fs1, fs2, fs3, fs4]
        # fss = [fs1]
        # nx = ny = 200
        nx, ny = 400, 800
        nts = num_files
        ts = np.linspace(9.98,10.02,nts) # Need to actually get these
        xs = np.linspace(-0.5,0.5,nx) # These too...
        ys =  np.linspace(-1.0,1.0,ny)
        # X, Y = np.meshgrid(xs,ys)
        self.points = (ts,xs,ys)
        self.dx = (xs[-1] - xs[0])/nx
        self.dy = (ys[-1] - ys[0])/ny
        self.vxs = np.zeros((num_files, nx, ny))
        self.vys = np.zeros((num_files, nx, ny))
        self.ns = np.zeros((num_files, nx, ny))
        for counter in range(num_files):
            self.vxs[counter] = self.fs1[counter]['Primitive/v1'][:]
            self.vys[counter] = self.fs1[counter]['Primitive/v2'][:]
            self.ns[counter] = self.fs1[counter]['Primitive/n'][:]

    def u_n_values_from_hdf5(self, t_n,i,j):
    #     t_n = np.where(ts[:]==t)[0][0]
        return [self.fs1[t_n]['Auxiliary/W'][i,j], self.fs1[t_n]['Primitive/v1'][i,j], self.fs1[t_n]['Primitive/v2'][i,j]], self.fs1[t_n]['Primitive/n'][i,j]
    
    # gives the fluid's 4-velocity at any point in time and space
    def interpolate_u_n_coords(self, t,x,y):
        point = (t,x,y)
        n_interpd = interpn(self.points,self.ns,point)
        vx_interpd = interpn(self.points,self.vxs,point)
        vy_interpd = interpn(self.points,self.vys,point)
        W_interpd = 1/np.sqrt(1 - (vx_interpd**2 + vy_interpd**2))
        u_interpd = W_interpd, vx_interpd, vy_interpd
        return [u_interpd[0][0], u_interpd[1][0], u_interpd[2][0]], n_interpd[0]
    
    def interpolate_u_n_point(self, point):
        n_interpd = interpn(self.points,self.ns,point)
        vx_interpd = interpn(self.points,self.vxs,point)
        vy_interpd = interpn(self.points,self.vys,point)
        W_interpd = 1/np.sqrt(1 - (vx_interpd**2 + vy_interpd**2))
        u_interpd = W_interpd, vx_interpd, vy_interpd
        return [u_interpd[0][0], u_interpd[1][0], u_interpd[2][0]], n_interpd[0]
    
    def Mink_dot(self,vec1,vec2):
        dot = -vec1[0]*vec2[0] # time component
        for i in range(1,len(vec1)):
            dot += vec1[i]*vec2[i] # spatial components
        return dot
    
    
    def get_U_mu(self, Vx_Vy):
        # get observer U from observer Vx, Vy
        Vx, Vy = Vx_Vy[0], Vx_Vy[1]
        W = 1/np.sqrt(1-Vx**2-Vy**2)
        U_mu = [W,Vx,Vy]
        return U_mu
    
    def get_U_mu_MagTheta(self, Vmag_Vtheta):
        Vmag, Vtheta = Vmag_Vtheta[0], Vmag_Vtheta[1]
        return self.get_U_mu([Vmag*np.cos(Vtheta),Vmag*np.sin(Vtheta)])
    
    def construct_tetrad(self, U):
        e_x = np.array([0.0,1.0,0.0]) # 1 + 2D
        E_x = e_x + np.multiply(self.Mink_dot(U,e_x),U)
        E_x = E_x / np.sqrt(self.Mink_dot(E_x,E_x)) # normalization
        e_y = np.array([0.0,0.0,1.0])
        E_y = e_y + np.multiply(self.Mink_dot(U,e_y),U) - np.multiply(self.Mink_dot(E_x,e_y),E_x)
        E_y = E_y / np.sqrt(self.Mink_dot(E_y,E_y))
        return E_x, E_y
        
    def find_boundary_pts(self, E_x,E_y,P,L):
        c1 = P + (L/2)*(E_x + E_y)
        c2 = P + (L/2)*(E_x - E_y)
        c3 = P + (L/2)*(-E_x - E_y)
        c4 = P + (L/2)*(-E_x + E_y)
        corners = [c1,c2,c3,c4]
        return corners
    
    def residual(self, V0_V1,P,L):
        U = self.get_U_mu(V0_V1)
        # U = get_U_mu_MagTheta(V0_V1)
        E_x, E_y = self.construct_tetrad(U)
        corners = self.find_boundary_pts(E_x,E_y,P,L)
        flux = 0
        for i in range(3,-1,-1):
            surface = np.linspace(corners[i],corners[i-1],10)
            for coords in surface:
                u, n = self.interpolate_u_n_point(coords)
                n_mu = np.multiply(u,n) # construct particle drift
    #             print(n_mu,U)
    #             print(Mink_dot(n_mu,U))
                if i == 3:
                    flux += self.Mink_dot(n_mu,-E_x) # Project wrt orthonormal tetrad and sum (surface integral)
                elif i == 2:
                    flux += self.Mink_dot(n_mu,-E_y) 
                elif i == 1:
                    flux += self.Mink_dot(n_mu,E_x)
                elif i == 0:
                    flux += self.Mink_dot(n_mu,E_y)
        return abs(flux) # **2 for minimization rather than r-f'ing?
    
    
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
    
    def find_observers(self, t_range,x_range,y_range,L,n_ts,n_xs,n_ys,initial_guess):
        t_coords = np.linspace(t_range[0],t_range[-1],n_ts)
        x_coords = np.linspace(x_range[0],x_range[-1],n_xs)
        y_coords = np.linspace(y_range[0],y_range[-1],n_ys)
        funs = []
        vectors = []
        coord_list = []
        for t in t_coords:
            for x in x_coords:
                for y in y_coords:
                    u, n = self.interpolate_u_n_coords(t,x,y)
                    # guess_vx_vy = initial_guess
                    guess_vx_vy = [u[1]/u[0], u[2]/u[0]]
                    coords = [t,x,y]
                    sol = minimize(self.residual_ib,x0=guess_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
                    vectors.append(self.get_U_mu(sol.x))
                    funs.append(sol.fun)
                    coord_list.append(coords)
                    #guess_vx_vy = [sol.x[0],sol.x[1]]
                    #try:
                     #   sol = minimize(self.residual_ib,x0=guess_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=1e-6)#,method='CG')
                      #  vectors.append(self.get_U_mu(sol.x))
                       # funs.append(sol.fun)
                     #   coord_list.append(coords)
                     #   #guess_vx_vy = [sol.x[0],sol.x[1]]
                     #   if (sol.fun > 1e-5):
                     #       print("Warning! Residual is large: ",sol.fun)
                    #except:
                    #    print("Failed for ",coords)
                    #finally:
                    #    pass
        # f_to_write.write(str(coord_list)+str(vectors)+str(funs))
        # DON'T THINK THIS WAS DOING ANYTHING??
        # with open('KH_observers.pickle', 'wb') as handle:
        #     pickle.dump(np.array([coord_list, vectors, funs]), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return [coord_list, vectors, funs]
    
    
    def profile(self, fnc):
        """A decorator that uses cProfile to profile a function"""
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval
        return inner
