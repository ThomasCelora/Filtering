# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:59:20 2022

@author: mjh1n20
"""


# from system.Setup import setup

if __name__ == '__main__':

    from multiprocessing import Process, Pool
    import os
    import numpy as np
#    import matplotlib.pyplot as plt
    import pickle
    from timeit import default_timer as timer
    import h5py
    from scipy.interpolate import interpn
    from scipy.optimize import root, minimize
#    from mpl_toolkits.mplot3d import Axes3D
    from scipy.integrate import solve_ivp, quad
    import cProfile, pstats, io
    # from system.BaseFunctionality import *
    import system.BaseFunctionality
    from system.BaseFunctionality import Base
    from functools import partial
    from itertools import repeat
    
    system = Base()

    # cProfile.run('fnc()')                    

#     test_vx_vy = [0.0,0.0] # vx, vy
#     test_vmag_vtheta = [0.0,0.0] # vmag, vtheta
#     coords = [6.0,0.0,0.0]
#     L = 0.01
#     tol = 1e-6
    
#     start = timer()
#     sol = minimize(residual_ib,x0=test_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=tol)#,method='CG')
# #     sol = minimize(residual,x0=test_vx_vy,args=(coords,L),bounds=((-0.7,0.7),(-0.7,0.7)),tol=tol)#,method='CG')
# #     sol = minimize(residual_ib,x0=test_vmag_vtheta,args=(coords,L),bounds=((0.0,0.9),(-1.5*np.pi, 1.5*np.pi)),tol=tol)#,method='CG')
#     print("time, mins:", (end - start)/60) # Time in seconds, e.g. 5.38091952400282
#     print("root-found: ",get_U_mu(sol.x))
#     print("blabla")

    t_range = [6.0,6.0]
    x_range = [0.0,0.0]
    y_range = [0.0,0.0]
    initial_guess = [-0.38,0.0]
    L = 0.01
    f_rts = open("runtimes.txt", "a")
    f_obs = open("observers.txt", "a")
    args = [([tr, tr], [xr, xr], [yr, yr], L, 1, 1, 1, initial_guess) for tr in np.linspace(0.0, 15.0, 16) for xr in np.linspace(-0.4, 0.4, 41) for yr in np.linspace(-0.2, 0.2, 21)]
    # KH_observers = system.find_observers(t_range,x_range,y_range,L,1,1,1,initial_guess)
    # print(KH_observers)
    # with open('KH_observers.pickle', 'wb') as handle:
    #     pickle.dump(KH_observers, handle, protocol=pickle.HIGHEST_PROTOCOL)


    start = timer()
    with Pool(2) as p:
        # f_obs.write(str(p.starmap(system.find_observers, args)))
        f_obs.write(str(p.starmap(system.find_observers, args)))

    mid = timer()
    print("time, mins:", (mid - start)/60,"\n") # Time in seconds, e.g. 5.38091952400282
    f_rts.write("Parallel time, mins:"+ str((mid - start)/60) + "\n") # Time in seconds, e.g. 5.38091952400282
    
#    system.find_observers(t_range1, x_range,y_range,L,1,1,1,initial_guess)
#    system.find_observers(t_range2, x_range,y_range,L,1,1,1,initial_guess)
#    print(system.find_observers(t_range3, x_range,y_range,L,1,1,1,initial_guess))
#    print(system.find_observers(t_range4, x_range,y_range,L,1,1,1,initial_guess))
#    print(system.find_observers(t_range5, x_range,y_range,L,1,1,1,initial_guess))

    end = timer()
    print("time, mins:", (end - mid)/60,"\n") # Time in seconds, e.g. 5.38091952400282
    f_rts.write("Serial time, mins:"+str((end - mid)/60) + "\n") # Time in seconds, e.g. 5.38091952400282
    f_rts.close()
    f_obs.close()
        




