#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# define grid
ns,Ls = [256,256],[1,1]
gd = Grid(ns=ns,Ls=Ls)

# create diffusivity field
d0 = np.eye(2)*2.5
d = np.einsum('ab,ij->abij',np.ones(gd.ns),d0)

# initialize fields
np.random.seed(42)
Qs = np.array([[2,3],\
               [7,9]])
Cs = [np.cos(Q[0]*np.pi*gd.xxs[0])*np.cos(Q[1]*np.pi*gd.xxs[1]) for Q in Qs]

# run
cwd = os.getcwd()
for i,C in enumerate(Cs):
    # write inputs
    write_inputs(f'Q_{i}',{'GD':gd,'ghost':2},C,d)

    # initialize calculator
    os.chdir(f'Q_{i}')
    calc = diff_solver(**read_diffsolver_args())
    
    # calculation
    calc.run(etol=1e-5,ftol=1e-4,
             ls_args={'t0':1e-6,'tol':1e-4},
             Nstep=400,step=20)
    os.chdir(cwd)
    
