#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# define grid
ns,Ls = [128,128],[1,1]
gd = Grid(ns=ns,Ls=Ls)

# create diffusivity field
d0 = np.eye(2)*2.5
d = np.einsum('ab,ij->abij',np.ones(gd.ns),d0)

# initialize fields
np.random.seed(42)
Qs = np.random.rand(2,2)
Cs = [macro_C0(gd,Q) for Q in Qs]

# run
cwd = os.getcwd()
for i,C in enumerate(Cs):
    # write inputs
    write_inputs(f'Q_{i}',{'GD':gd},C,d)

    # initialize calculator
    os.chdir(f'Q_{i}')
    calc = diff_solver(**read_diffsolver_args())
    
    # calculation
    calc.run(etol=1e-5,ftol=1e-4,
             ls_args={'tol':1e-3},
             Nstep=200,step=10)
    os.chdir(cwd)
    
