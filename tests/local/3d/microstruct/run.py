#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# initialize fields
Qs = np.eye(3)

# run
cwd = os.getcwd()
for i,Q in enumerate(Qs):

    # initialize calculator
    os.chdir(f'Q_{i}')
    calc = diff_solver(**read_diffsolver_args())
    
    # normalize parameters
    d_mean, F_max = normalize_parameters(calc)

    # calculation
    calc.run(etol=1e-5,ftol=F_max*1e-2,
             ls_args={'tol':1e-2},
             Nstep=400,step=50,restart=True,clean_old=True)
    os.chdir(cwd)
    
