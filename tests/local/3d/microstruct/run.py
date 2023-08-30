#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# loads microstructure data: ns, Ls, gphi, z_active, eulers
locals().update(np.load('../../../microstruct.npz'))
# cutting size in half to save time
gphi = gphi[:72,:72,:72]
ns = [72,72,72]
gd = Grid(ns=ns,Ls=Ls)
dim = len(ns)

# create diffusivity field
db,dgb = np.eye(dim), np.eye(dim)*10
d = np.einsum('abc,ij->abcij',gphi,db)+\
    np.einsum('abc,ij->abcij',1-gphi,dgb)

# initialize fields
Qs = np.eye(dim)
Cs = [macro_C0(gd,Q) for Q in Qs]

# run
cwd = os.getcwd()
for i,C in enumerate(Cs):
    # write inputs
    write_inputs(f'Q_{i}',{'Xs':gd.xs},C,d)

    # initialize calculator
    os.chdir(f'Q_{i}')
    calc = diff_solver(**read_diffsolver_args())
    
    # normalize parameters
    d_mean, F_max = normalize_parameters(calc)

    # calculation
    calc.run(etol=1e-5,ftol=F_max*1e-2,
             ls_args={'tol':1e-2},
             Nstep=400,step=50)
    os.chdir(cwd)
    
