#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# loads microstructure data: ns, Ls, gphi, z_active, eulers
locals().update(np.load('../../microstruct.npz'))
ns,Ls = ns[:2],Ls[:2]
gphi = gphi[:,:,0]     # take a slice from a 3-d microstructure
gd = Grid(ns=ns,Ls=Ls)

# create diffusivity field
db,dgb = np.eye(2), np.eye(2)*10
d = np.einsum('ab,ij->abij',gphi,db)+\
    np.einsum('ab,ij->abij',1-gphi,dgb)

# initialize fields
Qs = np.eye(2)
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
             Nstep=400,step=20)
    os.chdir(cwd)
    
