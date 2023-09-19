#!/usr/bin/env python
import numpy as np
import os
from FDSolverPy.diffusion.DiffSolver import *
from FDSolverPy.math.space import *

# loads microstructure data: ns, Ls, gphi, z_active, eulers
locals().update(np.load('../../microstruct.npz'))
gd = Grid(ns=ns,Ls=Ls)
dim = len(ns)


for dd in range(2,5):
    # create diffusivity field
    db,dgb = np.eye(dim), np.eye(dim)*dd
    d = np.einsum('abc,ij->abcij',gphi,db)+\
        np.einsum('abc,ij->abcij',1-gphi,dgb)

    # write inputs
    path = f'd_{dd}'
    write_d_eff_inputs(path,d,gd)
    

