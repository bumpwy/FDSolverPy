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

write_d_eff_inputs('./',d,gd)

