# run script for local machine
import subprocess, os
from FDSolverPy.diffusion.DiffSolver import *

cwd = os.getcwd()
for dd in range(2,5):
    for i in range(3):
        os.chdir(f'd_{dd}/Q_{i}')
        # create calculator
        calc = diff_solver(**read_diffsolver_args())
        # normalize parameters
        d_mean, F_max = normalize_parameters(calc)
        # run calculation
        calc.run(Nstep=1,ftol=1e-2*F_max)
        # return
        os.chdir(cwd)
