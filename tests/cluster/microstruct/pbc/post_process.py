from FDSolverPy.diffusion.DiffSolver_pbc import *
import os
cwd = os.getcwd()
for dd in range(2,5):
    os.chdir(f'd_{dd}')
    D = calculate_D()
    print(D)
    os.chdir(cwd)
