#!/usr/bin/env python
import numpy
from FDSolverPy.diffusion.DiffSolver import *

paths = ['Q_0','Q_1','Q_2']
D = calculate_D(paths)

print(D)
