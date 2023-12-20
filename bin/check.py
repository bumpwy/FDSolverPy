#!/usr/bin/env python
import json, os, sys
import numpy as np
import itertools as it
from FDSolverPy.diffusion.DiffSolver import *

# import path iterators
if len(sys.argv)>1:
    it_file = sys.argv[1]
else:
    it_file = 'iterators.json'
iterators = json.load(open(it_file,'r'))
combos = list(it.product(*iterators.values()))

Total,complete = 0,0
for combo in combos:
    path = os.path.join(*combo)
    locals().update(check_d_eff_outputs(path))
    Total += N
    complete += completion
    print(f'{path}...{completion}/{N}')
print(f'In total, {complete}/{Total} calculations completed')


