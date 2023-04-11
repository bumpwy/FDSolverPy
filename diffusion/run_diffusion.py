#!/usr/bin/env python
import json, sys
from argparse import ArgumentParser
from FDSolverPy.diffusion.DiffSolver import *

# Parse arguments.
parser = ArgumentParser(usage='usage: run diffusion calculation with user supplied options')
parser.add_argument('-e','--etol',dest='etol',default=1e-5,type=float,\
                  help='energy tolerance, default: %(default).2e')
parser.add_argument('-f','--ftol',dest='ftol',default=1e-5,type=float,\
                  help='force max tolerance, default=%(default).2e')
parser.add_argument('-N','--Nstep',dest='Nstep',default=100,type=int,\
                  help='total number of steps, default=%(default)i')
parser.add_argument('-s','--step',dest='step',default=10,type=int,\
                  help='dumping interval, default=%(default)i')
parser.add_argument('-l','--ls_args',dest='ls_args',default='{"t0":1e-2,"tol":1e-5}',type=json.loads,
                  help='line search arguments including \
                        trial step size t0 and tolerance tol, default: %(default)s')
parser.add_argument('-r','--restart',dest='restart',default=False,type=bool,\
                  help='whether or not restarting from previous run, default=%(default)s')

# error message
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# create calculator object
args = vars(parser.parse_args())
calc = diff_solver(**read_diffsolver_args())

# for record, I'd also create a run.py file in the directory
if calc.rank == 0:
    f = open('run.py','wt')
    f.write('''#!/usr/bin/env python
from FDSolverPy.diffusion.DiffSolver import *
calc = diffsolver(**read_diffsolver_args())\n''')
    ss = 'calc.run('
    arg_strs = f',\n{" "*len(ss)}'.join([f'{key} = {value}' for key, value in args.items()])
    f.write(ss+arg_strs+')\n\n')
    f.close()

# run calculation
calc.run(**args)
