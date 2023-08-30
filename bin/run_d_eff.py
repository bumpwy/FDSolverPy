#!/usr/bin/env python
import json, sys
from distutils.util import strtobool
from argparse import ArgumentParser
from FDSolverPy.diffusion.DiffSolver import *

# Parse arguments.
def bool_arg_type(x):
    return bool(strtobool(x))

parser = ArgumentParser(usage='usage: run calculation for effective diffusivity with user supplied options')
parser.add_argument('-e','--etol',dest='etol',default=1e-5,type=float,\
                  help='energy tolerance, default: %(default).2e')
parser.add_argument('-f','--ftol',dest='ftol',default=1e-2,type=float,\
                  help='force max tolerance, default=%(default).2e')
parser.add_argument('-N','--Nstep',dest='Nstep',default=100,type=int,\
                  help='total number of steps, default=%(default)i')
parser.add_argument('-s','--step',dest='step',default=10,type=int,\
                  help='dumping interval, default=%(default)i')
parser.add_argument('-l','--ls_args',dest='ls_args',default='{"t0":1e-2,"tol":1e-5}',type=json.loads,
                  help='line search arguments including \
                        trial step size t0 and tolerance tol, default: %(default)s')
parser.add_argument('-r','--restart',dest='restart',default=False,type=bool_arg_type,\
                  help='whether or not restarting from previous run, default=%(default)s')
parser.add_argument('-nn','--normalize',dest='normalize',default=True,type=bool_arg_type,\
                    help='whether or not normalize diffusivity for \
                          better numerical precision. This will change ftol --> ftol*F_max. default=%(default)s')
parser.add_argument('-dim','--dimension',dest='dimension',default=3,type=int,\
                    help='the dimension of the problem e.g. 1-, 2-, or 3-d. default=%(default)s' )

# error message
if len(sys.argv)==100:
    parser.print_help(sys.stderr)
    sys.exit(1)

# setup arguments 
run_args = vars(parser.parse_args())
normalize = run_args['normalize']
dim = run_args['dimension']
ftol = run_args['ftol']
del run_args['normalize'], run_args['dimension']

cwd = os.getcwd()
for i in range(dim):
    path = f'Q_{i}'
    os.chdir(path)
    
    # initialize calculators
    dat = read_diffsolver_args()
    calc = diff_solver(**read_diffsolver_args())

    # normalize parameters for numerical precision?
    if normalize:
        d_mean, F_max = normalize_parameters(calc)
        run_args['ftol'] = ftol*F_max

    # run calculation
    calc.run(**run_args)
    os.chdir(cwd)