#!/usr/bin/env python
import json, sys
from argparse import ArgumentParser
from FDSolverPy.diffusion.DiffSolver import *
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
parser.add_argument('-r','--restart',dest='restart',default=False,action='store_true',\
                  help='whether or not restarting from previous run, default=%(default)')
parser.add_argument('-nn','--normalize',dest='normalize',default=True,action='store_true',\
                    help='whether or not normalize diffusivity for \
                          better numerical precision. This will change ftol --> ftol*F_max. default=%(default)s')
parser.add_argument('-dim','--dimension',dest='dimension',default=3,type=int,\
                    help='the dimension of the problem e.g. 1-, 2-, or 3-d. default=%(default)s' )
parser.add_argument('-pbc','--pbc',dest='pbc',default=False,action='store_true')
parser.add_argument('-npbc','--non-pbc',dest='pbc',default=False,action='store_false')
parser.add_argument('-clean','--clean_old',dest='clean_old',action='store_true')

# error message
if len(sys.argv)==100:
    parser.print_help(sys.stderr)
    sys.exit(1)

# setup arguments 
run_args = vars(parser.parse_args())
normalize = run_args['normalize']
dim = run_args['dimension']
ftol = run_args['ftol']
pbc = run_args['pbc']
del run_args['normalize'], run_args['dimension'], run_args['pbc']

cwd = os.getcwd()
if rank==0:
    print(cwd)
Qfs = [f'Q_{i}' for i in range(dim)]
for Qf in Qfs:
    os.chdir(Qf)
    run = True
    
    # initialize calculators
    dat = read_diffsolver_args()
    if pbc: 
        calc = diff_solver_pbc(**read_diffsolver_args())
    else:
        calc = diff_solver(**read_diffsolver_args())
    
    # normalize parameters for numerical precision?
    if normalize:
        d_mean, F_max = normalize_parameters(calc)
        #run_args['ftol'] = ftol*F_max
    
    # if restarting, check if etol, ftol are met
    if run_args['restart']:
        # check if macro_vars.json file exists.
        # Should exist if properly finished previous run
        macro_vars_file = 'data/macro_vars.json'
        if os.path.isfile(macro_vars_file):
            output_vars = json.load(open(macro_vars_file,'r'))
            if output_vars['etol_current']<=run_args['etol'] and\
                output_vars['ftol_current']<=run_args['ftol']:
                calc.parprint(f'current ftol={output_vars["ftol_current"]}, while ftol_target={run_args["ftol"]}')
                run = False

    # run calculation
    if run: calc.run(**run_args)
    os.chdir(cwd)

if calc.rank==0:
    calculate_D(Qfs)
