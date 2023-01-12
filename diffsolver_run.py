#!/usr/bin/env python 
from optparse import OptionParser
from diffusion.DiffSolver import *

parser = OptionParser(usage='usage: executable to run DiffSolver calculations with user supplied run parameters')
parser.add_option('-o','--outdir',dest='outdir',default='./data',\
                   help='directory for calculated outputs, default=%default')
parser.add_option('-e','--etol',dest='etol',type='float',default=1e-4,\
                  help='tolerance for total energy, default=%default')
parser.add_option('-f','--ftol',dest='ftol',type='float',default=1e-4,\
                  help='tolerance for max force, default=%default')
parser.add_option('-n','--ndump',dest='step',type='int',default=10,\
                  help="dumps every 'step' iterations, default=%default")
parser.add_option('-N','--Ntotal',dest='Nstep',type='int',default=100,\
                  help="total number of iterations, default=%default")

(options,args) = parser.parse_args()

# set calculator
dct = read_diffsolver_args('./diff_solver.pckl')
calc = diff_solver(**dct)
calc.run(**options.__dict__)
