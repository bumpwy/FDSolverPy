#!/usr/bin/env python
# run script for clusters
from FDSolverPy.tools.job_scripts import *

##### slurm job allocation #####
alloc = {'A':'hhitemp','n':64,\
         'p':'pdebug','t':'0:10:00',\
         'o':'output'}
env = {}
run_script = '~/lib/python-packages/FDSolverPy/bin/run_d_eff.py'
run = f'srun python -u {run_script}'+\
       ''' -N 400 -s 100 -e 1e-5 -f 5e-3 -l '{"t0":1e-2,"tol":1e-4}' -nn -pbc -clean'''
##### slurm job allocation #####


##### file directories / parameters to iterate #####
# - Loop over grain size d
for dd in range(2,5):
    sub_path = f'd_{dd}'
    alloc['J'] = sub_path
    
    submit_job(sub_path,'job.sh',alloc,env,run)
    
