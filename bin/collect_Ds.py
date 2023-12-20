#!/usr/bin/env python
import json, os, sys
import itertools as it
import numpy as np

# build nested dictionary
def collect_d(dct,path,it_levels):
    lev = next(it_levels,-1)
    if lev==-1:
        d_eff_file = os.path.join(path,'D_eff.json')
        if os.path.isfile(d_eff_file):
            dat = json.load(open(d_eff_file,'r'))
            dct['Ds'] = dat['Ds']
            dct['D_par'] = dat['D_par']
            dct['D_ser'] = dat['D_ser']
            return 1
        else:
            return 0
    else:
        path = os.path.join(path,lev)
        if lev not in dct:
            dct[lev]={}
        return collect_d(dct[lev],path,it_levels)

# import path iterators
if len(sys.argv)>1:
    it_file = sys.argv[1]
else:
    it_file = 'iterators.json'
iterators = json.load(open(it_file,'r'))
combos = list(it.product(*iterators.values()))

dct = {}
empties = []
for combo in combos:
    path = os.path.join(*combo)
    d_eff_path = os.path.join(path,'D_eff.json')

    val = collect_d(dct,'./',iter(combo))
    if val==0:
        empties.append(path)
# print status
if empties==[]:
    print('all calculations completed')
else:
    print('The following calculations are incomplete')
    list(map(print,empties))
json.dump(dct,open('D_eff_all.json','w'),indent=4)



