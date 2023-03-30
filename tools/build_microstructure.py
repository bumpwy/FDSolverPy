#!/usr/bin/env python 
import numpy as np
from FDSolverPy.math.convolve import *
from FDSolverPy.math.space import Grid
from scipy.signal import oaconvolve

# creates interface phase from gaussian smearing
def create_interface(phis,width,grid,pbc=True):

    # reduce width to gaussian-sigma value
    sigma = width/2.25

    # create gaussian kernel
    dxs = grid.dxs
    wns = [int(width/dx)*4 for dx in dxs]
    ls = [dx*wn for dx,wn in zip(dxs,wns)]
    gd = Grid(ns=wns,Ls=ls)
    r2 = np.sum([(xx-c)**2 for xx,c in zip(gd.xxs,gd.center)],axis=0)
    w = np.exp(-r2/2/sigma**2)
    w /= np.sum(w)

    # convolve the phi's with gaussian kernel
    if pbc:
        zetas = np.array([convolve_fft(phi,w) for phi in phis])**2
    else:
        zetas = np.array([oaconvolve(phi,w,mode='full') for phi in phis])**2
    z_sum = np.sum(zetas,axis=0)

    # create the final field
    gphi = (z_sum - z_sum.min())/(z_sum.max()-z_sum.min())

    return gphi

