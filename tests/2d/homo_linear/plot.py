#!/usr/bin/env python
import numpy
import matplotlib.pyplot as plt
from FDSolverPy.diffusion.DiffSolver import *

# plotting canvas
fig, axs = plt.subplots(1,3,figsize=(18,6))


# load/plot
for i in range(2):

    # load results
    calc, c, q, j = read_diffsolver(f'Q_{i}')
    dF_dc = np.ones_like(calc.c)
    F,err = calc.dF(calc.c,dF_dc)
    print(err,dF_dc.shape)

    # plotting
    axs[i+1].imshow(c.T,origin='lower',cmap='Blues')
    #axs[i+1].plot(c[:,50])

# plot diffusivity
axs[0].imshow(calc.d[...,0,0].T,origin='lower')

plt.show()

